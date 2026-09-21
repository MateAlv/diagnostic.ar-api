# Authenticating /extract-hpo

**Decision (Mate, 2026-09-21): `/extract-hpo` should require a JWT, like every
other call.** This records what that means in practice, because the endpoint is
not currently shaped like the calls that already do it.

## Why it has to happen before the recognizer ships

Today the gateway has no authentication at all, and the browser calls it
directly: `frontend/src/app/nuevo/page.tsx:100` builds the URL from
`NEXT_PUBLIC_HPO_API_URL` and fetches it from the client.

That is survivable while extraction runs a local Ollama model, where an
abusive caller costs CPU on a machine we own. The Spanish recognizer replaces
that with a paid LLM call per mention, which turns an anonymous endpoint into
an open billing hole. **Auth is a prerequisite for the swap, not a follow-up.**

## The complication: the JWT is issued by a different service

| | Issues / verifies JWTs | Language |
|---|---|---|
| Backend | yes — `backend/src/services/authService.ts` | Node / Express |
| HPO gateway | no | Python / FastAPI |

`backend/src/middleware/authMiddleware.ts` accepts a token from either an
`Authorization: Bearer` header or an `access_token` cookie, verifies it with
`verifyAccessToken`, and — for the routes that need it — re-reads the user from
the database so a demoted or logged-out user loses access immediately rather
than when their token expires.

So "use JWT like every other call" is not a one-line addition to FastAPI: the
gateway would need the signing secret and a compatible verifier, and it has
neither.

## Two ways to do it

### A. Proxy through the backend (recommended)

The browser stops calling the gateway. It calls the Node backend, which already
authenticates, and the backend calls the gateway over the internal network.

- Reuses `authenticate` exactly as written — no second implementation of token
  verification, no shared secret in a second service, no drift between them.
- The gateway stops being publicly reachable at all, which is the stronger
  property: it is not just authenticated, it is not exposed.
- Gives one place to enforce a per-user rate limit, which matters once each
  call costs money. The recognizer's own limiter protects the provider budget;
  it does not identify who is spending it.
- Costs a frontend change (`NEXT_PUBLIC_HPO_API_URL` → the backend route) and a
  new backend route. The 10k `text_es` cap and CSRF behaviour the mock already
  enforces would apply to it like any other backend route.

### B. Verify the JWT inside the gateway, with a public key (Mate's preference)

**Decision (Mate, 2026-09-21): sign in the backend with a private key, verify
in the gateway with the public half.** This is the right shape for verifying
in a second service, and it removes the main objection to doing so — the
gateway never holds anything that can mint a token.

It needs one change first. **The backend signs with HS256 today**
(`backend/src/services/authService.ts:22,33`), which is symmetric: there is no
keypair, only `JWT_ACCESS_SECRET`, and whatever can verify can also sign.
Copying that secret to brainy would give the GPU node the ability to issue
tokens for any user, including an admin. So the flow Mate wants requires
moving access tokens to **RS256** first.

What that involves:

1. Generate an RSA keypair. The private key stays in the backend's `.env`; the
   public key ships to the gateway and is not a secret.
2. `signAccessToken` switches to `algorithm: 'RS256'` with the private key;
   `verifyAccessToken` to the public key. Both are single lines in
   `authService.ts`, and the payload (`sub`, `role`, `tokenVersion`) is
   unchanged.
3. The gateway verifies with `PyJWT` and the public key, pinning
   `algorithms=["RS256"]`. **Pinning matters:** a verifier that accepts the
   algorithm named in the token can be handed `alg: none`, or an HS256 token
   signed with the public key as if it were a shared secret. Both are standard
   JWT attacks and both are defeated by naming the algorithm explicitly.
4. Existing sessions carry HS256 tokens and would fail verification at the
   cutover. Either accept that everyone signs in again, or verify both
   algorithms in the backend for one token lifetime — the gateway should still
   only ever accept RS256.

Refresh tokens can stay HS256: they are only ever seen by the backend.

What this still does not give you, and A does:

- **Revocation.** `requireFreshAccessToken` re-reads `token_version` from the
  database, so a logged-out or demoted user loses access immediately. The
  gateway has no database, so a stolen token stays valid there until it
  expires. Keeping access-token lifetime short is the mitigation.
- **A place to rate-limit per user.** The gateway can do it per `sub` once it
  verifies tokens, but it has no shared state, so a second replica would keep
  its own counters.

### C. (Original B) Verify with the shared HS256 secret

Mentioned only to rule it out: the signing secret has to reach a second
service, the verification logic exists twice in two languages, and the
database-backed freshness check does
  not — a revoked token would keep working against the gateway until expiry.

**B (RS256) is what Mate wants and is sound**, provided the algorithm is
pinned on the verifying side and the keypair migration happens first. Its one
real gap against A is revocation: the gateway cannot check `token_version`, so
a stolen or logged-out token works there until it expires. Short access-token
lifetimes are the mitigation.

A remains the stronger shape on the merits — one verifier, immediate
revocation, and the gateway not publicly reachable at all — and the two are
not exclusive: RS256 verification in the gateway is worth having even behind a
proxying backend, as defence in depth for a service on a different host.

## Either way

- The endpoint answers 401 without a token and 403 with an invalid one, so it
  matches what the backend already returns and the frontend already handles.
- Decide what an anonymous caller gets. The wizard is usable before sign-up
  today; requiring a token changes that flow, and that is a product decision
  rather than a security one.

## Rate limiting without auth

Yes, and it is worth doing regardless of which option above lands: auth says
*who* may call, a rate limit says *how often*, and an authenticated user can
still run up a bill. These are independent controls.

Three that need no identity, strongest first:

### 1. Cache on the text hash — already built, costs nothing

The gateway already caches on a hash of the input (`app/cache.py`, keyed via
`utils.hash_text`). Resubmitting the same history is then free no matter who
sends it, which removes the cheapest abuse — a loop on one payload — without
knowing anything about the caller. Make sure the recognizer path goes through
the same cache rather than around it.

### 2. Per-IP limit, but read the forwarded header

Ordinary `slowapi`/`limits` middleware on FastAPI. One trap: the gateway
currently reads `http_request.client.host` (`main.py:177`), which **behind
Traefik is the proxy's address, not the caller's**. Every request looks like it
comes from one IP, so a limiter keyed on it throttles everyone at once or
nobody.

Traefik does forward the real address (`X-Forwarded-For`), so key the limiter
on the leftmost entry of that header *and only when the peer is the proxy* —
trusting the header unconditionally lets a caller spoof it and skip the limit.
The same fix improves the `source_ip` recorded in the audit log, which today
records Traefik on every row.

Caveat: an IP limit is coarse. A university or hospital NAT shares one address
between many legitimate users, which is a real population for this platform.
Set the ceiling by cost, not by what a single user plausibly needs.

### 3. A global concurrency cap — the actual spend ceiling

The bound that matters for billing is not requests per caller but LLM calls in
flight. A semaphore limiting concurrent extractions, with a queue and a fast
503 past a depth, caps worst-case spend whoever is calling and however many
IPs they have. This is the one that makes the endpoint *safe* rather than
merely *tidy*; the recognizer's own `RateLimitedProvider` already does the
per-process version of it.

### What none of these replace

A rate limit bounds the damage; it does not stop an anonymous stranger from
spending your Anthropic budget at the permitted rate. For a paid endpoint,
limits plus auth — in that order of urgency for cost, and with auth still
required before staging is publicly routable.

## Not yet implemented

This is a design note. The change spans `diagnostic.ar` (frontend + backend)
and this repo, and `diagnostic.ar` is checked out by another session, so
nothing here has been wired. Both repos also need Mate's sign-off before
staging, per the platform `CLAUDE.md`.
