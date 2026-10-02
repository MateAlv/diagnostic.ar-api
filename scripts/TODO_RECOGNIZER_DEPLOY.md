# TODO before USE_RECOGNIZER can be turned on

The recognizer path is merged and off by default. These are the things that
have to be settled before the flag is set anywhere, with the blocking one
first.

## 1. BLOCKER — the CLI providers cannot authenticate in a container

`RECOGNIZER_PROVIDER=claude-cli` shells out to the `claude` CLI, which
authenticates as a logged-in user through `~/.claude/.credentials.json` on the
machine it runs on. The gateway runs as a container on brainy, which has no
such file and no logged-in user. **It will fail on the first request**, not at
startup, so nothing catches it until someone submits a history.

`codex` has the same problem, plus two of its own: it refuses `claude-opus-5`
on a ChatGPT account, and its subscription quota ran out here after a handful
of calls.

Four ways out, none yet chosen:

| Option | Cost |
|---|---|
| Mount the credentials into the container | A user credential in a container image or volume; it also expires and needs re-issuing |
| Proxy LLM calls back to TARS | Makes a dev box load-bearing for production |
| `LLM_PROVIDER=anthropic` with an API key | Per-token billing, but it is the only one that is properly deployable |
| Bedrock | Same as above under an existing AWS account |

The provider seam means this is a config change rather than a code change:
`RECOGNIZER_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` set is all it takes.

Mate's current preference is the Claude subscription while the userbase is
small, which is reasonable for **local** use and does not resolve the deployed
case.

## 2. Latency does not fit a synchronous request

A single history measured 20-40s end to end on `claude-opus-5`, against ~18s
for the incumbent. The browser calls `/extract-hpo` directly and waits.

Batching the mapping stage was tried and made it worse -- short concurrent
calls beat one long one, 8.7s against 30.9s for five mentions -- so the cheap
fix is ruled out. What remains is an async job plus polling, or accepting the
wait with a clear progress state. That is a product decision.

## 3. Order of operations for staging

1. Deploy the backend with `JWT_ACCESS_PRIVATE_KEY`/`PUBLIC_KEY` set. Access
   tokens become RS256. Keep `JWT_ACCEPT_LEGACY_HS256=true` for one token
   lifetime (15m) so existing sessions are not signed out mid-use, then remove
   it.
2. Deploy the gateway with `JWT_PUBLIC_KEY` and `TRUSTED_PROXIES` set to
   Traefik's address. **`TRUSTED_PROXIES` is not optional**: without it the
   per-IP limiter keys on Traefik and throttles every caller as one.
3. Turn on `REQUIRE_AUTH=true`, then the limits, then `USE_RECOGNIZER=true`
   last. Each step is independently reversible.

Setting `REQUIRE_AUTH=true` before step 1 makes every request 401.

## 4. Smaller things

- **No revocation at the gateway.** It has no database, so `token_version` is
  unreadable and a logged-out user's token works there until it expires. Keep
  the access-token lifetime short.
- **Local dev cannot authenticate** against a real gateway: backend on :4000
  and gateway on :8000 are different origins, and the cookie is `sameSite=lax`.
  `npm run dev:mock` serves both on one port and is unaffected.
- **The recognizer's assets are not in git.** It needs `hp-es.babelon.tsv`,
  the ClinLinker encoder (485MB) and the HPO embedding batches (62MB). The
  encoder is downloaded at build time; the batches and the labels come ready
  made from a diagnostic.ar-model release (`hpo-embeddings-vX.Y.Z`, pinned in
  `services/api-gateway/embeddings.lock`, fetched by `fetch_embeddings.sh`).
  Computing them took ~80 s on TARS but ~11 min on a GitHub runner, on every
  build; the build now checks the package against its encoder, labels and
  generator script and refuses a mismatch.
- **Nothing has been measured against the incumbent.** The recognizer scored
  micro F1 0.676 on its own 8-history dataset; the incumbent was never scored,
  by decision, since the recognizer replaces it rather than having to beat it.

## 5. Staging runs it on nodo0, not on brainy (2026-09-24)

Staging and production reach **the same gateway** on brainy: both hpo-tunnels
point at nodo0:18100. Setting `USE_RECOGNIZER` there turns it on for
production. Staging therefore runs its own gateway, built from
`services/api-gateway/Dockerfile.recognizer`, as `hpo-recognizer` in the
`diagnosticar-staging` compose project on nodo0. A Traefik router takes only
`/hpo-api/extract-hpo`; search and specialties still go to brainy.

It needs no GPU: the LLM is remote and the ClinLinker encoder loads on CPU in
under a second, ~450 MB resident. The image is ~3 GB and is built on TARS,
because the recognizer's assets are not in git (see the Dockerfile header).
The recognizer code comes from the private copy
`MateAlv/spanish-symptoms-recognizer`.

Auth for staging is a subscription token from `claude setup-token`, passed as
`CLAUDE_CODE_OAUTH_TOKEN`. Mounting `~/.claude/.credentials.json` works too,
but the CLI refreshes and rewrites that file, and two machines sharing one
login can sign each other out.

Found while building it, all fixed on this branch:

- **The flag could not be turned on.** Swapping the whole pipeline left
  startup (`phenotyper.ensure_ready`), `/healthz`, `/info`, `/hpo/es/search`
  and the audit record calling attributes the recognizer does not have: the
  gateway would not boot, and past that every extraction would 500 in the
  audit `finally`. The incumbent now always serves those; the flag only
  chooses who answers `/extract-hpo`.
- **`PyJWT` was missing from `requirements.txt`**, so an image built from
  `main` fails importing `app/auth.py`.
- **`X-Forwarded-For` was read from the left**, which is whatever the caller
  sent. It is now read from the right, skipping trusted hops, and
  `TRUSTED_PROXIES` accepts CIDRs, since Traefik's address on a Docker network
  changes when its container is recreated.
- Startup now checks the provider has a credential and loads the encoder;
  `/healthz` reports `extract_hpo.ready` and the reason when it is not.
