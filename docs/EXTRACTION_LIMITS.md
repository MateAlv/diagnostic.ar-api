# Limits on free-text extraction

`/extract-hpo` is the one endpoint here that spends money: with the Spanish
symptom recognizer, every call runs several requests to a paid LLM. Five
layers decide whether a request gets that far, cheapest first:

| Layer | Setting | Refuses with | Keyed on |
|---|---|---|---|
| Authentication | `REQUIRE_AUTH`, `JWT_PUBLIC_KEY` | 401 | — |
| Per address | `RATE_LIMIT_PER_MINUTE` | 429 `Rate limit exceeded` | client IP (`TRUSTED_PROXIES`) |
| Per account | `USER_EXTRACTIONS_PER_HOUR`, `USER_EXTRACTIONS_PER_DAY` | 429 `{"error": "user-rate-limit"}` | the token's `sub` |
| Concurrency | `MAX_CONCURRENT_EXTRACTIONS`, `MAX_EXTRACTION_QUEUE` | 503 | — |
| **Global budget** | `GLOBAL_EXTRACTIONS_PER_5H`, `GLOBAL_EXTRACTIONS_PER_WEEK` | 429 `{"error": "global-budget"}` | the whole service |

Every setting defaults to 0, which disables it; each deployment sets its own.

**The global budget is the ceiling on spend.** The other layers are evaded:
addresses and accounts are free (sign-up on diagnostic.ar is open), and
concurrency bounds the rate, not the total. The budget counts an extraction
only once it holds a concurrency slot and is about to call the model, so a
request turned away earlier costs nothing against it.

Structured refusals carry `detail = {"error", "retry_after", "message"}` and a
`Retry-After` header. The wizard (`frontend/src/lib/extract-errors.ts` in
diagnostic.ar) shows a different message for each, and for both 429s points to
the symptom search, which spends nothing.

## Windows

Per account: a sliding hour and day. Global: a sliding 5 hours and 7 days,
the two windows a Claude subscription meters. Sliding, not fixed, so the
budget cannot be spent twice across a window boundary.

The global count survives restarts when `EXTRACTION_BUDGET_STATE` names a
writable file (staging: a named volume at `/var/lib/recognizer`); without it a
redeploy resets the week. If the file exists and cannot be read, the service
refuses to start rather than begin again from zero. Per-account and per-address
counts are in memory and reset on restart. All of it assumes one instance; more
than one needs the counts in Redis.

## What an extraction costs

Measured on staging on 2026-09-28, `claude-opus-5` through `claude -p`, the
recognizer's response cache off:

| History | Characters | LLM calls | Tokens (incl. cached) | API-equivalent USD | Time |
|---|---|---|---|---|---|
| short (3 findings) | 128 | 4 | 69,535 | 0.34 | 77 s |
| typical (8 findings) | 267 | 9 | 158,932 | 0.60 | 52 s |
| long (27 findings) | 1,202 | 29 | 509,961 | 1.89 | 82 s |

Mean **US$0.94** per extraction. The mapping stage makes one call per finding,
so cost grows with the length of the history.

**Most of those tokens are not the history.** `claude -p` sends Claude Code's
own system prompt with every call — about 17,000 tokens, mostly served from the
prompt cache but still counted — written for a coding agent, not for this.
Replacing it (`--system-prompt`, no tools) is likely the largest single saving
available, but it changes what the model sees, so the recognizer's quality has
to be measured again before and after.

## Staging: 40 per 5 hours on a Claude Max subscription

Staging runs on Mate's Claude Max (US$100) subscription token. His
requirements (2026-09-28): **at least 40 extractions per 5 hours**, and
ideally no more than 15% of the subscription's limits — **when the two
conflict, the 40 wins.** So staging runs `GLOBAL_EXTRACTIONS_PER_5H=40` and
`GLOBAL_EXTRACTIONS_PER_WEEK=1344` (the same rate over a week, 40 × 168 / 5,
which adds no limit of its own); `.env.staging` can override both
(`RECOGNIZER_GLOBAL_PER_5H`, `RECOGNIZER_GLOBAL_PER_WEEK`).

At the costs above, 40 extractions are about US$38 of API-equivalent usage
per 5 hours. Anthropic does not publish the subscription's limits in tokens,
so whether that is inside 15% can only be measured. If it is not, the lever is
the cost per extraction (the system prompt, above), not the cap.

### Calibration

1. Make sure nothing else uses the account for the next ~10 minutes —
   including Claude Code sessions, which count against the same limits.
2. Note the current session and weekly percentages on claude.ai → Settings →
   Usage.
3. From diagnostic.ar-api:
   ```bash
   ssh nodo0-diagnostic.ar 'docker exec -i diagnosticar-staging-hpo-recognizer-1 python -' \
       < scripts/calibrate_extraction_budget.py
   ```
   Five typical histories, called directly (the budget does not stop them),
   with token usage per call.
4. Note both percentages again. With `s` and `w` the session and weekly
   increases divided by 5, 40 extractions use `40 × s` % of a session and a
   full week at that rate `1344 × w` %. Compare with 15%: at or under, nothing
   to do; over, reduce the cost per extraction rather than the cap, since the
   40 is the requirement. A long history costs about three times the typical
   one used here.

## With an API key

The copy in the cloud (`docs/CLOUD_DEPLOY.md` in diagnostic.ar) pays per token
instead. There the budget is money: `GLOBAL_EXTRACTIONS_PER_WEEK` = weekly
budget ÷ cost per extraction — US$0.94 on average, US$1.90 for a long history,
at the prices of 2026-09-28. `GLOBAL_EXTRACTIONS_PER_5H` then limits how fast a
week's budget can go.
