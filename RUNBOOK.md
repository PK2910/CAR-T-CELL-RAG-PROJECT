# Runbook: what to do when an alert goes off

The server checks its logbook every 5 minutes (`api/alerts.py`). Current alerts are listed under `"alerts"` at
`/health` (empty when all is well). Each alert's start and end also go to the server's logs and to `alerts.log`
next to the logbook. Check by hand with `python api/alerts.py`.

First step for every alert: `python api/log_report.py --days 1`. It shows speed per stage (search, relevance check,
writing), errors by type, fallbacks, cost and ratings. On the online server, run these inside the container,
where the logbook is.

## slow

**Means:** the slowest 5% of the last hour's answers took over 25 s. Normal is about 19 s: Claude alone takes ~15 s.

**Check:** which stage is slow in the log report.
- **Relevance check or writing slow:** Claude is slow. See Anthropic's status page. Nothing to fix on our side.
  Users still get the passages if Claude passes the time limit (20 s for the check, 45 s for writing).
- **Search slow (normally under 1 s):** the server is overloaded. The load test showed waits growing above ~50
  people at once (`eval/RESULTS.md`, stress test). Check the host's CPU and memory graphs.
- **Everything slow, only now and then:** the host's machine may be shared and busy. Watch whether it continues.

**Fix:** if Claude is slow, wait. If the server is overloaded, restart it, then consider a bigger machine. Several
server processes would first need a Qdrant server and a shared database: the local index and the SQLite files
are opened by one process.

## failing

**Means:** over 5% of the last hour's questions got no answer at all, or only the passages (the fallback).

**Check:** "Errors" in the log report.
- `AuthenticationError` / `PermissionDeniedError`: the API key is wrong, expired or out of credit. Check the
  Anthropic console. The key is set in the host's settings (never in the code).
- `APITimeoutError` / `APIConnectionError` / `InternalServerError` / `OverloadedError`: Claude is down or slow.
  See Anthropic's status page.
- `RateLimitError`: too many requests for the account's limits. Usually short-lived; check the limits in the
  Anthropic console.
- `NoAPIKey`: the server has no key. Set `ANTHROPIC_API_KEY` in the host's settings and restart.

**Fix:** follow the cause above. Users keep getting the found passages meanwhile, and refused questions are not
counted against their daily limit.

## budget

**Means:** today's public spending passed $0.80 of the $1.00 daily limit (`api/usage.py`). At $1.00 the demo
stops answering public questions until midnight UTC, by itself. The owner's key still works.

**Check:** the log report's cost and "most asked questions". Is it real interest, or one user or script asking a
lot? (Each person can ask at most 10 a day; users are told apart by a scrambled code of their address.)

**Fix:** usually nothing; the limit protects the money. If it happens every day and the use is real, decide whether
to raise `DAILY_BUDGET_USD`. If it looks like abuse, lower `QUESTIONS_PER_DAY`.

## thumbs-down

**Means:** over 30% of the last week's ratings (at least 5) were thumbs down.

**Check:** `python api/log_report.py --export-down` saves the thumbs-down answers and comments to
`eval/results/thumbs_down.json`.
- Did something change just before? Compare the `versions` of good and bad answers: index, models, prompts,
  guardrails.
- Are they about one topic? Then the papers may be missing it (README, Keeping it up to date).

**Fix:** if a change caused it, go back to the earlier version. Otherwise turn the bad answers into test questions
(write answer keys first), fix, and check on the test sets before switching (README, Evaluation).

## bad-citations

**Means:** an answer in the last day cited a passage it was not given, e.g. `[9]` with 5 passages. The user saw a
warning added to the answer (the citation check). Before launch this happened once in 610 answers.

**Check:** look up the logbook ids in the alert (`data/api_log.sqlite`, table `answers`). Does the question or
answer look unusual? Did the answer model or prompt change (`versions`)?

**Fix:** a single one: note it. Several a day: compare with the last good version, and re-run
`python eval/citation_report.py`.

## alerts-broken

**Means:** the alerts themselves could not run, usually because the logbook could not be read.

**Check:** the server's logs for the error. On the host: is the storage volume (`RAG_STATE_DIR`) attached and not
full?

**Fix:** reattach or enlarge the volume and restart. Answers keep working: a broken logbook never stops an answer.

## The server is down

**Means:** `/health` does not answer at all. Alerts cannot report this themselves, so an outside uptime monitor
should watch `/health`.

**Check:** the host's logs. `Startup check failed, not starting` means the search setup does not match the index
(e.g. a changed `RAG_INDEX` or a missing index). The message says which check failed.

**Fix:** put back the matching index or settings and restart. If you just switched to a new index, go back to the
previous one (README, Keeping it up to date).
