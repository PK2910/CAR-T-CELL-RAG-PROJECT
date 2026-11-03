"""
Load test (free): how many people can ask at the same time before answers slow down or fail?

The real server runs (startup checks, real hybrid search over the papers, guardrails, logbook, cache), but Claude is
replaced by a pretend one that takes about as long as the real one: CHECK_S for the relevance check, then an answer
streamed over ANSWER_S. Rounds of 1, 5, 10 and 20 different questions are sent at the same moment to /ask/stream (as
the web page does), and for each the time to the first words and to the full answer is measured.

Safe to run: the server gets a fake API key (so no real Claude call could be made even if the stand-in failed), a
throwaway owner key (so the daily limit does not stop the test), and a temporary folder for the logbook, cache and
usage counts (RAG_STATE_DIR), so the real logbook is not touched. Stop the API first (one process per index).

Run:  python eval/load_test.py [--rounds 1,5,10,20]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8004
OWNER_KEY = "load-test-owner-key"
CHECK_S = 6.0     # the relevance check took ~6-8 s in the logbook
ANSWER_S = 8.0    # writing took ~8-9 s
PIECES = 80       # the answer arrives in this many streamed pieces


def serve() -> None:
    """The API with Claude replaced (run in a separate process by main)."""
    sys.path[:0] = [str(ROOT / "api"), str(ROOT / "src")]
    import generate
    import relevance_check
    import uvicorn

    usage = types.SimpleNamespace(input_tokens=9000, output_tokens=300)

    def check_batch(client, question, sources_text, n):
        time.sleep(CHECK_S if "1" in sources_text[:30] else 0)  # one round of checking
        return [{"verdict": "yes" if i < 8 else "no", "reason": "stand-in"} for i in range(n)], usage

    text = ("ICANS was reported in roughly 20-60% of patients across studies [1][2]. Severe cases were less common "
            "[3]. ") * 4

    class Response:
        stop_reason, stop_details, model = "end_turn", None, "stand-in"
        content = [types.SimpleNamespace(type="text", text=text)]
        usage = types.SimpleNamespace(input_tokens=9000, output_tokens=420)

    class Messages:
        def create(self, **kwargs):
            time.sleep(ANSWER_S)
            return Response()

        def stream(self, **kwargs):
            class Stream:
                def __enter__(s): return s
                def __exit__(s, *exc): return False
                @property
                def text_stream(s):
                    step = len(text) // PIECES + 1
                    for i in range(0, len(text), step):
                        time.sleep(ANSWER_S / PIECES)
                        yield text[i:i + step]
                def get_final_message(s): return Response()
            return Stream()

    class Claude:
        api_key, auth_token = "stand-in", None
        def __init__(self, *a, **k): self.beta = types.SimpleNamespace(messages=Messages())
        def with_options(self, **k): return self

    relevance_check.check_batch = check_batch
    generate.anthropic.Anthropic = Claude
    import main

    # Time the work done after an answer is written (saving it in the cache and the logbook, preparing the reply),
    # one line per call in profile.jsonl, to see which step slows down when many answers finish together
    profile = Path(os.environ["RAG_STATE_DIR"]) / "profile.jsonl"

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                with open(profile, "a") as f:
                    f.write(json.dumps({"step": name, "s": time.perf_counter() - t}) + "\n")
        return wrapper

    main.cache.put = timed("cache.put", main.cache.put)
    main.logbook.record = timed("logbook.record", main.logbook.record)
    main.public = timed("public", main.public)
    main.owner_debug = timed("owner_debug", main.owner_debug)
    uvicorn.run(main.app, host="127.0.0.1", port=PORT, log_level="warning")


def ask(question: str, results: list, start_line: threading.Barrier) -> None:
    import httpx
    start_line.wait()  # every request of a round starts together
    t0 = time.perf_counter()
    first, row = None, {"question": question, "ok": False}
    try:
        with httpx.stream("POST", f"http://127.0.0.1:{PORT}/ask/stream", json={"question": question},
                          headers={"X-Owner-Key": OWNER_KEY}, timeout=180) as r:
            if r.status_code != 200:
                row["error"] = f"HTTP {r.status_code}"
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "text" and first is None:
                    first = time.perf_counter() - t0
                elif event["type"] == "done":
                    row.update(ok=not event["fallback"], fallback=event["fallback"],
                               timing=(event.get("debug") or {}).get("timing") or {})  # owner key: seconds per step
                elif event["type"] == "error":
                    row["error"] = event["text"]
    except Exception as e:  # noqa: BLE001  (any failure counts against the server)
        row["error"] = f"{type(e).__name__}: {e}"
    row.update(first_s=first, total_s=time.perf_counter() - t0)
    results.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="1,5,10,20")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        return serve()

    sys.path.insert(0, str(ROOT / "eval"))
    from question_sets import load_questions
    questions = [q["question"] for q in load_questions() if q["answerable"] is True]
    state = tempfile.mkdtemp(prefix="rag-load-test-")
    env = {**os.environ, "ANTHROPIC_API_KEY": "fake-key-for-load-test", "RAG_OWNER_KEY": OWNER_KEY,
           "RAG_STATE_DIR": state}
    server = subprocess.Popen([sys.executable, __file__, "--serve"], env=env, cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        import httpx
        t0 = time.perf_counter()
        while True:  # wait for the startup checks and warm-up
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if server.poll() is not None:
                sys.exit("The server did not start:\n" + server.stderr.read()[-2000:])
            time.sleep(1)
        print(f"Server ready in {time.perf_counter() - t0:.0f} s. Stand-in Claude: check {CHECK_S:.0f} s + answer "
              f"{ANSWER_S:.0f} s, so one question alone takes at least {CHECK_S + ANSWER_S:.0f} s.\n")
        print(f"{'at once':>7} {'answered':>9} {'first words: median / slowest':>31} {'full answer: median / slowest':>31}"
              f"   median seconds per step (search / check / writing)")
        used = 0
        for n in [int(x) for x in args.rounds.split(",")]:
            batch, used = questions[used:used + n], used + n  # different questions: no saved answers from the cache
            results, start_line = [], threading.Barrier(n)
            threads = [threading.Thread(target=ask, args=(q, results, start_line)) for q in batch]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            ok = [r for r in results if r["ok"]]
            firsts = [r["first_s"] for r in results if r["first_s"] is not None]
            totals = [r["total_s"] for r in ok]
            fmt = lambda xs: f"{statistics.median(xs):5.1f} s / {max(xs):5.1f} s" if xs else "-"
            steps = " / ".join(f"{statistics.median(r['timing'].get(s, 0) for r in ok):.1f}"
                               for s in ("search_s", "check_s", "answer_s")) if ok else "-"
            print(f"{n:>7} {len(ok):>4} of {n:<3} {fmt(firsts):>31} {fmt(totals):>31}   {steps}")
            for r in results:
                if not r["ok"]:
                    print(f"        failed: {r.get('error') or 'fallback answer'}  ({r['question'][:60]})")
            calls = [json.loads(l) for l in (Path(state) / "profile.jsonl").read_text().splitlines()][-4 * n:]
            print("        after writing, per answer: " + ", ".join(
                f"{s} {statistics.median(c['s'] for c in calls if c['step'] == s):.2f} s"
                for s in ("cache.put", "logbook.record", "owner_debug", "public")))
    finally:
        server.terminate()
        server.wait(timeout=30)
    errors = [l for l in server.stderr.read().splitlines() if "Error" in l or "Traceback" in l]
    if errors:
        print("\nServer errors:\n  " + "\n  ".join(errors[-10:]))


if __name__ == "__main__":
    main()
