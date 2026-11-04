"""
Load test (free): how many people can ask at the same time before answers slow down or fail?

The real server runs (startup checks, real hybrid search over the papers, guardrails, logbook), but Claude is replaced
by a pretend one that takes as long as the real one did (stand_in: timings from the logbook and the final test set).
Three tests:

  python eval/load_test.py                  spike: rounds of 1, 5, 10 and 20 people asking at the same moment
  python eval/load_test.py --locust steady  a busy period: people arrive 2 a second up to 25; each asks, reads the
                                            answer for 3-8 s and asks again, for 3.5 minutes (Locust)
  python eval/load_test.py --locust stress  keep adding people (10, 25, 50, 75, 100) to find where it breaks

Questions come from a pool of 100 different hand-written and generated questions. The cache is switched off, so
every question is answered from scratch (the worst case). The report compares the results with TARGETS, set before
the test; the key number is the time the server adds on top of Claude's own time (the part we control).

Safe to run: the server gets a fake API key (so no real Claude call could be made even if the stand-in failed), a
throwaway owner key (so the daily limits do not stop the test), and a temporary folder for the logbook and usage
counts (RAG_STATE_DIR), so the real logbook is not touched. Stop the API first (one process per index).
Results are saved to eval/results/load_test_<test>.json.
"""

import argparse
import json
import math
import os
import random
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
URL = f"http://127.0.0.1:{PORT}"
OWNER_KEY = "load-test-owner-key"
POOL_SIZE = 100

# Pass targets, set before running. The full-answer target is the project's speed target; it depends mostly on
# Claude (the pretend one alone takes ~15 s, like the real one), so the server's own added time is the fair test.
TARGETS = {"errors_pct": 1.0, "added_p95_s": 2.0, "first_p95_s": 10.0, "full_p95_s": 15.0}

# Locust tests: (people, seconds until the end of that stage); people arrive RATE a second at the start of a stage
SHAPES = {"steady": {"stages": [(25, 210)], "rate": 2},
          "stress": {"stages": [(10, 90), (25, 180), (50, 270), (75, 360), (100, 450)], "rate": 5},
          "smoke": {"stages": [(3, 45)], "rate": 3}}  # a quick check that the test itself works


def stand_in(question: str) -> dict:
    """How long the pretend Claude takes for this question, copied from the real answers: the relevance check took
    5-8 s (logbook), the first words came ~1 s after it, and answers of 1,300-3,700 characters (final test set) were
    written at 300-420 characters a second (logbook). The same question always gets the same times, so the report
    can work out how much time the server added."""
    r = random.Random(question)
    chars = r.randint(1300, 3700)
    t = {"check_s": r.uniform(5, 8), "first_token_s": r.uniform(0.7, 1.2), "chars": chars,
         "write_s": chars / r.uniform(300, 420)}
    t["claude_s"] = t["check_s"] + t["first_token_s"] + t["write_s"]
    return t


SENTENCES = ["ICANS was reported in roughly 20-60% of patients across studies [1][2]. ",
             "Severe cases were less common [3]. ", "Most cases resolved with corticosteroids [4]. ",
             "Onset was usually within the first week after infusion [2][5]. ",
             "Higher disease burden was linked to more severe toxicity [6]. "]


def serve() -> None:
    """The API with Claude replaced (run in a separate process by main)."""
    sys.path[:0] = [str(ROOT / "api"), str(ROOT / "src")]
    import generate
    import relevance_check
    import uvicorn

    def check_batch(client, question, sources_text, n):
        time.sleep(stand_in(question)["check_s"] if "1" in sources_text[:30] else 0)  # one round of checking
        usage = types.SimpleNamespace(input_tokens=9000, output_tokens=300)
        return [{"verdict": "yes" if i < 8 else "no", "reason": "stand-in"} for i in range(n)], usage

    def asked(kwargs) -> str:  # the question, from the end of the prompt (generate.py: "...\n\nQuestion: ...")
        return kwargs["messages"][-1]["content"].split("\n\nQuestion: ", 1)[-1].split("\n")[0]

    def reply(t):
        text = ""
        while len(text) < t["chars"]:
            text += SENTENCES[len(text) % len(SENTENCES)]
        text = text[:t["chars"]]
        return text, types.SimpleNamespace(
            stop_reason="end_turn", stop_details=None, model="stand-in",
            content=[types.SimpleNamespace(type="text", text=text)],
            usage=types.SimpleNamespace(input_tokens=9000, output_tokens=int(len(text) / 2.7)))  # ~2.7 chars/token

    class Messages:
        def create(self, **kwargs):
            t = stand_in(asked(kwargs))
            time.sleep(t["first_token_s"] + t["write_s"])
            return reply(t)[1]

        def stream(self, **kwargs):
            t = stand_in(asked(kwargs))
            text, response = reply(t)

            class Stream:
                def __enter__(s): return s
                def __exit__(s, *exc): return False
                @property
                def text_stream(s):
                    time.sleep(t["first_token_s"])
                    start, step = time.perf_counter(), 12  # pieces of ~4 tokens, as Claude sends them
                    for i in range(0, len(text), step):
                        time.sleep(max(0.0, start + i / len(text) * t["write_s"] - time.perf_counter()))
                        yield text[i:i + step]
                def get_final_message(s): return response
            return Stream()

    class Claude:
        api_key, auth_token = "stand-in", None
        def __init__(self, *a, **k): self.beta = types.SimpleNamespace(messages=Messages())
        def with_options(self, **k): return self

    relevance_check.check_batch = check_batch
    generate.anthropic.Anthropic = Claude
    import main

    main.cache.get = lambda question: None  # every question answered from scratch (the worst case)

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


def pct(xs: list, p: float):
    """The p-th percentile (nearest rank): pct(xs, 95) = 95% of the values are at or below it."""
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)] if xs else None


def start_server(state: str) -> subprocess.Popen:
    import httpx
    env = {**os.environ, "ANTHROPIC_API_KEY": "fake-key-for-load-test", "RAG_OWNER_KEY": OWNER_KEY,
           "RAG_STATE_DIR": state}
    server = subprocess.Popen([sys.executable, __file__, "--serve"], env=env, cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.perf_counter()
    while True:  # wait for the startup checks and warm-up
        try:
            if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if server.poll() is not None:
            sys.exit("The server did not start:\n" + server.stderr.read()[-2000:])
        time.sleep(1)
    print(f"Server ready in {time.perf_counter() - t0:.0f} s.")
    return server


# ---------- spike: everyone at the same moment ----------

def ask(question: str, results: list, start_line: threading.Barrier) -> None:
    import httpx
    start_line.wait()  # every request of a round starts together
    t0 = time.perf_counter()
    first, row = None, {"question": question, "ok": False}
    try:
        with httpx.stream("POST", f"{URL}/ask/stream", json={"question": question},
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


def spike(questions: list, rounds: str, state: str) -> None:
    print(f"{'at once':>7} {'answered':>9} {'full answer: median / slowest':>31} {'added by server: median / slowest':>35}"
          f"   median seconds per step (search / check / writing)")
    used = 0
    for n in [int(x) for x in rounds.split(",")]:
        batch, used = questions[used:used + n], used + n  # different questions in every round
        results, start_line = [], threading.Barrier(n)
        threads = [threading.Thread(target=ask, args=(q, results, start_line)) for q in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = [r for r in results if r["ok"]]
        totals = [r["total_s"] for r in ok]
        added = [r["total_s"] - stand_in(r["question"])["claude_s"] for r in ok]
        fmt = lambda xs: f"{statistics.median(xs):5.1f} s / {max(xs):5.1f} s" if xs else "-"
        steps = " / ".join(f"{statistics.median(r['timing'].get(s, 0) for r in ok):.1f}"
                           for s in ("search_s", "check_s", "answer_s")) if ok else "-"
        print(f"{n:>7} {len(ok):>4} of {n:<3} {fmt(totals):>31} {fmt(added):>35}   {steps}")
        for r in results:
            if not r["ok"]:
                print(f"        failed: {r.get('error') or 'fallback answer'}  ({r['question'][:60]})")
        calls = [json.loads(l) for l in (Path(state) / "profile.jsonl").read_text().splitlines()][-4 * n:]
        print("        after writing, per answer: " + ", ".join(
            f"{s} {statistics.median(c['s'] for c in calls if c['step'] == s):.2f} s"
            for s in ("cache.put", "logbook.record", "owner_debug", "public")))


# ---------- Locust: people arriving gradually and pausing between questions ----------

def watch(server: subprocess.Popen, samples: list, stop: threading.Event) -> None:
    """The server's CPU (100% = one core fully busy) and memory, once a second."""
    import psutil
    p = psutil.Process(server.pid)
    p.cpu_percent()
    while not stop.wait(1):
        try:
            samples.append({"t": time.time(), "cpu": p.cpu_percent(), "mb": p.memory_info().rss / 1e6})
        except psutil.Error:
            return


def summary(rows: list, samples: list, seconds: float) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    errors = {}
    for r in rows:
        if r["status"] != "ok":
            errors[r["status"]] = errors.get(r["status"], 0) + 1
    first, full = [r["first_s"] for r in ok], [r["total_s"] for r in ok]
    added = [r["total_s"] - stand_in(r["question"])["claude_s"] for r in ok]
    per_token = [(r["total_s"] - r["first_s"]) / r["tokens"] * 1000 for r in ok if r["tokens"]]
    q = lambda xs: {"median": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99)}
    return {"requests": len(rows), "answered": len(ok), "errors": errors,
            "errors_pct": 100 * (len(rows) - len(ok)) / len(rows) if rows else 0.0,
            "answers_per_s": len(ok) / seconds if seconds else None,
            "first_s": q(first), "full_s": q(full), "added_s": q(added), "ms_per_token": q(per_token),
            "search_s": q([r["search_s"] for r in ok if r.get("search_s") is not None]),
            "cpu_pct": {"median": pct([s["cpu"] for s in samples], 50), "max": max((s["cpu"] for s in samples), default=None)},
            "memory_mb_max": max((s["mb"] for s in samples), default=None)}


def check(s: dict) -> list:
    """(target, result, passed) for each pass target."""
    return [(f"errors under {TARGETS['errors_pct']:.0f}%", f"{s['errors_pct']:.1f}%", s["errors_pct"] < TARGETS["errors_pct"]),
            *[(f"{name}: 95% under {TARGETS[key + '_p95_s']:.0f} s", f"{s[key + '_s']['p95']:.1f} s",
               s[key + "_s"]["p95"] is not None and s[key + "_s"]["p95"] < TARGETS[key + "_p95_s"])
              for name, key in (("time added by the server", "added"), ("first words", "first"),
                                ("full answer", "full"))]]


def run_locust(test: str, questions: list, state: str, server: subprocess.Popen) -> None:
    shape = SHAPES[test]
    pool = Path(state) / "pool.json"
    pool.write_text(json.dumps(questions[:POOL_SIZE]))
    records = Path(state) / "requests.jsonl"
    env = {**os.environ, "LOAD_TEST": test, "LOAD_POOL": str(pool), "LOAD_RESULTS": str(records),
           "LOAD_OWNER_KEY": OWNER_KEY}
    samples, stop = [], threading.Event()
    threading.Thread(target=watch, args=(server, samples, stop), daemon=True).start()
    minutes = shape["stages"][-1][1] / 60
    print(f"Running the {test} test with Locust for about {minutes:.1f} minutes...")
    with open(Path(state) / "locust.log", "w") as log:
        subprocess.run([sys.executable, "-m", "locust", "-f", str(ROOT / "eval/locustfile.py"), "--headless",
                        "--host", URL, "--stop-timeout", "120", "--only-summary"],
                       env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    lines = [json.loads(l) for l in records.read_text().splitlines()]
    t0 = next(l["start"] for l in lines if "start" in l)
    rows = [l for l in lines if "status" in l]

    # Each stage is judged on the questions asked after everyone in it has arrived (not during the ramp-up)
    stages, begin, prev_users = [], 0.0, 0
    for users, end in shape["stages"]:
        settled = begin + (users - prev_users) / shape["rate"]
        part = [r for r in rows if settled <= r["t_start"] - t0 < end]
        done = [r for r in rows if r["status"] == "ok" and settled + 20 <= r["t_start"] - t0 + r["total_s"] < end]
        cpu = [s for s in samples if settled <= s["t"] - t0 < end]
        s = summary(part, cpu, 0)
        s["answers_per_s"] = len(done) / (end - settled - 20)  # finished per second, once the stage is running
        stages.append({"users": users, **s})
        begin, prev_users = end, users

    out = {"test": test, "targets": TARGETS, "shape": shape, "pool": POOL_SIZE, "all": summary(rows, samples, 0),
           "stages": stages, "requests": rows}
    claude = [stand_in(q)["claude_s"] for q in questions[:POOL_SIZE]]
    print(f"\nPretend Claude alone takes {pct(claude, 50):.1f} s (median over the pool), so the full answer cannot be "
          f"faster than that.\nEach person asks again 3-8 s after an answer, so with no slowdown N people would get "
          f"about N / {statistics.mean(claude) + 5.5:.1f} answers a second ('expected').\n")
    for s in stages:
        s["expected_per_s"] = s["users"] / (statistics.mean(claude) + 5.5)
    print(f"{'people':>6} {'asked':>6} {'answered/s':>10} {'expected':>8} {'errors':>7} {'first words p95':>15} {'full p95 / p99':>15} "
          f"{'added p95':>10} {'search p95':>10} {'ms/token':>8} {'CPU med/max':>12}")
    for s in stages:
        f = lambda x: f"{x:.1f}" if x is not None else "-"
        print(f"{s['users']:>6} {s['requests']:>6} {s['answers_per_s']:>10.2f} {s['expected_per_s']:>8.2f} "
              f"{s['errors_pct']:>6.1f}% "
              f"{f(s['first_s']['p95']):>15} {f(s['full_s']['p95']) + ' / ' + f(s['full_s']['p99']):>15} "
              f"{f(s['added_s']['p95']):>10} {f(s['search_s']['p95']):>10} {f(s['ms_per_token']['median']):>8} "
              f"{f(s['cpu_pct']['median']) + ' / ' + f(s['cpu_pct']['max']):>12}")
    a = out["all"]
    print(f"\nWhole test: {a['requests']} questions, {a['answered']} answered; errors by type: "
          f"{a['errors'] or 'none'}; most memory used: {a['memory_mb_max']:.0f} MB")
    for s in stages:
        print(f"\nPass targets with {s['users']} people:")
        for target, result, passed in check(s):
            print(f"  {'PASS' if passed else 'FAIL'}  {target:<42} {result}")
    path = ROOT / f"eval/results/load_test_{test}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nSaved to {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="1,5,10,20")
    parser.add_argument("--locust", choices=list(SHAPES))
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        return serve()

    sys.path.insert(0, str(ROOT / "eval"))
    from question_sets import load_questions
    questions = [q["question"] for q in load_questions() if q["answerable"] is True]
    random.Random(0).shuffle(questions)  # a mix of hand-written and generated questions
    state = tempfile.mkdtemp(prefix="rag-load-test-")
    server = start_server(state)
    try:
        if args.locust:
            run_locust(args.locust, questions, state, server)
        else:
            spike(questions, args.rounds, state)
    finally:
        server.terminate()
        server.wait(timeout=30)
    errors = [l for l in server.stderr.read().splitlines() if "Error" in l or "Traceback" in l]
    if errors:
        print("\nServer errors:\n  " + "\n  ".join(errors[-10:]))


if __name__ == "__main__":
    main()
