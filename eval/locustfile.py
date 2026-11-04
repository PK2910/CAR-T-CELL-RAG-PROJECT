"""
The pretend people for the Locust load tests. Run through `python eval/load_test.py --locust steady|stress`, which
starts the server with the pretend Claude, then reads what this file records and writes the report.

Each person asks a random question from the pool on /ask/stream (as the web page does), reads the streamed answer,
waits 3-8 s (reading it) and asks again. One line per question goes to LOAD_RESULTS: when it was asked, how many
people were active, the time to the first words and to the full answer, and the outcome (ok, or the kind of error).
"""

import json
import os
import random
import sys
import time
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, events, task
from requests.exceptions import ConnectionError, Timeout

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import SHAPES  # noqa: E402

TEST = os.environ.get("LOAD_TEST", "steady")
POOL = json.loads(Path(os.environ["LOAD_POOL"]).read_text())
RESULTS = Path(os.environ["LOAD_RESULTS"])
OWNER_KEY = os.environ["LOAD_OWNER_KEY"]
TIMEOUT_S = 60  # no data from the server for this long: counted as a timeout


def save(row: dict) -> None:
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")


@events.test_start.add_listener
def _started(environment, **kwargs):
    save({"start": time.time()})


class Person(HttpUser):
    wait_time = between(3, 8)  # reading the answer before asking again

    @task
    def ask(self):
        question = random.choice(POOL)
        row = {"t_start": time.time(), "users": self.environment.runner.user_count, "question": question,
               "first_s": None, "tokens": None, "search_s": None, "status": "incomplete"}
        t0 = time.perf_counter()
        try:
            with self.client.post("/ask/stream", json={"question": question}, headers={"X-Owner-Key": OWNER_KEY},
                                  stream=True, timeout=TIMEOUT_S, catch_response=True, name="/ask/stream") as r:
                if r.status_code == 0:  # no response at all: Locust keeps the reason in r.error
                    row["status"] = "timeout" if isinstance(r.error, Timeout) else "connection"
                elif r.status_code == 429:
                    row["status"] = "429 too many requests"
                elif r.status_code >= 500:
                    row["status"] = "5xx server error"
                elif r.status_code != 200:
                    row["status"] = f"HTTP {r.status_code}"
                else:
                    for line in r.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        if event["type"] == "text" and row["first_s"] is None:
                            row["first_s"] = time.perf_counter() - t0
                        elif event["type"] == "done":
                            debug = event.get("debug") or {}  # owner key: tokens and seconds per step
                            row["tokens"] = (debug.get("tokens") or [None, None])[1]
                            row["search_s"] = (debug.get("timing") or {}).get("search_s")
                            row["status"] = "fallback (passages only)" if event["fallback"] else "ok"
                        elif event["type"] == "error":
                            row["status"] = "error message"
                if row["status"] == "ok":
                    r.success()
                else:
                    r.failure(row["status"])
        except Timeout:
            row["status"] = "timeout"
        except ConnectionError:
            row["status"] = "connection"
        row["total_s"] = time.perf_counter() - t0
        save(row)


class Stages(LoadTestShape):
    """People arrive RATE a second at the start of each stage; the test ends after the last stage."""

    def tick(self):
        t = self.get_run_time()
        for users, end in SHAPES[TEST]["stages"]:
            if t < end:
                return users, SHAPES[TEST]["rate"]
        return None
