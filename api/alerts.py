"""
Alerts (README, Alerts): the server checks its own logbook every CHECK_EVERY_S and warns when something needs a
look, so nobody has to watch it. What to do for each alert: RUNBOOK.md.

  slow            the slowest 5% of answers in the last hour took over SLOW_P95_S (normal: ~19 s)
  failing         over FAILING_PCT% of the last hour's questions got no answer or only the passages (Claude down,
                  too slow, or the key not working)
  budget          today's public spending passed BUDGET_WARN_USD of the daily limit (api/usage.py)
  thumbs_down     over THUMBS_DOWN_PCT% of the last week's ratings were thumbs down
  bad_citations   an answer in the last day cited a passage it was not given (the citation check)

Rules on shares and percentiles wait for a few answers (MIN_ANSWERS, MIN_RATINGS): one slow answer is not an alarm.
Saved answers from the cache are left out of the speed and failure rules (they take no time and call no model).

Where they show: /health lists the current alerts; when one starts or ends, a line is printed (the hosting service's
logs) and added to alerts.log next to the logbook. An uptime service watching /health can email when "alerts" is
not empty.
Check by hand:  python api/alerts.py
"""

import json
import math
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logbook  # noqa: E402
import usage  # noqa: E402

CHECK_EVERY_S = 300
SLOW_P95_S = 25.0
FAILING_PCT = 5.0
BUDGET_WARN_USD = 0.80
THUMBS_DOWN_PCT = 30.0
MIN_ANSWERS = 5
MIN_RATINGS = 5


def _p95(xs: list) -> float:
    xs = sorted(xs)
    return xs[math.ceil(0.95 * len(xs)) - 1]


def _since(now: datetime, **ago) -> str:
    return (now - timedelta(**ago)).isoformat(timespec="seconds")  # the logbook's time format


def check(now: datetime | None = None) -> list[dict]:
    """The alerts that are on right now: [{"name", "message", "runbook"}]."""
    now = now or datetime.now(timezone.utc)
    week = logbook.entries(_since(now, days=7))
    day = [e for e in week if e["time"] >= _since(now, days=1)]
    hour = [e for e in day if e["time"] >= _since(now, hours=1) and not e["cached"]]
    alerts = []

    def alert(name, message):
        alerts.append({"name": name, "message": message, "runbook": f"RUNBOOK.md#{name.replace('_', '-')}"})

    times = [e["seconds_total"] for e in hour if e["answer"] is not None and e["seconds_total"] is not None]
    if len(times) >= MIN_ANSWERS and _p95(times) > SLOW_P95_S:
        alert("slow", f"95% of the last hour's {len(times)} answers took up to {_p95(times):.0f} s "
                      f"(alarm above {SLOW_P95_S:.0f} s)")

    failed = [e for e in hour if e["answer"] is None or e["fallback"]]
    if len(hour) >= MIN_ANSWERS and 100 * len(failed) / len(hour) > FAILING_PCT:
        alert("failing", f"{len(failed)} of the last hour's {len(hour)} questions got no answer or only the "
                         f"passages (alarm above {FAILING_PCT:.0f}%)")

    spent = usage.spent_today()
    if spent >= BUDGET_WARN_USD:
        alert("budget", f"${spent:.2f} of today's ${usage.DAILY_BUDGET_USD:.2f} limit spent")

    ratings = [e["rating"] for e in week if e["rating"] is not None]
    down = ratings.count(-1)
    if len(ratings) >= MIN_RATINGS and 100 * down / len(ratings) > THUMBS_DOWN_PCT:
        alert("thumbs_down", f"{down} of the last week's {len(ratings)} ratings were thumbs down "
                             f"(alarm above {THUMBS_DOWN_PCT:.0f}%)")

    bad = [e for e in day if json.loads(e["bad_citations"] or "[]")]
    if bad:
        alert("bad_citations", f"{len(bad)} answer(s) in the last day cited a passage they were not given "
                               f"(logbook ids: {', '.join(e['id'] for e in bad[:5])})")
    return alerts


def safe_check() -> list[dict]:
    """check(), but a broken logbook never breaks /health: it becomes an alert itself."""
    try:
        return check()
    except Exception as e:  # noqa: BLE001
        return [{"name": "alerts_broken", "message": f"the alerts could not be checked: {type(e).__name__}: {e}",
                 "runbook": "RUNBOOK.md#alerts-broken"}]


def _note(text: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {text}"
    print(line)
    try:
        with open(logbook.STATE_DIR / "alerts.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def watch(stop: threading.Event) -> None:
    """Run in the background while the server runs: note each alert when it starts and when it ends."""
    on: dict[str, str] = {}
    while True:
        now = {a["name"]: a["message"] for a in safe_check()}
        for name in now.keys() - on.keys():
            _note(f"ALERT {name}: {now[name]} (what to do: RUNBOOK.md#{name.replace('_', '-')})")
        for name in on.keys() - now.keys():
            _note(f"RESOLVED {name}")
        on = now
        if stop.wait(CHECK_EVERY_S):
            return


if __name__ == "__main__":
    found = check()
    for a in found:
        print(f"ALERT {a['name']}: {a['message']}\n      what to do: {a['runbook']}")
    print("No alerts." if not found else "")
