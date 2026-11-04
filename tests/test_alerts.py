"""The alerts (api/alerts.py), on made-up logbook entries in a temporary logbook."""

import alerts
import logbook
import usage

ANSWER = {"answer": "ICANS 40% [1].", "cost_usd": 0.02}


def log(n, seconds=15.0, **result):
    """n made-up answered questions in the logbook; returns their ids."""
    return [logbook.record(f"question {i}", {**ANSWER, **result}, "user", "ask", seconds) for i in range(n)]


def names():
    return [a["name"] for a in alerts.check()]


def test_no_alerts_when_all_is_well(pipeline):
    log(10)
    assert names() == []


def test_slow_answers(pipeline):
    log(4, seconds=40)
    assert names() == []  # too few answers to judge
    log(2, seconds=40)
    assert names() == ["slow"]


def test_failing(pipeline):
    log(9)
    logbook.record("question", None, "user", "ask", 1.0, error=RuntimeError("Claude down"))  # 1 in 10
    assert names() == ["failing"]


def test_fallback_counts_as_failing(pipeline):
    log(8)
    log(2, degraded=True)
    assert names() == ["failing"]


def test_budget(pipeline):
    usage.add_spend(0.79)
    assert names() == []
    usage.add_spend(0.02)
    assert names() == ["budget"]


def test_thumbs_down(pipeline):
    ids = log(5)
    for i in ids[:4]:
        logbook.add_feedback(i, "down")
    assert names() == []  # 4 ratings: too few to judge
    logbook.add_feedback(ids[4], "up")
    assert names() == ["thumbs_down"]  # 4 of 5 down


def test_bad_citation(pipeline):
    log(5)
    [bad] = log(1, citations={"invalid": [9]})
    found = alerts.check()
    assert [a["name"] for a in found] == ["bad_citations"] and bad in found[0]["message"]
    assert found[0]["runbook"] == "RUNBOOK.md#bad-citations"


def test_health_lists_alerts(client):
    assert client.get("/health").json()["alerts"] == []
    usage.add_spend(0.90)
    body = client.get("/health").json()
    assert [a["name"] for a in body["alerts"]] == ["budget"] and body["status"] == "ok"  # still up: no restart


def test_background_check_notes_new_alerts(pipeline, monkeypatch, tmp_path):
    import threading
    monkeypatch.setattr(logbook, "STATE_DIR", tmp_path)
    usage.add_spend(0.90)
    stop = threading.Event()
    stop.set()  # check once, then stop
    alerts.watch(stop)
    assert "ALERT budget" in (tmp_path / "alerts.log").read_text()
