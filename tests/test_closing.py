"""Tests for atomic billing-cycle closing and immutable close snapshots.

Like settlement tests, these run against a temporary *file-backed* SQLite
database (WAL, real write locks) so same-revision batch/close races exercise
the actual serialization guarantees.
"""
from __future__ import annotations

import os
import threading
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402

from app import closing, settlement  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    BillingCycle,
    CycleClosure,
    CycleInterval,
    UsageBatch,
    configure_sqlite_engine,
    seed,
)
from app.main import app  # noqa: E402
from app.pricing import PricingError  # noqa: E402
from app.schemas import (  # noqa: E402
    CycleCloseRequest,
    QuoteIntervalIn,
)

V = "us-ny-demo-v2024"
# 2024-07-06 is a Saturday in NY -> only the `leisure` rule
# (tiers: 100 @0.15, 300 @0.18, open @0.22).
DAY1 = {
    "start": "2024-07-06T00:00:00-04:00",
    "end": "2024-07-06T01:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
DAY2 = {
    "start": "2024-07-06T01:00:00-04:00",
    "end": "2024-07-06T02:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
DAY3 = {
    "start": "2024-07-06T02:00:00-04:00",
    "end": "2024-07-06T03:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}


@pytest.fixture()
def tmp_engine(tmp_path):
    path = tmp_path / "close.db"
    eng = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng)
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        seed(s)
        s.commit()
    yield eng, path
    eng.dispose()


def _submit(eng, cycle, key, rev, intervals, version=V):
    class _Req:
        def __init__(self):
            self.version = version
            self.idempotency_key = key
            self.expected_revision = rev
            self.intervals = [QuoteIntervalIn(**iv) for iv in intervals]

    return settlement.submit_batch(cycle, _Req(), engine=eng)


def _close(eng, cycle, close_id, rev, total):
    req = CycleCloseRequest(close_id=close_id, expected_revision=rev,
                            expected_total_amount=Decimal(total))
    return closing.close_cycle(cycle, req, engine=eng)


def _two_batch_cycle(eng, cycle="c1"):
    _submit(eng, cycle, "k1", 0, [DAY1])
    _submit(eng, cycle, "k2", 1, [DAY2])
    return cycle


# ------------------------------------------------------------- happy paths


def test_close_saves_immutable_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    res = _close(eng, "c1", "close-1", 2, "18.60")
    assert res.replayed is False
    snap = res.body
    assert snap["close_id"] == "close-1"
    assert snap["cycle_id"] == "c1"
    assert snap["version"] == V
    assert snap["revision"] == 2  # closing does not advance revision
    assert snap["status"] == "closed"
    assert snap["closed_at"]
    assert Decimal(snap["total_amount"]) == Decimal("18.60")
    assert Decimal(snap["total_energy"]) == Decimal("120")
    # batches/intervals stable-sorted
    assert [b["revision"] for b in snap["batches"]] == [1, 2]
    assert [b["idempotency_key"] for b in snap["batches"]] == ["k1", "k2"]
    assert [(i["revision"], i["seq"]) for i in snap["intervals"]] == [
        (1, 0), (2, 1)]
    # fresh full quote of the whole cycle stored in the snapshot
    assert snap["quote"]["version"] == V
    assert Decimal(snap["quote"]["total_amount"]) == Decimal("18.60")
    seg_energy = sum(Decimal(s["energy"]) for s in snap["quote"]["segments"])
    assert seg_energy == Decimal("120.000000")
    # digest recomputes identically from the same rows
    with Session(eng) as s:
        cycle = s.get(BillingCycle, "c1")
        assert cycle.status == "closed"
        assert cycle.closed_at == snap["closed_at"]
        batches = s.scalars(select(UsageBatch)).all()
        intervals = s.scalars(select(CycleInterval)).all()
        digest = closing._detail_digest(
            "c1", V, 2, cycle.total_energy, "18.60", batches, intervals)
        assert digest == snap["detail_digest"]
        assert len(digest) == 64


def test_get_cycle_surfaces_closed_status(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    state = settlement.get_cycle("c1", engine=eng)
    assert state["status"] == "open"
    assert state["close_id"] is None
    assert state["closed_at"] is None
    _close(eng, "c1", "close-1", 2, "18.60")
    state = settlement.get_cycle("c1", engine=eng)
    assert state["status"] == "closed"
    assert state["close_id"] == "close-1"
    assert state["closed_at"]


def test_detail_digest_covers_batch_content(tmp_engine):
    """Two cycles with identical totals still get different digests when
    their batch content (idempotency keys) differs."""
    eng, _ = tmp_engine
    _two_batch_cycle(eng, "cA")
    _two_batch_cycle(eng, "cB")
    a = _close(eng, "cA", "ca", 2, "18.60").body
    b = _close(eng, "cB", "cb", 2, "18.60").body
    assert a["detail_digest"] != b["detail_digest"]


def test_stable_digest_regardless_of_row_order(tmp_engine):
    """Digest is computed over stable-sorted rows, so unscheduled DB row
    iteration order cannot change it."""
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    res = _close(eng, "c1", "close-1", 2, "18.60")
    with Session(eng) as s:
        cycle = s.get(BillingCycle, "c1")
        batches = list(reversed(s.scalars(select(UsageBatch)).all()))
        intervals = list(reversed(s.scalars(select(CycleInterval)).all()))
        digest = closing._detail_digest(
            "c1", V, 2, cycle.total_energy, "18.60", batches, intervals)
    assert digest == res.body["detail_digest"]


# ------------------------------------------------------------- idempotency


def test_same_close_id_same_content_replays_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    first = _close(eng, "c1", "close-1", 2, "18.60").body
    replay = _close(eng, "c1", "close-1", 2, "18.60")
    assert replay.replayed is True
    assert replay.body == first
    with Session(eng) as s:
        assert s.scalars(select(CycleClosure)).all().__len__() == 1


def test_same_close_id_different_revision_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 1, "18.60")
    assert ei.value.code == "close_conflict"
    assert ei.value.details["close_id"] == "close-1"
    assert ei.value.details["existing"]["expected_revision"] == 2
    assert ei.value.details["received"]["expected_revision"] == 1


def test_same_close_id_different_amount_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 2, "18.59")
    assert ei.value.code == "close_conflict"


def test_same_close_id_other_cycle_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng, "cA")
    _two_batch_cycle(eng, "cB")
    _close(eng, "cA", "shared-id", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _close(eng, "cB", "shared-id", 2, "18.60")
    assert ei.value.code == "close_conflict"
    assert ei.value.details["existing"]["cycle_id"] == "cA"
    assert ei.value.details["received"]["cycle_id"] == "cB"
    # the second cycle stays open, no second closure row
    with Session(eng) as s:
        assert s.get(BillingCycle, "cB").status == "open"
        assert len(s.scalars(select(CycleClosure)).all()) == 1


# ------------------------------------------------------------- rejections


def test_close_unknown_cycle_404(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        _close(eng, "missing", "x", 0, "0.00")
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"
    assert s_count(eng, CycleClosure) == 0


def s_count(eng, model):
    with Session(eng) as s:
        return len(s.scalars(select(model)).all())


def test_revision_mismatch_rejects_and_changes_nothing(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 1, "18.60")
    assert ei.value.code == "revision_conflict"
    assert ei.value.details["current_revision"] == 2
    with Session(eng) as s:
        c = s.get(BillingCycle, "c1")
        assert c.status == "open"
        assert c.closed_at is None
        assert c.revision == 2
    assert s_count(eng, CycleClosure) == 0


def test_amount_mismatch_rejects_and_changes_nothing(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 2, "9.99")
    assert ei.value.code == "total_amount_mismatch"
    det = ei.value.details
    assert det["expected_total_amount"] == "9.99"
    assert det["current_total_amount"] == "18.60"
    with Session(eng) as s:
        c = s.get(BillingCycle, "c1")
        assert c.status == "open"
        assert c.revision == 2
    assert s_count(eng, CycleClosure) == 0


def test_second_close_with_new_id_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-2", 2, "18.60")
    assert ei.value.code == "cycle_closed"
    assert ei.value.details["existing_close_id"] == "close-1"
    with Session(eng) as s:
        rows = s.scalars(select(CycleClosure)).all()
        assert [r.close_id for r in rows] == ["close-1"]


def test_new_batch_rejected_after_close(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _submit(eng, "c1", "k3", 2, [DAY3])
    assert ei.value.code == "cycle_closed"
    # cycle untouched by the failed batch
    with Session(eng) as s:
        c = s.get(BillingCycle, "c1")
        assert c.status == "closed"
        assert c.revision == 2
        assert len(s.scalars(select(UsageBatch)).all()) == 2
        assert len(s.scalars(select(CycleInterval)).all()) == 2


def test_batch_replay_still_works_after_close(tmp_engine):
    eng, _ = tmp_engine
    first = _submit(eng, "c1", "k1", 0, [DAY1]).body
    _submit(eng, "c1", "k2", 1, [DAY2])
    _close(eng, "c1", "close-1", 2, "18.60")
    # idempotent replay (even with a stale expected_revision) still returns
    # the original first-batch response
    replay = _submit(eng, "c1", "k1", 99, [DAY1])
    assert replay.replayed is True
    assert replay.body == first


def test_invalid_close_id(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    for bad in ("../x", "-lead", "x y", "a/b", "中文id"):
        with pytest.raises(PricingError) as ei:
            _close(eng, "c1", bad, 2, "18.60")
        assert ei.value.code == "invalid_close_id"
    assert s_count(eng, CycleClosure) == 0


def test_get_closure_not_found_states(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        closing.get_closure("missing", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"

    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        closing.get_closure("c1", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "closure_not_found"


def test_get_closure_returns_stored_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    snap = _close(eng, "c1", "close-1", 2, "18.60").body
    assert closing.get_closure("c1", engine=eng) == snap


# ------------------------------------------------------------- concurrency


def test_concurrent_close_and_batch_only_one_wins(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "race", "base", 0, [DAY1])  # revision 1, total 9.00
    start = threading.Event()
    outcomes: dict[str, object] = {}

    def do_close():
        start.wait()
        try:
            outcomes["close"] = _close(eng, "race", "close-1", 1, "9.00")
        except PricingError as e:
            outcomes["close"] = e

    def do_batch():
        start.wait()
        try:
            outcomes["batch"] = _submit(eng, "race", "kb", 1, [DAY2])
        except PricingError as e:
            outcomes["batch"] = e

    t1 = threading.Thread(target=do_close)
    t2 = threading.Thread(target=do_batch)
    t1.start(); t2.start()
    start.set(); t1.join(); t2.join()

    close_res = outcomes["close"]
    batch_res = outcomes["batch"]
    with Session(eng) as s:
        cycle = s.get(BillingCycle, "race")
        closures = s.scalars(select(CycleClosure)).all()
        batches = s.scalars(select(UsageBatch)).all()

        close_won = not isinstance(close_res, PricingError)
        batch_won = not isinstance(batch_res, PricingError)
        assert close_won != batch_won, outcomes  # exactly one wins

        if close_won:
            assert close_res.body["revision"] == 1
            assert isinstance(batch_res, PricingError)
            assert batch_res.code == "cycle_closed"
            assert cycle.status == "closed"
            assert cycle.revision == 1
            assert len(closures) == 1
            assert len(batches) == 1  # loser batch wrote nothing
        else:
            assert close_res.code in ("revision_conflict", "cycle_closed")
            assert batch_res.body["revision"] == 2
            assert cycle.status == "open"
            assert cycle.revision == 2
            assert len(closures) == 0
            assert len(batches) == 2


def test_concurrent_distinct_closes_only_one_wins(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "race", "base", 0, [DAY1])
    start = threading.Event()
    outcomes: dict[str, object] = {}

    def worker(name):
        start.wait()
        try:
            outcomes[name] = _close(eng, "race", f"close-{name}", 1, "9.00")
        except PricingError as e:
            outcomes[name] = e

    threads = [threading.Thread(target=worker, args=(n,))
               for n in ("a", "b")]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    ok = [r for r in outcomes.values() if not isinstance(r, PricingError)]
    err = [r for r in outcomes.values() if isinstance(r, PricingError)]
    assert len(ok) == 1, outcomes
    assert len(err) == 1
    assert err[0].code == "cycle_closed"
    assert s_count(eng, CycleClosure) == 1


def test_concurrent_same_close_id_same_content_both_replay(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "idem", "base", 0, [DAY1])
    start = threading.Event()
    outcomes: list = []

    def worker():
        start.wait()
        try:
            outcomes.append(_close(eng, "idem", "close-1", 1, "9.00"))
        except PricingError as e:
            outcomes.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert all(not isinstance(r, PricingError) for r in outcomes), outcomes
    assert outcomes[0].body == outcomes[1].body
    assert s_count(eng, CycleClosure) == 1


# ------------------------------------------------------------- persistence


def test_closure_survives_restart(tmp_engine):
    eng, path = tmp_engine
    _two_batch_cycle(eng)
    snap = _close(eng, "c1", "close-1", 2, "18.60").body
    eng.dispose()

    eng2 = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng2)
    try:
        # snapshot query after restart
        assert closing.get_closure("c1", engine=eng2) == snap
        # same close_id same content replays
        replay = _close(eng2, "c1", "close-1", 2, "18.60")
        assert replay.replayed is True
        assert replay.body == snap
        # new batches remain rejected
        with pytest.raises(PricingError) as ei:
            _submit(eng2, "c1", "k3", 2, [DAY3])
        assert ei.value.code == "cycle_closed"
        state = settlement.get_cycle("c1", engine=eng2)
        assert state["status"] == "closed"
        assert state["revision"] == 2
    finally:
        eng2.dispose()


# ------------------------------------------------------------- HTTP surface


def test_http_close_full_flow():
    client = TestClient(app)
    with client:
        cid = "http-close-cycle"
        r = client.post(f"/cycles/{cid}/batches", json={
            "version": V,
            "idempotency_key": "k1",
            "expected_revision": 0,
            "intervals": [{
                "start": "2024-07-06T00:00:00-04:00",
                "end": "2024-07-06T01:00:00-04:00",
                "timezone": "America/New_York",
                "energy": "60",
            }],
        })
        assert r.status_code == 201, r.text

        # close
        r = client.post(f"/cycles/{cid}/close", json={
            "close_id": "http-close-1",
            "expected_revision": 1,
            "expected_total_amount": "9.00",
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "closed"
        assert body["revision"] == 1
        assert body["detail_digest"]
        assert body["batches"][0]["idempotency_key"] == "k1"
        assert body["quote"]["total_amount"] == "9.00"

        # replay -> 200, byte-identical body
        r2 = client.post(f"/cycles/{cid}/close", json={
            "close_id": "http-close-1",
            "expected_revision": 1,
            "expected_total_amount": "9.00",
        })
        assert r2.status_code == 200, r2.text
        assert r2.json() == body

        # GET snapshot
        rg = client.get(f"/cycles/{cid}/close")
        assert rg.status_code == 200
        assert rg.json() == body

        # amount mismatch replay -> 409 close_conflict
        r3 = client.post(f"/cycles/{cid}/close", json={
            "close_id": "http-close-1",
            "expected_revision": 1,
            "expected_total_amount": "9.01",
        })
        assert r3.status_code == 409
        assert r3.json()["error"]["code"] == "close_conflict"

        # new batch rejected
        r4 = client.post(f"/cycles/{cid}/batches", json={
            "version": V,
            "idempotency_key": "k2",
            "expected_revision": 1,
            "intervals": [{
                "start": "2024-07-06T01:00:00-04:00",
                "end": "2024-07-06T02:00:00-04:00",
                "timezone": "America/New_York",
                "energy": "60",
            }],
        })
        assert r4.status_code == 409
        assert r4.json()["error"]["code"] == "cycle_closed"

        # GET state shows closed
        rs = client.get(f"/cycles/{cid}")
        assert rs.json()["status"] == "closed"
        assert rs.json()["close_id"] == "http-close-1"


def test_http_close_revision_and_amount_conflicts():
    client = TestClient(app)
    with client:
        cid = "http-close-conflict-cycle"
        client.post(f"/cycles/{cid}/batches", json={
            "version": V, "idempotency_key": "k1", "expected_revision": 0,
            "intervals": [{
                "start": "2024-07-06T00:00:00-04:00",
                "end": "2024-07-06T01:00:00-04:00",
                "timezone": "America/New_York",
                "energy": "60",
            }],
        })
        r = client.post(f"/cycles/{cid}/close", json={
            "close_id": "x1", "expected_revision": 5,
            "expected_total_amount": "9.00",
        })
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "revision_conflict"

        r = client.post(f"/cycles/{cid}/close", json={
            "close_id": "x1", "expected_revision": 1,
            "expected_total_amount": "8.00",
        })
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "total_amount_mismatch"

        # failed closes left the cycle open and no snapshot behind
        rg = client.get(f"/cycles/{cid}/close")
        assert rg.status_code == 404
        assert rg.json()["error"]["code"] == "closure_not_found"


def test_http_close_unknown_and_unclosed():
    client = TestClient(app)
    with client:
        r = client.post("/cycles/no-such-cycle/close", json={
            "close_id": "x", "expected_revision": 0,
            "expected_total_amount": "0.00",
        })
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "cycle_not_found"

        r = client.get("/cycles/no-such-cycle/close")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "cycle_not_found"
