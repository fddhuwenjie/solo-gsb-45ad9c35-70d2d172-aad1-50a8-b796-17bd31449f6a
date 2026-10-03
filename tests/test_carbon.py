"""Tests for traceable carbon-emission snapshots of billing cycles.

Like the settlement/closing tests, these run against a temporary
*file-backed* SQLite database (WAL, real write locks) so concurrent report
generation exercises the actual serialization guarantees.
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

from app import carbon, closing, settlement  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    BillingCycle,
    CarbonReport,
    CycleInterval,
    UsageBatch,
    configure_sqlite_engine,
    seed,
)
from app.main import app  # noqa: E402
from app.pricing import PricingError  # noqa: E402
from app.schemas import (  # noqa: E402
    CarbonReportRequest,
    CycleCloseRequest,
    QuoteIntervalIn,
)

V = "us-ny-demo-v2024"
# 2024-07-06 is a Saturday in NY -> only the `leisure` rule
# (tiers: 100 @0.15, 300 @0.18, open @0.22).
# 00:00-01:00 -04:00 == 04:00-05:00 UTC, etc.
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
# Flat factor covering both DAY1 and DAY2 (04:00-06:00 UTC).
FLAT = [{"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.4"}]


@pytest.fixture()
def tmp_engine(tmp_path):
    path = tmp_path / "carbon.db"
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


def _gen(eng, cycle, rev, factors, factor_version="grid-2024-v1"):
    req = CarbonReportRequest(
        factor_version=factor_version, expected_revision=rev,
        factors=[{"start": f["start"], "end": f["end"],
                  "factor": f["factor"]} for f in factors])
    return carbon.generate_report(cycle, req, engine=eng)


def _two_batch_cycle(eng, cycle="c1"):
    _submit(eng, cycle, "k1", 0, [DAY1])
    _submit(eng, cycle, "k2", 1, [DAY2])
    return cycle


def _count(eng, model):
    with Session(eng) as s:
        return len(s.scalars(select(model)).all())


# ------------------------------------------------------------- happy paths


def test_generate_report_flat_factor(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    res = _gen(eng, "c1", 2, FLAT)
    assert res.replayed is False
    body = res.body
    assert body["cycle_id"] == "c1"
    assert body["revision"] == 2
    assert body["cycle_status"] == "open"
    assert body["version"] == V
    assert body["factor_version"] == "grid-2024-v1"
    assert body["factor_unit"] == "kgCO2e/kWh"
    assert body["emission_unit"] == "kgCO2e"
    assert Decimal(body["total_energy"]) == Decimal("120")
    # 120 kWh x 0.4 kgCO2e/kWh = 48 kgCO2e
    assert Decimal(body["total_emissions"]) == Decimal("48.000000")
    assert len(body["content_digest"]) == 64
    assert len(body["report_digest"]) == 64
    assert body["created_at"]
    # second interval crosses the 100 kWh cumulative tier boundary, so the
    # pricing engine splits it: 3 atomic segments (60 | 40 tier1, 20 tier2)
    assert len(body["segments"]) == 3
    metas = [(s["interval_seq"], s["revision"], s["batch_key"],
              s["tier_ordinal"], Decimal(s["energy"]),
              Decimal(s["emissions"])) for s in body["segments"]]
    assert metas == [
        (0, 1, "k1", 1, Decimal("60.000000"), Decimal("24.000000")),
        (1, 2, "k2", 1, Decimal("40.000000"), Decimal("16.000000")),
        (1, 2, "k2", 2, Decimal("20.000000"), Decimal("8.000000")),
    ]
    for seg in body["segments"]:
        assert seg["rule_code"] == "leisure"
        assert seg["factor"] == "0.4"
    assert [s["seq"] for s in body["segments"]] == [0, 1, 2]
    # report persisted
    assert _count(eng, CarbonReport) == 1


def test_factor_boundary_splits_interval(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "c1", "k1", 0, [DAY1])  # 04:00-05:00 UTC, 60 kWh
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T04:30:00Z",
         "factor": "0.4"},
        {"start": "2024-07-06T04:30:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.6"},
    ]
    body = _gen(eng, "c1", 1, factors).body
    segs = body["segments"]
    assert len(segs) == 2
    assert [s["factor"] for s in segs] == ["0.4", "0.6"]
    assert [Decimal(s["energy"]) for s in segs] == [
        Decimal("30.000000"), Decimal("30.000000")]
    assert [Decimal(s["emissions"]) for s in segs] == [
        Decimal("12.000000"), Decimal("18.000000")]
    assert segs[0]["end"] == segs[1]["start"] == "2024-07-06T04:30:00Z"
    assert Decimal(body["total_emissions"]) == Decimal("30.000000")


def test_tier_boundary_segments_carry_rule_and_tier(tmp_engine):
    eng, _ = tmp_engine
    big = dict(DAY1, end="2024-07-06T02:00:00-04:00", energy="150")
    _submit(eng, "c1", "k1", 0, [big])  # 04:00-06:00 UTC, 150 kWh
    factors = [{"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T06:00:00Z",
                "factor": "0.5"}]
    body = _gen(eng, "c1", 1, factors).body
    segs = body["segments"]
    # cumulative tier boundary at 100 kWh splits the quote into two segments
    assert [s["tier_ordinal"] for s in segs] == [1, 2]
    assert [s["rule_code"] for s in segs] == ["leisure", "leisure"]
    assert [Decimal(s["energy"]) for s in segs] == [
        Decimal("100.000000"), Decimal("50.000000")]
    assert [Decimal(s["emissions"]) for s in segs] == [
        Decimal("50.000000"), Decimal("25.000000")]
    assert Decimal(body["total_emissions"]) == Decimal("75.000000")


def test_union_of_all_boundaries(tmp_engine):
    """Usage, pricing-segment and factor boundaries all become cuts."""
    eng, _ = tmp_engine
    big = dict(DAY1, end="2024-07-06T02:00:00-04:00", energy="150")
    _submit(eng, "c1", "k1", 0, [big])  # tier boundary at 05:20 UTC
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.4"},
        {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.6"},
    ]
    body = _gen(eng, "c1", 1, factors).body
    # cuts at 04:00 | 05:00 (factor) | 05:20 (tier) | 06:00 -> 3 atomic segs
    spans = [(s["start"], s["end"]) for s in body["segments"]]
    assert spans == [
        ("2024-07-06T04:00:00Z", "2024-07-06T05:00:00Z"),
        ("2024-07-06T05:00:00Z", "2024-07-06T05:20:00Z"),
        ("2024-07-06T05:20:00Z", "2024-07-06T06:00:00Z"),
    ]
    assert [s["factor"] for s in body["segments"]] == ["0.4", "0.6", "0.6"]
    assert [s["tier_ordinal"] for s in body["segments"]] == [1, 1, 2]
    total_e = sum(Decimal(s["energy"]) for s in body["segments"])
    total_c = sum(Decimal(s["emissions"]) for s in body["segments"])
    assert total_e == Decimal("150.000000")
    # 75x0.4 + 25x0.6 + 50x0.6 = 30 + 15 + 30 = 75
    assert total_c == Decimal("75.000000")
    assert Decimal(body["total_emissions"]) == total_c


def test_energy_and_emissions_conserved_with_remainder(tmp_engine):
    """Six-decimal residuals land on the last atomic segment of each
    original interval; per-interval sums reconcile exactly."""
    eng, _ = tmp_engine
    tiny = {
        "start": "2024-07-06T00:00:00-04:00",
        "end": "2024-07-06T00:00:03-04:00",
        "timezone": "America/New_York",
        "energy": "1",
    }
    _submit(eng, "c1", "k1", 0, [tiny])  # 04:00:00-04:00:03 UTC, 1 kWh
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T04:00:01Z",
         "factor": "0.1"},
        {"start": "2024-07-06T04:00:01Z", "end": "2024-07-06T04:00:02Z",
         "factor": "0.2"},
        {"start": "2024-07-06T04:00:02Z", "end": "2024-07-06T04:00:03Z",
         "factor": "0.3"},
    ]
    body = _gen(eng, "c1", 1, factors).body
    segs = body["segments"]
    assert len(segs) == 3
    energies = [Decimal(s["energy"]) for s in segs]
    # 1/3 kWh per second at 6 dp; residual on the last segment
    assert energies == [Decimal("0.333333"), Decimal("0.333333"),
                        Decimal("0.333334")]
    assert sum(energies) == Decimal("1.000000")
    emissions = [Decimal(s["emissions"]) for s in segs]
    # exact total = 0.333333*0.1 + 0.333333*0.2 + 0.333334*0.3 = 0.2000001
    # -> 0.200000 at 6 dp; first two segments rounded, residual on last
    assert emissions[0] == Decimal("0.033333")
    assert emissions[1] == Decimal("0.066667")
    assert sum(emissions) == Decimal("0.200000")
    assert Decimal(body["total_emissions"]) == sum(emissions)


def test_factor_order_and_equivalent_decimals_replay(tmp_engine):
    """Factor list order and equivalent decimal spellings do not change
    the content digest: the second request replays the first report."""
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors_ab = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.40"},
        {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "5e-1"},
    ]
    first = _gen(eng, "c1", 2, factors_ab)
    factors_ba = [
        {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.5"},
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.4"},
    ]
    replay = _gen(eng, "c1", 2, factors_ba)
    assert replay.replayed is True
    assert replay.body == first.body
    assert _count(eng, CarbonReport) == 1


# ------------------------------------------------------------- idempotency


def test_same_content_replays_report(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    first = _gen(eng, "c1", 2, FLAT).body
    replay = _gen(eng, "c1", 2, FLAT)
    assert replay.replayed is True
    assert replay.body == first
    assert _count(eng, CarbonReport) == 1


def test_same_revision_different_factors_conflict(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _gen(eng, "c1", 2, FLAT)
    changed = [dict(FLAT[0], factor="0.45")]
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, changed)
    assert ei.value.code == "carbon_conflict"
    assert ei.value.status_code == 409
    det = ei.value.details
    assert det["existing"]["factor_version"] == "grid-2024-v1"
    assert det["received"]["content_digest"] != det["existing"]["content_digest"]
    assert _count(eng, CarbonReport) == 1


def test_same_revision_different_factor_version_conflict(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _gen(eng, "c1", 2, FLAT, factor_version="grid-2024-v1")
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, FLAT, factor_version="grid-2024-v2")
    assert ei.value.code == "carbon_conflict"
    assert ei.value.details["received"]["factor_version"] == "grid-2024-v2"
    assert _count(eng, CarbonReport) == 1


def test_different_revision_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _submit(eng, "c1", "k1", 0, [DAY1])
    _gen(eng, "c1", 1, FLAT[:1] and [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.4"}])
    _submit(eng, "c1", "k2", 1, [DAY2])  # revision advances to 2
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, FLAT)
    assert ei.value.code == "revision_conflict"
    assert ei.value.details["current_revision"] == 1
    assert _count(eng, CarbonReport) == 1


# ------------------------------------------------------------- rejections


def test_revision_mismatch_rejects_and_writes_nothing(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 1, FLAT)
    assert ei.value.code == "revision_conflict"
    assert ei.value.details["expected_revision"] == 1
    assert ei.value.details["current_revision"] == 2
    assert _count(eng, CarbonReport) == 0


def test_unknown_cycle_404(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        _gen(eng, "missing", 0, FLAT)
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"
    assert _count(eng, CarbonReport) == 0


def test_factor_gap_at_end_rejects(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)  # usage 04:00-06:00 UTC
    factors = [{"start": "2024-07-06T04:00:00Z",
                "end": "2024-07-06T05:30:00Z", "factor": "0.4"}]
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, factors)
    assert ei.value.code == "factor_gap"
    assert ei.value.status_code == 409
    assert ei.value.details["at"] == "2024-07-06T05:30:00Z"
    assert ei.value.details["interval_seq"] == 1
    assert _count(eng, CarbonReport) == 0


def test_factor_gap_in_middle_rejects(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T04:30:00Z",
         "factor": "0.4"},
        {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.4"},
    ]
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, factors)
    assert ei.value.code == "factor_gap"
    assert ei.value.details["at"] == "2024-07-06T04:30:00Z"
    assert _count(eng, CarbonReport) == 0


def test_factor_overlap_rejects(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.4"},
        {"start": "2024-07-06T04:30:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.5"},
    ]
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, factors)
    assert ei.value.code == "factor_overlap"
    assert ei.value.status_code == 400
    assert ei.value.details["factors"] == [0, 1]
    assert _count(eng, CarbonReport) == 0


def test_factor_touching_intervals_allowed(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.4"},
        {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.5"},
    ]
    body = _gen(eng, "c1", 2, factors).body
    assert Decimal(body["total_emissions"]) == Decimal("54.000000")


def test_factor_naive_time_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors = [{"start": "2024-07-06T04:00:00",
                "end": "2024-07-06T06:00:00Z", "factor": "0.4"}]
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, factors)
    assert ei.value.code == "offset_required"
    assert ei.value.status_code == 400
    assert _count(eng, CarbonReport) == 0


def test_factor_invalid_interval_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors = [{"start": "2024-07-06T06:00:00Z",
                "end": "2024-07-06T04:00:00Z", "factor": "0.4"}]
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, factors)
    assert ei.value.code == "invalid_interval"
    assert ei.value.status_code == 400
    assert _count(eng, CarbonReport) == 0


def test_pricing_failure_rejects_and_writes_nothing(tmp_engine, monkeypatch):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)

    def _boom(version, payload, **kwargs):
        raise PricingError("规则缺口", code="rule_gap", status_code=409,
                           details={"at": "2024-07-06T04:00:00Z"})

    monkeypatch.setattr(carbon.pricing, "quote", _boom)
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, FLAT)
    assert ei.value.code == "rule_gap"
    assert ei.value.status_code == 409
    assert _count(eng, CarbonReport) == 0


# ------------------------------------------------------------- closed cycle


def test_closed_cycle_reads_immutable_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    res = _gen(eng, "c1", 2, FLAT)
    assert res.replayed is False
    body = res.body
    assert body["cycle_status"] == "closed"
    assert body["revision"] == 2
    assert Decimal(body["total_emissions"]) == Decimal("48.000000")
    # segments trace back to the batches frozen in the close snapshot
    # (second interval crosses the 100 kWh tier boundary -> two segments)
    assert [s["batch_key"] for s in body["segments"]] == ["k1", "k2", "k2"]
    assert [s["revision"] for s in body["segments"]] == [1, 2, 2]
    # replay on the closed cycle
    replay = _gen(eng, "c1", 2, FLAT)
    assert replay.replayed is True
    assert replay.body == body


def test_closed_cycle_wrong_revision_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 1, FLAT)
    assert ei.value.code == "revision_conflict"
    assert ei.value.details["current_revision"] == 2
    assert _count(eng, CarbonReport) == 0


def test_closed_cycle_same_revision_different_factors_conflict(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "close-1", 2, "18.60")
    _gen(eng, "c1", 2, FLAT)
    with pytest.raises(PricingError) as ei:
        _gen(eng, "c1", 2, [dict(FLAT[0], factor="0.9")])
    assert ei.value.code == "carbon_conflict"
    assert _count(eng, CarbonReport) == 1


# --------------------------------------------------------------- read model


def test_get_report_not_found_states(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        carbon.get_report("missing", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"

    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        carbon.get_report("c1", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "carbon_report_not_found"


def test_get_report_returns_stored_body(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    body = _gen(eng, "c1", 2, FLAT).body
    assert carbon.get_report("c1", engine=eng) == body


def test_generation_does_not_touch_cycle_state(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    before = settlement.get_cycle("c1", engine=eng)
    _gen(eng, "c1", 2, FLAT)
    after = settlement.get_cycle("c1", engine=eng)
    assert before == after  # revision/totals/updated_at all unchanged
    # a later batch still advances the revision normally
    day3 = dict(DAY2, start="2024-07-06T02:00:00-04:00",
                end="2024-07-06T03:00:00-04:00")
    res = _submit(eng, "c1", "k3", 2, [day3])
    assert res.body["revision"] == 3


# ------------------------------------------------------------- concurrency


def test_concurrent_same_content_generates_one_report(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng, "race")
    start = threading.Event()
    outcomes: list = []

    def worker():
        start.wait()
        try:
            outcomes.append(_gen(eng, "race", 2, FLAT))
        except PricingError as e:
            outcomes.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    oks = [r for r in outcomes if not isinstance(r, PricingError)]
    assert len(oks) == 4, outcomes
    assert sum(1 for r in oks if not r.replayed) == 1
    assert all(r.body == oks[0].body for r in oks)
    assert _count(eng, CarbonReport) == 1


def test_concurrent_different_factors_only_one_wins(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng, "race")
    start = threading.Event()
    outcomes: list = []

    def worker(factor):
        start.wait()
        try:
            outcomes.append(_gen(eng, "race", 2, [dict(FLAT[0],
                                                       factor=factor)]))
        except PricingError as e:
            outcomes.append(e)

    threads = [threading.Thread(target=worker, args=(f,))
               for f in ("0.4", "0.5", "0.6", "0.7")]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    oks = [r for r in outcomes if not isinstance(r, PricingError)]
    errs = [r for r in outcomes if isinstance(r, PricingError)]
    assert len(oks) == 1, outcomes
    assert all(e.code == "carbon_conflict" for e in errs)
    assert _count(eng, CarbonReport) == 1


# ------------------------------------------------------------- persistence


def test_report_survives_restart(tmp_engine):
    eng, path = tmp_engine
    _two_batch_cycle(eng)
    body = _gen(eng, "c1", 2, FLAT).body
    eng.dispose()

    eng2 = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng2)
    try:
        assert carbon.get_report("c1", engine=eng2) == body
        replay = _gen(eng2, "c1", 2, FLAT)
        assert replay.replayed is True
        assert replay.body == body
        with pytest.raises(PricingError) as ei:
            _gen(eng2, "c1", 2, [dict(FLAT[0], factor="0.9")])
        assert ei.value.code == "carbon_conflict"
    finally:
        eng2.dispose()


# ------------------------------------------------------------- HTTP surface


def test_http_carbon_full_flow():
    client = TestClient(app)
    with client:
        cid = "http-carbon-cycle"
        for key, rev, iv in (("k1", 0, DAY1), ("k2", 1, DAY2)):
            r = client.post(f"/cycles/{cid}/batches", json={
                "version": V,
                "idempotency_key": key,
                "expected_revision": rev,
                "intervals": [iv],
            })
            assert r.status_code == 201, r.text

        payload = {
            "factor_version": "grid-2024-v1",
            "expected_revision": 2,
            "factors": [{
                "start": "2024-07-06T04:00:00Z",
                "end": "2024-07-06T06:00:00Z",
                "factor": "0.4",
            }],
        }
        r = client.post(f"/cycles/{cid}/carbon", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["revision"] == 2
        assert body["cycle_status"] == "open"
        assert body["total_emissions"] == "48.000000"
        # second interval crosses the 100 kWh tier boundary -> 3 segments
        assert len(body["segments"]) == 3

        # replay -> 200, identical body
        r2 = client.post(f"/cycles/{cid}/carbon", json=payload)
        assert r2.status_code == 200, r2.text
        assert r2.json() == body

        # GET -> same body
        rg = client.get(f"/cycles/{cid}/carbon")
        assert rg.status_code == 200
        assert rg.json() == body

        # same revision, different factors -> 409 carbon_conflict
        r3 = client.post(f"/cycles/{cid}/carbon", json={
            **payload,
            "factors": [{"start": "2024-07-06T04:00:00Z",
                         "end": "2024-07-06T06:00:00Z", "factor": "0.5"}],
        })
        assert r3.status_code == 409
        assert r3.json()["error"]["code"] == "carbon_conflict"

        # different revision -> 409 revision_conflict
        r4 = client.post(f"/cycles/{cid}/carbon", json={
            **payload, "expected_revision": 1})
        assert r4.status_code == 409
        assert r4.json()["error"]["code"] == "revision_conflict"


def test_http_carbon_validation_and_gap_errors():
    client = TestClient(app)
    with client:
        cid = "http-carbon-errors"
        r = client.post(f"/cycles/{cid}/batches", json={
            "version": V, "idempotency_key": "k1", "expected_revision": 0,
            "intervals": [DAY1],
        })
        assert r.status_code == 201, r.text

        # unknown cycle
        r = client.post("/cycles/no-such-cycle/carbon", json={
            "factor_version": "g", "expected_revision": 0,
            "factors": [{"start": "2024-07-06T04:00:00Z",
                         "end": "2024-07-06T05:00:00Z", "factor": "0.4"}],
        })
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "cycle_not_found"
        r = client.get("/cycles/no-such-cycle/carbon")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "cycle_not_found"

        # report not yet generated
        r = client.get(f"/cycles/{cid}/carbon")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "carbon_report_not_found"

        # factor gap -> 409
        r = client.post(f"/cycles/{cid}/carbon", json={
            "factor_version": "g", "expected_revision": 1,
            "factors": [{"start": "2024-07-06T04:00:00Z",
                         "end": "2024-07-06T04:30:00Z", "factor": "0.4"}],
        })
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "factor_gap"

        # factor overlap -> 400
        r = client.post(f"/cycles/{cid}/carbon", json={
            "factor_version": "g", "expected_revision": 1,
            "factors": [
                {"start": "2024-07-06T04:00:00Z",
                 "end": "2024-07-06T05:00:00Z", "factor": "0.4"},
                {"start": "2024-07-06T04:30:00Z",
                 "end": "2024-07-06T05:00:00Z", "factor": "0.5"},
            ],
        })
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "factor_overlap"

        # negative factor -> 422
        r = client.post(f"/cycles/{cid}/carbon", json={
            "factor_version": "g", "expected_revision": 1,
            "factors": [{"start": "2024-07-06T04:00:00Z",
                         "end": "2024-07-06T05:00:00Z", "factor": "-0.1"}],
        })
        assert r.status_code == 422

        # nothing was persisted by the failures
        r = client.get(f"/cycles/{cid}/carbon")
        assert r.status_code == 404


def test_http_carbon_after_close():
    client = TestClient(app)
    with client:
        cid = "http-carbon-closed"
        for key, rev, iv in (("k1", 0, DAY1), ("k2", 1, DAY2)):
            r = client.post(f"/cycles/{cid}/batches", json={
                "version": V, "idempotency_key": key,
                "expected_revision": rev, "intervals": [iv],
            })
            assert r.status_code == 201, r.text
        r = client.post(f"/cycles/{cid}/close", json={
            "close_id": "close-carbon-1", "expected_revision": 2,
            "expected_total_amount": "18.60",
        })
        assert r.status_code == 201, r.text

        r = client.post(f"/cycles/{cid}/carbon", json={
            "factor_version": "grid-2024-v1", "expected_revision": 2,
            "factors": [{"start": "2024-07-06T04:00:00Z",
                         "end": "2024-07-06T06:00:00Z", "factor": "0.4"}],
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["cycle_status"] == "closed"
        assert body["total_emissions"] == "48.000000"

        rg = client.get(f"/cycles/{cid}/carbon")
        assert rg.status_code == 200
        assert rg.json() == body
