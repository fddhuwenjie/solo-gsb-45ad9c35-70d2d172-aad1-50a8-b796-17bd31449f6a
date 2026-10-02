"""Tests for traceable, immutable billing-cycle carbon snapshots.

Like settlement/closing tests, these run against a temporary *file-backed*
SQLite database (WAL, real write locks) so same-revision races between carbon
snapshots, batches and closes exercise the real serialization guarantees.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
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
    CarbonSnapshot,
    configure_sqlite_engine,
    seed,
)
from app.importer import canonical_json  # noqa: E402
from app.main import app  # noqa: E402
from app.pricing import PricingError  # noqa: E402
from app.schemas import (  # noqa: E402
    CarbonFactorIn,
    CarbonSnapshotRequest,
    CycleCloseRequest,
    QuoteIntervalIn,
)
import hashlib  # noqa: E402

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
# UTC: DAY1 = 04:00Z-05:00Z, DAY2 = 05:00Z-06:00Z (same July day, EDT -04:00).
F_FLAT = lambda v="0.300": [  # noqa: E731
    {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T06:00:00Z",
     "factor": v}]
# Two abutting factors split at the usage midpoint (05:00Z).
F_TWO = [
    {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
     "factor": "0.300"},
    {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
     "factor": "0.500"},
]


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


def _carbon(eng, cycle, factor_version, rev, factors):
    req = CarbonSnapshotRequest(
        expected_revision=rev, factor_version=factor_version,
        factors=[CarbonFactorIn(**f) for f in factors])
    return carbon.create_carbon_snapshot(cycle, req, engine=eng)


def _close(eng, cycle, close_id, rev, total):
    req = CycleCloseRequest(close_id=close_id, expected_revision=rev,
                            expected_total_amount=Decimal(total))
    return closing.close_cycle(cycle, req, engine=eng)


def _two_batch_cycle(eng, cycle="c1"):
    _submit(eng, cycle, "k1", 0, [DAY1])
    _submit(eng, cycle, "k2", 1, [DAY2])
    return cycle


def _sums(report):
    return (sum(Decimal(s["energy"]) for s in report["segments"]),
            sum(Decimal(s["amount"]) for s in report["segments"]),
            sum(Decimal(s["emission"]) for s in report["segments"]))


# ------------------------------------------------------------- happy paths


def test_carbon_snapshot_basic_segments_and_totals(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    res = _carbon(eng, "c1", "grid-v1", 2, F_TWO)
    assert res.replayed is False
    rep = res.body
    assert rep["factor_version"] == "grid-v1"
    assert rep["cycle_id"] == "c1"
    assert rep["version"] == V
    assert rep["revision"] == 2
    assert rep["status"] == "open"
    assert rep["emission_unit"] == "kgCO2e"
    assert rep["energy_unit"] == "kWh"
    assert Decimal(rep["total_energy"]) == Decimal("120")
    assert Decimal(rep["total_amount"]) == Decimal("18.60")
    # 60 kWh @0.3 + 40 @0.5 (tier1) + 20 @0.5 (tier2) = 18+20+10 = 48
    assert Decimal(rep["total_emission"]) == Decimal("48.000000")

    segs = rep["segments"]
    # factor boundary 05:00Z + tier boundary at 100 kWh (05:40Z)
    assert [(s["start"][11:16], s["end"][11:16]) for s in segs] == [
        ("04:00", "05:00"), ("05:00", "05:40"), ("05:40", "06:00")]
    assert [s["factor"] for s in segs] == ["0.3", "0.5", "0.5"]
    assert [s["tier_ordinal"] for s in segs] == [1, 1, 2]
    assert [Decimal(s["emission"]) for s in segs] == [
        Decimal("18.000000"), Decimal("20.000000"), Decimal("10.000000")]
    # batch attribution
    assert segs[0]["batch_revision"] == 1
    assert segs[0]["idempotency_key"] == "k1"
    assert segs[0]["interval_seq"] == 0
    assert segs[1]["batch_revision"] == 2
    assert segs[1]["idempotency_key"] == "k2"
    assert segs[1]["interval_seq"] == 1
    # traceable factor provenance window
    assert segs[0]["factor_start"].endswith("04:00:00Z")
    assert segs[0]["factor_end"].endswith("05:00:00Z")
    # segments half-open partition (adjacent share an endpoint)
    for a, b in zip(segs, segs[1:]):
        assert a["end"] == b["start"]


def test_energy_amount_emission_conservation(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    rep = _carbon(eng, "c1", "g", 2, F_TWO).body
    energy, amount, emission = _sums(rep)
    assert energy == Decimal("120.000000")
    assert amount == Decimal("18.600000")
    assert emission == Decimal(rep["total_emission"])
    # per-original-interval energy conservation (grouped by interval_seq)
    per_e = defaultdict(Decimal)
    per_em = defaultdict(Decimal)
    per_am = defaultdict(Decimal)
    for s in rep["segments"]:
        per_e[s["interval_seq"]] += Decimal(s["energy"])
        per_em[s["interval_seq"]] += Decimal(s["emission"])
        per_am[s["interval_seq"]] += Decimal(s["amount"])
    assert per_e == {0: Decimal("60.000000"), 1: Decimal("60.000000")}
    # per-interval emissions: 60@0.3=18 ; 40@0.5+20@0.5=30
    assert per_em == {0: Decimal("18.000000"), 1: Decimal("30.000000")}
    # per-interval amounts tie to the quote tiers: interval0 all tier1 9.00;
    # interval1 = 40@0.15 + 20@0.18 = 6+3.6 = 9.60
    assert per_am == {0: Decimal("9.000000"), 1: Decimal("9.600000")}


def test_flat_factor_matches_quote_segments(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    rep = _carbon(eng, "c1", "g", 2, F_FLAT("0.300")).body
    quote = rep["quote"]
    # with a single factor spanning all usage, carbon segments mirror the
    # quote segments one-to-one
    assert len(rep["segments"]) == len(quote["segments"])
    for cs, qs in zip(rep["segments"], quote["segments"]):
        assert cs["start"] == qs["start"] and cs["end"] == qs["end"]
        assert Decimal(cs["energy"]) == Decimal(qs["energy"])
        assert Decimal(cs["amount"]) == Decimal(qs["amount"])
        assert cs["rule_code"] == qs["rule_code"]
        assert cs["tier_ordinal"] == qs["tier_ordinal"]
        # emission == energy * factor
        assert Decimal(cs["emission"]) == (
            Decimal(cs["energy"]) * Decimal("0.300")).quantize(Decimal("0.000001"))
    # total 120 * 0.3 = 36
    assert Decimal(rep["total_emission"]) == Decimal("36.000000")


def test_factors_may_extend_beyond_usage(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    factors = [
        {"start": "2024-07-01T00:00:00Z", "end": "2024-07-31T00:00:00Z",
         "factor": "0.250"}]
    rep = _carbon(eng, "c1", "g", 2, factors).body
    assert Decimal(rep["total_emission"]) == Decimal("30.000000")  # 120*0.25


def test_out_of_order_batches_attribute_by_utc_time(tmp_engine):
    """Batches may arrive in non-chronological order; pricing interval_index
    is assigned after UTC (start,end) sort, so carbon segments must attribute
    batch/seq by UTC time, not by storage (revision, seq) order."""
    eng, _ = tmp_engine
    # submit local hours 3, 1, 2 in scrambled order (= UTC 07, 05, 06 start)
    plan = [("k3", 3, 1), ("k1", 1, 2), ("k2", 2, 3)]
    for key, hour, rev_after in plan:
        iv = {
            "start": f"2024-07-06T0{hour}:00:00-04:00",
            "end": f"2024-07-06T0{hour + 1}:00:00-04:00",
            "timezone": "America/New_York", "energy": "10"}
        r = _submit(eng, "coo", key, rev_after - 1, [iv])
        assert r.body["revision"] == rev_after
    rep = _carbon(eng, "coo", "g", 3, [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.200"},
        {"start": "2024-07-06T06:00:00Z", "end": "2024-07-06T08:00:00Z",
         "factor": "0.800"}]).body
    segs = rep["segments"]
    # UTC ordering with correct owning batch key
    got = [(s["start"][11:16], s["idempotency_key"], s["batch_revision"])
           for s in segs]
    assert got == [
        ("05:00", "k1", 2), ("06:00", "k2", 3), ("07:00", "k3", 1)]
    # emissions: 10@0.2 + 10@0.8 + 10@0.8 = 2+8+8 = 18
    assert Decimal(rep["total_emission"]) == Decimal("18.000000")
    # energy still conserved overall
    assert sum(Decimal(s["energy"]) for s in segs) == Decimal("30.000000")


def test_factor_boundary_splits_pricing_segment_amount(tmp_engine):
    """A pricing segment cut by a factor boundary keeps amount & energy
    conserved across the pieces."""
    eng, _ = tmp_engine
    _submit(eng, "c1", "k1", 0, [{  # single 2h interval, 120 kWh
        "start": "2024-07-06T00:00:00-04:00",
        "end": "2024-07-06T02:00:00-04:00",
        "timezone": "America/New_York", "energy": "120"}])
    # factor change at 04:30Z (inside the first hour / first pricing tier)
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T04:30:00Z",
         "factor": "0.100"},
        {"start": "2024-07-06T04:30:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.200"}]
    rep = _carbon(eng, "c1", "g", 1, factors).body
    energy, amount, emission = _sums(rep)
    assert energy == Decimal("120.000000")
    assert amount == Decimal("18.600000")
    # 30 kWh @0.1 + 90 kWh @0.2 = 3 + 18 = 21
    assert emission == Decimal("21.000000")
    assert Decimal(rep["total_emission"]) == Decimal("21.000000")


# ------------------------------------------------------------- idempotency


def test_same_factor_version_same_content_replays(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    first = _carbon(eng, "c1", "gv", 2, F_TWO).body
    replay = _carbon(eng, "c1", "gv", 2, F_TWO)
    assert replay.replayed is True
    assert replay.body == first
    with Session(eng) as s:
        rows = s.scalars(select(CarbonSnapshot)).all()
        assert len(rows) == 1


def test_same_factor_version_different_factor_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _carbon(eng, "c1", "gv", 2, F_TWO)
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 2, F_FLAT("0.999"))
    assert ei.value.code == "carbon_conflict"
    det = ei.value.details
    assert det["factor_version"] == "gv"
    assert det["existing"]["revision"] == 2
    assert "content_digest" in det["existing"]
    with Session(eng) as s:
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 1


def test_same_factor_version_different_revision_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _carbon(eng, "c1", "gv", 2, F_TWO)
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 1, F_TWO)
    assert ei.value.code == "carbon_conflict"
    assert ei.value.details["received"]["revision"] == 1


def test_different_factor_versions_coexist(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    a = _carbon(eng, "c1", "gv-a", 2, F_FLAT("0.3")).body
    b = _carbon(eng, "c1", "gv-b", 2, F_FLAT("0.5")).body
    assert Decimal(a["total_emission"]) == Decimal("36.000000")
    assert Decimal(b["total_emission"]) == Decimal("60.000000")
    with Session(eng) as s:
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 2
    # each retrievable independently and byte-identical
    assert carbon.get_carbon_snapshot("c1", "gv-a", engine=eng) == a
    assert carbon.get_carbon_snapshot("c1", "gv-b", engine=eng) == b


def test_replay_is_byte_identical_ignoring_request_order(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    first = _carbon(eng, "c1", "gv", 2, list(reversed(F_TWO))).body
    # factors supplied in different order -> normalized, same digest, replay
    replay = _carbon(eng, "c1", "gv", 2, F_TWO)
    assert replay.replayed is True
    assert replay.body == first


# ------------------------------------------------------------- revision gate


def test_open_cycle_requires_matching_revision(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 1, F_TWO)
    assert ei.value.code == "revision_conflict"
    assert ei.value.details["current_revision"] == 2
    assert ei.value.details["expected_revision"] == 1
    with Session(eng) as s:
        assert s.get(BillingCycle, "c1").status == "open"
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 0


def test_unknown_cycle_404(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "missing", "gv", 0, F_TWO)
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"


# ------------------------------------------------------------- factor errors


def test_factor_gap_rejects_and_persists_nothing(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    gap = [{"start": "2024-07-06T04:00:00Z",
            "end": "2024-07-06T05:00:00Z", "factor": "0.3"}]
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 2, gap)
    assert ei.value.code == "factor_gap"
    assert ei.value.details["gap_start"].endswith("05:00:00Z")
    with Session(eng) as s:
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 0


def test_factor_overlap_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    overlap = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:30:00Z",
         "factor": "0.3"},
        {"start": "2024-07-06T05:00:00Z", "end": "2024-07-06T06:00:00Z",
         "factor": "0.4"}]
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 2, overlap)
    assert ei.value.code == "factor_overlap"
    assert "factor_a" in ei.value.details


def test_factor_requires_utc_offset(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    naive = [{"start": "2024-07-06T04:00:00",
              "end": "2024-07-06T06:00:00", "factor": "0.3"}]
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 2, naive)
    assert ei.value.code == "offset_required"


def test_invalid_factor_interval_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    bad = [{"start": "2024-07-06T06:00:00Z",
            "end": "2024-07-06T04:00:00Z", "factor": "0.3"}]
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 2, bad)
    assert ei.value.code == "invalid_interval"


def test_invalid_factor_version_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    for bad in ("../x", "-lead", "a b", "a/b"):
        with pytest.raises(PricingError) as ei:
            _carbon(eng, "c1", bad, 2, F_TWO)
        assert ei.value.code == "invalid_factor_version"


def test_negative_factor_rejected_at_schema():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CarbonFactorIn(start="2024-07-06T04:00:00Z",
                       end="2024-07-06T06:00:00Z", factor="-0.001")


# ----------------------------------------------------- pricing failure passthrough


def test_rule_gap_rejects_whole_batch(tmp_engine):
    """If the carbon re-quote hits a pricing rule_gap, the whole snapshot is
    rejected and nothing is persisted."""
    eng, _ = tmp_engine
    from app.database import (
        CycleInterval, Holiday, RateRule, TariffVersion, Tier)
    # Custom version with a single whole-day rule covering only 2024-07-06.
    with Session(eng) as s:
        ver = TariffVersion(code="gap-v1", currency="USD",
                            energy_unit="kWh", description="single day")
        rule = RateRule(
            code="all", name="only 7/6",
            effective_from="2024-07-06T00:00:00",
            effective_to="2024-07-07T00:00:00",
            timezone="America/New_York", weekdays="", day_scope="all",
            time_on="00:00", time_off="24:00")
        rule.tiers.append(Tier(ordinal=1, cumulative_cap=None,
                               unit_price="0.10"))
        ver.rules.append(rule)
        s.add(ver)
        s.commit()
    _submit(eng, "cgap", "k1", 0, [{
        "start": "2024-07-06T00:00:00-04:00",
        "end": "2024-07-06T01:00:00-04:00",
        "timezone": "America/New_York", "energy": "10"}], version="gap-v1")
    # Append a stored interval outside the rule window: batch submit for that
    # day would itself fail, so insert the row directly to make the carbon
    # re-quote encounter rule_gap.
    with Session(eng) as s:
        s.add(CycleInterval(
            cycle_id="cgap", revision=1, batch_id=None,
            start_utc="2024-07-08T04:00:00Z", end_utc="2024-07-08T05:00:00Z",
            timezone="America/New_York", energy="5", seq=1))
        cyc = s.get(BillingCycle, "cgap")
        cyc.total_energy = "15"
        s.commit()
    factors = [
        {"start": "2024-07-06T04:00:00Z", "end": "2024-07-06T05:00:00Z",
         "factor": "0.3"},
        {"start": "2024-07-08T04:00:00Z", "end": "2024-07-08T05:00:00Z",
         "factor": "0.3"}]
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "cgap", "gv", 1, factors)
    assert ei.value.code == "rule_gap"
    with Session(eng) as s:
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 0


# --------------------------------------------------------------- closed cycle


def test_closed_cycle_reads_immutable_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    first = _carbon(eng, "c1", "gv", 2, F_TWO).body
    _close(eng, "c1", "cl1", 2, "18.60")
    # same content still replays after close (revision pinned at closure)
    replay = _carbon(eng, "c1", "gv", 2, F_TWO)
    assert replay.replayed is True
    assert replay.body == first
    # GET returns the same immutable report
    assert carbon.get_carbon_snapshot("c1", "gv", engine=eng) == first
    assert replay.body["status"] == "open"  # status captured at creation


def test_closed_cycle_new_factor_version_rejected(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _carbon(eng, "c1", "gv", 2, F_TWO)
    _close(eng, "c1", "cl1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv-new", 2, F_TWO)
    assert ei.value.code == "cycle_closed"
    with Session(eng) as s:
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 1


def test_closed_cycle_changed_factor_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _carbon(eng, "c1", "gv", 2, F_TWO)
    _close(eng, "c1", "cl1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "gv", 2, F_FLAT("0.4"))
    assert ei.value.code == "carbon_conflict"


def test_carbon_after_close_then_replay_status(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    _close(eng, "c1", "cl1", 2, "18.60")
    # no snapshot exists and cycle already closed -> cycle_closed
    with pytest.raises(PricingError) as ei:
        _carbon(eng, "c1", "late", 2, F_TWO)
    assert ei.value.code == "cycle_closed"


def test_get_snapshot_not_found_states(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        carbon.get_carbon_snapshot("missing", "gv", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"
    _two_batch_cycle(eng)
    with pytest.raises(PricingError) as ei:
        carbon.get_carbon_snapshot("c1", "gv", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "carbon_snapshot_not_found"


# ------------------------------------------------------------- digest/report


def test_report_digest_covers_full_report(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    rep = _carbon(eng, "c1", "gv", 2, F_TWO).body
    doc = dict(rep)
    doc.pop("report_digest")
    expect = hashlib.sha256(
        canonical_json(doc).encode("utf-8")).hexdigest()
    assert rep["report_digest"] == expect
    assert len(rep["report_digest"]) == 64
    assert rep["content_digest"] != rep["report_digest"]


def test_content_digest_changes_with_factor(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng, "cA")
    _two_batch_cycle(eng, "cB")
    a = _carbon(eng, "cA", "gv", 2, F_FLAT("0.3")).body
    b = _carbon(eng, "cB", "gv", 2, F_FLAT("0.4")).body
    assert a["content_digest"] != b["content_digest"]


def test_report_self_contained_quote(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    rep = _carbon(eng, "c1", "gv", 2, F_TWO).body
    q = rep["quote"]
    assert q["version"] == V
    assert Decimal(q["total_amount"]) == Decimal("18.60")
    assert Decimal(q["total_energy"]) == Decimal("120")
    assert q["matched_rules"] == ["leisure"]


# ------------------------------------------------------------- concurrency


def test_concurrent_same_factor_version_same_content_one_wins(tmp_engine):
    eng, _ = tmp_engine
    _two_batch_cycle(eng)
    start = threading.Event()
    outcomes: list = []

    def worker():
        start.wait()
        try:
            outcomes.append(_carbon(eng, "c1", "gv", 2, F_TWO))
        except PricingError as e:
            outcomes.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert all(not isinstance(o, PricingError) for o in outcomes), outcomes
    bodies = {o.body["report_digest"] for o in outcomes}
    assert len(bodies) == 1
    created = [o for o in outcomes if not o.replayed]
    assert len(created) == 1
    with Session(eng) as s:
        assert len(s.scalars(select(CarbonSnapshot)).all()) == 1


def test_concurrent_carbon_and_batch_serialize(tmp_engine):
    """A carbon snapshot does NOT advance the revision (it is a committal
    read), so under the process-wide write lock the two serialize: whichever
    goes first succeeds; the snapshot is always a consistent record of exactly
    one revision. The loser's outcome is determined by ordering, never a
    half-written state.

    * batch first (rev 1->2): the snapshot then reads rev 2 but expected 1 ->
      revision_conflict;
    * carbon first: snapshot recorded at rev 1, then batch advances to rev 2
      and both succeed; the stored snapshot stays pinned to revision 1.
    """
    eng, _ = tmp_engine
    _submit(eng, "race", "base", 0, [DAY1])  # revision 1
    start = threading.Event()
    outcomes: dict = {}

    def do_carbon():
        start.wait()
        try:
            outcomes["carbon"] = _carbon(eng, "race", "gv", 1, [
                {"start": "2024-07-06T04:00:00Z",
                 "end": "2024-07-06T06:00:00Z", "factor": "0.3"}])
        except PricingError as e:
            outcomes["carbon"] = e

    def do_batch():
        start.wait()
        try:
            outcomes["batch"] = _submit(eng, "race", "k2", 1, [DAY2])
        except PricingError as e:
            outcomes["batch"] = e

    t1 = threading.Thread(target=do_carbon)
    t2 = threading.Thread(target=do_batch)
    t1.start(); t2.start()
    start.set(); t1.join(); t2.join()

    carbon_res, batch_res = outcomes["carbon"], outcomes["batch"]
    # the batch can always commit (carbon never moves the revision)
    assert not isinstance(batch_res, PricingError), outcomes
    assert batch_res.body["revision"] == 2
    with Session(eng) as s:
        cycle = s.get(BillingCycle, "race")
        snaps = s.scalars(select(CarbonSnapshot)).all()
        assert cycle.revision == 2
        if not isinstance(carbon_res, PricingError):
            # carbon ran first: exactly one snapshot pinned to revision 1
            assert carbon_res.body["revision"] == 1
            assert len(snaps) == 1
            assert snaps[0].revision == 1
        else:
            # batch ran first: expected_revision 1 is stale -> conflict
            assert carbon_res.code == "revision_conflict"
            assert len(snaps) == 0


def test_concurrent_carbon_and_close_serialize(tmp_engine):
    """Unlike a batch, a close freezes the cycle, so the snapshot/close pair
    has a strict winner determined under the write lock:

    * close first: cycle becomes closed; a *new* factor version cannot create
      a snapshot -> cycle_closed;
    * carbon first (open): snapshot recorded at rev 1, then close proceeds and
      pins the cycle at rev 1; both succeed, snapshot remains replayable.
    """
    eng, _ = tmp_engine
    _submit(eng, "race", "base", 0, [DAY1])  # revision 1, total 9.00
    start = threading.Event()
    outcomes: dict = {}

    def do_carbon():
        start.wait()
        try:
            outcomes["carbon"] = _carbon(eng, "race", "gv", 1, [
                {"start": "2024-07-06T04:00:00Z",
                 "end": "2024-07-06T05:00:00Z", "factor": "0.3"}])
        except PricingError as e:
            outcomes["carbon"] = e

    def do_close():
        start.wait()
        try:
            outcomes["close"] = _close(eng, "race", "cl1", 1, "9.00")
        except PricingError as e:
            outcomes["close"] = e

    t1 = threading.Thread(target=do_carbon)
    t2 = threading.Thread(target=do_close)
    t1.start(); t2.start()
    start.set(); t1.join(); t2.join()

    carbon_res, close_res = outcomes["carbon"], outcomes["close"]
    # close always wins its own transition (it sets status=closed)
    assert not isinstance(close_res, PricingError), outcomes
    with Session(eng) as s:
        cycle = s.get(BillingCycle, "race")
        snaps = s.scalars(select(CarbonSnapshot)).all()
        assert cycle.status == "closed"
        assert cycle.revision == 1
        if not isinstance(carbon_res, PricingError):
            # carbon first: snapshot at rev 1 survives closure, replayable
            assert carbon_res.body["revision"] == 1
            assert len(snaps) == 1
            replay = _carbon(eng, "race", "gv", 1, [
                {"start": "2024-07-06T04:00:00Z",
                 "end": "2024-07-06T05:00:00Z", "factor": "0.3"}])
            assert replay.replayed is True
        else:
            # close first: new factor version on a closed cycle -> cycle_closed
            assert carbon_res.code == "cycle_closed"
            assert len(snaps) == 0


# ------------------------------------------------------------- persistence


def test_snapshot_survives_restart(tmp_engine):
    eng, path = tmp_engine
    _two_batch_cycle(eng)
    rep = _carbon(eng, "c1", "gv", 2, F_TWO).body
    eng.dispose()

    eng2 = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng2)
    try:
        assert carbon.get_carbon_snapshot("c1", "gv", engine=eng2) == rep
        replay = _carbon(eng2, "c1", "gv", 2, F_TWO)
        assert replay.replayed is True
        assert replay.body == rep
        with pytest.raises(PricingError) as ei:
            _carbon(eng2, "c1", "gv", 2, F_FLAT("0.1"))
        assert ei.value.code == "carbon_conflict"
    finally:
        eng2.dispose()


# ------------------------------------------------------------- DST behavior


def test_dst_spring_forward_uses_actual_duration(tmp_engine):
    eng, _ = tmp_engine
    # 2024-03-09 23:30 -> 2024-03-10 05:30 NY local spans 5 UTC hours (18000s).
    _submit(eng, "dst", "k1", 0, [{
        "start": "2024-03-09T23:30:00-05:00",
        "end": "2024-03-10T05:30:00-04:00",
        "timezone": "America/New_York", "energy": "10"}])
    # flat factor with the spring-forward gap inside; energy is split by the
    # engine's UTC-second proportions, so it still conserves to 10 kWh.
    factors = [{"start": "2024-03-10T04:30:00Z",
                "end": "2024-03-10T09:30:00Z", "factor": "0.300"}]
    rep = _carbon(eng, "dst", "gv", 1, factors).body
    energy, _, emission = _sums(rep)
    assert energy == Decimal("10.000000")
    assert emission == Decimal("3.000000")
    total_dur = sum(s["duration_seconds"] for s in rep["segments"])
    assert total_dur == 18000


# ------------------------------------------------------------- HTTP surface


def test_http_carbon_full_flow():
    client = TestClient(app)
    with client:
        cid = "http-carbon-cycle"
        r = client.post(f"/cycles/{cid}/batches", json={
            "version": V, "idempotency_key": "k1", "expected_revision": 0,
            "intervals": [{
                "start": "2024-07-06T00:00:00-04:00",
                "end": "2024-07-06T02:00:00-04:00",
                "timezone": "America/New_York", "energy": "120"}]})
        assert r.status_code == 201, r.text

        payload = {"expected_revision": 1, "factor_version": "gv",
                   "factors": F_TWO}
        r1 = client.post(f"/cycles/{cid}/carbon", json=payload)
        assert r1.status_code == 201, r1.text
        body = r1.json()
        assert body["total_emission"] == "48.000000"
        assert body["revision"] == 1
        assert body["report_digest"]
        assert len(body["segments"]) == 3

        # replay -> 200 byte-identical
        r2 = client.post(f"/cycles/{cid}/carbon", json=payload)
        assert r2.status_code == 200
        assert r2.json() == body

        # GET
        rg = client.get(f"/cycles/{cid}/carbon/gv")
        assert rg.status_code == 200
        assert rg.json() == body

        # different factor -> 409
        p = dict(payload, factors=F_FLAT("0.4"))
        r3 = client.post(f"/cycles/{cid}/carbon", json=p)
        assert r3.status_code == 409
        assert r3.json()["error"]["code"] == "carbon_conflict"

        # stale revision against a *new* factor version -> revision_conflict
        r4 = client.post(f"/cycles/{cid}/carbon", json={
            "expected_revision": 0, "factor_version": "fresh",
            "factors": F_TWO})
        assert r4.status_code == 409
        assert r4.json()["error"]["code"] == "revision_conflict"

        # stale revision against the *existing* factor version -> immutable
        # snapshot conflict (cannot replay with a different pinned revision)
        r4b = client.post(f"/cycles/{cid}/carbon", json={
            "expected_revision": 0, "factor_version": "gv", "factors": F_TWO})
        assert r4b.status_code == 409
        assert r4b.json()["error"]["code"] == "carbon_conflict"

        # factor gap -> 409 factor_gap, nothing persisted for that fv
        r5 = client.post(f"/cycles/{cid}/carbon", json={
            "expected_revision": 1, "factor_version": "gappy",
            "factors": [{"start": "2024-07-06T04:00:00Z",
                         "end": "2024-07-06T05:00:00Z", "factor": "0.3"}]})
        assert r5.status_code == 409
        assert r5.json()["error"]["code"] == "factor_gap"
        r6 = client.get(f"/cycles/{cid}/carbon/gappy")
        assert r6.status_code == 404
        assert r6.json()["error"]["code"] == "carbon_snapshot_not_found"


def test_http_carbon_unknown_cycle_and_offset():
    client = TestClient(app)
    with client:
        r = client.post("/cycles/nope/carbon", json={
            "expected_revision": 0, "factor_version": "gv",
            "factors": F_TWO})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "cycle_not_found"

        r2 = client.get("/cycles/nope/carbon/gv")
        assert r2.status_code == 404
        assert r2.json()["error"]["code"] == "cycle_not_found"
