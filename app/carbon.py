"""已提交账期的可追溯碳排快照。

为账期（billing cycle）生成并持久化一份**不可变碳排报告**：

1. 请求提交 ``factor_version``、``expected_revision`` 与一组按 **UTC 半开
   区间**给出的排放因子（kgCO2e/kWh，区间互不重叠、必须完整覆盖账期
   用量）。
2. 账期未关账时 ``expected_revision`` 必须与当前 revision 一致，读取
   实时区间；已关账账期改为读取**不可变关账快照**中的区间与批次
   （revision 同样必须匹配关账时的值）。
3. 计算时把**用量区间、计价分段与因子边界的并集**切成原子段：能量按
   实际 UTC 时长分摊（六位小数，余量归入该区间的末段），逐段给出所属
   批次、命中的费率规则、排放因子与排放量；排放量同样按原用量区间守恒
   （六位小数余量归入末段），总排放量为各段之和。
4. 因子缺口（409 ``factor_gap``）、因子重叠（400 ``factor_overlap``）
   或计价失败（``rule_gap`` / ``rule_conflict``）整批拒绝——不写入
   任何数据。
5. 报告以规范摘要（SHA-256）持久化且不可变：同账期 + 同 revision +
   同 factor_version + 同内容的请求重放原报告（HTTP 200）；revision
   不同 → 409 ``revision_conflict``，revision 相同但因子版本或因子
   内容不同 → 409 ``carbon_conflict``。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from . import pricing
from .database import (
    BillingCycle,
    CarbonReport,
    CycleClosure,
    CycleInterval,
    UsageBatch,
    engine as default_engine,
)
from .importer import canonical_json
from .pricing import Q6, ZERO, PricingError, load_version
from .schemas import CarbonReportRequest

FACTOR_UNIT = "kgCO2e/kWh"
EMISSION_UNIT = "kgCO2e"

# 与批次提交/关账相同的写锁竞争重试策略：延迟事务升级写锁撞上
# SQLITE_BUSY 时，在全新事务里重读状态后整笔重跑。
_MAX_LOCK_RETRIES = 20


@dataclass(frozen=True)
class ReportResult:
    body: dict
    replayed: bool


@dataclass(frozen=True)
class FactorInterval:
    index: int                 # 原始输入下标（0 起）
    start: datetime            # UTC
    end: datetime              # UTC
    factor: Decimal


@dataclass(frozen=True)
class _IntervalView:
    """账期用量区间的统一视图（开放账期来自数据行，已关账来自关账快照）。"""

    seq: int
    revision: int
    batch_key: str | None
    start: datetime
    end: datetime
    tz_name: str
    energy: Decimal


class _StoredIntervalPayload:
    """与 ``settlement`` 中同名类等价的最小区间负载（时间已是 UTC ``Z``）。"""

    def __init__(self, start_utc: datetime, end_utc: datetime,
                 tz_name: str, energy: Decimal):
        self.start = _utc_z(start_utc)
        self.end = _utc_z(end_utc)
        self.timezone = tz_name
        self._energy = energy

    def energy_decimal(self) -> Decimal:
        return self._energy


def _utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canon_decimal(value: Decimal) -> str:
    """数值规范形：``0.40`` / ``0.4`` / ``4e-1`` 统一为 ``"0.4"``。"""
    return format(value.normalize(), "f")


# --------------------------------------------------------------------- errors


def _cycle_not_found(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期不存在: {cycle_id}",
        code="cycle_not_found", status_code=404,
        details={"cycle_id": cycle_id})


def _revision_conflict(cycle_id: str, expected: int,
                       current: int) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 expected_revision 与当前 revision 不匹配："
        f"期望 {expected}，当前为 {current}，拒绝生成碳排报告",
        code="revision_conflict", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_revision": expected,
                 "current_revision": current})


def _carbon_conflict(cycle_id: str, existing: CarbonReport,
                     req: CarbonReportRequest, digest: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 已存在 revision={existing.revision} 的不可变碳排"
        f"报告（factor_version={existing.factor_version!r}）；同 revision "
        "下不能用不同的因子版本或因子内容重新生成",
        code="carbon_conflict", status_code=409,
        details={"cycle_id": cycle_id,
                 "existing": {
                     "revision": existing.revision,
                     "factor_version": existing.factor_version,
                     "content_digest": existing.content_digest,
                     "report_digest": existing.report_digest,
                 },
                 "received": {
                     "revision": req.expected_revision,
                     "factor_version": req.factor_version,
                     "content_digest": digest,
                 }})


def _factor_gap(at: datetime, iv: _IntervalView) -> PricingError:
    return PricingError(
        f"排放因子未完整覆盖账期用量：时刻 {_utc_z(at)}（用量区间 "
        f"seq={iv.seq}，revision={iv.revision}）没有任何因子区间覆盖",
        code="factor_gap", status_code=409,
        details={"at": _utc_z(at),
                 "interval_seq": iv.seq,
                 "revision": iv.revision})


# ------------------------------------------------------------- factor parsing


def _parse_offset_instant(value: str, *, index: int) -> datetime:
    """解析必须携带 UTC 偏移量的 ISO-8601 时刻并换算成 UTC。"""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PricingError(
            f"factors[{index}] 的时间格式无效: {value}",
            code="invalid_datetime", status_code=400,
            details={"factor_index": index, "value": value},
        ) from exc
    if dt.tzinfo is None:
        raise PricingError(
            f"factors[{index}] 必须携带 UTC 偏移量（如 Z / +08:00）",
            code="offset_required", status_code=400,
            details={"factor_index": index, "value": value},
        )
    return dt.astimezone(timezone.utc)


def _parse_factors(req: CarbonReportRequest) -> list[FactorInterval]:
    """解析并按 UTC 起点排序因子区间；重叠（首尾相接允许）即拒绝。"""
    factors: list[FactorInterval] = []
    for idx, item in enumerate(req.factors):
        start = _parse_offset_instant(item.start, index=idx)
        end = _parse_offset_instant(item.end, index=idx)
        if end <= start:
            raise PricingError(
                f"第 {idx} 段因子区间必须满足 end > start（半开区间）",
                code="invalid_interval", status_code=400,
                details={"factor_index": idx},
            )
        factors.append(FactorInterval(idx, start, end, item.factor))

    factors.sort(key=lambda f: (f.start, f.end))
    for k in range(1, len(factors)):
        if factors[k].start < factors[k - 1].end:
            raise PricingError(
                f"第 {factors[k - 1].index} 段与第 {factors[k].index} 段"
                "排放因子区间重叠；因子区间必须互不重叠（首尾相接允许）",
                code="factor_overlap", status_code=400,
                details={"factors": [factors[k - 1].index,
                                     factors[k].index]},
            )
    return factors


def _ensure_coverage(views: list[_IntervalView],
                     factors: list[FactorInterval]) -> None:
    """因子区间（已排序、互不重叠）必须完整覆盖每一段用量区间。"""
    for iv in sorted(views, key=lambda v: (v.start, v.end)):
        cursor = iv.start
        for f in factors:
            if f.end <= cursor:
                continue
            if f.start > cursor:
                raise _factor_gap(cursor, iv)
            cursor = f.end
            if cursor >= iv.end:
                break
        if cursor < iv.end:
            raise _factor_gap(cursor, iv)


# --------------------------------------------------------------------- digest


def _content_digest(factor_version: str,
                    factors: list[FactorInterval]) -> str:
    """幂等内容摘要：因子版本 + 规范化因子区间（按 UTC 起点排序）。

    ``expected_revision`` 是并发控制令牌，刻意不参与摘要；因子列表的
    输入顺序与等价数值写法不影响摘要。
    """
    doc = {
        "factor_version": factor_version,
        "factors": [
            {"start": _utc_z(f.start),
             "end": _utc_z(f.end),
             "factor": _canon_decimal(f.factor)}
            for f in factors
        ],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


# ------------------------------------------------------------- segment build


def _find_quote_segment(segments, a: datetime, b: datetime):
    """覆盖原子段 [a, b) 的计价分段；能量为正的原子段必然被覆盖。"""
    for s in segments:
        if s.start <= a and s.end >= b:
            return s
    raise PricingError(
        f"时刻 {_utc_z(a)} 没有计价分段覆盖（规则缺口）",
        code="rule_gap", status_code=409,
        details={"at": _utc_z(a)})


def _find_factor(factors: list[FactorInterval], at: datetime) -> FactorInterval:
    for f in factors:
        if f.start <= at < f.end:
            return f
    # 覆盖性已预检，理论不可达。
    raise PricingError(
        f"时刻 {_utc_z(at)} 没有任何因子区间覆盖（因子缺口）",
        code="factor_gap", status_code=409,
        details={"at": _utc_z(at)})


def _build_segments(version, views: list[_IntervalView],
                    factors: list[FactorInterval]):
    """取用量、计价分段与因子边界的并集切原子段，逐段分摊能量与排放量。

    返回 ``(segments, total_emissions)``：每条原用量区间内能量与排放量
    都严格守恒（六位小数余量归入该区间的末个原子段）。
    """
    payload = [
        _StoredIntervalPayload(v.start, v.end, v.tz_name, v.energy)
        for v in views
    ]
    # 复用计价引擎对账期整体重算：结构问题/规则缺口在此抛出（409），
    # 因早于任何写入，失败天然不留下半成品。
    quote_result = pricing.quote(version, payload)
    segs_by_index: dict[int, list] = {}
    for s in quote_result.segments:
        segs_by_index.setdefault(s.interval_index, []).append(s)

    segments: list[dict] = []
    total_emissions = ZERO
    for vi, iv in sorted(enumerate(views),
                         key=lambda t: (t[1].start, t[1].end)):
        quote_segments = segs_by_index.get(vi, [])
        cuts = {iv.start, iv.end}
        for s in quote_segments:
            cuts.add(s.start)
            cuts.add(s.end)
        for f in factors:
            if iv.start < f.start < iv.end:
                cuts.add(f.start)
            if iv.start < f.end < iv.end:
                cuts.add(f.end)
        points = sorted(cuts)

        # 第一遍：与计价引擎同款的能量分摊——按实际 UTC 秒数比例、
        # 六位小数，余量归入该区间的末段；<=0 的碎片跳过（同引擎）。
        total_dur = Decimal((iv.end - iv.start).total_seconds())
        pieces: list[tuple] = []
        emitted = ZERO
        for a, b in zip(points, points[1:]):
            if b == iv.end:
                energy = iv.energy - emitted
            else:
                energy = (iv.energy * Decimal((b - a).total_seconds())
                          / total_dur).quantize(Q6)
            if energy <= 0:
                continue
            emitted += energy
            qseg = _find_quote_segment(quote_segments, a, b)
            fac = _find_factor(factors, a)
            pieces.append((a, b, energy, qseg, fac))

        # 第二遍：排放量按原区间守恒——逐段 energy × factor 取六位小数，
        # 余量归入该区间的末个原子段，各段之和等于区间总排放。
        exacts = [(p, p[2] * p[4].factor) for p in pieces]
        interval_total = sum((e for _, e in exacts), ZERO).quantize(Q6)
        acc = ZERO
        for j, ((a, b, energy, qseg, fac), exact_e) in enumerate(exacts):
            if j < len(exacts) - 1:
                em = exact_e.quantize(Q6)
            else:
                em = interval_total - acc
            acc += em
            segments.append({
                "seq": len(segments),
                "interval_seq": iv.seq,
                "revision": iv.revision,
                "batch_key": iv.batch_key,
                "start": _utc_z(a),
                "end": _utc_z(b),
                "duration_seconds": int((b - a).total_seconds()),
                "energy": str(energy.quantize(Q6)),
                "rule_code": qseg.rule_code,
                "rule_name": qseg.rule_name,
                "tier_ordinal": qseg.tier_ordinal,
                "unit_price": str(qseg.unit_price),
                "factor": _canon_decimal(fac.factor),
                "emissions": str(em.quantize(Q6)),
            })
        total_emissions += interval_total
    return segments, total_emissions


def _report_body(*, cycle: BillingCycle, version, factor_version: str,
                 factors: list[FactorInterval], views: list[_IntervalView],
                 cycle_status: str, created_at: str) -> dict:
    segments, total_emissions = _build_segments(version, views, factors)
    total_energy = sum((v.energy for v in views), ZERO)
    content = {
        "cycle_id": cycle.cycle_id,
        "revision": cycle.revision,
        "cycle_status": cycle_status,
        "version": cycle.version_code,
        "factor_version": factor_version,
        "factor_unit": FACTOR_UNIT,
        "emission_unit": EMISSION_UNIT,
        "energy_unit": version.energy_unit,
        "content_digest": _content_digest(factor_version, factors),
        "total_energy": str(total_energy),
        "total_emissions": str(total_emissions.quantize(Q6)),
        "factors": [
            {"start": _utc_z(f.start),
             "end": _utc_z(f.end),
             "factor": _canon_decimal(f.factor)}
            for f in factors
        ],
        "segments": segments,
    }
    # 报告摘要只覆盖确定性内容（created_at 不参与），供离线核对。
    digest = hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
    return {**content, "report_digest": digest, "created_at": created_at}


# -------------------------------------------------------------- idempotency


def _replay_or_conflict(existing: CarbonReport, cycle_id: str,
                        req: CarbonReportRequest,
                        digest: str) -> ReportResult:
    if (existing.revision == req.expected_revision
            and existing.factor_version == req.factor_version
            and existing.content_digest == digest):
        return ReportResult(json.loads(existing.report_json), True)
    if existing.revision != req.expected_revision:
        raise _revision_conflict(cycle_id, req.expected_revision,
                                 existing.revision)
    raise _carbon_conflict(cycle_id, existing, req, digest)


# ----------------------------------------------------------------- main flow


def generate_report(cycle_id: str, req: CarbonReportRequest, *,
                    engine=None) -> ReportResult:
    """生成账期的不可变碳排报告，返回 ``(报告体, 是否幂等重放)``。

    任何校验失败都抛出 :class:`PricingError` 且整笔事务回滚——不写入
    报告。``engine`` 可注入（测试用临时文件库），默认应用全局引擎。
    """
    # 先解析/校验因子区间（偏移量、半开、重叠在此暴露），合法才进事务。
    factors = _parse_factors(req)
    digest = _content_digest(req.factor_version, factors)
    eng = engine or default_engine

    attempt = 0
    while True:
        attempt += 1
        try:
            return _generate_once(eng, cycle_id, req, factors, digest)
        except OperationalError as exc:  # SQLITE_BUSY write-lock race
            if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                raise
            # 全新事务重读报告/账期状态后整笔重跑。


def _is_lock_error(exc: OperationalError) -> bool:
    msg = str(exc.orig).lower() if exc.orig is not None else ""
    return ("database is locked" in msg
            or "database table is locked" in msg)


def _generate_once(eng, cycle_id: str, req: CarbonReportRequest,
                   factors: list[FactorInterval], digest: str) -> ReportResult:
    with Session(eng) as session:
        try:
            with session.begin():
                # 1) 幂等优先：同账期已有报告时，同 revision + 同因子版本 +
                #    同内容重放原报告；revision 不同或因子不同即冲突。
                existing = session.get(CarbonReport, cycle_id)
                if existing is not None:
                    return _replay_or_conflict(existing, cycle_id, req, digest)

                # 2) 账期必须存在；expected_revision 必须匹配当前 revision
                #    （关账后 revision 冻结，同一检查天然覆盖两种状态）。
                cycle = session.get(BillingCycle, cycle_id)
                if cycle is None:
                    raise _cycle_not_found(cycle_id)
                if req.expected_revision != cycle.revision:
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, cycle.revision)

                # 3) 未关账读实时区间；已关账读取不可变关账快照。
                if cycle.status == "closed":
                    views = _views_from_closure(session, cycle_id)
                    cycle_status = "closed"
                else:
                    views = _views_from_rows(session, cycle_id)
                    cycle_status = "open"

                # 4) 因子必须完整覆盖账期用量；计价失败（rule_gap /
                #    rule_conflict）在 _report_body 内抛出，整批拒绝。
                _ensure_coverage(views, factors)
                version = load_version(session, cycle.version_code)
                created_at = _now()
                body = _report_body(
                    cycle=cycle, version=version,
                    factor_version=req.factor_version, factors=factors,
                    views=views, cycle_status=cycle_status,
                    created_at=created_at)

                session.add(CarbonReport(
                    cycle_id=cycle_id,
                    revision=cycle.revision,
                    factor_version=req.factor_version,
                    content_digest=digest,
                    report_digest=body["report_digest"],
                    total_energy=body["total_energy"],
                    total_emissions=body["total_emissions"],
                    report_json=json.dumps(body, ensure_ascii=False),
                    created_at=created_at,
                ))
                return ReportResult(body, False)

        except IntegrityError as exc:
            # 主键冲突（并发同账期生成）：只读重查归类为重放 / 冲突，
            # 绝不写任何数据。
            return _classify_integrity_error(eng, cycle_id, req, digest, exc)


def _classify_integrity_error(eng, cycle_id, req, digest, exc):
    with Session(eng) as session:
        row = session.get(CarbonReport, cycle_id)
        if row is not None:
            return _replay_or_conflict(row, cycle_id, req, digest)
    raise exc


def _views_from_rows(session: Session, cycle_id: str) -> list[_IntervalView]:
    rows = session.scalars(
        select(CycleInterval)
        .where(CycleInterval.cycle_id == cycle_id)
        .order_by(CycleInterval.seq)
    ).all()
    keys = {
        b.revision: b.idempotency_key
        for b in session.scalars(
            select(UsageBatch).where(UsageBatch.cycle_id == cycle_id)).all()
    }
    return [
        _IntervalView(seq=r.seq, revision=r.revision,
                      batch_key=keys.get(r.revision),
                      start=_parse_z(r.start_utc), end=_parse_z(r.end_utc),
                      tz_name=r.timezone, energy=Decimal(r.energy))
        for r in rows
    ]


def _views_from_closure(session: Session,
                        cycle_id: str) -> list[_IntervalView]:
    """已关账账期从不可变关账快照中读取区间与批次（而非实时数据行）。"""
    closure = session.scalar(
        select(CycleClosure).where(CycleClosure.cycle_id == cycle_id))
    if closure is None:
        raise PricingError(
            f"账期 {cycle_id} 已关账但缺少关账快照",
            code="closure_not_found", status_code=404,
            details={"cycle_id": cycle_id})
    snap = json.loads(closure.snapshot_json)
    keys = {b["revision"]: b["idempotency_key"] for b in snap["batches"]}
    return [
        _IntervalView(seq=i["seq"], revision=i["revision"],
                      batch_key=keys.get(i["revision"]),
                      start=_parse_z(i["start"]), end=_parse_z(i["end"]),
                      tz_name=i["timezone"], energy=Decimal(i["energy"]))
        for i in snap["intervals"]
    ]


# --------------------------------------------------------------- read model


def get_report(cycle_id: str, *, engine=None) -> dict:
    """返回账期的不可变碳排报告。

    账期不存在 → 404 ``cycle_not_found``；账期还没有碳排报告 →
    404 ``carbon_report_not_found``。
    """
    eng = engine or default_engine
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise _cycle_not_found(cycle_id)
        row = session.get(CarbonReport, cycle_id)
        if row is None:
            raise PricingError(
                f"账期 {cycle_id} 还没有碳排报告",
                code="carbon_report_not_found", status_code=404,
                details={"cycle_id": cycle_id})
        return json.loads(row.report_json)
