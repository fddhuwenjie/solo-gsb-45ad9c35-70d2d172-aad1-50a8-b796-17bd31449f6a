"""已提交账期的可追溯碳排快照。

为账期（``billing cycle``）已接收的用量生成**可追溯、不可变**的碳排快照。
调用方提交 ``cycle_id``、``expected_revision``、``factor_version`` 与一组
按 **UTC 半开区间** 给出的排放因子（单位 ``kgCO2e/kWh``）。流程在单个
SQLite 写事务内完成：

1. **因子区间校验**：端点必须携带 UTC 偏移量；区间合法（end > start）、
   按时间排序后**互不重叠**（首尾相接允许），且其并集必须**完整覆盖**账期
   全部用量时刻（允许因子超出用量边界）。因子缺口 → 409 ``factor_gap``，
   因子重叠 → 400 ``factor_overlap``，整批拒绝。
2. **账期状态 / revision**：账期不存在 → 404 ``cycle_not_found``。账期
   **未关账**（open）时 ``expected_revision`` 必须与当前 revision 一致，
   否则 409 ``revision_conflict``；账期**已关账**（closed）后数据冻结，
   只能读取已持久化的不可变快照。
3. **计价**：复用计价引擎对账期全部区间整体重算——任何规则缺口/结构问题
   （409 ``rule_gap`` / ``rule_conflict``）都在写入前抛出，整批拒绝。
4. **并集切分**：取用量区间端点、**计价分段**边界与因子边界的 UTC 并集，
   沿 UTC 时间线切成原子半开段；能量按各原子段相对其原区间的**实际时长**
   比例分摊（六位小数），舍入余量归入该原区间的**末段**，因此能量按原区间
   严格守恒。
5. **排放量**：每个原子段按其起点命中唯一因子区间，``排放 = 能量 × 因子``
   量化到六位小数；同样在每个原区间的末段吸收舍入余量，使**排放量按原区间
   守恒**。总排放为各段排放之和。
6. **持久化**：以规范摘要（SHA-256）保存自包含的不可变报告。同账期 + 同
   ``factor_version`` + 同内容（revision、因子、用量、计价一致）重放原报告
   （HTTP 200，不写库）；换 revision 或换因子内容 → 409 ``carbon_conflict``。
   换一个 ``factor_version`` 可再存一份独立快照。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from . import pricing
from .database import (
    BillingCycle,
    CarbonSnapshot,
    CycleInterval,
    UsageBatch,
    cycle_write_lock,
    engine as default_engine,
)
from .importer import canonical_json
from .pricing import (
    Q2,
    Q6,
    ZERO,
    PricingError,
    load_version,
)
from .schemas import CarbonSnapshotRequest
from .settlement import build_quote_response

FACTOR_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
EMISSION_UNIT = "kgCO2e"

# 与结算/关账相同的写锁竞争重试策略。
_MAX_LOCK_RETRIES = 20


@dataclass(frozen=True)
class FactorInterval:
    """规范化后的因子区间（UTC 半开）。"""
    start: datetime
    end: datetime
    factor: Decimal


@dataclass(frozen=True)
class CarbonResult:
    body: dict
    replayed: bool


class _StoredIntervalPayload:
    """与 ``settlement``/``closing`` 中同名类等价的最小区间负载。"""

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


def validate_factor_version(factor_version: str) -> None:
    if not FACTOR_VERSION_RE.fullmatch(factor_version):
        raise PricingError(
            f"非法 factor_version: {factor_version!r}；只允许字母/数字开头，"
            "后跟至多 127 个字母、数字、下划线或连字符",
            code="invalid_factor_version", status_code=400,
            details={"factor_version": factor_version})


# ------------------------------------------------------------- factor parsing


def _parse_utc(value: str, index: int, field: str) -> datetime:
    """因子端点必须是携带 UTC 偏移量的 ISO-8601 时刻。"""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PricingError(
            f"factors[{index}].{field} 时间格式无效: {value}",
            code="invalid_datetime", status_code=400,
            details={"index": index, "field": field, "value": value},
        ) from exc
    if dt.tzinfo is None:
        raise PricingError(
            f"factors[{index}].{field} 必须携带 UTC 偏移量（如 Z / +00:00）",
            code="offset_required", status_code=400,
            details={"index": index, "field": field, "value": value},
        )
    return dt.astimezone(timezone.utc)


def parse_factors(req: CarbonSnapshotRequest) -> list[FactorInterval]:
    """解析并校验请求因子区间：合法、互不重叠；**不**在此校验覆盖（覆盖需要
    账期用量范围）。"""
    parsed: list[FactorInterval] = []
    for idx, item in enumerate(req.factors):
        start = _parse_utc(item.start, idx, "start")
        end = _parse_utc(item.end, idx, "end")
        if end <= start:
            raise PricingError(
                f"factors[{idx}] 必须满足 end > start（UTC 半开区间）",
                code="invalid_interval", status_code=400,
                details={"factor_index": idx,
                         "start": _utc_z(start), "end": _utc_z(end)},
            )
        parsed.append(FactorInterval(start, end, item.factor))

    parsed.sort(key=lambda f: (f.start, f.end))
    for k in range(1, len(parsed)):
        prev, cur = parsed[k - 1], parsed[k]
        if cur.start < prev.end:
            raise PricingError(
                f"排放因子区间在 UTC 时间线上重叠：{_utc_z(prev.start)} ~ "
                f"{_utc_z(prev.end)} 与 {_utc_z(cur.start)} ~ "
                f"{_utc_z(cur.end)}；因子区间不得重叠（首尾相接允许）",
                code="factor_overlap", status_code=400,
                details={"factor_a": {"start": _utc_z(prev.start),
                                      "end": _utc_z(prev.end),
                                      "factor": str(prev.factor)},
                         "factor_b": {"start": _utc_z(cur.start),
                                      "end": _utc_z(cur.end),
                                      "factor": str(cur.factor)}},
            )
    return parsed


def _factor_at(instant: datetime, factors: list[FactorInterval]):
    """返回覆盖 ``instant`` 的唯一因子区间（半开 [start, end)）。"""
    for f in factors:
        if f.start <= instant < f.end:
            return f
    return None


def _ensure_factors_cover(usage: list[CycleInterval],
                          factors: list[FactorInterval]) -> None:
    """因子并集必须完整覆盖每一段用量区间（因子可超出用量边界）。"""
    for row in usage:
        u_start, u_end = _parse_z(row.start_utc), _parse_z(row.end_utc)
        # 用量区间互不重叠，逐段在已排序因子上推进指针即可。
        covered_from = u_start
        for f in factors:
            if f.end <= covered_from:
                continue
            if f.start > covered_from:
                break  # 出现缺口
            if f.start <= covered_from:
                if f.end >= u_end:
                    covered_from = u_end
                    break
                covered_from = f.end
        if covered_from < u_end:
            raise PricingError(
                f"排放因子未完整覆盖账期用量：自 {_utc_z(covered_from)} 起"
                f"（用量区间 {row.start_utc} ~ {row.end_utc}）缺少因子；"
                "因子区间必须完整覆盖全部用量时刻",
                code="factor_gap", status_code=409,
                details={"gap_start": _utc_z(covered_from),
                         "usage_start": row.start_utc,
                         "usage_end": row.end_utc},
            )


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
        f"期望 {expected}，当前为 {current}，拒绝生成碳排快照",
        code="revision_conflict", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_revision": expected,
                 "current_revision": current})


def _cycle_closed(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 已关账，用量已冻结；请用已持久化的因子版本读取"
        "不可变碳排快照，不能基于新因子版本重新生成",
        code="cycle_closed", status_code=409,
        details={"cycle_id": cycle_id})


def _carbon_conflict(*, factor_version: str, existing: CarbonSnapshot,
                     received_revision: int, received_digest: str,
                     ) -> PricingError:
    return PricingError(
        f"账期 {existing.cycle_id} 的因子版本 {factor_version!r} 已有内容"
        f"不同的碳排快照：首次 revision={existing.revision}，"
        f"摘要={existing.content_digest[:12]}…；本次 revision="
        f"{received_revision}，摘要={received_digest[:12]}…；快照不可变，"
        "同账期同因子版本仅允许同内容重放",
        code="carbon_conflict", status_code=409,
        details={"cycle_id": existing.cycle_id,
                 "factor_version": factor_version,
                 "existing": {
                     "revision": existing.revision,
                     "content_digest": existing.content_digest,
                     "report_digest": existing.report_digest,
                 },
                 "received": {
                     "revision": received_revision,
                     "content_digest": received_digest,
                 }})


# --------------------------------------------------------------------- digest


def _factor_content_digest(factors: list[FactorInterval]) -> str:
    """请求因子的规范化内容摘要（UTC 端点、因子值，按起点排序）。

    ``expected_revision`` 是并发/一致性控制令牌，刻意不参与因子内容摘要，
    但会进入下面的快照 ``content_digest``（因此换 revision 仍冲突）。
    """
    doc = {
        "factors": [
            {"start": _utc_z(f.start), "end": _utc_z(f.end),
             "factor": format(f.factor.normalize(), "f")}
            for f in factors
        ],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def _snapshot_content_digest(*, cycle_id: str, factor_version: str,
                             revision: int, factor_digest: str,
                             total_energy: str, total_amount: str,
                             total_emission: str,
                             intervals: list[CycleInterval]) -> str:
    """同键冲突判定摘要：因子内容 + 账期身份/revision/明细 + 计算结果。

    只要 revision、因子、用量或计价/排放结果任一不同，摘要即不同。
    """
    doc = {
        "cycle_id": cycle_id,
        "factor_version": factor_version,
        "revision": revision,
        "factor_digest": factor_digest,
        "total_energy": total_energy,
        "total_amount": total_amount,
        "total_emission": total_emission,
        "intervals": [
            {"seq": r.seq, "revision": r.revision,
             "start": r.start_utc, "end": r.end_utc,
             "timezone": r.timezone, "energy": r.energy}
            for r in sorted(intervals, key=lambda r: (r.revision, r.seq))
        ],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


# ------------------------------------------------------------- core splitting


@dataclass(frozen=True)
class _Atom:
    """一段原子半开窗口 [start, end)，归属一条原用量区间。"""
    start: datetime
    end: datetime
    usage: CycleInterval          # 原用量区间（含批次/seq 归属）
    usage_index: int              # 原区间在本次重算输入中的下标
    total_seconds: int            # 原区间总时长（秒）


def _build_atoms(usage_rows: list[CycleInterval],
                 quote, factors: list[FactorInterval]) -> list[_Atom]:
    """取用量端点、计价分段边界与因子边界的并集，切出原子段。

    只在**用量覆盖到的时间线**内切分（用量之间的空隙不产生段）。
    """
    # 计价分段（quote.segments）已按 UTC 时间排序，interval_index 对应
    # parse_usage_intervals 对输入（按起点排序）分配的下标。
    price_bounds: set[datetime] = set()
    for s in quote.segments:
        price_bounds.add(s.start)
        price_bounds.add(s.end)
    factor_bounds: set[datetime] = set()
    for f in factors:
        factor_bounds.add(f.start)
        factor_bounds.add(f.end)

    atoms: list[_Atom] = []
    # usage_rows 已按 (revision, seq) 稳定排序；构造给 quote 的输入也按同样
    # 顺序，因此下标与 quote 段的 interval_index 一一对应。
    for idx, row in enumerate(usage_rows):
        u_start, u_end = _parse_z(row.start_utc), _parse_z(row.end_utc)
        cuts = {u_start, u_end}
        for b in price_bounds:
            if u_start < b < u_end:
                cuts.add(b)
        for b in factor_bounds:
            if u_start < b < u_end:
                cuts.add(b)
        ordered = sorted(cuts)
        total_seconds = int((u_end - u_start).total_seconds())
        for a, b in zip(ordered, ordered[1:]):
            atoms.append(_Atom(a, b, row, idx, total_seconds))
    atoms.sort(key=lambda x: (x.start, x.end))
    return atoms


def _price_segment_map(quote):
    """按 interval_index 分组计价段（各自按时间排序），供原子段反查规则。"""
    grouped: dict[int, list] = {}
    for s in quote.segments:
        grouped.setdefault(s.interval_index, []).append(s)
    for segs in grouped.values():
        segs.sort(key=lambda s: (s.start, s.end))
    return grouped


def _find_price_segment(segs: list, start: datetime, end: datetime):
    """定位覆盖原子段 [start, end) 的计价段。

    正常情况下原子段端点取自计价分段边界，必然命中包含其起点的整段；
    仅当计价引擎在某窗口产生零能量分段（极端舍入碎片）时，用“起点之前
    最近的分段”确定性回退，保证仍可追溯到费率规则与阶梯。
    """
    containing = None
    for s in segs:
        if s.start <= start < s.end:
            containing = s
            break
    if containing is not None:
        return containing
    fallback = None
    for s in segs:
        if s.start <= start:
            fallback = s
    return fallback


def compute_carbon(*, usage_rows: list[CycleInterval],
                   batch_by_id: dict[int, UsageBatch],
                   quote, factors: list[FactorInterval]):
    """切分并计算逐段能量/排放/费用，返回 ``(segments, total_emission)``。

    三条守恒线（余量都吸收到各自的**末段**）：

    * **能量**按原用量区间守恒：除末原子段外按相对原区间的实际 UTC 时长
      比例量化到六位小数，末段取残差；
    * **排放**按原用量区间守恒：各段 ``能量 × 因子``，区间末段取该区间
      排放总量（量化后）减去前面各段的残差；
    * **费用**按计价分段守恒：计价分段可能被因子边界进一步切成多个原子段，
      各段按 ``能量 × 费率`` 量化，末段吸收残差，使各原子段金额之和恰为
      该计价分段的金额。
    """
    atoms = _build_atoms(usage_rows, quote, factors)
    grouped = _price_segment_map(quote)

    # 预先解析每个原子段的计价分段与因子。
    usage_positions: dict[int, list[int]] = {}
    ps_positions: dict[tuple[int, int], list[int]] = {}
    resolved: list[tuple] = []
    for pos, atom in enumerate(atoms):
        usage_positions.setdefault(atom.usage_index, []).append(pos)
        segs = grouped.get(atom.usage_index, [])
        price_seg = _find_price_segment(segs, atom.start, atom.end)
        factor_seg = _factor_at(atom.start, factors)
        if factor_seg is None:
            # 覆盖已在 _ensure_factors_cover 校验过；此处为不变量保护。
            raise PricingError(
                f"时刻 {_utc_z(atom.start)} 缺少排放因子",
                code="factor_gap", status_code=409,
                details={"at": _utc_z(atom.start)})
        ps_idx = _price_segment_index(segs, price_seg)
        ps_positions.setdefault((atom.usage_index, ps_idx), []).append(pos)
        resolved.append((atom, price_seg, factor_seg))

    usage_last = {pos[-1] for pos in usage_positions.values()}
    price_last = {pos[-1] for pos in ps_positions.values()}

    # 第一遍：能量（原区间守恒）。
    energies: dict[int, Decimal] = {}
    energy_used: dict[int, Decimal] = {}
    for pos, (atom, _, _) in enumerate(resolved):
        uidx = atom.usage_index
        interval_energy = Decimal(atom.usage.energy)
        if pos in usage_last:
            energy = interval_energy - energy_used.get(uidx, ZERO)
        else:
            duration = int((atom.end - atom.start).total_seconds())
            energy = (interval_energy * Decimal(duration)
                      / Decimal(atom.total_seconds)).quantize(Q6)
        energy_used[uidx] = energy_used.get(uidx, ZERO) + energy
        energies[pos] = energy

    # 第二遍：排放（原区间守恒）与费用（计价分段守恒）。
    emission_used: dict[int, Decimal] = {}
    amount_accounted: dict[tuple[int, int], Decimal] = {}
    segments_out: list[dict] = []
    total_emission = ZERO

    for pos, (atom, price_seg, factor_seg) in enumerate(resolved):
        uidx = atom.usage_index
        row = atom.usage
        energy = energies[pos]

        exact_emission = energy * factor_seg.factor
        if pos in usage_last:
            interval_total = sum(
                (energies[p] * resolved[p][2].factor
                 for p in usage_positions[uidx]), ZERO
            ).quantize(Q6, rounding=ROUND_HALF_UP)
            emission = interval_total - emission_used.get(uidx, ZERO)
        else:
            emission = exact_emission.quantize(Q6, rounding=ROUND_HALF_UP)
        emission_used[uidx] = emission_used.get(uidx, ZERO) + emission
        total_emission += emission

        ps_idx = _price_segment_index(grouped.get(uidx, []), price_seg)
        ps_key = (uidx, ps_idx)
        if pos in price_last:
            amount = price_seg.amount - amount_accounted.get(ps_key, ZERO)
        else:
            amount = (energy * price_seg.unit_price).quantize(
                Q6, rounding=ROUND_HALF_UP)
            amount_accounted[ps_key] = (
                amount_accounted.get(ps_key, ZERO) + amount)

        batch = batch_by_id[row.batch_id] if row.batch_id is not None else None
        segments_out.append({
            "start": atom.start,
            "end": atom.end,
            "duration_seconds": int((atom.end - atom.start).total_seconds()),
            "batch_revision": row.revision,
            "idempotency_key": (batch.idempotency_key
                                if batch is not None else ""),
            "interval_seq": row.seq,
            "energy": energy,
            "rule_code": price_seg.rule_code,
            "rule_name": price_seg.rule_name,
            "tier_ordinal": price_seg.tier_ordinal,
            "unit_price": price_seg.unit_price,
            "amount": amount,
            "factor": factor_seg.factor,
            "factor_start": factor_seg.start,
            "factor_end": factor_seg.end,
            "emission": emission,
        })

    segments_out.sort(key=lambda s: (s["start"], s["end"], s["interval_seq"]))
    return segments_out, total_emission


def _price_segment_index(segs: list, seg) -> int:
    for idx, s in enumerate(segs):
        if (s.start == seg.start and s.end == seg.end
                and s.rule_code == seg.rule_code
                and s.tier_ordinal == seg.tier_ordinal):
            return idx
    return -1


# ----------------------------------------------------------------- main flow


def create_carbon_snapshot(cycle_id: str, req: CarbonSnapshotRequest, *,
                           engine=None) -> CarbonResult:
    """生成（或重放）账期碳排快照，返回 ``(报告体, 是否幂等重放)``。

    任何校验/计价失败都抛出 :class:`PricingError`，且整笔事务回滚，不写入
    快照。``engine`` 可注入（测试用临时文件库）。
    """
    validate_factor_version(req.factor_version)
    factors = parse_factors(req)
    factor_digest = _factor_content_digest(factors)
    eng = engine or default_engine

    # 进程内串行化同账期的读-改-写临界区（与批次提交/关账共用一把锁），
    # 且锁必须覆盖整个 SQLITE_BUSY 重试循环：同 revision 的快照与并发批次/
    # 关账只有一个生效，落败方在全新事务里读到新 revision/状态后得到
    # revision_conflict / cycle_closed / 重放。
    with cycle_write_lock:
        attempt = 0
        while True:
            attempt += 1
            try:
                return _create_once(eng, cycle_id, req, factors, factor_digest)
            except OperationalError as exc:  # SQLITE_BUSY write-lock race
                if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                    raise
                # 全新事务重读 revision / 快照记录后整笔重跑（仍持锁）。


def _is_lock_error(exc: OperationalError) -> bool:
    msg = str(exc.orig).lower() if exc.orig is not None else ""
    return ("database is locked" in msg
            or "database table is locked" in msg)


def _replay_or_conflict(*, row: CarbonSnapshot, factor_version: str,
                        received_revision: int,
                        received_digest: str) -> CarbonResult:
    if row.revision == received_revision and row.content_digest == received_digest:
        return CarbonResult(json.loads(row.report_json), True)
    raise _carbon_conflict(
        factor_version=factor_version, existing=row,
        received_revision=received_revision, received_digest=received_digest)


def _create_once(eng, cycle_id: str, req: CarbonSnapshotRequest,
                 factors: list[FactorInterval],
                 factor_digest: str) -> CarbonResult:
    with Session(eng) as session:
        try:
            with session.begin():
                # 1) 幂等优先：同账期 + 同 factor_version 先于一切状态判定。
                existing = session.scalar(
                    select(CarbonSnapshot).where(
                        CarbonSnapshot.cycle_id == cycle_id,
                        CarbonSnapshot.factor_version == req.factor_version,
                    )
                )

                cycle = session.get(BillingCycle, cycle_id)
                if cycle is None:
                    raise _cycle_not_found(cycle_id)

                # 重新读取账期明细（稳定排序），后续覆盖校验/计价/摘要都用它。
                intervals = list(session.scalars(
                    select(CycleInterval)
                    .where(CycleInterval.cycle_id == cycle_id)
                    .order_by(CycleInterval.revision, CycleInterval.seq)
                ).all())
                batches = list(session.scalars(
                    select(UsageBatch)
                    .where(UsageBatch.cycle_id == cycle_id)
                ).all())

                version = load_version(session, cycle.version_code)
                # 计价引擎按 (start, end) 排序后给输入分配 interval_index；
                # 碳排切分必须用**同一排序**，原子段才能据 interval_index
                # 正确反查计价分段。（批次可乱序提交，故不能用 revision/seq
                # 的存储顺序。）
                pricing_rows = sorted(
                    intervals,
                    key=lambda r: (_parse_z(r.start_utc),
                                  _parse_z(r.end_utc), r.seq))
                payload = [
                    _StoredIntervalPayload(
                        _parse_z(r.start_utc), _parse_z(r.end_utc),
                        r.timezone, Decimal(r.energy))
                    for r in pricing_rows
                ]
                # 复用计价引擎整体重算（结构/规则缺口在此暴露，先于写入）。
                quote_result = pricing.quote(version, payload)

                # 因子覆盖必须完整（因子可超出用量边界）。
                _ensure_factors_cover(intervals, factors)

                total_amount = quote_result.total_amount.quantize(
                    Q2, rounding=ROUND_HALF_UP)
                batch_by_id = {b.id: b for b in batches}
                segments, total_emission = compute_carbon(
                    usage_rows=pricing_rows, batch_by_id=batch_by_id,
                    quote=quote_result, factors=factors)

                content_digest = _snapshot_content_digest(
                    cycle_id=cycle_id, factor_version=req.factor_version,
                    revision=req.expected_revision, factor_digest=factor_digest,
                    total_energy=cycle.total_energy,
                    total_amount=str(total_amount),
                    total_emission=str(total_emission.quantize(Q6)),
                    intervals=intervals)

                # 已有同因子版本快照：同 revision + 同内容重放，否则冲突
                # （revision 以请求的 expected_revision 为准——它是客户端声明
                # 的一致性令牌；无论账期此后是否又推进了 revision）。
                if existing is not None:
                    return _replay_or_conflict(
                        row=existing, factor_version=req.factor_version,
                        received_revision=req.expected_revision,
                        received_digest=content_digest)

                # 新快照：关账账期冻结，拒绝用新因子版本重新生成。
                if cycle.status == "closed":
                    raise _cycle_closed(cycle_id)

                # 开放账期：expected_revision 必须匹配当前 revision。
                if req.expected_revision != cycle.revision:
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, cycle.revision)

                body = _report_body(
                    cycle=cycle, req=req, factors=factors,
                    currency=version.currency,
                    energy_unit=version.energy_unit,
                    total_amount=total_amount,
                    total_emission=total_emission,
                    content_digest=content_digest,
                    segments=segments, quote=quote_result, version=version,
                    created_at=_now())

                # 写锁守护：用一个**真正改写行**的条件更新（updated_at 标记为
                # 快照生成时刻）在单写者锁上与并发批次/关账串行化。若账期已被
                # 同 revision 的批次推进或被关账，rowcount==0；若对方持写锁，
                # 本事务在升级写锁时收到 SQLITE_BUSY，外层以全新事务重试。
                guarded = session.execute(
                    update(BillingCycle)
                    .where(BillingCycle.cycle_id == cycle_id,
                           BillingCycle.status == "open",
                           BillingCycle.revision == req.expected_revision)
                    .values(updated_at=body["created_at"])
                )
                if guarded.rowcount != 1:
                    refreshed = session.execute(
                        select(BillingCycle)
                        .where(BillingCycle.cycle_id == cycle_id)
                        .execution_options(populate_existing=True)
                    ).scalar_one()
                    if refreshed.status == "closed":
                        raise _cycle_closed(cycle_id)
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, refreshed.revision)

                session.add(CarbonSnapshot(
                    cycle_id=cycle_id,
                    factor_version=req.factor_version,
                    revision=cycle.revision,
                    version_code=cycle.version_code,
                    currency=version.currency,
                    energy_unit=version.energy_unit,
                    total_energy=cycle.total_energy,
                    total_amount=str(total_amount),
                    total_emission=str(total_emission.quantize(Q6)),
                    content_digest=content_digest,
                    report_digest=body["report_digest"],
                    report_json=json.dumps(body, ensure_ascii=False),
                    status=cycle.status,
                    created_at=body["created_at"],
                ))
                return CarbonResult(body, False)

        except IntegrityError as exc:
            # 唯一约束冲突（并发竞态）：只读重查归类为重放 / 冲突，绝不写库。
            return _classify_integrity_error(
                eng, cycle_id, req, factor_digest, exc)


def _classify_integrity_error(eng, cycle_id, req, factor_digest, exc):
    """唯一约束冲突（并发竞态）后的只读归类，绝不写任何数据。

    同账期 + 同 factor_version 的并发：先到者已插入快照行，后到者按持久化
    行重算因子摘要做重放/冲突判定；否则按账期当前状态归类为
    ``cycle_closed`` / ``revision_conflict``（与并发批次/关账竞争落败）。
    """
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise _cycle_not_found(cycle_id) from exc
        row = session.scalar(
            select(CarbonSnapshot).where(
                CarbonSnapshot.cycle_id == cycle_id,
                CarbonSnapshot.factor_version == req.factor_version,
            )
        )
        if row is not None:
            same_factor = _extract_factor_digest(row) == factor_digest
            same_revision = row.revision == req.expected_revision
            if same_factor and same_revision:
                return CarbonResult(json.loads(row.report_json), True)
            raise _carbon_conflict(
                factor_version=req.factor_version, existing=row,
                received_revision=req.expected_revision,
                received_digest=row.content_digest)
        if cycle.status == "closed":
            raise _cycle_closed(cycle_id) from exc
        raise _revision_conflict(
            cycle_id, req.expected_revision, cycle.revision) from exc


def _extract_factor_digest(row: CarbonSnapshot) -> str:
    """从持久化报告中回算因子内容摘要（供并发归类比较）。"""
    report = json.loads(row.report_json)
    factors = [
        FactorInterval(_parse_z(f["start"]), _parse_z(f["end"]),
                       Decimal(f["factor"]))
        for f in report["factors"]
    ]
    return _factor_content_digest(factors)


# ---------------------------------------------------------------- report body


def _report_body(*, cycle: BillingCycle, req: CarbonSnapshotRequest,
                 factors: list[FactorInterval], currency: str,
                 energy_unit: str, total_amount: Decimal,
                 total_emission: Decimal, content_digest: str,
                 segments: list[dict], quote, version,
                 created_at: str) -> dict:
    total_emission_q = total_emission.quantize(Q6)
    body = {
        "cycle_id": cycle.cycle_id,
        "factor_version": req.factor_version,
        "version": cycle.version_code,
        "currency": currency,
        "energy_unit": energy_unit,
        "emission_unit": EMISSION_UNIT,
        "revision": cycle.revision,
        "status": cycle.status,
        "total_energy": cycle.total_energy,
        "total_amount": str(total_amount),
        "total_emission": str(total_emission_q),
        "content_digest": content_digest,
        "created_at": created_at,
        "factors": [
            {"start": _utc_z(f.start), "end": _utc_z(f.end),
             "factor": format(f.factor.normalize(), "f")}
            for f in factors
        ],
        "segments": [
            {
                "seq": idx,
                "start": _utc_z(s["start"]),
                "end": _utc_z(s["end"]),
                "duration_seconds": s["duration_seconds"],
                "batch_revision": s["batch_revision"],
                "idempotency_key": s["idempotency_key"],
                "interval_seq": s["interval_seq"],
                "energy": str(s["energy"].quantize(Q6)),
                "rule_code": s["rule_code"],
                "rule_name": s["rule_name"],
                "tier_ordinal": s["tier_ordinal"],
                "unit_price": str(s["unit_price"]),
                "amount": str(s["amount"].quantize(Q6)),
                "factor": format(s["factor"].normalize(), "f"),
                "factor_start": _utc_z(s["factor_start"]),
                "factor_end": _utc_z(s["factor_end"]),
                "emission": str(s["emission"].quantize(Q6)),
            }
            for idx, s in enumerate(segments)
        ],
        "quote": build_quote_response(version, quote).model_dump(),
    }
    body["report_digest"] = hashlib.sha256(
        canonical_json(_digest_doc(body)).encode("utf-8")).hexdigest()
    return body


def _digest_doc(body: dict) -> dict:
    """报告摘要覆盖的规范文档：剔除报告自身的摘要字段后对完整内容取摘要。"""
    doc = dict(body)
    doc.pop("report_digest", None)
    return doc


# --------------------------------------------------------------- read model


def get_carbon_snapshot(cycle_id: str, factor_version: str, *,
                        engine=None) -> dict:
    """返回账期指定因子版本的不可变碳排报告。

    账期不存在 → 404 ``cycle_not_found``；账期存在但该因子版本没有快照 →
    404 ``carbon_snapshot_not_found``。
    """
    eng = engine or default_engine
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise _cycle_not_found(cycle_id)
        row = session.scalar(
            select(CarbonSnapshot).where(
                CarbonSnapshot.cycle_id == cycle_id,
                CarbonSnapshot.factor_version == factor_version,
            )
        )
        if row is None:
            raise PricingError(
                f"账期 {cycle_id} 没有因子版本 {factor_version!r} 的碳排快照",
                code="carbon_snapshot_not_found", status_code=404,
                details={"cycle_id": cycle_id,
                         "factor_version": factor_version})
        return json.loads(row.report_json)
