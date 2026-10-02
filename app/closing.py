"""已接收账期的原子关账与不可变快照。

调用方提交 ``cycle_id``、``expected_revision``、唯一 ``close_id`` 与
``expected_total_amount``，关账在**单个 SQLite 写事务**内完成：

1. 先按 ``close_id`` 判幂等：同 close_id 同内容（账期、expected_revision、
   expected_total_amount 一致）原样返回首次快照（HTTP 200），换内容或换
   账期 → 409 ``close_conflict``；
2. 重新读取账期：不存在 404；已被其他 close_id 关账 → 409 ``cycle_closed``；
3. ``revision`` 与重新读取的账期 revision 不一致 → 409
   ``revision_conflict``；重新读取的累计金额（四舍五入到分）与
   ``expected_total_amount`` 不一致 → 409 ``total_amount_mismatch``；
4. 重新读取**全部批次与区间**，按批次（revision）与区间（revision, seq）
   **稳定排序**构造明细文档并计算 SHA-256（``detail_digest``），同时复用
   计价引擎对账期整体重算，保存自包含的**不可变快照**；
5. 以条件更新 ``WHERE status='open' AND revision=:expected`` 把账期置为
   ``closed``：rowcount==0 说明并发批次或并发关账抢先，整笔事务回滚，
   账期状态不发生任何变化。

因此并发的批次提交与关账只有一个生效：落败方在新事务里重读状态后得到
``revision_conflict`` / ``cycle_closed``。关账成功后批次接口拒绝新批次
（见 :mod:`settlement`）。
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
    CycleClosure,
    CycleInterval,
    UsageBatch,
    engine as default_engine,
)
from .importer import canonical_json
from .pricing import (
    Q2,
    PricingError,
    load_version,
)
from .schemas import CycleCloseRequest
from .settlement import build_quote_response

CLOSE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

# 与批次提交相同的写锁竞争重试策略：延迟事务升级写锁撞上 SQLITE_BUSY 时，
# 在全新事务里重读状态后整笔重跑。
_MAX_LOCK_RETRIES = 20


@dataclass(frozen=True)
class CloseResult:
    body: dict
    replayed: bool


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


def validate_close_id(close_id: str) -> None:
    if not CLOSE_ID_RE.fullmatch(close_id):
        raise PricingError(
            f"非法 close_id: {close_id!r}；只允许字母/数字开头，后跟至多 "
            "127 个字母、数字、下划线或连字符",
            code="invalid_close_id", status_code=400,
            details={"close_id": close_id})


# --------------------------------------------------------------------- errors


def _cycle_not_found(cycle_id: str) -> PricingError:
    return PricingError(
        f"账期不存在: {cycle_id}",
        code="cycle_not_found", status_code=404,
        details={"cycle_id": cycle_id})


def _cycle_closed(cycle_id: str, existing_close_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 已关账（close_id={existing_close_id!r}）；"
        "关账后拒绝新批次与重复或冲突关账，请用原 close_id 重放或查询快照",
        code="cycle_closed", status_code=409,
        details={"cycle_id": cycle_id, "existing_close_id": existing_close_id})


def _revision_conflict(cycle_id: str, expected: int,
                       current: int) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 expected_revision 与当前 revision 不匹配："
        f"期望 {expected}，当前为 {current}，拒绝关账",
        code="revision_conflict", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_revision": expected,
                 "current_revision": current})


def _amount_mismatch(cycle_id: str, expected: Decimal,
                     current: Decimal) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的累计金额与 expected_total_amount 不一致："
        f"期望 {expected}，账期当前为 {current}，拒绝关账",
        code="total_amount_mismatch", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_total_amount": str(expected),
                 "current_total_amount": str(current)})


def _close_conflict(close_id: str, existing: CycleClosure,
                    cycle_id: str, expected_revision: int,
                    expected_total: Decimal) -> PricingError:
    return PricingError(
        f"close_id {close_id!r} 已用于内容不同的关账：首次关账账期 "
        f"{existing.cycle_id}（revision={existing.expected_revision}，"
        f"总额={existing.expected_total_amount}），本次请求账期 {cycle_id}"
        f"（revision={expected_revision}，总额={expected_total}）；"
        "同 close_id 仅允许同内容重放",
        code="close_conflict", status_code=409,
        details={"close_id": close_id,
                 "existing": {
                     "cycle_id": existing.cycle_id,
                     "expected_revision": existing.expected_revision,
                     "expected_total_amount": existing.expected_total_amount,
                     "detail_digest": existing.detail_digest,
                 },
                 "received": {
                     "cycle_id": cycle_id,
                     "expected_revision": expected_revision,
                     "expected_total_amount": str(expected_total),
                 }})


# -------------------------------------------------------------- idempotency


def _request_matches(close_id: str, row: CycleClosure, cycle_id: str,
                     expected_revision: int,
                     expected_total: Decimal) -> bool:
    return (
        row.close_id == close_id
        and row.cycle_id == cycle_id
        and row.expected_revision == expected_revision
        and Decimal(row.expected_total_amount) == expected_total
    )


def _replay_or_conflict(close_id: str, row: CycleClosure, cycle_id: str,
                        expected_revision: int,
                        expected_total: Decimal) -> CloseResult:
    if _request_matches(close_id, row, cycle_id, expected_revision,
                        expected_total):
        return CloseResult(json.loads(row.snapshot_json), True)
    raise _close_conflict(close_id, row, cycle_id, expected_revision,
                          expected_total)


def _closed_cycle_outcome(session: Session, cycle_id: str,
                          req: CycleCloseRequest,
                          expected_total: Decimal):
    """账期已 closed 时的归类：优先按本次 close_id 找快照——同 id 同内容
    重放、同 id 异内容冲突；否则用别的 close_id 关账 → cycle_closed。"""
    same = session.get(CycleClosure, req.close_id)
    if same is not None:
        return _replay_or_conflict(
            req.close_id, same, cycle_id, req.expected_revision,
            expected_total)
    prior = session.scalar(
        select(CycleClosure).where(CycleClosure.cycle_id == cycle_id))
    raise _cycle_closed(cycle_id,
                        prior.close_id if prior else req.close_id)


# ---------------------------------------------------------------- snapshot


def _detail_digest(cycle_id: str, version_code: str, revision: int,
                   total_energy: str, total_amount: str,
                   batches: list[UsageBatch],
                   intervals: list[CycleInterval]) -> str:
    """按批次（revision）与区间（revision, seq）稳定排序后的明细 SHA-256。"""
    doc = {
        "cycle_id": cycle_id,
        "version": version_code,
        "revision": revision,
        "total_energy": total_energy,
        "total_amount": total_amount,
        "batches": [
            {
                "revision": b.revision,
                "idempotency_key": b.idempotency_key,
                "content_digest": b.content_digest,
                "batch_energy": b.batch_energy,
                "batch_amount": str(Decimal(b.batch_amount).quantize(Q2)),
            }
            for b in sorted(batches, key=lambda b: (b.revision, b.id))
        ],
        "intervals": [
            {
                "seq": r.seq,
                "revision": r.revision,
                "start": r.start_utc,
                "end": r.end_utc,
                "timezone": r.timezone,
                "energy": r.energy,
            }
            for r in sorted(intervals, key=lambda r: (r.revision, r.seq))
        ],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def _snapshot_body(*, close_id: str, cycle: BillingCycle, currency: str,
                   energy_unit: str, total_amount: Decimal,
                   batches: list[UsageBatch],
                   intervals: list[CycleInterval],
                   quote, version, closed_at: str) -> dict:
    return {
        "close_id": close_id,
        "cycle_id": cycle.cycle_id,
        "version": cycle.version_code,
        "currency": currency,
        "energy_unit": energy_unit,
        "revision": cycle.revision,
        "total_energy": cycle.total_energy,
        "total_amount": str(total_amount),
        "detail_digest": _detail_digest(
            cycle.cycle_id, cycle.version_code, cycle.revision,
            cycle.total_energy, str(total_amount), batches, intervals),
        "status": "closed",
        "closed_at": closed_at,
        "batches": [
            {
                "revision": b.revision,
                "idempotency_key": b.idempotency_key,
                "content_digest": b.content_digest,
                "batch_energy": b.batch_energy,
                "batch_amount": str(Decimal(b.batch_amount).quantize(Q2)),
                "created_at": b.created_at,
            }
            for b in sorted(batches, key=lambda b: (b.revision, b.id))
        ],
        "intervals": [
            {
                "seq": r.seq,
                "revision": r.revision,
                "start": r.start_utc,
                "end": r.end_utc,
                "timezone": r.timezone,
                "energy": r.energy,
            }
            for r in sorted(intervals, key=lambda r: (r.revision, r.seq))
        ],
        "quote": build_quote_response(version, quote).model_dump(),
    }


# ----------------------------------------------------------------- main flow


def close_cycle(cycle_id: str, req: CycleCloseRequest, *,
                engine=None) -> CloseResult:
    """原子关账，返回 ``(快照响应体, 是否幂等重放)``。

    任何校验失败都抛出 :class:`PricingError` 且整笔事务回滚——账期状态、
    revision 与批次/区间均不改变。``engine`` 可注入（测试用临时文件库）。
    """
    validate_close_id(req.close_id)
    eng = engine or default_engine
    expected_total = req.expected_total_amount.quantize(Q2)

    attempt = 0
    while True:
        attempt += 1
        try:
            return _close_once(eng, cycle_id, req, expected_total)
        except OperationalError as exc:  # SQLITE_BUSY write-lock race
            if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                raise
            # 全新事务重读 revision / 关账记录后整笔重跑。


def _is_lock_error(exc: OperationalError) -> bool:
    msg = str(exc.orig).lower() if exc.orig is not None else ""
    return ("database is locked" in msg
            or "database table is locked" in msg)


def _close_once(eng, cycle_id: str, req: CycleCloseRequest,
                expected_total: Decimal) -> CloseResult:
    with Session(eng) as session:
        try:
            with session.begin():
                # 1) 幂等优先：同 close_id 重放/冲突先于一切状态判定。
                existing = session.get(CycleClosure, req.close_id)
                if existing is not None:
                    return _replay_or_conflict(
                        req.close_id, existing, cycle_id,
                        req.expected_revision, expected_total)

                # 2) 重新读取账期；不存在 404；已关账时按本次 close_id 归类
                #    为重放 / 冲突 / cycle_closed。
                cycle = session.get(BillingCycle, cycle_id)
                if cycle is None:
                    raise _cycle_not_found(cycle_id)
                if cycle.status == "closed":
                    return _closed_cycle_outcome(
                        session, cycle_id, req, expected_total)

                # 3) 事务内重新比对 revision 与累计金额（到分）。
                if req.expected_revision != cycle.revision:
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, cycle.revision)
                current_total = Decimal(cycle.total_amount).quantize(
                    Q2, rounding=ROUND_HALF_UP)
                if current_total != expected_total:
                    raise _amount_mismatch(
                        cycle_id, expected_total, current_total)

                # 4) 重新读取全部批次与区间（稳定排序后入快照/摘要）。
                batches = session.scalars(
                    select(UsageBatch)
                    .where(UsageBatch.cycle_id == cycle_id)
                ).all()
                intervals = session.scalars(
                    select(CycleInterval)
                    .where(CycleInterval.cycle_id == cycle_id)
                ).all()

                version = load_version(session, cycle.version_code)
                payload = [
                    _StoredIntervalPayload(
                        _parse_z(r.start_utc), _parse_z(r.end_utc),
                        r.timezone, Decimal(r.energy))
                    for r in sorted(intervals, key=lambda r: (r.revision,
                                                              r.seq))
                ]
                # 复用计价引擎对账期整体重算（结构/规则缺口问题在此暴露，
                # 因早于任何写入，失败天然不改变账期）。
                quote_result = pricing.quote(version, payload) if payload else None

                closed_at = _now()
                body = _snapshot_body(
                    close_id=req.close_id, cycle=cycle,
                    currency=version.currency, energy_unit=version.energy_unit,
                    total_amount=current_total, batches=batches,
                    intervals=intervals, quote=quote_result, version=version,
                    closed_at=closed_at)

                # 5) 条件置 closed：只有仍是 open 且 revision 匹配才生效；
                #    rowcount==0 即并发批次/并发关账抢先，回滚后重读归类。
                guarded = session.execute(
                    update(BillingCycle)
                    .where(BillingCycle.cycle_id == cycle_id,
                           BillingCycle.status == "open",
                           BillingCycle.revision == req.expected_revision)
                    .values(status="closed", closed_at=closed_at,
                            updated_at=closed_at)
                )
                if guarded.rowcount != 1:
                    refreshed = session.execute(
                        select(BillingCycle)
                        .where(BillingCycle.cycle_id == cycle_id)
                        .execution_options(populate_existing=True)
                    ).scalar_one()
                    if refreshed.status == "closed":
                        # 并发关账抢先生效（同一 close_id 同内容重放原快照，
                        # 异内容/异 close_id 冲突）。
                        return _closed_cycle_outcome(
                            session, cycle_id, req, expected_total)
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, refreshed.revision)

                session.add(CycleClosure(
                    close_id=req.close_id,
                    cycle_id=cycle_id,
                    revision=cycle.revision,
                    expected_revision=req.expected_revision,
                    expected_total_amount=str(expected_total),
                    total_energy=cycle.total_energy,
                    total_amount=str(current_total),
                    detail_digest=body["detail_digest"],
                    snapshot_json=json.dumps(body, ensure_ascii=False),
                    created_at=closed_at,
                ))
                return CloseResult(body, False)

        except IntegrityError as exc:
            # 唯一约束冲突（并发竞态）：只读重查归类为重放 / 冲突 /
            # cycle_closed / revision_conflict，绝不写任何数据。
            return _classify_integrity_error(
                eng, cycle_id, req, expected_total, exc)


def _classify_integrity_error(eng, cycle_id, req, expected_total, exc):
    with Session(eng) as session:
        row = session.get(CycleClosure, req.close_id)
        if row is not None:
            return _replay_or_conflict(
                req.close_id, row, cycle_id, req.expected_revision,
                expected_total)
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is not None and cycle.status == "closed":
            return _closed_cycle_outcome(
                session, cycle_id, req, expected_total)
        if cycle is not None:
            raise _revision_conflict(
                cycle_id, req.expected_revision, cycle.revision) from exc
    raise exc


# --------------------------------------------------------------- read model


def get_closure(cycle_id: str, *, engine=None) -> dict:
    """返回账期的不可变关账快照。

    账期不存在 → 404 ``cycle_not_found``；账期尚未关账 →
    404 ``closure_not_found``。
    """
    eng = engine or default_engine
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise _cycle_not_found(cycle_id)
        row = session.scalar(
            select(CycleClosure).where(CycleClosure.cycle_id == cycle_id))
        if row is None:
            raise PricingError(
                f"账期 {cycle_id} 尚未关账，没有关账快照",
                code="closure_not_found", status_code=404,
                details={"cycle_id": cycle_id})
        return json.loads(row.snapshot_json)
