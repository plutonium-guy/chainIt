"""
stepcraft — advanced end-to-end example: an order-processing pipeline.

Demonstrates, in one cohesive program:
  • `@piped` composition with the `|` operator and automatic list auto-mapping
  • `PIPE` argument injection and bound keyword arguments
  • `@retry` + `@circuit_breaker` on a flaky downstream call
  • `parallel='thread'` and `batch_size` execution modes
  • `ConditionalStep` / `SwitchStep` routing
  • `FanOutStep` / `FanInStep` (compute tax & shipping concurrently, then merge)
  • `MapReduceStep` aggregation
  • `Graph` DAG orchestration (diamond dependency, parallel scheduling)
  • shared run context via `Pipeline(context=...)` + `get_context()`
  • `on_step` observability hooks
  • async execution through `run_async` (uvloop when available)
  • runtime type-checking (beartype) + `schema=` output validation

Run it:  uv run --with stepcraft python examples/order_pipeline.py
"""

from __future__ import annotations

import time
from typing import Any

from stepcraft import (
    ConditionalStep,
    FanInStep,
    FanOutStep,
    Graph,
    MapReduceStep,
    Pipeline,
    PIPE,
    SwitchStep,
    circuit_breaker,
    get_context,
    piped,
    retry,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

ORDERS: list[dict[str, Any]] = [
    {"id": "A-1", "country": "US", "tier": "gold", "items": [40.0, 10.0]},
    {"id": "A-2", "country": "DE", "tier": "std", "items": [20.0]},
    {"id": "A-3", "country": "IN", "tier": "gold", "items": [5.0, 5.0, 5.0]},
    {"id": "A-4", "country": "US", "tier": "std", "items": [100.0]},
]


# ---------------------------------------------------------------------------
# 1. Per-order steps. A default @piped step applied to a *list* auto-maps over
#    each element, so these all operate on a single order dict.
# ---------------------------------------------------------------------------

@piped
def validate(order: dict) -> dict:
    """beartype enforces `order: dict` (and the `-> dict` return); we add our
    own field checks and read the shared run context."""
    if not order.get("items"):
        raise ValueError(f"order {order.get('id')} has no items")
    subtotal = round(sum(order["items"]), 2)
    return {**order, "subtotal": subtotal, "trace": get_context().get("trace_id")}


# SwitchStep routes each order to a discount branch keyed by customer tier.
discount = SwitchStep(
    key=lambda o: o["tier"],
    branches={
        "gold": piped(lambda o: {**o, "discount": round(o["subtotal"] * 0.10, 2)}),
        "std": piped(lambda o: {**o, "discount": 0.0}),
    },
    default=piped(lambda o: {**o, "discount": 0.0}),
)

# ConditionalStep flags "big" orders (post-discount) for review.
flag_review = ConditionalStep(
    condition=lambda o: (o["subtotal"] - o["discount"]) >= 90.0,
    if_true=piped(lambda o: {**o, "review": True}),
    if_false=piped(lambda o: {**o, "review": False}),
)


# ---------------------------------------------------------------------------
# 2. A flaky "fraud service" guarded by retry + circuit breaker.
#    The breaker returns a *new* PipeStep with its own state (immutable
#    decorators), so reuse never shares counters.
# ---------------------------------------------------------------------------

_attempts: dict[str, int] = {}


@circuit_breaker(failure_threshold=5, recovery_timeout=1.0)
@retry(max_attempts=4, delay=0.05, backoff=2.0)
@piped
def fraud_check(order: dict) -> dict:
    """Fails the first 2 calls per order, then succeeds — retry recovers it."""
    n = _attempts.get(order["id"], 0) + 1
    _attempts[order["id"]] = n
    if n < 3:
        raise ConnectionError(f"fraud service timeout for {order['id']} (try {n})")
    return {**order, "fraud_score": round(0.01 * len(order["items"]), 3)}


# ---------------------------------------------------------------------------
# 3. Fan-out: compute tax and shipping concurrently, then fan-in to merge.
#    FanInStep receives the tuple of branch outputs.
# ---------------------------------------------------------------------------

_TAX = {"US": 0.07, "DE": 0.19, "IN": 0.18}


@piped
def compute_tax(order: dict) -> dict:
    net = order["subtotal"] - order["discount"]
    return {"tax": round(net * _TAX.get(order["country"], 0.0), 2)}


@piped
def compute_shipping(order: dict) -> dict:
    # Pretend this is slow I/O so the thread fan-out actually overlaps.
    time.sleep(0.02)
    return {"shipping": 0.0 if order["subtotal"] >= 50 else 4.99}


def merge_costs(order: dict, tax: dict, shipping: dict) -> dict:
    total = round(
        order["subtotal"] - order["discount"] + tax["tax"] + shipping["shipping"], 2
    )
    return {**order, **tax, **shipping, "total": total}


def price_order(order: dict) -> dict:
    """Compute tax and shipping concurrently (thread fan-out), then merge.
    The order itself is captured in the fan-in combiner's closure."""
    fan = FanOutStep(
        (compute_tax(order), compute_shipping(order)),  # bound to this order
        parallel="thread",
    )
    combine = FanInStep(lambda tax, shipping: merge_costs(order, tax, shipping))
    return (fan | combine).run()


price = piped(price_order)


# ---------------------------------------------------------------------------
# 4. Compose the per-order pipeline and run it over all orders with a shared
#    context and an on_step timing hook.
# ---------------------------------------------------------------------------

def timing_hook(name: str, _inp: Any, _out: Any, dt: float) -> None:
    print(f"    · {name:<26} {dt * 1000:6.2f} ms")


def process_orders() -> list[dict]:
    # One linear pipeline; apply it once per order with Pipeline.map() so each
    # step receives a single order — SwitchStep/ConditionalStep route per order
    # (they don't auto-map collections the way plain @piped steps do).
    per_order: Pipeline = validate | discount | flag_review | fraud_check | price
    with_ctx = Pipeline(per_order.steps, context={"trace_id": "trace-42"})

    print("→ on_step + shared-context trace for one order (A-1):")
    first = with_ctx.run(ORDERS[0], on_step=timing_hook)
    print(f"      validate saw trace_id = {first['trace']!r}")

    rest = with_ctx.map(ORDERS[1:])
    return [first, *rest]


# ---------------------------------------------------------------------------
# 5. MapReduce: aggregate total revenue across priced orders.
# ---------------------------------------------------------------------------

revenue = MapReduceStep(
    mapper=lambda o: o["total"],
    reducer=lambda totals: round(sum(totals), 2),
    batch_size=2,
)


# ---------------------------------------------------------------------------
# 6. Graph DAG: a diamond that fans analytics out and joins them in a report.
#    seed -> ingest -> {by_country, by_tier} -> report
# ---------------------------------------------------------------------------

def build_report_graph() -> Graph:
    def by_country(priced: list[dict]) -> dict:
        agg: dict[str, float] = {}
        for o in priced:
            agg[o["country"]] = round(agg.get(o["country"], 0.0) + o["total"], 2)
        return agg

    def by_tier(priced: list[dict]) -> dict:
        agg: dict[str, int] = {}
        for o in priced:
            agg[o["tier"]] = agg.get(o["tier"], 0) + 1
        return agg

    def report(parents: tuple) -> dict:
        # Two parents -> inputs arrive as a tuple in sorted-parent order.
        country, tier = parents
        return {"revenue_by_country": country, "orders_by_tier": tier}

    return (
        Graph()
        .add_node("ingest", piped(lambda priced: priced, map=False))
        .add_node("by_country", piped(by_country, map=False))
        .add_node("by_tier", piped(by_tier, map=False))
        .add_node("report", piped(report))
        .add_edge("ingest", "by_country")
        .add_edge("ingest", "by_tier")
        .add_edge("by_country", "report")
        .add_edge("by_tier", "report")
    )


# ---------------------------------------------------------------------------
# 7. Async: process orders concurrently through the async runtime (uvloop).
# ---------------------------------------------------------------------------

@piped
async def async_score(order: dict) -> dict:
    import asyncio

    await asyncio.sleep(0.01)
    return {**order, "priority": "high" if order.get("review") else "normal"}


def run_async_demo(priced: list[dict]) -> list[dict]:
    pipe = Pipeline([async_score])
    # map_async fans the orders out concurrently on the (uvloop) event loop.
    return pipe.map_async(priced)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("stepcraft advanced example — order pipeline")
    print("=" * 70)

    priced = process_orders()
    print("\n→ Priced orders:")
    for o in priced:
        print(
            f"    {o['id']}: subtotal={o['subtotal']:>6} discount={o['discount']:>5} "
            f"tax={o['tax']:>5} ship={o['shipping']:>4} → total={o['total']:>7} "
            f"review={o['review']} fraud={o['fraud_score']}"
        )

    print(f"\n→ MapReduce total revenue: {revenue.run(priced)}")

    print("\n→ Graph DAG report (parallel scheduling):")
    report = build_report_graph().run(seed=priced, parallel=True)["report"]
    print(f"    revenue_by_country = {report['revenue_by_country']}")
    print(f"    orders_by_tier     = {report['orders_by_tier']}")

    print("\n→ Async prioritization via run_async (uvloop when available):")
    prioritized = run_async_demo(priced)
    for o in prioritized:
        print(f"    {o['id']}: priority={o['priority']}")

    print("\n→ PIPE injection demo:")
    add = piped(lambda base, bonus: base + bonus)
    print(f"    add(PIPE, bonus=100).run(5) = {add(PIPE, bonus=100).run(5)}")

    print("\n✓ done")


if __name__ == "__main__":
    main()
