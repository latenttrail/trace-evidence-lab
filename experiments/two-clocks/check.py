"""Trace Evidence Lab experiment 02: separate first-token and token-gap clocks.

This checker validates the recorded comparison and its uncertainty boundaries. It
does not rerun the serving campaign or infer a universal concurrency limit.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent


def load_observations() -> dict:
    """Load the committed observations for the two-clock comparison."""
    return json.loads((ROOT / "data" / "observations.json").read_text())


def pct_change(old: float, new: float) -> float:
    """Return the percentage change from *old* to *new*."""
    return (new - old) / old * 100


def main() -> None:
    """Assert the reported changes and print a compact evidence receipt."""
    data = load_observations()
    runs = {row["name"]: row for row in data["runs"]}
    control = runs["comp-400"]
    high = runs["comp-600"]
    raised_limit = runs["comp-600-ci128"]
    bands = data["repeat_bands"]

    ttft_change = pct_change(control["ttft_p95_ms"], high["ttft_p95_ms"])
    itl_change = pct_change(control["itl_p95_ms"], high["itl_p95_ms"])
    throughput_change = pct_change(control["throughput_sse_chunks_s"], high["throughput_sse_chunks_s"])
    limit_throughput_change = pct_change(high["throughput_sse_chunks_s"], raised_limit["throughput_sse_chunks_s"])

    assert ttft_change > bands["ttft_p95_pct"]
    assert abs(itl_change) > bands["itl_p95_pct"]
    assert abs(throughput_change) > bands["throughput_pct"]
    assert abs(limit_throughput_change) > bands["throughput_pct"]

    mid = runs["comp-450"]
    assert abs(pct_change(control["ttft_p95_ms"], mid["ttft_p95_ms"])) <= bands["ttft_p95_pct"]
    assert abs(pct_change(control["itl_p95_ms"], mid["itl_p95_ms"])) <= bands["itl_p95_pct"]
    assert abs(pct_change(control["throughput_sse_chunks_s"], mid["throughput_sse_chunks_s"])) <= bands["throughput_pct"]

    print("Trace Evidence Lab / two-clocks")
    print(f"  TTFT p95: {control['ttft_p95_ms']:,} -> {high['ttft_p95_ms']:,} ms ({ttft_change:+.1f}%)")
    print(f"  ITL p95: {control['itl_p95_ms']:.1f} -> {high['itl_p95_ms']:.1f} ms ({itl_change:+.1f}%)")
    print(f"  throughput: {control['throughput_sse_chunks_s']:,} -> {high['throughput_sse_chunks_s']:,} SSE chunks/s ({throughput_change:+.1f}%)")
    print(f"  raised admission limit: {high['throughput_sse_chunks_s']:,} -> {raised_limit['throughput_sse_chunks_s']:,} ({limit_throughput_change:+.1f}%)")
    print("  450-user run: all three changes remain inside the recorded bands")
    print("  status: all checks passed")
    print("  boundary: recorded campaign; queueing is a derived diagnosis, not a directly measured wait")


if __name__ == "__main__":
    main()
