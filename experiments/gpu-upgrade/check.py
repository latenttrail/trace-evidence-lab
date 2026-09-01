"""Check the recorded GPU-upgrade observations and emit a reproducible receipt."""

import json
from pathlib import Path


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "observations.json"


def pct_change(old: float, new: float) -> float:
    """Return percentage change from old to new."""
    return (new - old) / old * 100.0


def main() -> None:
    """Validate the recorded comparisons and print their boundaries."""
    data = json.loads(DATA.read_text())
    h100 = data["hardware_swap"]["h100"]
    h200 = data["hardware_swap"]["h200"]
    eager = data["compiled_counterfactual"]["eager"]
    compiled = data["compiled_counterfactual"]["compiled"]
    itl_delta = pct_change(h100["itl_p95_ms"], h200["itl_p95_ms"])
    throughput_delta = pct_change(
        h100["throughput_sse_chunks_per_s"], h200["throughput_sse_chunks_per_s"]
    )
    assert abs(itl_delta) > data["repeat_bands"]["itl_pct"]
    assert abs(throughput_delta) < data["repeat_bands"]["throughput_pct"]
    assert compiled["itl_p95_ms"] < eager["itl_p95_ms"]
    assert compiled["throughput_sse_chunks_per_s"] > eager["throughput_sse_chunks_per_s"]
    print("Trace Evidence Lab / gpu-upgrade")
    print("  H200 ITL p95: 5.7 -> 5.3 ms")
    print("  H200 throughput: 16,187 -> 15,260 SSE chunks/s")
    print("  throughput change is inside the recorded +/-7% band")
    print("  compiled ITL p95: 19.2 -> 5.5 ms at 200 users")
    print("  compiled throughput: 5,434 -> 8,438 SSE chunks/s at 200 users")
    print("  status: all checks passed")
    print("  boundary: recorded campaign; checker validates arithmetic, not GPU execution")


if __name__ == "__main__":
    main()
