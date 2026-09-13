"""
Trace Evidence · Post 3 (stable research ID 9) - proof object
=============================================================

Claim under test
----------------
An engine that admits more requests than its KV cache can hold does not fail.
It admits one too many, runs out of blocks, evicts a request that was partway
through, and re-admits it later - at which point that request starts again from
its prompt.

I expected over-admission to reduce completions. It does not. Completions
PLATEAU while repeated work climbs, which is harder to diagnose: the system
looks like it has simply reached capacity, and the scheduler never once runs
out of admitted work - because it is reprocessing prompts it has already
processed.

Two corrections this script encodes
-----------------------------------
1. An earlier version counted discarded work as `preemptions * output_length`,
   which assumes every evicted request had generated a full output. It had not.
   The simulator now records the tokens actually lost at the moment the KV is
   dropped, and the true decode figure is roughly 37x smaller than that estimate.

2. With the honest accounting, the waste is not mostly decode at all. It is
   PREFILL: each eviction forces a whole prompt to be processed again. Decode
   loss stays small and does not grow; re-prefill grows without limit.

   Prefill and decode tokens are NOT summed here. A prefill token and a decode
   token do not cost the same - prefill is compute-bound and processes the
   prompt in parallel, decode is memory-bound and strictly sequential. Adding
   them would invent a total that means nothing.

Evidence class : Teaching miniature · simulator-derived
Provenance     : Ankit run against `sim/serving_sim.py`. Not a hardware measurement.

Scope boundary
--------------
A simulation. Steps, not milliseconds. It reproduces the mechanism and its
shape; it is not a benchmark of any engine, and no engine is named or ranked.

NO UTILISATION CLAIM IS MADE HERE, and an earlier version of this receipt made
one. sim/serving_sim.py states that it "does NOT model kernels, precision, real
timings, or any specific engine", so it cannot speak to GPU utilisation at all.
What it observes is a scheduler property: the set of admitted requests is never
empty, so the engine always has work to issue. Whether that keeps a real device
busy is a separate question this artifact does not answer.
Real schedulers use watermarks, swapping and smarter victim selection - all of
which change the numbers and none of which remove the trade-off.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 preemption_thrash.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))   # repo: <root>/sim
from serving_sim import Engine, Request   # noqa: E402

STEPS, N_REQ, PROMPT, OUTPUT = 400, 24, 256, 200
BLOCKS, BLOCK_SIZE = 192, 16


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def run(max_batch, blocks=BLOCKS):
    e = Engine(total_blocks=blocks, block_size=BLOCK_SIZE, prefill_budget=512,
               max_batch=max_batch, chunked_prefill=True)
    sched = {}
    for i in range(N_REQ):
        sched.setdefault(i * 4, []).append(Request(i, i * 4, PROMPT, OUTPUT))
    e.run(sched, STEPS)
    return e


def report(e):
    allr = e.done + e.running
    return {
        "done": len([r for r in e.done if r.done_at is not None]),
        "hit": len([r for r in allr if r.preemptions]),
        "ev": sum(r.preemptions for r in allr),
        "kept": sum(r.generated for r in allr),
        "dec": sum(r.redone_decode for r in allr),
        "pre": sum(r.redone_prefill for r in allr),
    }


if __name__ == "__main__":
    print(__doc__.split("Run\n---")[0].strip())

    rule("SAME LOAD, SAME CACHE, ONE SETTING CHANGED: HOW MANY TO ADMIT")
    print(f"\n  {'max batch':>10} | {'completed':>10} | {'evictions':>10} | "
          f"{'prompt tokens re-prefilled':>27} | {'decode tokens lost':>19}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*27}-+-{'-'*19}")
    rows = []
    for mb in (4, 8, 16, 24, 32):
        r = report(run(mb))
        rows.append((mb, r))
        print(f"  {mb:>10} | {r['done']:>10} | {r['ev']:>10} | "
              f"{r['pre']:>27,} | {r['dec']:>19,}")

    plateau = [(mb, r) for mb, r in rows if mb >= 8]
    completions = {r["done"] for _, r in plateau}
    ev_lo, ev_hi = plateau[0][1]["ev"], plateau[-1][1]["ev"]
    pre_lo, pre_hi = plateau[0][1]["pre"], plateau[-1][1]["pre"]
    dec_lo, dec_hi = plateau[0][1]["dec"], plateau[-1][1]["dec"]

    print(f"""
  Completions stop improving at max batch {plateau[0][0]} and never move again: {plateau[0][1]['done']} for
  every setting above it.

  Over the same range evictions go {ev_lo} -> {ev_hi}, and prompt tokens re-prefilled
  go {pre_lo:,} -> {pre_hi:,}. A {pre_hi/pre_lo:.0f}x increase in repeated work bought exactly nothing.

  Nothing errored. No request was rejected. The queue was served throughout, and
  the engine never ran out of admitted work to issue.
""")
    assert len(completions) == 1, "completions should plateau across the over-admitted range"
    assert ev_hi > ev_lo * 3, "evictions should climb sharply while completions do not"
    assert pre_hi > pre_lo * 3, "re-prefill should climb with evictions"

    rule("THE WASTE IS PREFILL, NOT DECODE")
    top = plateau[-1][1]
    print(f"""
  decode tokens delivered                    {top['kept']:>8,}
  decode tokens generated then evicted       {top['dec']:>8,}
  prompt tokens processed a second time      {top['pre']:>8,}

  The intuition is that eviction throws away generated text. It mostly does not.
  At this setting a request loses about {top['dec']/top['ev']:.0f} generated tokens per eviction, and the
  decode loss does NOT grow with the admission limit: it goes {dec_lo:,} -> {dec_hi:,} while
  evictions go {ev_lo} -> {ev_hi}.

  What grows is re-prefill. Every eviction sends a request back to the start of
  its {PROMPT}-token prompt, so the engine reprocesses {top['pre']:,} prompt tokens to deliver
  {top['kept']:,} tokens of output - {top['pre']/top['kept']:.1f} prompt tokens reprocessed per token delivered.

  These two are deliberately not added together. Prefill is compute-bound and
  parallel; decode is memory-bound and sequential. A combined 'wasted tokens'
  total would be a number with no unit behind it.
""")
    assert top["pre"] > top["kept"] * 5, "re-prefill should dominate delivered output"
    assert top["dec"] < top["kept"], "decode loss should be small next to delivered output"
    assert dec_hi <= dec_lo, "decode loss should not grow as admission rises"

    rule("RAISING THE LIMIT MAKES EACH REQUEST PROGRESS LESS")
    print(f"\n  {'max batch':>10} | {'evictions':>10} | {'decode tokens per eviction':>27}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*27}")
    for mb, r in plateau:
        print(f"  {mb:>10} | {r['ev']:>10} | {r['dec']/r['ev']:>27.1f}")
    first, last = plateau[0][1], plateau[-1][1]
    print(f"""
  This is the part a dashboard cannot show. Raising the limit does not only
  cause more evictions. It causes each request to get less far before it is
  knocked back: from {first['dec']/first['ev']:.0f} generated tokens per eviction down to {last['dec']/last['ev']:.0f}.

  The engine is admitting work, letting it produce a handful of tokens, evicting
  it, reprocessing its whole prompt, and repeating. The scheduler stays saturated
  throughout, because churn and progress are the same shape from where it sits.
""")
    assert first["dec"] / first["ev"] > 4 * (last["dec"] / last["ev"]), \
        "progress per eviction should collapse as the admission limit rises"

    rule("WHAT THE LIMIT SHOULD HAVE BEEN SET FROM")
    import math
    slots = BLOCKS * BLOCK_SIZE
    at_admit = math.ceil(PROMPT / BLOCK_SIZE)
    at_done = math.ceil((PROMPT + OUTPUT) / BLOCK_SIZE)
    fits_admit, fits_done = BLOCKS // at_admit, BLOCKS // at_done
    print(f"""
  cache                                   {BLOCKS} blocks x {BLOCK_SIZE} tokens = {slots:,} token slots
  a request holding its prompt            ceil({PROMPT}/{BLOCK_SIZE})       = {at_admit} blocks
  the same request when it finishes       ceil({PROMPT + OUTPUT}/{BLOCK_SIZE}) = {at_done} blocks

  So the cache fits {fits_admit} requests at admission and {fits_done} at completion.

  Admission control reads the first number. The system is bound by the second. A
  request is not a fixed-size object: it grows for as long as it runs, so a set
  that fits on the way in does not fit on the way out.

  The plateau sits at max batch 8, between {fits_done} and {fits_admit}, which is what this
  arithmetic predicts. Sizing the limit on terminal occupancy rather than
  admission-time occupancy is what watermarking exists to approximate.
""")
    assert fits_done < fits_admit, "a request must grow while it runs, or there is no trade-off"
    assert fits_done <= plateau[0][0] <= fits_admit, \
        f"the plateau ({plateau[0][0]}) should fall between terminal ({fits_done}) and admission ({fits_admit}) capacity"

    rule("WHY THE OBVIOUS FIX MAKES IT WORSE")
    print("""
  The instinct on seeing a queue is to admit more, to clear it. That is exactly
  the move that pushes the cache past its limit and starts the eviction loop, so
  the queue grows, which invites admitting more still.

  The queue is the symptom. The cache is the constraint, and it is fixed.
""")

    rule("RESULT")
    print(f"""
  Raising the admission limit from {plateau[0][0]} to {plateau[-1][0]} changed completions by zero, raised
  evictions from {ev_lo} to {ev_hi}, and reprocessed {pre_hi:,} prompt tokens to deliver
  {top['kept']:,} tokens of output.

  A serving system that admits work it cannot hold does not degrade gracefully.
  It converts throughput into repeated work while never appearing to stall: the
  queue keeps draining and refilling, and a counter that only watches whether
  work is being issued reports a healthy system the whole time.

  Admission control is not a queue-length problem. It is a memory problem.
""")
