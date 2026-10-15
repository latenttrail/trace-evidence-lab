# Over-admission does not reduce completions. It converts them into repeated work.

**Evidence class:** Teaching miniature — simulator-derived, **not** Measured by us
**Provenance:** my own run against [`sim/serving_sim.py`](../../sim/serving_sim.py)

## The question

If a serving engine's KV cache is the thing that runs out, what does raising the admission limit
actually buy? The answer is not "fewer completions". It is the same completions, paid for again.

![Preemption thrash](visual/preemption-thrash-dark.gif)

A paper-ground version of the same loop is at [`visual/preemption-thrash-paper.gif`](visual/preemption-thrash-paper.gif).

```
$ python3 preemption_thrash.py

   max batch | completed | evictions | prompt tokens re-prefilled
           4 |         5 |         0 |          0
           8 |         8 |        17 |      4,352
          16 |         8 |       161 |     41,216
          24 |         8 |       215 |     55,040
          32 |         8 |       215 |     55,040

  decode tokens delivered                2,409
  decode tokens generated then evicted     792
  prompt tokens reprocessed             55,040
```

## What it shows

1. **Completions plateau at 8 and never move again**, from max batch 8 through 32. Nothing errors
   and no request is rejected. Every run is a fixed window of 400 scheduler steps, and every count
   is what happened inside it.
2. **The waste is prefill, not decode.** An evicted request loses its cache and restarts from its
   prompt, so the engine reprocessed 55,040 prompt tokens to deliver 2,409 output tokens — 22.8
   reprocessed per token delivered. Only 792 generated tokens were actually thrown away. Decode
   loss does grow with the limit, but slowly (581 → 792) while re-prefill grows 13×.
3. **Each request gets less far before it is knocked back**: 34.2 decode tokens per eviction at
   max batch 8, 4.9 at 16, 3.7 at 24 and 32.
4. **The limit should have been set from the cache, not the queue.** 3,072 token slots fit 12
   requests at admission and 6 at completion, because a request grows for as long as it runs.
   Admission control reads the first number; the system is bound by the second. Stepping the
   limit one at a time, the first eviction appears at max batch 7: one above the 6 that fit at
   completion. The script asserts this.

## Boundary

**This says nothing about GPU utilisation.** `serving_sim.py` states that it "does NOT model
kernels, precision, real timings, or any specific engine", so it cannot. What it observes is a
scheduler property: the set of admitted requests is never empty, so the engine always has work to
issue. Whether that keeps a real device busy is a different question, and this artifact does not
answer it. An earlier version of this receipt claimed otherwise.

Steps, not milliseconds. It reproduces a mechanism and its shape; it is not a benchmark, and no
engine is named or ranked. Real schedulers use watermarks, swapping and smarter victim selection,
all of which change the numbers and none of which remove the trade-off.

This run has **no prefix cache**. 55,040 is this configuration's re-prefill cost. What a
cache-enabled engine would pay is not measured here, and no bound is claimed on it.

## Corrections

Three things this experiment got wrong before the post went out. Each is disclosed rather than
quietly replaced.

1. **The finding is a plateau, not a reversal.** The original hypothesis was that over-admission
   reduces completions. The simulation said otherwise. A plateau is the harder result to spot:
   a reversal shows on a dashboard, whereas a plateau looks like a system that has simply reached
   capacity.
2. **Decode waste was overestimated about 24×.** Discarded work was first computed as
   evictions × output length, which assumes every evicted request had generated a full 200
   tokens. They hadn't. That estimate said 19,400 decode tokens were thrown away; the
   instrumented figure is 792. It moved the finding: the waste isn't decode, it's re-prefill.
3. **Evictions were undercounted (fixed 2026-10-15).** The script tallied only finished and
   running requests. An evicted request goes back to the waiting queue, so every request still
   waiting when the run stopped, and every eviction it had suffered, was missing. At the top
   setting that hid more than half the evictions (97 counted, 215 real), and it made decode loss
   appear to fall as admission rose. The figures above are the corrected ones.

## Why prefill and decode are never added together

A prefill token and a decode token do not cost the same thing. Prefill reads the whole prompt in
parallel and is compute-bound; decode produces one token at a time and is memory-bound. A combined
"wasted tokens" total would carry no meaningful unit, so the two figures stay separate everywhere.

## Prefix caching

A real engine with prefix caching would face a smaller re-prefill bill than this run. It does not
remove the mechanism, because the blocks holding that prefix are the same blocks the engine just
reclaimed to make room. Prefix caching helps most when memory is not under pressure, and this
experiment is about what happens when it is. What a cache-enabled engine would pay is not measured
here, so no bound is claimed.

## Run it

```bash
python3 preemption_thrash.py
```

Standard library only. The script asserts its own headline figures.
