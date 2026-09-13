"""
Trace Evidence · Post 4 (stable research ID 8) - proof object
=====================================================

Claim under test
----------------
A long prompt arriving from a new user delays the replies of users already
mid-conversation. Nothing is overloaded, nothing errors, and the affected users
never sent the prompt that slowed them down.

The cause is that prefill and decode compete for the same forward pass. A step
spent reading a 4,000-token prompt is a step in which nobody in flight gets a
token.

Evidence class : Teaching miniature - simulator-derived, never 'Measured by us'
Provenance     : Ankit run against `sim/serving_sim.py`, written for this series.

Scope boundary
--------------
A simulation, not a benchmark. Time is measured in scheduler steps, not
milliseconds - converting would be inventing a number. It reproduces the SHAPE
of the interaction and the mechanism that causes it. It is not a measurement of
any engine, and no engine is named or ranked.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 long_prompt_stall.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))   # repo: <root>/sim
from serving_sim import Engine, Request, itl_series   # noqa: E402

STEPS = 140


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def scenario(chunked, big_prompt=4096):
    e = Engine(total_blocks=4096, block_size=16, prefill_budget=512,
               max_batch=16, chunked_prefill=chunked)
    # Four conversations already in flight, short prompts, long replies.
    schedule = {0: [Request(i, 0, 128, 100) for i in range(4)]}
    # At step 40, one new user arrives with a very large prompt.
    schedule[40] = [Request(99, 40, big_prompt, 20)]
    e.run(schedule, STEPS)
    return e


def gap_report(e, label):
    victims = [r for r in e.done + e.running if r.rid < 4]
    gaps = []
    for r in victims:
        gaps += itl_series(r)
    worst = max(gaps) if gaps else 0
    typical = sorted(gaps)[len(gaps) // 2] if gaps else 0
    return typical, worst, len(gaps)


if __name__ == "__main__":
    print(__doc__.split("Run\n---")[0].strip())

    rule("FOUR CONVERSATIONS IN FLIGHT. AT STEP 40, ONE 4,096-TOKEN PROMPT ARRIVES.")

    for chunked, label in ((False, 'prefill takes the whole step'),
                           (True, 'prefill chunked alongside decode')):
        e = scenario(chunked)
        typ, worst, n = gap_report(e, label)
        stalls = sum(1 for l in e.log if l.get('stalled'))
        print(f"""
  {label}
    steps where nobody in flight got a token   {stalls:>4}
    typical gap between tokens                 {typ:>4} step{'s' if typ != 1 else ''}
    worst gap between tokens                   {worst:>4} steps
""")
        if not chunked:
            base_worst = worst
        else:
            chunked_worst = worst

    print(f"""  The four original users sent 128-token prompts and asked for nothing unusual.
  Their worst pause went from {chunked_worst} step{'s' if chunked_worst != 1 else ''} to {base_worst} steps because somebody else
  arrived with a long prompt.
""")
    assert base_worst > chunked_worst * 2, "an unchunked long prefill should visibly stall decode"

    rule("AND IT SCALES WITH THE PROMPT, NOT WITH THE LOAD")
    print(f"  {'new prompt':>12} | {'worst gap for the users already there':>38}")
    print(f"  {'-'*12}-+-{'-'*38}")
    for p in (256, 1024, 4096, 16384):
        e = scenario(False, big_prompt=p)
        _, worst, _ = gap_report(e, '')
        print(f"  {p:>12,} | {worst:>32} steps")
    rule("THE THREE CLAIMS THAT WERE ASSERTED BUT NEVER COMPUTED")
    print("""
  The caption says throughput did not change, that the long prompt's own numbers
  look fine, and that each affected user looks unremarkable on their own. Those
  were prose in an earlier version of this receipt and nothing here calculated
  them. They are calculated and asserted below, or they do not ship.
""")

    # A control run with no long prompt, against the same seed and schedule.
    ctrl = Engine(total_blocks=4096, block_size=16, prefill_budget=512,
                  max_batch=16, chunked_prefill=False)
    ctrl.run({0: [Request(i, 0, 128, 100) for i in range(4)]}, STEPS)
    stalled = scenario(False)

    def delivered(e):
        return sum(r.generated for r in e.done + e.running if r.rid < 4)

    d_ctrl, d_stall = delivered(ctrl), delivered(stalled)
    drop = (d_ctrl - d_stall) / d_ctrl * 100 if d_ctrl else 0.0

    print(f"  decode tokens delivered to the four original users")
    print(f"    no long prompt arrives  {d_ctrl:>6}")
    print(f"    long prompt at step 40  {d_stall:>6}   {drop:+.1f} %")
    print()

    # The long prompt's own experience, which is the point about per-request views.
    big = [r for r in stalled.done + stalled.running if r.rid == 99][0]
    big_ttft = (big.first_token_at - big.arrival) if big.first_token_at is not None else None
    big_gaps = itl_series(big)
    big_worst = max(big_gaps) if big_gaps else 0

    print(f"  the long request's own numbers")
    print(f"    time to its first token {big_ttft:>6} steps")
    print(f"    its own worst gap       {big_worst:>6} steps")
    print()

    # Each victim on its own: median gap versus worst gap.
    print(f"  each affected user, seen alone")
    print(f"    {'user':>6} | {'median gap':>10} | {'worst gap':>9}")
    print(f"    {'-'*6}-+-{'-'*10}-+-{'-'*9}")
    victim_medians = []
    for r in sorted([r for r in stalled.done + stalled.running if r.rid < 4], key=lambda r: r.rid):
        g = itl_series(r)
        med = sorted(g)[len(g) // 2] if g else 0
        victim_medians.append(med)
        print(f"    {r.rid:>6} | {med:>10} | {max(g) if g else 0:>9}")

    print(f"""
  Throughput to the four is unchanged at {drop:+.1f} %: the tokens still arrive,
  they arrive later. The long request's own view is clean. Every affected user's
  MEDIAN gap is {max(victim_medians)} step or better while their worst is {base_worst}. Nothing in a
  per-request average shows this; only the gap inside one conversation does.
""")
    assert abs(drop) < 1.0, "throughput to the original users should be unchanged, not merely similar"
    assert big_worst <= 1, "the long request's own inter-token gaps should look unremarkable"
    assert max(victim_medians) <= 1, "each victim's median gap should look unremarkable alone"
    assert base_worst >= 8, "the worst gap should be the outlier the medians hide"

    rule("RESULT")
    print(f"""
  A user who was already talking to the system experienced a {base_worst}-step pause
  caused entirely by a request that was not theirs.

  This is why per-request latency averages hide the problem, and the section
  above computes it rather than asserting it: the long prompt's own numbers look
  fine, and the cost was paid by four other people.

  Chunked prefill does not make the work disappear. It spreads it, so the pause
  is distributed instead of landing on one turn.

  When you measure tail latency, measure it per conversation - not per request.
""")
