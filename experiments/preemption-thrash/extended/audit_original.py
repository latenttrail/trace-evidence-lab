"""
Audit of the ORIGINAL simulator (sim/serving_sim.py) as used by ../preemption_thrash.py.

Nothing here changes the original. `Probe` is a line-for-line copy of
Engine.step with read-only counters added; the script first asserts that the
probe reproduces the original engine's per-step log exactly, so every number
below describes the original engine's behaviour, not a modified one.

Run:  python3 audit_original.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sim"))
from serving_sim import Engine, Request   # noqa: E402

STEPS, N_REQ, PROMPT, OUTPUT, BLOCKS, BS = 400, 24, 256, 200, 192, 16


class Probe(Engine):
    def step(self, arrivals=()):
        for r in arrivals:
            self.waiting.append(r)
        n0 = len(self.running)
        self._admit()
        if len(self.running) - n0 > 1:
            self.stats['steps_admitting_several'] += 1
        budget, prefilled_this_step = self.prefill_budget, 0
        for r in list(self.running):
            if r.is_prefilling and budget > 0:
                take = min(budget, r.prompt - r.prefilled)
                r.prefilled += take; budget -= take; prefilled_this_step += take
        used = self.blocks_used()
        self.stats['peak_blocks_after_prefill'] = max(self.stats['peak_blocks_after_prefill'], used)
        if used > self.total_blocks:
            self.stats['steps_overcommitted_after_prefill'] += 1
        decoded = 0
        stalled = (not self.chunked_prefill) and prefilled_this_step > 0
        for r in list(self.running):
            if stalled or r.is_prefilling or r.cancelled_at is not None:
                continue
            if self._need(r) > self.blocks_free():
                v = self._preempt_one()
                if v is None:
                    continue
                if v is r:
                    self.stats['tokens_to_self_evicted_request'] += 1
            r.generated += 1; decoded += 1
            r.token_steps.append(self.step_no)
            if r.first_token_at is None:
                r.first_token_at = self.step_no
            if r.generated >= r.output:
                r.done_at = self.step_no
                self.running.remove(r)
                self.done.append(r)
        self.log.append({'step': self.step_no, 'running': len(self.running), 'waiting': len(self.waiting),
                         'blocks_used': self.blocks_used(), 'prefilled': prefilled_this_step,
                         'decoded': decoded, 'stalled': stalled})
        self.step_no += 1
        return self.log[-1]


def run(cls, mb, steps=STEPS):
    e = cls(total_blocks=BLOCKS, block_size=BS, prefill_budget=512, max_batch=mb, chunked_prefill=True)
    e.stats = dict(steps_admitting_several=0, peak_blocks_after_prefill=0,
                   steps_overcommitted_after_prefill=0, tokens_to_self_evicted_request=0)
    sched = {}
    for i in range(N_REQ):
        sched.setdefault(i * 4, []).append(Request(i, i * 4, PROMPT, OUTPUT))
    e.run(sched, steps)
    return e


if __name__ == '__main__':
    for mb in (4, 7, 8, 16, 24, 32):
        assert run(Engine, mb).log == run(Probe, mb).log, f"probe diverges from Engine at max batch {mb}"
    print("Probe reproduces the original Engine's per-step log exactly (max batch 4, 7, 8, 16, 24, 32).\n")

    print("A1-A3: behaviour inside one step, 400-step window")
    print(f"  {'max batch':>9} | {'peak blocks after prefill':>25} | {'steps over capacity':>19} | "
          f"{'steps admitting >1':>18} | {'tokens to self-evicted req':>26}")
    for mb in (4, 7, 8, 16, 24, 32):
        s = run(Probe, mb).stats
        print(f"  {mb:>9} | {s['peak_blocks_after_prefill']:>19} / {BLOCKS} | {s['steps_overcommitted_after_prefill']:>19} | "
              f"{s['steps_admitting_several']:>18} | {s['tokens_to_self_evicted_request']:>26}")

    print("\nA4: step cost. The original engine advances one step per call regardless of work:")
    e = run(Engine, 32)
    pf = [row['prefilled'] for row in e.log]
    print(f"  max batch 32: prefill tokens per step range {min(pf)}-{max(pf)}, every step costs 1.")
    print("  Re-prefill therefore costs no time; only the 512-token prefill budget limits it.")

    print("\nA6: workload ceiling. Only 24 requests exist, so any limit >= 24 is 'unlimited':")
    logs = {mb: run(Engine, mb).log for mb in (12, 16, 24, 32)}
    print("  per-step logs identical for max batch 24 and 32:", logs[24] == logs[32])
    print("  Because admission reserves nothing (A2), far more requests run at once than the")
    print("  cache holds even as bare prompts (12 x 16 blocks = 192):")
    for mb in (12, 16, 32):
        print(f"    max batch {mb:>2}: most requests ever running at once = {max(row['running'] for row in logs[mb])}")

    print("\nA7: completion counts are window-bound. Completions by step:")
    for mb in (4, 8, 32):
        print(f"  max batch {mb:>2}: " + ", ".join(
            f"{steps}: {len([r for r in run(Engine, mb, steps).done if r.done_at is not None])}"
            for steps in (400, 800, 1600)))
