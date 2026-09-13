"""
A continuous-batching serving simulator, small enough to read.

Deterministic, seeded, CPU-only, standard library only. It models the handful of
things that actually decide serving behaviour:

  - a KV cache made of fixed-size blocks, which is a finite resource
  - a step loop that must choose, every step, between admitting new work
    (prefill) and continuing existing work (decode)
  - a prefill token budget per step, because a prompt is not free
  - preemption, because when blocks run out something has to give
  - cancellation, because clients disconnect

It does NOT model kernels, precision, real timings, or any specific engine. Step
counts are the unit; converting them to milliseconds would be inventing a number.
What it reproduces is the SHAPE of scheduling behaviour, which is what the posts
that use it are about.

Used by Trace Evidence posts 9, 13 and 19.
"""

from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    arrival: int
    prompt: int
    output: int
    prefilled: int = 0
    generated: int = 0
    admitted_at: int | None = None
    first_token_at: int | None = None
    done_at: int | None = None
    cancelled_at: int | None = None
    preemptions: int = 0
    # Work thrown away by eviction, accumulated at the moment the KV is dropped.
    # Counting `preemptions * output` instead would overstate this: a request
    # evicted early had not yet generated a full output.
    redone_decode: int = 0
    redone_prefill: int = 0
    token_steps: list = field(default_factory=list)

    @property
    def is_prefilling(self):
        return self.prefilled < self.prompt

    @property
    def finished(self):
        return self.done_at is not None or self.cancelled_at is not None

    def blocks(self, block_size):
        used = self.prefilled + self.generated
        return (used + block_size - 1) // block_size if used else 0


class Engine:
    """One step = one forward pass. Prefill and decode share it."""

    def __init__(self, total_blocks=64, block_size=16, prefill_budget=256,
                 max_batch=16, preempt='newest', release_on_cancel=True,
                 chunked_prefill=True):
        # chunked_prefill=False models the simpler scheduler: a step that does
        # any prefill does no decode. That is the contention Post 9 is about.
        self.chunked_prefill = chunked_prefill
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.prefill_budget = prefill_budget
        self.max_batch = max_batch
        self.preempt = preempt
        self.release_on_cancel = release_on_cancel
        self.waiting, self.running, self.done = [], [], []
        self.step_no = 0
        self.log = []

    # ---------------------------------------------------------------- blocks
    def blocks_used(self):
        return sum(r.blocks(self.block_size) for r in self.running)

    def blocks_free(self):
        return self.total_blocks - self.blocks_used()

    def _need(self, r):
        """Blocks this request will need after one more token."""
        used = r.prefilled + r.generated + 1
        return (used + self.block_size - 1) // self.block_size - r.blocks(self.block_size)

    # ------------------------------------------------------------ scheduling
    def _admit(self):
        while self.waiting and len(self.running) < self.max_batch:
            r = self.waiting[0]
            need = (min(r.prompt, self.prefill_budget) + self.block_size - 1) // self.block_size
            if need > self.blocks_free():
                break
            self.waiting.pop(0)
            r.admitted_at = self.step_no
            self.running.append(r)

    def _preempt_one(self):
        """Blocks ran out. Evict someone and put them back in the queue."""
        cands = [r for r in self.running if not r.is_prefilling]
        if not cands:
            return None
        victim = cands[-1] if self.preempt == 'newest' else cands[0]
        self.running.remove(victim)
        victim.preemptions += 1
        victim.redone_decode += victim.generated
        victim.redone_prefill += victim.prefilled
        victim.prefilled, victim.generated = 0, 0     # its KV is gone
        self.waiting.insert(0, victim)
        return victim

    def cancel(self, rid):
        for r in list(self.running):
            if r.rid == rid:
                r.cancelled_at = self.step_no
                if self.release_on_cancel:
                    self.running.remove(r)
                    self.done.append(r)
                return True
        return False

    # ------------------------------------------------------------- the step
    def step(self, arrivals=()):
        for r in arrivals:
            self.waiting.append(r)
        self._admit()

        budget = self.prefill_budget
        prefilled_this_step = 0
        for r in list(self.running):
            if r.is_prefilling and budget > 0:
                take = min(budget, r.prompt - r.prefilled)
                r.prefilled += take
                budget -= take
                prefilled_this_step += take

        # Decode only runs for requests that have finished their prompt, and a
        # step spent on prefill is a step those requests did not get a token in.
        decoded = 0
        stalled = (not self.chunked_prefill) and prefilled_this_step > 0
        for r in list(self.running):
            if stalled or r.is_prefilling or r.cancelled_at is not None:
                continue
            if self._need(r) > self.blocks_free():
                if self._preempt_one() is None:
                    continue
            r.generated += 1
            decoded += 1
            r.token_steps.append(self.step_no)
            if r.first_token_at is None:
                r.first_token_at = self.step_no
            if r.generated >= r.output:
                r.done_at = self.step_no
                self.running.remove(r)
                self.done.append(r)

        self.log.append({
            'step': self.step_no,
            'running': len(self.running),
            'waiting': len(self.waiting),
            'blocks_used': self.blocks_used(),
            'prefilled': prefilled_this_step,
            'decoded': decoded,
            'stalled': stalled,
        })
        self.step_no += 1
        return self.log[-1]

    def run(self, schedule, steps, hooks=None):
        """schedule: {step: [Request, ...]}   hooks: {step: fn(engine)}"""
        for s in range(steps):
            if hooks and s in hooks:
                hooks[s](self)
            self.step(schedule.get(s, ()))
        return self.log


def itl_series(r):
    """Gaps between consecutive generated tokens, in steps."""
    return [b - a for a, b in zip(r.token_steps, r.token_steps[1:])]
