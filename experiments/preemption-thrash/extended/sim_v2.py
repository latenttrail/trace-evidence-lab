"""
serving_sim_v2 - a corrected, timed continuous-batching simulator.

This does NOT replace ../../../sim/serving_sim.py. The original experiment in
../preemption_thrash.py still runs against the original simulator, unchanged.
This file exists because an audit of the original found behaviour that a
stronger experiment cannot rest on (see ../extended/README.md, "Audit"):

  A1  Prefill allocates KV without checking free blocks. After the prefill
      phase the original engine can hold 224 of 192 blocks; the decode phase
      then preempts its way back under capacity.
  A2  Admission does not reserve memory: an admitted request holds zero blocks
      until it prefills, so one step can admit many requests against the same
      free blocks.
  A3  A request that evicts itself still receives a token that step.
  A4  Every step costs the same, so re-prefill is free in time. A throughput
      plateau is therefore guaranteed by construction, not discovered.
  A5  Eviction discards generated tokens and makes the request re-decode them.
      vLLM V1 keeps generated tokens and rebuilds their KV by prefill
      (scheduler.py `_preempt_request`: num_computed_tokens = 0, output kept,
      request prepended to the waiting queue).

What v2 models, and the assumption behind each choice
-----------------------------------------------------
- KV blocks are allocated before a step runs, for exactly the tokens that step
  computes. Nothing ever exceeds capacity; a test asserts it every step.
- Scheduling order follows vLLM V1's documented chunked-prefill policy: running
  requests first, in admission order, then waiting requests while the per-step
  token budget lasts. Decode costs one budget token; prefill is chunked.
- Victim selection: the newest running request (vLLM V1 FCFS: running.pop()).
  Victims go to the FRONT of the waiting queue.
- Resume mode 'recompute' (default, vLLM V1-like): generated tokens are kept and
  their KV is rebuilt by prefill. Mode 'restart' reproduces the original
  simulator's semantics (generated tokens discarded and re-decoded).
- Admission policies:
    cap        admit while running < cap and the first prefill chunk fits
               (optimistic, vLLM-like; cap plays the role of max_num_seqs)
    watermark  as cap, but only if free blocks after admission stay >= w * total
    reserve    admit only if every running request could reach prompt +
               max_tokens without preemption (conservative; never preempts
               while outputs respect max_tokens)
- Step time (ASSUMED, in abstract units "u", not milliseconds):
      dt = c0 + a * prefill_tokens + b * decode_seqs
  c0 = 1 is one weight pass; a is the marginal cost of one prefill token; b the
  marginal cost of one more decoding sequence. Defaults a = 0.004, b = 0.01 are
  order-of-magnitude choices for a ~8B dense model on one modern GPU (a 256-token
  prompt costs about one decode step). They are swept, not trusted. a = b = 0
  reproduces the original simulator's "every step costs the same" assumption.

What v2 still does NOT model
----------------------------
Prefix caching, swapping to host memory, kernels, attention cost growth with
context length, CUDA graphs, tensor parallelism, speculative decoding, or any
named engine. Time units are not milliseconds and must never be converted.

Standard library only. Deterministic for a given workload.
"""

from dataclasses import dataclass, field
import math


@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    output: int            # true output length (unknown to the scheduler)
    max_tokens: int        # declared cap; only the 'reserve' policy reads it
    generated: int = 0     # output tokens produced and kept
    computed: int = 0      # tokens whose KV is resident
    alloc: int = 0         # tokens KV blocks are allocated for (>= computed)
    high_water: int = 0    # most tokens ever resident; prefill below it is recompute
    first_admit: float | None = None
    first_token: float | None = None
    done_at: float | None = None
    evictions: int = 0
    recompute_tokens: int = 0
    discarded_decode: int = 0
    kv_destroyed: int = 0  # resident tokens dropped by evictions; must equal recompute_tokens
    queued: float = 0.0    # total time spent in the waiting queue
    wait_since: float = 0.0
    admit_t: float = 0.0   # time of the most recent admission
    cache_hit_tokens: int = 0   # tokens restored from the prefix cache instead of recomputed
    bids: list = field(default_factory=list)   # block ids held (prefix_cache mode only)
    gen_at_admit: int = 0  # generated count at the most recent admission

    @property
    def context(self):
        return self.prompt + self.generated

    @property
    def prefilling(self):
        return self.computed < self.context


class EngineV2:
    def __init__(self, total_blocks=192, block_size=16, token_budget=512,
                 policy='cap', cap=16, watermark=0.0, resume='recompute',
                 c0=1.0, a=0.004, b=0.01, victim='newest', requeue='front',
                 prefix_cache=False):
        # victim / requeue default to vLLM V1 FCFS behaviour. The alternatives exist
        # only to test the ping-pong mechanism; they are not claims about any engine.
        assert policy in ('cap', 'watermark', 'reserve')
        assert resume in ('recompute', 'restart')
        assert victim in ('newest', 'least_progress')
        assert requeue in ('front', 'back')
        self.victim, self.requeue = victim, requeue
        # prefix_cache: a request's own freed blocks keep their contents until the
        # block is reallocated. Free blocks are reused least-recently-freed first, and
        # a request's blocks are freed tail-first, so its prefix survives longest.
        # This follows vLLM V1's block pool design as I understand it; cross-request
        # prefix sharing (e.g. common system prompts) is NOT modelled.
        assert not (prefix_cache and resume != 'recompute')
        self.prefix_cache = prefix_cache
        from collections import OrderedDict
        self.free_q = OrderedDict((b, None) for b in range(total_blocks))
        self.content = {}     # block id -> (rid, index) last written
        self.where = {}       # (rid, index) -> block id last used
        self.evlog = []   # (t, rid, tokens resident, mid_prefill, time since admission, tokens generated since admission)
        self.total_blocks, self.bs, self.budget0 = total_blocks, block_size, token_budget
        self.policy, self.cap, self.wm, self.resume = policy, cap, watermark, resume
        self.c0, self.a, self.b = c0, a, b
        self.waiting, self.running, self.done = [], [], []
        self.t = 0.0
        self.log = []

    # ------------------------------------------------------------------ blocks
    def nblocks(self, tokens):
        return -(-tokens // self.bs)

    def used(self):
        return sum(self.nblocks(r.alloc) for r in self.running)

    def free(self):
        return self.total_blocks - self.used()

    def _preempt(self, r):
        self.running.remove(r)
        r.evictions += 1
        r.kv_destroyed += r.computed
        self.evlog.append((self.t, r.rid, r.computed, r.prefilling,
                           self.t - r.admit_t, r.generated - r.gen_at_admit))
        if self.resume == 'restart':
            r.discarded_decode += r.generated
            r.generated = 0
        for j, blk in enumerate(r.bids):                    # only full blocks stay cached
            if (j + 1) * self.bs > r.computed:
                self.content[blk] = None
        r.computed = r.alloc = 0
        self._sync(r)
        r.wait_since = self.t
        if self.requeue == 'front':
            self.waiting.insert(0, r)
        else:
            self.waiting.append(r)
        self._step_evictions += 1

    def _reserve_ok(self, cand):
        need = lambda r: self.nblocks(r.prompt + r.max_tokens)
        return sum(need(r) for r in self.running) + need(cand) <= self.total_blocks

    # ------------------------------------------------- block identity (prefix cache)
    def _sync(self, r):
        """Make r's held block ids match r.alloc. No-op unless prefix_cache."""
        if not self.prefix_cache:
            return
        want = self.nblocks(r.alloc)
        while len(r.bids) < want:
            b, _ = self.free_q.popitem(last=False)          # least recently freed
            j = len(r.bids)
            self.content[b] = (r.rid, j)
            self.where[(r.rid, j)] = b
            r.bids.append(b)
        while len(r.bids) > want:
            self.free_q[r.bids.pop()] = None                # tail block freed first

    def _cached_prefix(self, r):
        """Full blocks of r's own earlier KV still intact in the free pool."""
        if not self.prefix_cache:
            return 0
        h = 0
        while (h + 1) * self.bs <= r.context - 1:          # always compute >= 1 token
            b = self.where.get((r.rid, h))
            if b is None or b not in self.free_q or self.content.get(b) != (r.rid, h):
                break
            h += 1
        return h

    # -------------------------------------------------------------------- step
    def step(self):
        budget = self.budget0
        sched = []                       # (request, tokens computed this step)
        self._step_evictions = 0

        # 1. running requests, oldest admission first
        i = 0
        while i < len(self.running) and budget > 0:
            r = self.running[i]
            n = 1 if not r.prefilling else min(r.context - r.computed, budget)
            evicted_self = False
            while self.nblocks(r.computed + n) - self.nblocks(r.alloc) > self.free():
                if self.victim == 'newest':
                    victim = self.running[-1]
                else:   # least resident KV among requests not yet scheduled this step
                    victim = min(self.running[i:], key=lambda q: (q.computed, -self.running.index(q)))
                self._preempt(victim)
                if victim is r:
                    evicted_self = True
                    break
            if evicted_self:
                continue                             # running[i] is now the next request
            r.alloc = r.computed + n
            self._sync(r)
            sched.append((r, n))
            budget -= n
            i += 1

        # 2. admission, FIFO with head-of-line blocking
        while self.waiting and budget > 0:
            r = self.waiting[0]
            if self.policy in ('cap', 'watermark') and len(self.running) >= self.cap:
                break
            h = self._cached_prefix(r)
            hit = h * self.bs
            n = min(r.context - hit, budget)
            need = self.nblocks(hit + n)
            if need > self.free():
                break
            if self.policy == 'watermark' and self.free() - need < self.wm * self.total_blocks:
                break
            if self.policy == 'reserve' and not self._reserve_ok(r):
                break
            self.waiting.pop(0)
            r.queued += self.t - r.wait_since
            if r.first_admit is None:
                r.first_admit = self.t
            if h:
                for j in range(h):                          # reclaim cached blocks in place
                    b = self.where[(r.rid, j)]
                    del self.free_q[b]
                    r.bids.append(b)
                r.computed = hit
                r.cache_hit_tokens += hit
            r.alloc = hit + n
            self._sync(r)
            r.admit_t, r.gen_at_admit = self.t, r.generated
            self.running.append(r)
            sched.append((r, n))
            budget -= n

        assert self.used() <= self.total_blocks, "KV over-committed"
        if self.prefix_cache:
            assert len(self.free_q) == self.free(), "block identity out of sync"

        # 3. execute
        pf = sum(n for r, n in sched if r.prefilling)
        dec = sum(1 for r, n in sched if not r.prefilling)
        dt = self.c0 + self.a * pf + self.b * dec
        self.t += dt
        recompute = 0
        for r, n in sched:
            if r.prefilling:
                rc = max(0, min(r.computed + n, r.high_water) - r.computed)
                recompute += rc
                r.recompute_tokens += rc
                r.computed += n
            else:
                r.computed += 1
                r.generated += 1
                if r.first_token is None:
                    r.first_token = self.t
                if r.generated >= r.output:
                    r.done_at = self.t
                    self.running.remove(r)
                    self.done.append(r)
            r.high_water = max(r.high_water, r.computed)
            r.alloc = r.computed if r in self.running else 0
            self._sync(r)
        self.log.append((self.t, dt, len(self.running), len(self.waiting), self.used(),
                         pf, recompute, dec, self._step_evictions))

    def run(self, reqs, max_steps=2_000_000):
        """Open-loop: requests enter the waiting queue at their arrival time."""
        pending = sorted(reqs, key=lambda r: (r.arrival, r.rid))
        cap_tokens = self.total_blocks * self.bs
        for r in pending:
            assert r.prompt + r.output <= cap_tokens, f"request {r.rid} can never fit"
        k, steps = 0, 0
        while (k < len(pending) or self.waiting or self.running) and steps < max_steps:
            while k < len(pending) and pending[k].arrival <= self.t:
                pending[k].wait_since = pending[k].arrival
                self.waiting.append(pending[k]); k += 1
            if not self.waiting and not self.running:
                self.t = pending[k].arrival          # idle until the next arrival
                continue
            self.step()
            steps += 1
        if steps >= max_steps:
            raise RuntimeError("did not drain: the run is unstable at this load")
        return self


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(e, measure=None):
    """measure: optional set of rids to compute latency stats over (steady state)."""
    reqs = [r for r in e.done if measure is None or r.rid in measure]
    lat = [r.done_at - r.arrival for r in reqs]
    ttft = [r.first_token - r.arrival for r in reqs]
    span = e.t - min(r.arrival for r in e.done)
    total_dt = sum(row[1] for row in e.log)
    busy_prefill = sum(e.a * row[5] for row in e.log)
    recompute_time = sum(e.a * row[6] for row in e.log)
    kv_tw = sum(row[1] * row[4] for row in e.log) / total_dt / e.total_blocks
    useful_out = sum(r.output for r in e.done)
    return {
        'n': len(reqs),
        'makespan': e.t,
        'out_tok_per_u': useful_out / span,
        'req_per_ku': 1000 * len(e.done) / span,
        'lat_p50': pct(lat, 50), 'lat_p95': pct(lat, 95), 'lat_p99': pct(lat, 99),
        'ttft_p50': pct(ttft, 50), 'ttft_p99': pct(ttft, 99),
        'evictions': sum(r.evictions for r in e.done),
        'max_evictions_one_req': max(r.evictions for r in e.done),
        'recompute_tokens': sum(r.recompute_tokens for r in e.done),
        'discarded_decode': sum(r.discarded_decode for r in e.done),
        'recompute_time_share': recompute_time / total_dt,
        'prefill_time_share': busy_prefill / total_dt,
        'kv_util_tw': kv_tw,
    }
