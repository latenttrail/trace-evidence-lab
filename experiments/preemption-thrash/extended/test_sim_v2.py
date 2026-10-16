"""Invariant tests for sim_v2. Run: python3 test_sim_v2.py"""
import random
from sim_v2 import EngineV2, Req, summarize


def burst(n=24, prompt=256, output=200, gap=4.0, max_tokens=200):
    return [Req(i, i * gap, prompt, output, max_tokens) for i in range(n)]


def mixed(seed, n=120):
    rng = random.Random(seed)
    out, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(1 / 30)
        p = min(1024, max(16, int(rng.lognormvariate(5.3, 0.6))))
        o = min(1000, max(8, int(rng.lognormvariate(4.8, 0.9))))
        out.append(Req(i, t, p, o, 1000))
    return out


def test_single_request_timing():
    e = EngineV2(a=0, b=0).run([Req(0, 0.0, 256, 200, 200)])
    r = e.done[0]
    # one step prefills the whole prompt (budget 512), then 200 decode steps
    assert r.done_at == 201, r.done_at
    assert r.evictions == 0 and r.recompute_tokens == 0


def test_never_overcommits_and_conserves_tokens():
    for policy, kw in (('cap', {'cap': 32}), ('watermark', {'cap': 32, 'watermark': 0.1}),
                       ('reserve', {})):
        for resume in ('recompute', 'restart'):
            for seed in range(5):
                reqs = mixed(seed)
                e = EngineV2(policy=policy, resume=resume, **kw).run(reqs)  # asserts capacity each step
                assert len(e.done) == len(reqs)
                assert all(r.generated == r.output for r in e.done)
                dec_steps = sum(row[7] for row in e.log)
                expect = sum(r.output for r in e.done) + sum(r.discarded_decode for r in e.done)
                assert dec_steps == expect, (policy, resume, seed, dec_steps, expect)
                if resume == 'recompute':
                    assert sum(r.discarded_decode for r in e.done) == 0


def test_reserve_never_preempts_when_outputs_respect_max_tokens():
    for seed in range(5):
        e = EngineV2(policy='reserve').run(mixed(seed))
        assert sum(r.evictions for r in e.done) == 0


def test_recompute_accounting():
    # every token of KV an eviction destroys is rebuilt exactly once, and nothing else counts
    for resume in ('recompute', 'restart'):
        for reqs in (burst(), mixed(0), mixed(1)):
            e = EngineV2(policy='cap', cap=32, resume=resume).run(reqs)
            assert sum(r.evictions for r in e.done) > 0
            for r in e.done:
                assert r.recompute_tokens == r.kv_destroyed if resume == 'recompute' else True
                assert r.recompute_tokens <= r.kv_destroyed


def test_prefix_cache_accounting():
    # with a prefix cache, destroyed KV is either recomputed or restored from cache, never both
    for reqs in (burst(), mixed(0), mixed(1)):
        e = EngineV2(policy='cap', cap=32, prefix_cache=True).run(reqs)   # asserts id/count sync each step
        assert sum(r.cache_hit_tokens for r in e.done) > 0
        for r in e.done:
            assert r.recompute_tokens + r.cache_hit_tokens == r.kv_destroyed
            assert r.cache_hit_tokens % 16 == 0


def test_prefix_cache_off_is_unchanged():
    a = EngineV2(policy='cap', cap=32).run(mixed(3))
    b = EngineV2(policy='cap', cap=32, prefix_cache=False).run(mixed(3))
    assert a.log == b.log


def test_zero_cost_matches_constant_step():
    e = EngineV2(a=0, b=0, cap=8).run(burst())
    assert all(row[1] == 1.0 for row in e.log)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok ', name)
