"""
Seeded workload generators for sim_v2. Every workload is a list of Req and is
fully determined by (name, seed, parameters).

  burst24   the original experiment: 24 requests, 256-token prompts, 200-token
            outputs, one arrival every 4 time units. No randomness.
  homog     Poisson arrivals, every request 256 in / 200 out. Isolates the effect
            of arrival timing alone.
  chat      Poisson arrivals, lognormal prompts (median ~200 tokens) and lognormal
            outputs (median ~120, heavy right tail), clipped so every request can
            fit the cache alone. max_tokens = 1000 for every request, which is
            what the 'reserve' policy budgets for. The shape is ASSUMED, chosen to
            resemble chat traffic; it is not fitted to any trace.
  bimodal   Poisson arrivals, 80% short answers (out ~ 40) and 20% long ones
            (out ~ 700). Tests whether one static cap can suit both.

The cache in every experiment is 192 blocks of 16 tokens (3,072 tokens), as in
the original. That is deliberately tiny next to a real GPU; what transfers is
ratios (request footprint / cache), not absolute counts.
"""
import random
from sim_v2 import Req


def burst24():
    return [Req(i, i * 4.0, 256, 200, 200) for i in range(24)]


def _poisson(rng, n, rate):
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def homog(seed, n=400, rate=0.05):
    rng = random.Random(seed)
    return [Req(i, t, 256, 200, 200) for i, t in enumerate(_poisson(rng, n, rate))]


def chat(seed, n=400, rate=0.05, max_tokens=1000):
    rng = random.Random(seed)
    reqs = []
    for i, t in enumerate(_poisson(rng, n, rate)):
        p = min(1024, max(16, int(rng.lognormvariate(5.3, 0.6))))     # median ~200
        o = min(max_tokens, max(8, int(rng.lognormvariate(4.8, 0.9))))  # median ~120
        reqs.append(Req(i, t, p, o, max_tokens))
    return reqs


def bimodal(seed, n=400, rate=0.05, max_tokens=1000):
    rng = random.Random(seed)
    reqs = []
    for i, t in enumerate(_poisson(rng, n, rate)):
        p = min(1024, max(16, int(rng.lognormvariate(5.3, 0.6))))
        if rng.random() < 0.8:
            o = max(8, int(rng.gauss(40, 10)))
        else:
            o = min(max_tokens, max(200, int(rng.gauss(700, 120))))
        reqs.append(Req(i, t, p, o, max_tokens))
    return reqs


WORKLOADS = {'homog': homog, 'chat': chat, 'bimodal': bimodal}
