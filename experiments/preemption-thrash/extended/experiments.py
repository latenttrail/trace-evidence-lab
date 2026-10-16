"""
Extended preemption-thrash experiments on sim_v2. Simulation only: every number
is in abstract time units (u), never milliseconds, and describes this simulator.

Run everything:      python3 experiments.py all
Run one experiment:  python3 experiments.py e3_admission
Each experiment writes results/<name>.json (one row per run) and results/<name>.txt.

Shared conditions unless an experiment says otherwise
-----------------------------------------------------
cache 192 blocks x 16 tokens ("x1") or 768 blocks ("x4", with arrival rate and
caps scaled by 4); token budget 512 per step; step cost dt = 1 + 0.004*prefill
tokens + 0.01*decoding sequences; vLLM-V1-like scheduling (newest running
request is the victim, victims go to the front of the queue, generated tokens
survive preemption and their KV is rebuilt by prefill); seeds 0-4.
"""
import json, sys, math, random, statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sim_v2 import EngineV2, Req, summarize, pct
from workloads import WORKLOADS, burst24

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
SEEDS = range(5)


# ---------------------------------------------------------------------------- one run
def make_reqs(spec):
    if spec['wl'] == 'burst24':
        return burst24()
    if spec['wl'] == 'chat_profile':
        return chat_profile(spec['seed'], spec['phases'], spec['scale'])
    s = spec['scale']
    return WORKLOADS[spec['wl']](spec['seed'], n=spec['n'] * s, rate=spec['rate'] * s)


def chat_profile(seed, phases, scale, max_tokens=1000):
    """Piecewise-Poisson arrivals with the `chat` length model."""
    rng = random.Random(seed); reqs = []; t0 = 0.0; i = 0
    for dur, rate in phases:
        t = t0
        while True:
            t += rng.expovariate(rate * scale)
            if t >= t0 + dur:
                break
            p = min(1024, max(16, int(rng.lognormvariate(5.3, 0.6))))
            o = min(max_tokens, max(8, int(rng.lognormvariate(4.8, 0.9))))
            reqs.append(Req(i, t, p, o, max_tokens)); i += 1
        t0 += dur
    return reqs


def run_one(spec):
    """spec: workload fields + engine kwargs under 'eng'. Returns one result row."""
    reqs = make_reqs(spec)
    s = spec.get('scale', 1)
    eng = dict(spec['eng'])
    eng.setdefault('total_blocks', 192 * s)
    e = EngineV2(**eng).run(reqs)
    meas = None
    if 'measure' in spec:
        lo, hi = spec['measure']
        meas = {r.rid for r in sorted(reqs, key=lambda r: r.rid)[lo * s:hi * s]}
    row = dict(spec)
    row.update(summarize(e, meas))
    n = len(e.done)
    ev = sorted((r.evictions for r in e.done), reverse=True)
    tot = sum(ev)
    row['evictions_per_req'] = tot / n
    row['top1pct_eviction_share'] = (sum(ev[:max(1, n // 100)]) / tot) if tot else 0.0
    row['zero_progress_share'] = (sum(1 for x in e.evlog if x[5] == 0) / len(e.evlog)) if e.evlog else 0.0
    row['mid_prefill_share'] = (sum(1 for x in e.evlog if x[3]) / len(e.evlog)) if e.evlog else 0.0
    destroyed = sum(r.kv_destroyed for r in e.done)
    row['kv_restored_from_cache'] = (sum(r.cache_hit_tokens for r in e.done) / destroyed) if destroyed else 0.0
    # saturation throughput: decode tokens per u over steps with a non-empty queue
    back = [x for x in e.log if x[3] > 0]
    bt = sum(x[1] for x in back)
    row['backlog_tok_per_u'] = sum(x[7] for x in back) / bt if bt else float('nan')
    row['backlog_recompute_share'] = sum(e.a * x[6] for x in back) / bt if bt else float('nan')
    row['backlog_kv_util'] = sum(x[1] * x[4] for x in back) / bt / e.total_blocks if bt else float('nan')
    if spec.get('series'):
        row['series'] = [(round(x[0], 1), x[3], x[8]) for x in e.log[::25]]
    return row


def run_all(specs):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(run_one, specs, chunksize=1))


def save(name, rows, text):
    OUT.mkdir(exist_ok=True)
    (OUT / f'{name}.json').write_text(json.dumps(rows, indent=0, default=str))
    (OUT / f'{name}.txt').write_text(text)
    print(text)


def mean(rows, k):
    return st.mean(r[k] for r in rows)


def sd(rows, k):
    return st.stdev(r[k] for r in rows) if len(rows) > 1 else 0.0


def group(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] if not isinstance(r[k], dict) else json.dumps(r[k], sort_keys=True)
                             for k in keys), []).append(r)
    return out


# --------------------------------------------------------------------------- experiments
def e1_semantics():
    """Original workload, run to completion: does the plateau survive the fixes,
    a time model, and vLLM-like resume semantics?"""
    cfgs = [('original semantics, steps free', dict(resume='restart', a=0, b=0)),
            ('vLLM-like resume, steps free', dict(resume='recompute', a=0, b=0)),
            ('original semantics, timed', dict(resume='restart')),
            ('vLLM-like resume, timed', dict(resume='recompute'))]
    caps = (1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24)
    specs = [dict(wl='burst24', cfg=name, cap=c, eng=dict(policy='cap', cap=c, **kw))
             for name, kw in cfgs for c in caps]
    rows = run_all(specs)
    lines = ["E1  burst24 (24 requests, 256 in / 200 out, one every 4u), run until all finish",
             "    throughput = useful output tokens per u over the whole run", ""]
    for name, _ in cfgs:
        lines.append(f"  {name}")
        lines.append("    cap | makespan | out tok/u | evictions | KV rebuilt | decode redone | p50 lat | p99 lat")
        for r in [r for r in rows if r['cfg'] == name]:
            lines.append(f"    {r['cap']:>3} | {r['makespan']:>8.1f} | {r['out_tok_per_u']:>9.2f} | {r['evictions']:>9} | "
                         f"{r['recompute_tokens']:>10,} | {r['discarded_decode']:>13,} | {r['lat_p50']:>7.1f} | {r['lat_p99']:>7.1f}")
        lines.append("")
    save('e1_semantics', rows, "\n".join(lines))


def e2_cost_sweep():
    """At saturation, does admitting past 'fits at completion' pay, as a function of
    what one prefill token costs relative to a decode step?"""
    A = (0.0, 0.002, 0.004, 0.01, 0.02, 0.04)
    specs = []
    for wl in ('homog', 'chat'):
        for a in A:
            for resume in ('recompute', 'restart'):
                for cap in (6, 8, 12, 48):
                    for seed in SEEDS:
                        specs.append(dict(wl=wl, seed=seed, n=300, rate=0.2, scale=1, a=a, resume=resume, cap=cap,
                                          eng=dict(policy='cap', cap=cap, a=a, resume=resume)))
    rows = run_all(specs)
    g = group(rows, 'wl', 'a', 'resume', 'cap')
    lines = ["E2  saturation (arrivals 0.2/u, 300 requests, x1 cache): useful output tok/u, mean of 5 seeds",
             "    cap 6 = 'fits at completion' sizing for 256/200 requests; cap 48 = effectively unlimited", ""]
    for wl in ('homog', 'chat'):
        lines.append(f"  {wl}")
        lines.append("        a |  vLLM-like resume: cap6  cap8 cap12 cap48 ratio48/6 |  original semantics: cap6 cap48 ratio")
        for a in A:
            v = {(res, c): mean(g[(wl, a, res, c)], 'out_tok_per_u') for res in ('recompute', 'restart') for c in (6, 8, 12, 48)}
            lines.append(f"    {a:5.3f} |  {v[('recompute',6)]:5.2f} {v[('recompute',8)]:5.2f} {v[('recompute',12)]:5.2f} "
                         f"{v[('recompute',48)]:5.2f}    {v[('recompute',48)]/v[('recompute',6)]:5.2f}  |  "
                         f"{v[('restart',6)]:5.2f} {v[('restart',48)]:5.2f} {v[('restart',48)]/v[('restart',6)]:5.2f}")
        lines.append("")
    save('e2_cost_sweep', rows, "\n".join(lines))


ADMISSION = ([('wm', w) for w in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30)] +
             [('cap', c) for c in (6, 8, 12)] + [('reserve', None)])


def adm_eng(kind, v, scale, prefix):
    if kind == 'wm':
        return dict(policy='watermark', cap=48 * scale, watermark=v, prefix_cache=prefix)
    if kind == 'cap':
        return dict(policy='cap', cap=v * scale, prefix_cache=prefix)
    return dict(policy='reserve', prefix_cache=prefix)


def adm_label(kind, v):
    return {'wm': f"headroom {int(round((v or 0) * 100))}%", 'cap': f"cap {v} (x scale)", 'reserve': 'reserve max_tokens'}[kind]


def e3_admission():
    """Saturation throughput across every admission knob, both cache scales,
    prefix cache off and on."""
    specs = []
    for scale in (1, 4):
        for prefix in (False, True):
            for wl in ('homog', 'chat', 'bimodal'):
                for kind, v in ADMISSION:
                    for seed in SEEDS:
                        specs.append(dict(wl=wl, seed=seed, n=300, rate=0.2, scale=scale, prefix=prefix,
                                          adm=adm_label(kind, v), eng=adm_eng(kind, v, scale, prefix)))
    rows = run_all(specs)
    g = group(rows, 'scale', 'prefix', 'wl', 'adm')
    lines = ["E3  saturation throughput by admission policy (arrivals 0.2/u x scale; decode tok/u while backlogged)",
             "    mean ± sd over 5 seeds; 'paired vs 0%' = mean of per-seed ratios to zero headroom", ""]
    for scale in (1, 4):
        for prefix in (False, True):
            lines.append(f"  cache x{scale}, prefix cache {'ON' if prefix else 'off'}")
            for wl in ('homog', 'chat', 'bimodal'):
                base = {r['seed']: r['backlog_tok_per_u'] for r in g[(scale, prefix, wl, 'headroom 0%')]}
                lines.append(f"    {wl}")
                for kind, v in ADMISSION:
                    rs = g[(scale, prefix, wl, adm_label(kind, v))]
                    paired = st.mean(r['backlog_tok_per_u'] / base[r['seed']] - 1 for r in rs)
                    lo = min(r['backlog_tok_per_u'] / base[r['seed']] - 1 for r in rs)
                    hi = max(r['backlog_tok_per_u'] / base[r['seed']] - 1 for r in rs)
                    lines.append(f"      {adm_label(kind, v):<20} {mean(rs,'backlog_tok_per_u'):6.2f} ± {sd(rs,'backlog_tok_per_u'):4.2f}"
                                 f"  paired vs 0% {paired:+6.1%} [{lo:+.1%}, {hi:+.1%}]"
                                 f"  | evictions/req {mean(rs,'evictions_per_req'):5.2f}  recompute {mean(rs,'backlog_recompute_share'):5.1%}"
                                 f"  KV usage {mean(rs,'backlog_kv_util'):.2f}  zero-progress evictions {mean(rs,'zero_progress_share'):4.0%}")
            lines.append("")
    save('e3_admission', rows, "\n".join(lines))


def e4_latency():
    """Latency vs offered load, x4 cache, prefix cache on: zero vs 5% headroom."""
    specs = []
    for wl in ('homog', 'chat'):
        for rate in (0.016, 0.020, 0.024, 0.026, 0.028):
            for w in (0.0, 0.05):
                for seed in SEEDS:
                    specs.append(dict(wl=wl, seed=seed, n=600, rate=rate, scale=4, wm=w, measure=(100, 500),
                                      eng=dict(policy='watermark', cap=192, watermark=w, prefix_cache=True)))
    rows = run_all(specs)
    g = group(rows, 'wl', 'rate', 'wm')
    lines = ["E4  latency by offered load; x4 cache; prefix cache ON; requests 400-1999 measured; 5 seeds",
             "    ratio = mean of per-seed (5% headroom / 0% headroom)", ""]
    for wl in ('homog', 'chat'):
        lines.append(f"  {wl}")
        for rate in (0.016, 0.020, 0.024, 0.026, 0.028):
            a0, a5 = g[(wl, rate, 0.0)], g[(wl, rate, 0.05)]
            pr = lambda k: [y[k] / x[k] for x, y in zip(sorted(a0, key=lambda r: r['seed']), sorted(a5, key=lambda r: r['seed']))]
            lines.append(f"    rate {rate:.3f}x4: p50 {mean(a0,'lat_p50'):6.0f} -> {mean(a5,'lat_p50'):6.0f} (x{st.mean(pr('lat_p50')):.2f})"
                         f"  p99 {mean(a0,'lat_p99'):6.0f} -> {mean(a5,'lat_p99'):6.0f} (x{st.mean(pr('lat_p99')):.2f})"
                         f"  evictions/req {mean(a0,'evictions_per_req'):.2f} -> {mean(a5,'evictions_per_req'):.2f}"
                         f"  KV usage {mean(a0,'kv_util_tw'):.2f} -> {mean(a5,'kv_util_tw'):.2f}")
        lines.append("")
    save('e4_latency', rows, "\n".join(lines))


def e4b_cap_remedy():
    """Latency of the remedy 'lower the concurrency cap until preemptions stop'.
    x4 cache, prefix cache on. cap 24 = 6 x 4, the original post's 'fits at
    completion' sizing for 256/200 requests; it all but eliminates preemption
    on both workloads in E3."""
    pols = [('no cap, no headroom', dict(policy='cap', cap=192)),
            ('cap 32', dict(policy='cap', cap=32)),
            ('cap 24 (fits at completion)', dict(policy='cap', cap=24)),
            ('no cap, 5% headroom', dict(policy='watermark', cap=192, watermark=0.05))]
    rates = (0.016, 0.020, 0.024, 0.026, 0.028)
    specs = []
    for wl in ('homog', 'chat'):
        for rate in rates:
            for name, kw in pols:
                for seed in SEEDS:
                    specs.append(dict(wl=wl, seed=seed, n=600, rate=rate, scale=4, pol=name, measure=(100, 500),
                                      eng=dict(prefix_cache=True, **kw)))
    rows = run_all(specs)
    g = group(rows, 'wl', 'rate', 'pol')
    lines = ["E4b latency of lowering the cap to stop preemption; x4 cache; prefix cache ON; requests 400-1999; 5 seeds", ""]
    for wl in ('homog', 'chat'):
        lines.append(f"  {wl}")
        for rate in rates:
            lines.append(f"    rate {rate:.3f}x4")
            for name, _ in pols:
                rs = g[(wl, rate, name)]
                lines.append(f"      {name:<28} p50 {mean(rs,'lat_p50'):6.0f}  p99 {mean(rs,'lat_p99'):6.0f}  TTFT p99 {mean(rs,'ttft_p99'):6.0f}"
                             f"  preemptions/req {mean(rs,'evictions_per_req'):.2f}  KV usage {mean(rs,'kv_util_tw'):.2f}")
        lines.append("")
    save('e4b_cap_remedy', rows, "\n".join(lines))


def e5_anatomy():
    """What an eviction is: progress since admission, concentration, and whether
    victim/requeue choices are the cause."""
    variants = [('vLLM-like (newest victim, requeue front)', {}),
                ('requeue to back', dict(requeue='back')),
                ('victim = least progress', dict(victim='least_progress')),
                ('least progress + requeue back', dict(victim='least_progress', requeue='back'))]
    specs = []
    for scale in (1, 4):
        for rate in (0.015, 0.022, 0.026):
            for name, kw in variants:
                for seed in SEEDS:
                    specs.append(dict(wl='chat', seed=seed, n=600, rate=rate, scale=scale, variant=name, measure=(100, 500),
                                      eng=dict(policy='cap', cap=48 * scale, **kw)))
    rows = run_all(specs)
    g = group(rows, 'scale', 'rate', 'variant')
    lines = ["E5  anatomy of an eviction; chat workload; optimistic admission (no headroom); 5 seeds", ""]
    for scale in (1, 4):
        for rate in (0.015, 0.022, 0.026):
            lines.append(f"  cache x{scale}, rate {rate} x{scale}")
            for name, _ in variants:
                rs = g[(scale, rate, name)]
                lines.append(f"    {name:<42} evictions/req {mean(rs,'evictions_per_req'):5.2f} | zero-progress {mean(rs,'zero_progress_share'):4.0%}"
                             f" | mid-prefill {mean(rs,'mid_prefill_share'):4.0%} | top-1% of requests hold {mean(rs,'top1pct_eviction_share'):4.0%}"
                             f" | worst request {max(r['max_evictions_one_req'] for r in rs):3d}x | recompute {mean(rs,'recompute_time_share'):4.1%}"
                             f" | p50 {mean(rs,'lat_p50'):5.0f} p99 {mean(rs,'lat_p99'):5.0f}")
            lines.append("")
    save('e5_anatomy', rows, "\n".join(lines))


def e6_recovery():
    """Recovery after a 1.5x burst at a base load near capacity."""
    base, specs = 0.024, []
    profiles = {'no burst': [(40000, base)], 'burst': [(3000, base), (1500, base * 1.5), (35500, base)]}
    for w in (0.0, 0.05):
        for label, ph in profiles.items():
            for seed in SEEDS:
                specs.append(dict(wl='chat_profile', seed=seed, scale=4, phases=ph, profile=label, wm=w, series=True,
                                  eng=dict(policy='watermark', cap=192, watermark=w, prefix_cache=True)))
    rows = run_all(specs)

    def qwin(r, a, b):
        pts = [p for p in r['series'] if a <= p[0] < b]
        return st.mean(p[1] for p in pts) if pts else float('nan')
    g = group(rows, 'wm', 'profile')
    lines = [f"E6  recovery after a burst; chat; x4 cache; prefix cache ON; base {base}x4, burst 1.5x for t=3000-4500",
             "    mean waiting-queue length per window (sampled every 25 steps), 5 seeds", ""]
    for w in (0.0, 0.05):
        for label in profiles:
            rs = g[(w, label)]
            cols = "  ".join(f"t{a//1000}-{b//1000}k {st.mean(qwin(r, a, b) for r in rs):5.1f}"
                             for a, b in ((0, 3000), (3000, 4500), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 30000), (30000, 40000)))
            lines.append(f"  headroom {int(w*100)}%, {label:<8}  {cols}   | backlogged decode tok/u {mean(rs,'backlog_tok_per_u'):.2f}")
        lines.append("")
    lines.append("  offered decode load at the base rate is ~17.1 tok/u (mean chat output ~178 tokens x 0.096 req/u)")
    save('e6_recovery', rows, "\n".join(lines))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')


def e7_metrics():
    """Which observable tracks lost throughput? Uses E3's runs (no new simulation).
    For each (scale, prefix, workload) the best admission policy defines 100%;
    every other policy's shortfall is compared with three observables."""
    rows = json.loads((OUT / 'e3_admission.json').read_text())
    g = group(rows, 'scale', 'prefix', 'wl', 'adm')
    pts = []
    for (scale, prefix, wl, adm), rs in g.items():
        pts.append(dict(scale=scale, prefix=prefix, wl=wl, adm=adm,
                        thr=mean(rs, 'backlog_tok_per_u'), ev=mean(rs, 'evictions_per_req'),
                        rc=mean(rs, 'backlog_recompute_share'), kv=mean(rs, 'backlog_kv_util')))
    best = {}
    for p in pts:
        k = (p['scale'], p['prefix'], p['wl'])
        best[k] = max(best.get(k, 0), p['thr'])
    for p in pts:
        p['shortfall'] = 1 - p['thr'] / best[(p['scale'], p['prefix'], p['wl'])]
    lines = ["E7  which observable tracks lost throughput? (from E3; 10 admission policies x 3 workloads x 2 scales x 2 prefix modes)",
             "    shortfall = 1 - throughput / best policy's throughput for the same workload, scale and prefix mode",
             "    Spearman rank correlation with shortfall (+1 = higher metric always means more throughput lost)", ""]
    sets = {'all policies': pts,
            'optimistic policies only (headroom 0-30%, no static cap, no reserve)': [p for p in pts if p['adm'].startswith('headroom')]}
    for name, ps in sets.items():
        sh = [p['shortfall'] for p in ps]
        lines.append(f"  {name} (n={len(ps)})")
        for key, label in (('ev', 'preemptions per request   (vllm:num_preemptions / requests)'),
                           ('kv', 'KV cache usage            (vllm:kv_cache_usage_perc)'),
                           ('rc', 'share of time recomputing evicted KV  (no direct vLLM metric)')):
            lines.append(f"    {label:<64} rho = {spearman([p[key] for p in ps], sh):+.2f}")
        lines.append("")
    # concrete counterexamples
    lines.append("  Counterexamples (same workload, scale, prefix mode):")
    for k in sorted({(p['scale'], p['prefix'], p['wl']) for p in pts}):
        grp = [p for p in pts if (p['scale'], p['prefix'], p['wl']) == k]
        top = max(grp, key=lambda p: p['thr'])
        calm = min(grp, key=lambda p: (p['ev'], -p['thr']))
        lines.append(f"    x{k[0]} prefix {'on ' if k[1] else 'off'} {k[2]:<8} best: {top['adm']:<20} {top['ev']:5.2f} preemptions/req, KV {top['kv']:.2f} | "
                     f"fewest preemptions: {calm['adm']:<20} {calm['ev']:5.2f}/req, KV {calm['kv']:.2f}, {calm['shortfall']:5.1%} below best")
    (OUT / 'e7_points.json').write_text(json.dumps(pts, indent=0))
    save('e7_metrics', pts, "\n".join(lines))


def headline():
    """Per-seed paired ratios behind the headline (reads E4b; no new simulation)."""
    rows = json.loads((OUT / 'e4b_cap_remedy.json').read_text())
    idx = {(r['wl'], r['rate'], r['pol'], r['seed']): r for r in rows}
    cap, free = 'cap 24 (fits at completion)', 'no cap, no headroom'
    pols = (free, 'cap 32', cap, 'no cap, 5% headroom')
    lines = ["HEADLINE: per-seed paired ratios, 'cap 24 (fits at completion)' / 'no cap, no headroom'",
             "x4 cache, prefix cache ON, 5 seeds (0-4), requests 400-1999 of 2400 measured", ""]
    for wl in ('homog', 'chat'):
        for rate in (0.020, 0.024, 0.026, 0.028):
            out = []
            for k in ('lat_p50', 'lat_p99'):
                v = [idx[(wl, rate, cap, s)][k] / idx[(wl, rate, free, s)][k] for s in SEEDS]
                out.append(f"{k[-3:]} x{st.mean(v):.2f} [{min(v):.2f}-{max(v):.2f}]")
            pc = st.mean(idx[(wl, rate, cap, s)]['evictions_per_req'] for s in SEEDS)
            pf = st.mean(idx[(wl, rate, free, s)]['evictions_per_req'] for s in SEEDS)
            kc = st.mean(idx[(wl, rate, cap, s)]['kv_util_tw'] for s in SEEDS)
            kf = st.mean(idx[(wl, rate, free, s)]['kv_util_tw'] for s in SEEDS)
            lines.append(f"  {wl:<6} rate {rate:.3f}x4: {'   '.join(out)}   preemptions/req {pc:.2f} vs {pf:.2f}   KV usage {kc:.2f} vs {kf:.2f}")
    lines += ["", "cap 24 median latency vs the best of the four policies at each load:"]
    for wl in ('homog', 'chat'):
        for rate in (0.024, 0.026, 0.028):
            m = {p: st.mean(idx[(wl, rate, p, s)]['lat_p50'] for s in SEEDS) for p in pols}
            best = min(m, key=m.get)
            lines.append(f"  {wl:<6} rate {rate:.3f}x4: best = {best:<22} p50 {m[best]:6.0f};  cap 24 p50 {m[cap]:6.0f}  (x{m[cap]/m[best]:.1f})")
    (OUT / 'headline.txt').write_text("\n".join(lines))
    print("\n".join(lines))


EXPERIMENTS = {f.__name__: f for f in (e1_semantics, e2_cost_sweep, e3_admission, e4_latency, e4b_cap_remedy,
                                        e5_anatomy, e6_recovery, e7_metrics, headline)}

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in EXPERIMENTS.items():
        if which in ('all', name):
            print(f"\n######## {name}")
            fn()
