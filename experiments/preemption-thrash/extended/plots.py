"""
SVG plots for the extended experiments, from results/*.json. Standard library only.

Run after experiments.py:  python3 plots.py
Writes plots/*.svg. Every point carries a <title> tooltip with its exact value;
the full numbers are in results/*.txt.

Palette: reference categorical slots 1-3 (blue, orange, aqua), validated
all-pairs for colour-vision deficiency. Aqua is below 3:1 contrast on the
surface, so every series is also direct-labelled and has its own marker shape.
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / 'results', HERE / 'plots'
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
MARKERS = ['circle', 'square', 'triangle']
FONT = "font-family=\"-apple-system,'Segoe UI',Helvetica,Arial,sans-serif\""


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def marker(kind, x, y, color, tip):
    t = f'<title>{esc(tip)}</title>'
    ring = f'stroke="{SURFACE}" stroke-width="2"'
    if kind == 'circle':
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" {ring}>{t}</circle>'
    if kind == 'square':
        return f'<rect x="{x-4.2:.1f}" y="{y-4.2:.1f}" width="8.4" height="8.4" rx="1.5" fill="{color}" {ring}>{t}</rect>'
    return (f'<polygon points="{x:.1f},{y-5.5:.1f} {x+5:.1f},{y+4:.1f} {x-5:.1f},{y+4:.1f}" '
            f'fill="{color}" {ring}>{t}</polygon>')


def nice_ticks(lo, hi, n=5):
    span = hi - lo or 1
    raw = span / n
    mag = 10 ** __import__('math').floor(__import__('math').log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = __import__('math').floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def panel(x0, y0, w, h, series, title, xlabel, ylabel, xlim=None, ylim=None, fmt_x=str, fmt_y=None,
          refs=(), lines=True, label_pos='end', R=18):
    """series: [(name, [(x, y, tip)], color_index)]. refs: [('h'|'v', value, label)]."""
    fmt_y = fmt_y or (lambda v: f"{v:g}")
    xs = [p[0] for _, pts, _ in series for p in pts]
    ys = [p[1] for _, pts, _ in series for p in pts]
    xlo, xhi = xlim or (min(xs), max(xs))
    ylo, yhi = ylim or (min(ys), max(ys))
    L, T, B = 64, 34, 46
    pw, ph = w - L - R, h - T - B
    X = lambda v: x0 + L + (v - xlo) / ((xhi - xlo) or 1) * pw
    Y = lambda v: y0 + T + ph - (v - ylo) / ((yhi - ylo) or 1) * ph
    o = [f'<text x="{x0+L}" y="{y0+18}" {FONT} font-size="14" font-weight="600" fill="{INK}">{esc(title)}</text>']
    for t in nice_ticks(ylo, yhi):
        o.append(f'<line x1="{x0+L}" x2="{x0+L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{x0+L-8}" y="{Y(t)+4:.1f}" {FONT} font-size="11" fill="{MUTED}" text-anchor="end">{esc(fmt_y(t))}</text>')
    for t in nice_ticks(xlo, xhi):
        o.append(f'<text x="{X(t):.1f}" y="{y0+T+ph+18}" {FONT} font-size="11" fill="{MUTED}" text-anchor="middle">{esc(fmt_x(t))}</text>')
    o.append(f'<line x1="{x0+L}" x2="{x0+L+pw}" y1="{y0+T+ph}" y2="{y0+T+ph}" stroke="{MUTED}" stroke-width="1"/>')
    o.append(f'<text x="{x0+L+pw/2:.1f}" y="{y0+h-6}" {FONT} font-size="12" fill="{INK2}" text-anchor="middle">{esc(xlabel)}</text>')
    o.append(f'<text transform="translate({x0+16},{y0+T+ph/2:.1f}) rotate(-90)" {FONT} font-size="12" fill="{INK2}" text-anchor="middle">{esc(ylabel)}</text>')
    for kind, v, lab in refs:
        if kind == 'h':
            o.append(f'<line x1="{x0+L}" x2="{x0+L+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 4"/>')
            o.append(f'<text x="{x0+L+pw-4}" y="{Y(v)-6:.1f}" {FONT} font-size="11" fill="{INK2}" text-anchor="end">{esc(lab)}</text>')
        else:
            o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y0+T}" y2="{y0+T+ph}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 4"/>')
            o.append(f'<text x="{X(v)+5:.1f}" y="{y0+T+12}" {FONT} font-size="11" fill="{INK2}">{esc(lab)}</text>')
    placed = []
    for name, pts, ci in series:
        c = SERIES[ci]
        pts = sorted(pts)
        if lines and len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(p[0]):.1f},{Y(p[1]):.1f}" for i, p in enumerate(pts))
            o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for p in pts:
            o.append(marker(MARKERS[ci], X(p[0]), Y(p[1]), c, p[2]))
        if label_pos and pts:
            px, py = (pts[-1] if label_pos == "end" else pts[0])[:2]
            ly = Y(py)
            while any(abs(ly - q) < 14 for q in placed):
                ly += 14
            placed.append(ly)
            anchor = 'start' if label_pos == 'end' else 'end'
            dx = 8 if label_pos == 'end' else -8
            o.append(f'<text x="{X(px)+dx:.1f}" y="{ly+4:.1f}" {FONT} font-size="12" fill="{INK}" text-anchor="{anchor}">{esc(name)}</text>')
    return "\n".join(o)


def legend(x, y, items):
    o, cx = [], x
    for name, ci in items:
        o.append(marker(MARKERS[ci], cx + 5, y - 4, SERIES[ci], name))
        o.append(f'<text x="{cx+16}" y="{y}" {FONT} font-size="12" fill="{INK2}">{esc(name)}</text>')
        cx += 26 + 7 * len(name)
    return "\n".join(o)


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">'
            f'<title>{esc(title)}</title><desc>{esc(desc)}</desc>'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>{body}</svg>')


def load(name):
    return json.loads((RES / f'{name}.json').read_text())


def mean_by(rows, keyf, val):
    g = {}
    for r in rows:
        g.setdefault(keyf(r), []).append(r[val])
    return {k: st.mean(v) for k, v in g.items()}


# ----------------------------------------------------------------------------- figures
def fig1_semantics():
    rows = load('e1_semantics')
    ser = []
    for i, (cfg, lab) in ((1, ('original semantics, timed', 'original semantics')),
                          (0, ('vLLM-like resume, timed', 'vLLM-like resume'))):
        pts = [(r['cap'], r['out_tok_per_u'], f"{lab}, cap {r['cap']}: {r['out_tok_per_u']:.2f} tok/u, {r['evictions']} evictions")
               for r in rows if r['cfg'] == cfg]
        ser.append((lab, pts, i))
    body = panel(0, 40, 760, 380, ser, "Same 24 requests, same cache: does admitting more help?",
                 "admission cap (max running requests)", "output tokens per time unit",
                 xlim=(0, 24), ylim=(0, 7), refs=[('v', 6, 'cap 6 = all fit at completion')], R=150)
    body += legend(64, 26, [('vLLM-like resume', 0), ('original semantics', 1)])
    (OUT / 'fig1_semantics.svg').write_text(svg(860, 430, body, 'Eviction semantics decide the sign',
        'Throughput versus admission cap for the original burst workload, under the original simulator semantics and vLLM-like resume semantics, both with step costs.'))


def fig2_cost():
    rows = load('e2_cost_sweep')
    m = mean_by(rows, lambda r: (r['wl'], r['a'], r['resume'], r['cap']), 'out_tok_per_u')
    body, x0 = '', 0
    for wl, title in (('homog', 'Uniform requests (256 in / 200 out)'), ('chat', 'Chat-like mix')):
        ser = []
        for i, (res, lab) in enumerate((('recompute', 'vLLM-like resume'), ('restart', 'original semantics'))):
            pts = []
            for a in sorted({r['a'] for r in rows}):
                v = m[(wl, a, res, 48)] / m[(wl, a, res, 6)]
                pts.append((a * 256, v, f"{lab}, 256-token re-prefill = {a*256:.2f} steps: unlimited / cap-6 throughput = {v:.2f}"))
            ser.append((lab, pts, i))
        body += panel(x0, 40, 470, 360, ser, title, "cost of re-prefilling a 256-token prompt (decode steps)",
                      "throughput: unlimited ÷ cap 6", ylim=(0.6, 1.35), fmt_y=lambda v: f"{v:.1f}",
                      refs=[('h', 1.0, 'break-even')], label_pos=None)
        x0 += 480
    body += legend(64, 26, [('vLLM-like resume', 0), ('original semantics', 1)])
    (OUT / 'fig2_cost.svg').write_text(svg(950, 410, body, 'When over-admission pays',
        'Ratio of saturation throughput with no admission cap to throughput with cap 6, versus the cost of one re-prefill, for two resume semantics and two workloads.'))


def fig3_headroom():
    rows = load('e3_admission')
    rows = [r for r in rows if r['scale'] == 4 and r['adm'].startswith('headroom')]
    base = {(r['prefix'], r['wl'], r['seed']): r['backlog_tok_per_u'] for r in rows if r['adm'] == 'headroom 0%'}
    body, x0 = '', 0
    for prefix, title in ((False, 'No prefix cache'), (True, 'With prefix cache (own blocks reusable)')):
        ser = []
        for i, wl in enumerate(('homog', 'chat', 'bimodal')):
            g = {}
            for r in rows:
                if r['prefix'] == prefix and r['wl'] == wl:
                    w = int(r['adm'].split()[1].rstrip('%'))
                    g.setdefault(w, []).append(100 * (r['backlog_tok_per_u'] / base[(prefix, wl, r['seed'])] - 1))
            pts = [(w, st.mean(v), f"{wl}, headroom {w}%: {st.mean(v):+.1f}% vs none (seeds {min(v):+.1f} to {max(v):+.1f})")
                   for w, v in sorted(g.items())]
            ser.append(({'homog': 'uniform', 'chat': 'chat mix', 'bimodal': 'bimodal'}[wl], pts, i))
        body += panel(x0, 40, 470, 360, ser, title, "KV blocks kept free at admission (%)",
                      "throughput vs zero headroom (%)", ylim=(-25, 15), fmt_y=lambda v: f"{v:+g}",
                      refs=[('h', 0, '')], label_pos=None)
        x0 += 480
    body += legend(64, 26, [('uniform', 0), ('chat mix', 1), ('bimodal', 2)])
    (OUT / 'fig3_headroom.svg').write_text(svg(960, 410, body, 'Admission headroom: an inverted U',
        'Paired change in saturation throughput versus percentage of KV blocks kept free at admission, x4 cache, three workloads, with and without a prefix cache.'))


def fig4_metrics():
    pts = json.loads((RES / 'e7_points.json').read_text())
    body, x0 = '', 0
    for key, xl, title in (('ev', 'preemptions per request', 'Preemption count'),
                           ('kv', 'KV cache usage (time-weighted)', 'KV cache usage'),
                           ('rc', 'share of time rebuilding evicted KV', 'Recompute time')):
        ser = []
        for i, wl in enumerate(('homog', 'chat', 'bimodal')):
            p = [(q[key], 100 * q['shortfall'],
                  f"{wl} x{q['scale']} prefix {'on' if q['prefix'] else 'off'}, {q['adm']}: {xl} {q[key]:.2f}, {100*q['shortfall']:.1f}% below best")
                 for q in pts if q['wl'] == wl]
            ser.append(({'homog': 'uniform', 'chat': 'chat mix', 'bimodal': 'bimodal'}[wl], p, i))
        fx = (lambda v: f"{v:.0%}") if key == 'rc' else (lambda v: f"{v:g}")
        body += panel(x0, 40, 330, 340, ser, title, xl, "throughput lost vs best policy (%)",
                      ylim=(0, 70), fmt_x=fx, lines=False, label_pos=None)
        x0 += 335
    body += legend(64, 26, [('uniform', 0), ('chat mix', 1), ('bimodal', 2)])
    (OUT / 'fig4_metrics.svg').write_text(svg(1010, 390, body, 'Which metric tracks lost throughput',
        'Scatter of throughput shortfall against three observables across 120 admission configurations.'))


def fig5_recovery():
    rows = [r for r in load('e6_recovery') if r['profile'] == 'burst']
    ser = []
    for i, w in ((0, 0.0), (2, 0.05)):          # same colours as fig6
        rs = [r for r in rows if r['wm'] == w]
        bins = {}
        for r in rs:
            for t, q, _ in r['series']:
                b = int(t // 1000)
                bins.setdefault(b, []).append(q)
        lab = 'no admission control' if w == 0 else '5% headroom at admission'
        pts = [(b + 0.5, st.mean(v), f"{lab}, t {b}-{b+1}k: mean queue {st.mean(v):.1f}") for b, v in sorted(bins.items()) if b < 40]
        ser.append((lab, pts, i))
    body = panel(0, 40, 860, 360, ser, "Waiting queue after a 1.5x burst (t = 3k-4.5k), base load near capacity",
                 "time (thousand units)", "requests waiting (mean of 5 seeds)", xlim=(0, 40),
                 refs=[('v', 3, 'burst'), ('v', 4.5, '')], label_pos=None)
    body += legend(64, 26, [('no admission control', 0), ('5% headroom at admission', 2)])
    (OUT / 'fig5_recovery.svg').write_text(svg(960, 410, body, 'Recovery after a burst',
        'Mean waiting-queue length over time after a burst, with zero and 5% admission headroom, chat workload, x4 cache, prefix cache on.'))


def fig6_cap_remedy():
    rows = load('e4b_cap_remedy')
    pols = (('no cap, no headroom', 'no admission control', 0),
            ('cap 24 (fits at completion)', 'cap = fits at completion (0 preemptions)', 1),
            ('no cap, 5% headroom', '5% headroom at admission', 2))
    body, x0 = '', 0
    for wl, title, ymax in (('homog', 'Uniform requests (256 in / 200 out)', 3500), ('chat', 'Chat-like mix', 1700)):
        ser = []
        for pol, lab, ci in pols:
            g = {}
            for r in rows:
                if r['wl'] == wl and r['pol'] == pol:
                    g.setdefault(r['rate'] * 4, []).append(r)
            pts = [(x, st.mean(r['lat_p50'] for r in rs),
                    f"{lab}, load {x:.3f} req/u: median latency {st.mean(r['lat_p50'] for r in rs):.0f}u, "
                    f"{st.mean(r['evictions_per_req'] for r in rs):.2f} preemptions per request, KV usage {st.mean(r['kv_util_tw'] for r in rs):.2f}")
                   for x, rs in sorted(g.items())]
            ser.append((lab, pts, ci))
        body += panel(x0, 40, 470, 360, ser, title, 'offered load (requests per time unit)', 'median latency (time units)',
                      ylim=(0, ymax), fmt_x=lambda v: f"{v:.2f}", label_pos=None)
        x0 += 480
    body += legend(64, 26, [(p[1], p[2]) for p in pols])
    (OUT / 'fig6_cap_remedy.svg').write_text(svg(960, 410, body, 'Eliminating preemption, at a price',
        'Median latency versus offered load for three admission policies, x4 cache, prefix cache on, mean of 5 seeds.'))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for f in (fig1_semantics, fig2_cost, fig3_headroom, fig4_metrics, fig5_recovery, fig6_cap_remedy):
        f(); print('wrote', f.__name__)
