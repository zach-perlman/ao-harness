"""Multi-run comparison dashboard — compare several AO runs against one baseline.

Purpose
-------
The standard `dashboard.py` compares ONE target run to the baseline. For a sweep
(e.g. C1's beta ∈ {0.1, 0.3, 0.5}) we want every variant side-by-side so we can
pick a winner per task. This builds a single self-contained HTML:

    Task | n | baseline | run_1 (Δ[CI]) | run_2 (Δ[CI]) | ... | best

Mechanism
---------
Reuses dashboard.py's loaders + metric machinery so numbers match exactly:
  * load_summaries / load_records read each run dir.
  * the HEADLINE (★ first) metric per task drives the comparison.
  * Δ = run − baseline, with a PAIRED bootstrap 95% CI (same items both runs);
    a CI straddling 0 is greyed + tagged n.s.  AUC-headline tasks (mmlu,
    missing_info) fall back to the analytic independent-SE interval.
  * "best" = the significant run with the largest improvement in the good
    direction (— if none beats baseline outside noise).
All headline metrics here are higher-is-better, so "improvement" = Δ > 0.

Usage
-----
    python -m ao_cli.sweep_dashboard \
        --baseline baseline_replication --runs c1v3 c1v4_b01 c1v4_b03 c1v4_b05 \
        --out artifacts/<model>/aobench_results/b_sweep_dashboard.html
Runs default to auto-detected c1v4_b* dirs (+ c1v3 as reference if present).
"""

from __future__ import annotations

import argparse
import html
import math
from datetime import datetime
from pathlib import Path

from . import ARTIFACTS, load_config, model_slug
from .dashboard import (
    TASKS, _CLUSTERED_TASKS, _Z, effective_n, fmt, headline_halfwidth,
    load_records, load_summaries, metric_n, metric_se, metric_value, normalize,
    paired_bootstrap_delta,
)

# Per-run bar colors: baseline green, c1v3 grey, betas on a warm→cool ramp; any
# extra runs cycle a fallback palette.
_RUN_COLORS = {
    "baseline_replication": "#7C9C59", "c1v3": "#9AA0A6",
    "c1v4_b01": "#2A6FDF", "c1v4_b03": "#8E44AD", "c1v4_b05": "#E67E22",
}
_FALLBACK = ["#2A6FDF", "#E67E22", "#8E44AD", "#16A085", "#C0392B", "#2C3E50"]


def _delta_ci(task: str, m: dict, recs_o: list, recs_b: list,
              sums_o: dict, sums_b: dict, neff: int | None):
    """(delta, lo, hi, ns) for run vs baseline on metric m. Paired bootstrap when
    available (rate/scale headline tasks), else analytic independent-SE interval."""
    vo, vb = metric_value(sums_o.get(task), m), metric_value(sums_b.get(task), m)
    if vo is None or vb is None:
        return None
    ci = paired_bootstrap_delta(task, recs_o, recs_b)
    if ci is not None:
        d, lo, hi = ci
    else:
        n = neff or metric_n(sums_o.get(task), m) or metric_n(sums_b.get(task), m)
        se_o, se_b = metric_se(sums_o.get(task), m, vo, n), metric_se(sums_b.get(task), m, vb, n)
        d = vo - vb
        if se_o is None or se_b is None:
            return (d, None, None, False)
        hw = _Z * math.sqrt(se_o ** 2 + se_b ** 2)
        lo, hi = d - hw, d + hw
    ns = (lo is None) or (lo <= 0.0 <= hi)
    return (d, lo, hi, ns)


def _cell(task: str, m: dict, vo, info) -> str:
    """A run's cell: headline value, then Δ[lo,hi] coloured by significance."""
    if vo is None:
        return '<td class="num tie">—</td>'
    if info is None:
        return f'<td class="num">{fmt(m["kind"], vo)}</td>'
    d, lo, hi, ns = info
    better = d > 0  # all headline metrics are higher-is-better
    cls = "tie" if (ns or abs(d) < 1e-9) else ("win" if better else "lose")
    sign = "+" if d >= 0 else ""
    citxt = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "(n/a)"
    tag = ' <span class="ns">n.s.</span>' if ns and abs(d) >= 1e-9 else ""
    return (f'<td class="num {cls}"><div class="val">{fmt(m["kind"], vo)}</div>'
            f'<div class="d">{sign}{d:.3f}<br><span class="ci">{citxt}</span>{tag}</div></td>')


def chart_variants(results_root: Path, order: list[str], sums: dict[str, dict]) -> str:
    """Grouped bar chart, mirroring the regular dashboard's headline chart but with
    one series PER RUN (baseline + every variant). Each task's headline (★) metric
    is normalized to [0=chance, 1=perfect] and drawn with a 95% CI whisker, so the
    bars are directly comparable across tasks AND runs. Plotly is embedded so the
    page stays self-contained."""
    import plotly.graph_objects as go

    tasks = [t for t, spec in TASKS.items()
             if any(metric_value(sums[r].get(t), spec["metrics"][0]) is not None for r in order)]
    labels = [TASKS[t]["title"].replace(" ", "<br>", 1) for t in tasks]

    # effective-n (distinct problems, not probe count) sizes CIs honestly for the
    # judge-clustered tasks; other tasks fall back to the raw metric n.
    neff = {r: {t: effective_n(t, load_records(results_root / r, t))
                for t in tasks if t in _CLUSTERED_TASKS} for r in order}

    fig = go.Figure()
    allv: list[float] = []
    for i, r in enumerate(order):
        ys, txt, err = [], [], []
        for t in tasks:
            m = TASKS[t]["metrics"][0]
            v = metric_value(sums[r].get(t), m)
            nv = normalize(m["kind"], v) if v is not None else None
            ys.append(round(nv, 4) if nv is not None else None)
            if nv is not None:
                allv.append(nv)
            txt.append(fmt(m["kind"], v))
            err.append(headline_halfwidth(sums[r].get(t), m, neff.get(r, {}).get(t)))
        color = _RUN_COLORS.get(r, _FALLBACK[i % len(_FALLBACK)])
        fig.add_bar(name=r, x=labels, y=ys, marker_color=color, customdata=txt,
                    error_y=dict(type="data", array=err, visible=True,
                                 color="rgba(20,30,50,.30)", thickness=1.1),
                    hovertemplate=f"{r} · %{{x}}<br>raw %{{customdata}} · "
                                  "norm %{y:.3f} ± %{error_y.array:.3f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dash", line_color="#888",
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_layout(
        barmode="group", template="plotly_white",
        title="Headline metric per task — normalized (0 = chance, 1 = perfect; whiskers = 95% CI). "
              "Hover for raw values.",
        yaxis_title="normalized score", legend_title_text="run",
        margin=dict(t=60, b=90, l=50, r=20), font=dict(size=12),
    )
    fig.update_yaxes(range=[min(-0.1, min(allv, default=0.0) - 0.05), 1.12])
    # plotly.js is loaded once in the page <head> (see build), so the fragment omits
    # it — matches the regular dashboard, which renders reliably.
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", default_height="540px")


def build(results_root: Path, baseline: str, runs: list[str], out_path: Path) -> Path:
    base_sums = load_summaries(results_root / baseline)
    run_sums = {r: load_summaries(results_root / r) for r in runs}
    cfg = load_config()
    model = cfg["model"]["name"]
    gen_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Top bar chart: baseline + every variant, one series each (like the regular
    # dashboard's headline chart).
    chart_order = [baseline] + runs
    chart = chart_variants(results_root, chart_order, {baseline: base_sums, **run_sums})
    import plotly.offline as pyo
    plotly_js = f"<script>{pyo.get_plotlyjs()}</script>"

    head_cells = "".join(f"<th>{html.escape(r)}<div class='sub'>value · Δ vs base [95% CI]</div></th>"
                         for r in runs)
    rows = []
    for task, spec in TASKS.items():
        if task not in base_sums and not any(task in run_sums[r] for r in runs):
            continue
        m = spec["metrics"][0]                         # headline (★) metric
        base_recs = load_records(results_root / baseline, task)
        neff = effective_n(task, base_recs)
        vb = metric_value(base_sums.get(task), m)

        cells, best_run, best_d = [], None, 0.0
        for r in runs:
            vo = metric_value(run_sums[r].get(task), m)
            recs_o = load_records(results_root / r, task)
            info = _delta_ci(task, m, recs_o, base_recs, run_sums[r], base_sums, neff) if vb is not None else None
            cells.append(_cell(task, m, vo, info))
            if info is not None:
                d, lo, hi, ns = info
                if (not ns) and d > best_d:            # significant improvement
                    best_d, best_run = d, r
        best = f'<b class="win">{html.escape(best_run)}</b> (+{best_d:.3f})' if best_run else '<span class="mut">— none</span>'
        nlabel = (f'{metric_n(base_sums.get(task), m) or "—"}'
                  + (f' <span class="mut">({neff} prob.)</span>' if neff else ''))
        rows.append(
            f'<tr><td class="task"><b>{html.escape(spec["title"])}</b>'
            f'<div class="mut">{html.escape(m["label"])} ★</div></td>'
            f'<td class="num">{nlabel}</td>'
            f'<td class="num base">{fmt(m["kind"], vb)}</td>'
            f'{"".join(cells)}<td class="best">{best}</td></tr>')

    style = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#1a1f29;margin:24px;max-width:1200px}
    h1{font-size:21px;margin:0 0 4px} .mut{color:#6b7480}
    table{border-collapse:collapse;width:100%;margin-top:14px} th,td{border:1px solid #e3e7ee;padding:7px 9px;text-align:left;vertical-align:top}
    th{background:#f6f8fa;font-size:12px} .sub{font-weight:400;color:#8a929e;font-size:10px}
    .num{text-align:right;font-variant-numeric:tabular-nums} .task{min-width:200px}
    .val{font-weight:600} .d{font-size:11px;margin-top:2px} .ci{color:#8a929e;font-size:10px}
    td.win{background:#eef7f0} td.win .d{color:#137333} td.lose{background:#fdf0ef} td.lose .d{color:#b3261e}
    td.tie .d{color:#6b7480} td.base{background:#fafbfc;font-weight:600} .ns{color:#b3261e;font-weight:600}
    .best{font-size:12px} .legend{margin:10px 0;padding:8px 12px;background:#f6f8fa;border-radius:6px;font-size:12px;color:#4a5260}
    """
    cfg_note = (f"pairs: n_tokens={cfg['contributions']['swap_test'].get('n_tokens')}, "
                f"pairs_per_token={cfg['contributions']['swap_test'].get('pairs_per_token')}")
    htmldoc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>C1 beta sweep — comparison</title><style>{style}</style>{plotly_js}</head><body>
<h1>C1 DPO sweep — {html.escape(model)}</h1>
<p class="mut">baseline <code>{html.escape(baseline)}</code> · runs {", ".join("<code>"+html.escape(r)+"</code>" for r in runs)} · {html.escape(cfg_note)} · generated {gen_ts}</p>
{chart}
<h2 style="font-size:16px;margin:18px 0 0">Per-task detail (Δ vs baseline)</h2>
<div class="legend">Each cell: headline (★) metric value, then <b>Δ vs baseline</b> with a paired-bootstrap 95% CI.
<span style="color:#137333">green</span>=significant gain, <span style="color:#b3261e">red</span>=significant drop,
grey + <span class="ns">n.s.</span>=within noise (CI crosses 0). <b>best</b> = significant run with the largest gain.</div>
<table><thead><tr><th>Task</th><th>n</th><th>baseline</th>{head_cells}<th>best</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="mut" style="margin-top:14px">Headline metrics are all higher-is-better. AUC tasks (mmlu, missing_info) use the
analytic independent-SE interval (no paired bootstrap); rate/scale tasks use the paired item bootstrap.</p>
</body></html>"""
    out_path.write_text(htmldoc)
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--baseline", default=None, help="baseline run dir (default: config eval.baseline_run)")
    p.add_argument("--runs", nargs="*", default=None, help="variant run dirs (default: auto-detect c1v4_b*)")
    p.add_argument("--out", default=None, help="output HTML (default: <results-dir>/b_sweep_dashboard.html)")
    args = p.parse_args(argv)

    cfg = load_config()
    root = Path(args.results_dir) if args.results_dir else (
        ARTIFACTS / model_slug(cfg["model"]["name"]) / "aobench_results")
    baseline = args.baseline or cfg.get("eval", {}).get("baseline_run") or "baseline_replication"
    if args.runs:
        runs = args.runs
    else:
        runs = sorted(p.name for p in root.glob("c1v4_b*") if p.is_dir())
        if (root / "c1v3").is_dir():
            runs = ["c1v3"] + runs                     # c1v3 = the pre-sweep reference
    runs = [r for r in runs if (root / r).is_dir()]
    if not runs:
        raise SystemExit(f"[sweep_dashboard] no variant runs found under {root}")
    out = Path(args.out) if args.out else root / "b_sweep_dashboard.html"
    build(root, baseline, runs, out)
    print(f"[sweep_dashboard] wrote {out}  (baseline={baseline}, runs={runs})")


if __name__ == "__main__":
    main()
