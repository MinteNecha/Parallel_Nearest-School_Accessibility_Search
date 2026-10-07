#!/usr/bin/env python3
"""Turn results/raw.csv (and results/dumps/) into summary tables and paper figures."""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STRATS = ["source", "graph", "scenario"]
STYLE = {  # fixed hue per strategy (validated categorical slots 1-3), plus marker as 2nd encoding
    "source": dict(color="#2a78d6", marker="o", ls="-"),
    "graph": dict(color="#eb6834", marker="s", ls="--"),
    "scenario": dict(color="#1baf7a", marker="^", ls="-."),
}
LABEL = {"source": "Source (schools split)", "graph": "Graph partition", "scenario": "Scenario batch"}
KEY = ["label", "alg", "k", "q", "B"]
CELL = KEY + ["strategy", "p"]


def load(raw):
    df = pd.read_csv(raw)
    ok = df[df.trial == 0].groupby(CELL).verified.min()
    meas = df[df.trial >= 1].copy()
    meas["lb"] = (meas.comp_max - meas.comp_min) / meas.comp_mean
    meas["wait_frac"] = meas.wait_mean / meas.wall
    meas["xfer_frac"] = meas.xfer_mean / meas.wall
    meas["comm_frac"] = meas.wait_frac + meas.xfer_frac
    g = meas.groupby(CELL)
    s = g.agg(wall=("wall", "median"), wall_min=("wall", "min"), wall_max=("wall", "max"),
              trials=("wall", "size"), comp=("comp_mean", "median"), wait=("wait_mean", "median"),
              xfer=("xfer_mean", "median"), lb=("lb", "median"), comm_frac=("comm_frac", "median"),
              wait_frac=("wait_frac", "median"), xfer_frac=("xfer_frac", "median"),
              msgs=("msgs", "median"), steps=("steps", "median"), edge_cut=("edge_cut", "median"),
              n=("n", "first"), m=("m", "first")).reset_index()
    s = s.join(ok.rename("verified"), on=CELL)
    base = s[s.strategy == "seq"].set_index(KEY)[["wall", "wall_min", "wall_max"]]
    base.columns = ["seq_wall", "seq_min", "seq_max"]
    s = s.join(base, on=KEY)
    s["speedup"] = s.seq_wall / s.wall
    s["speedup_lo"] = s.seq_wall / s.wall_max
    s["speedup_hi"] = s.seq_wall / s.wall_min
    s["efficiency"] = s.speedup / s.p
    return s


def defaults_for(design, graph):
    d = design["graphs"][graph]
    return d["k"], d["q"], d["B"]


def sel(s, graph, **kw):
    m = (s.label == graph)
    for k, v in kw.items():
        m &= s[k] == v
    return s[m]


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)


def plot_lines(ax, d, xcol, ycol, lo=None, hi=None):
    for st in STRATS:
        r = d[d.strategy == st].sort_values(xcol)
        if r.empty:
            continue
        ax.plot(r[xcol], r[ycol], lw=1.6, ms=5, label=LABEL[st], **STYLE[st])
        if lo:
            ax.fill_between(r[xcol], r[lo], r[hi], color=STYLE[st]["color"], alpha=0.15, lw=0)


def fig_speedup(s, design, out):
    graphs = [g for g in ("small", "medium", "large") if g in design["graphs"]]
    algs = ["dijkstra", "bfs"]
    fig, axes = plt.subplots(len(algs), len(graphs), figsize=(3.2 * len(graphs), 5.4), squeeze=False)
    for i, alg in enumerate(algs):
        for j, g in enumerate(graphs):
            k, q, B = defaults_for(design, g)
            d = sel(s, g, alg=alg, k=k, q=q, B=B)
            d = d[d.strategy != "seq"]
            ax = axes[i][j]
            plot_lines(ax, d, "p", "speedup", "speedup_lo", "speedup_hi")
            ps = sorted(d.p.unique())
            ax.plot(ps, ps, color="#8a8984", lw=0.8, ls=":", label="Ideal")
            ax.set_xticks(ps)
            ax.set_title(f"{alg.capitalize()}, {g} graph", fontsize=9)
            ax.set_xlabel("MPI processes")
            ax.set_ylabel("Speedup vs sequential")
            style_axes(ax)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_speedup.pdf"))
    plt.close(fig)


def fig_sweeps(s, design, out):
    g = "medium" if "medium" in design["graphs"] else next(iter(design["graphs"]))
    k0, q0, B0 = defaults_for(design, g)
    pmax = design["max_p"]
    sweeps = [("k", "Schools per scenario (k)", dict(q=q0, B=B0)),
              ("q", "Query origins (q)", dict(k=k0, B=B0)),
              ("B", "Scenarios (B)", dict(k=k0, q=q0))]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), squeeze=False)
    for i, alg in enumerate(("dijkstra", "bfs")):
        for j, (col, xl, fixed) in enumerate(sweeps):
            d = sel(s, g, alg=alg, p=pmax, **fixed)
            d = d[d.strategy != "seq"]
            ax = axes[i][j]
            plot_lines(ax, d, col, "speedup", "speedup_lo", "speedup_hi")
            if col != "B":
                ax.set_xscale("log")
            ax.axhline(1, color="#8a8984", lw=0.8, ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel(f"Speedup at p={pmax}")
            ax.set_title(f"{alg.capitalize()}, {g} graph", fontsize=9)
            style_axes(ax)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_sweeps.pdf"))
    plt.close(fig)


def fig_breakdown(s, design, out):
    graphs = list(design["graphs"])
    pmax = design["max_p"]
    fig, axes = plt.subplots(2, len(graphs), figsize=(3.2 * len(graphs), 5.0), squeeze=False)
    parts = [("comp", "Compute", "#2a78d6"), ("wait", "Sync wait", "#8a8984"), ("xfer", "Transfer", "#eb6834")]
    for i, alg in enumerate(("dijkstra", "bfs")):
        for j, g in enumerate(graphs):
            k, q, B = defaults_for(design, g)
            d = sel(s, g, alg=alg, k=k, q=q, B=B, p=pmax).set_index("strategy").reindex(STRATS)
            ax, bottom = axes[i][j], np.zeros(len(STRATS))
            for col, name, colr in parts:
                v = d[col].fillna(0).values
                ax.bar(STRATS, v, bottom=bottom, label=name, color=colr, width=0.55,
                       edgecolor="white", linewidth=1.5)
                bottom += v
            ax.set_title(f"{alg.capitalize()}, {g} graph, p={pmax}", fontsize=9)
            ax.set_ylabel("Mean seconds per rank")
            style_axes(ax)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_breakdown.pdf"))
    plt.close(fig)


def difference_tables(dumps, out):
    rows = []
    for dj in sorted(glob.glob(os.path.join(dumps, "*_dijkstra_k*.csv"))):
        bf = dj.replace("_dijkstra_", "_bfs_")
        if not os.path.exists(bf):
            continue
        a, b = pd.read_csv(dj), pd.read_csv(bf)
        m = a.merge(b, on=["qidx", "node"], suffixes=("_w", "_h"))
        graph, _, k = os.path.basename(dj)[:-4].split("_")
        m["m_per_hop"] = m.dist_w / m.dist_h.clip(lower=1)
        nz = m[(m.dist_h > 0) & (m.dist_w > 0)]
        rows.append(dict(graph=graph, k=int(k[1:]), origins=len(m),
                         nearest_school_differs=float((m.school_idx_w != m.school_idx_h).mean()),
                         spearman_dist_vs_hops=float(nz.dist_w.corr(nz.dist_h, method="spearman")),
                         median_m_per_hop=float(nz.m_per_hop.median()),
                         iqr_m_per_hop_lo=float(nz.m_per_hop.quantile(0.25)),
                         iqr_m_per_hop_hi=float(nz.m_per_hop.quantile(0.75)),
                         mean_road_km=float(m.dist_w.mean() / 1000), mean_hops=float(m.dist_h.mean())))
    diff = pd.DataFrame(rows)
    diff.to_csv(os.path.join(out, "diff_dijkstra_vs_bfs.csv"), index=False)
    return diff


def fig_diff(diff, dumps, out):
    if diff.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for g, r in diff.groupby("graph"):
        r = r.sort_values("k")
        ax.plot(r.k, 100 * r.nearest_school_differs, marker="o", lw=1.4, ms=4, label=g)
    ax.set_xscale("log")
    ax.set_xlabel("Schools per scenario (k)")
    ax.set_ylabel("Origins with a different nearest school (%)")
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False)
    g0 = diff.sort_values("k").iloc[-1]
    a = pd.read_csv(os.path.join(dumps, f"{g0.graph}_dijkstra_k{g0.k}.csv"))
    b = pd.read_csv(os.path.join(dumps, f"{g0.graph}_bfs_k{g0.k}.csv"))
    m = a.merge(b, on=["qidx", "node"], suffixes=("_w", "_h")).sample(min(4000, len(a)), random_state=42)
    axes[1].scatter(m.dist_h, m.dist_w / 1000, s=3, alpha=0.35, color="#2a78d6", lw=0)
    axes[1].set_xlabel("BFS hops to nearest school")
    axes[1].set_ylabel("Dijkstra road distance (km)")
    axes[1].set_title(f"{g0.graph} graph, k={g0.k}", fontsize=9)
    style_axes(axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_diff.pdf"))
    plt.close(fig)


def tex_table(path, header, rows, align):
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_tex(main, diff, design, out):
    os.makedirs(out, exist_ok=True)
    pmax = design["max_p"]
    rows = []
    for (g, alg, st), r in main[main.strategy != "seq"].groupby(["label", "alg", "strategy"]):
        r = r.set_index("p")
        t = lambda p: f"{r.wall[p]:.3f}" if p in r.index else "-"
        top = r.loc[pmax]
        rows.append([g, alg, st, t(1), t(2), t(pmax), f"{top.speedup:.2f}", f"{top.efficiency:.2f}",
                     f"{top.lb:.2f}", f"{100 * top.comm_frac:.0f}\\%"])
    tex_table(os.path.join(out, "main.tex"),
              ["Graph", "Alg.", "Strategy", "$T_1$", "$T_2$", f"$T_{{{pmax}}}$", f"$S_{{{pmax}}}$",
               f"$E_{{{pmax}}}$", "LB", "Comm."], rows, "lllrrrrrrr")
    if not diff.empty:
        tex_table(os.path.join(out, "diff.tex"),
                  ["Graph", "$k$", "Origins", "Differ (\\%)", "Spearman", "m/hop (median)"],
                  [[r.graph, r.k, r.origins, f"{100 * r.nearest_school_differs:.1f}",
                    f"{r.spearman_dist_vs_hops:.3f}", f"{r.median_m_per_hop:.0f}"]
                   for r in diff.sort_values(["graph", "k"]).itertuples()], "lrrrrr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    design = json.load(open(os.path.join(a.results, "design.json")))
    s = load(os.path.join(a.results, "raw.csv"))
    s.to_csv(os.path.join(a.results, "summary.csv"), index=False)
    figs = os.path.join(a.results, "figures")
    os.makedirs(figs, exist_ok=True)
    main_rows = []
    for g in design["graphs"]:
        k, q, B = defaults_for(design, g)
        main_rows.append(sel(s, g, k=k, q=q, B=B))
    main = pd.concat(main_rows).sort_values(["label", "alg", "strategy", "p"])
    main[["label", "alg", "strategy", "p", "k", "q", "B", "wall", "wall_min", "wall_max", "speedup",
          "efficiency", "lb", "comm_frac", "wait_frac", "xfer_frac", "steps", "msgs", "edge_cut",
          "verified"]].to_csv(os.path.join(a.results, "table_main.csv"), index=False)
    fig_speedup(s, design, figs)
    fig_sweeps(s, design, figs)
    fig_breakdown(s, design, figs)
    diff = difference_tables(os.path.join(a.results, "dumps"), a.results)
    fig_diff(diff, os.path.join(a.results, "dumps"), figs)
    write_tex(main, diff, design, os.path.join(a.results, "tables"))
    bad = s[(s.strategy != "seq") & (s.verified != 1)]
    print(f"cells={len(s)}  unverified={len(bad)}")
    if design.get("synthetic"):
        print("NOTE: synthetic data, development run only. Do not report as Gauteng results.")


if __name__ == "__main__":
    main()
