#!/usr/bin/env python3
"""Run the experiment matrix and append raw rows to a CSV.

  quick: small graph only, 3 trials, one cell per (algorithm, strategy, p)  (< 5 min)
  full : three graphs, plus sweeps over k (schools), q (queries) and B (scenarios)

Every cell is run once as: 1 warm-up trial + --trials measured trials (default 5),
and the first trial of every parallel cell is verified against the sequential result.
"""
import argparse
import csv
import itertools
import json
import math
import os
import platform
import subprocess
import sys

ALGS = ("dijkstra", "bfs")
STRATS = ("source", "graph", "scenario")


class GraphSpec:
    def __init__(self, name, args, n, pool):
        self.name, self.args, self.n, self.pool = name, args, n, pool


def real_graph(data, name):
    d = os.path.join(data, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    args = ["--graph", os.path.join(d, "graph.bin"), "--schools", os.path.join(d, "schools.bin")]
    return GraphSpec(name, args, meta["nodes"], meta["schools_in_pool"])


def synth_graph(name, target):
    side = math.ceil(math.sqrt(target))
    n = side * math.ceil(target / side)  # mirrors graph_synth in graph.c
    return GraphSpec(name, ["--synth", str(target)], n, max(50, n // 20))


def dedupe(values):
    return sorted({v for v in values if v >= 1})


def defaults(g, config="full"):
    quick = config == "quick"
    return dict(k=max(1, g.pool // 4), q=max(1, min(1000 if quick else 5000, g.n // 5)),
                B=4 if quick else 8)


def build_cells(config, graphs, ps):
    """Cells are dicts: graph, alg, strategy, p, k, q, B."""
    cells = []

    def add(g, k, q, B, strategies=STRATS, procs=ps):
        for alg in ALGS:
            for strat, p in itertools.product(strategies, procs):
                cells.append(dict(graph=g, alg=alg, strategy=strat, p=p, k=k, q=q, B=B))

    if config == "quick":
        g = graphs["small"]
        d = defaults(g, config)
        add(g, d["k"], d["q"], d["B"])
    else:
        for g in graphs.values():
            d = defaults(g)
            add(g, d["k"], d["q"], d["B"])
        g = graphs["medium"]
        d = defaults(g)
        for k in dedupe([g.pool // 16, g.pool // 8, g.pool // 4, g.pool // 2, g.pool]):
            add(g, k, d["q"], d["B"])
        for q in dedupe(min(v, g.n // 2) for v in (100, 1000, 10000, 50000)):
            add(g, d["k"], q, d["B"])
        for B in (1, 2, 4, 8, 16):
            add(g, d["k"], d["q"], B)
    seen, out = set(), []
    for c in cells:
        key = cell_key(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    for c in list(out):
        base = dict(c, strategy="seq", p=1)
        if cell_key(base) not in seen:
            seen.add(cell_key(base))
            out.append(base)
    return out


def cell_key(c):
    return (c["graph"].name, c["alg"], c["strategy"], c["p"], c["k"], c["q"], c["B"])


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["label"], r["alg"], r["strategy"], int(r["p"]), int(r["k"]), int(r["q"]), int(r["B"]))
                for r in csv.DictReader(f)}


def dump_jobs(config, graphs):
    jobs = []
    if config == "quick":
        graphs = {"small": graphs["small"]}
    for g in graphs.values():
        ks = {defaults(g, config)["k"]}
        if g.name == "medium":
            ks |= {g.pool // 16, g.pool // 4, g.pool}
        for alg, k in itertools.product(ALGS, sorted(k for k in ks if k >= 1)):
            jobs.append((g, alg, k))
    return jobs


def run(cmd, out_path, header):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"FAILED ({res.returncode}): {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    lines = res.stdout.strip().splitlines()
    with open(out_path, "a") as f:
        for line in lines[0 if header else 1:]:
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=("quick", "full"), required=True)
    ap.add_argument("--data", default="data", help="folder holding small/, medium/, large/")
    ap.add_argument("--synthetic", action="store_true", help="development only: synthetic grid graphs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--binary", default="./nsearch")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--mpi-args", default="", help="e.g. '--bind-to core' for OpenMPI")
    ap.add_argument("--max-p", type=int, default=4, help="highest process count (physical cores)")
    ap.add_argument("--trials", type=int)
    a = ap.parse_args()
    trials = a.trials or (3 if a.config == "quick" else 5)
    ps = [p for p in (1, 2, 4, 8) if p <= a.max_p]
    if a.max_p not in ps:
        ps.append(a.max_p)

    if a.synthetic:
        graphs = {"small": synth_graph("small", 5000), "medium": synth_graph("medium", 50000),
                  "large": synth_graph("large", 500000)}
    else:
        graphs = {n: real_graph(a.data, n) for n in ("small", "medium", "large")}
    if a.config == "quick":
        graphs = {"small": graphs["small"], "medium": graphs["small"]}

    os.makedirs(os.path.join(a.out, "dumps"), exist_ok=True)
    raw = os.path.join(a.out, "raw.csv")
    with open(os.path.join(a.out, "environment.txt"), "w") as f:
        f.write(f"{platform.platform()}\npython {platform.python_version()}\n")
        for cmd in ([a.mpiexec, "--version"], ["gcc", "--version"]):
            f.write(subprocess.run(cmd, capture_output=True, text=True).stdout.split("\n\n")[0] + "\n")
        f.write(f"mpi_args={a.mpi_args!r} config={a.config} trials={trials} synthetic={a.synthetic}\n")

    design = {g.name: dict(defaults(g, a.config), n=g.n, pool=g.pool) for g in graphs.values()}
    with open(os.path.join(a.out, "design.json"), "w") as f:
        json.dump({"config": a.config, "synthetic": a.synthetic, "max_p": a.max_p, "graphs": design}, f, indent=2)

    cells = build_cells(a.config, graphs, ps)
    finished = done_keys(raw)
    todo = [c for c in cells if cell_key(c) not in finished]
    print(f"{len(cells)} cells, {len(todo)} to run", file=sys.stderr)
    for i, c in enumerate(todo, 1):
        g = c["graph"]
        cmd = [a.mpiexec, "-n", str(c["p"])] + a.mpi_args.split() + [
            a.binary, *g.args, "--alg", c["alg"], "--strategy", c["strategy"], "--k", str(c["k"]),
            "--q", str(c["q"]), "--B", str(c["B"]), "--trials", str(trials), "--label", g.name,
            "--header"]
        if c["strategy"] != "seq":
            cmd.append("--verify")
        print(f"[{i}/{len(todo)}] {g.name} {c['alg']} {c['strategy']} p={c['p']} k={c['k']} "
              f"q={c['q']} B={c['B']}", file=sys.stderr)
        run(cmd, raw, header=not os.path.exists(raw) or os.path.getsize(raw) == 0)

    for g, alg, k in dump_jobs(a.config, graphs):
        path = os.path.join(a.out, "dumps", f"{g.name}_{alg}_k{k}.csv")
        if os.path.exists(path):
            continue
        q = min(20000, g.n // 2)
        res = subprocess.run([a.binary, *g.args, "--alg", alg, "--strategy", "seq", "--k", str(k),
                              "--q", str(q), "--B", "1", "--trials", "1", "--dump", path],
                             capture_output=True, text=True)
        if res.returncode:
            sys.exit(res.stderr)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
