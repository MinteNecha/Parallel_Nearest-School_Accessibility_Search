#!/usr/bin/env python3
"""End-to-end check on a tiny hand-built OSM extract.

Builds a PBF (grid of residential streets, some one-way, plus a footway that must be
ignored), runs prepare_data.py, then compares the C program (sequential Dijkstra and
BFS) with SciPy computed on the independently constructed TRAVEL graph.
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import osmium
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from prepare_data import haversine  # noqa: E402

N = 25
LON0, LAT0, STEP = 28.00, -26.20, 0.001


def nid(r, c):
    return 1 + r * N + c


def build_pbf(path):
    w = osmium.SimpleWriter(path)
    for r in range(N):
        for c in range(N):
            w.add_node(osmium.osm.mutable.Node(id=nid(r, c), location=(LON0 + c * STEP, LAT0 + r * STEP)))
    wid, travel = 1, []
    for r in range(N):
        tags = {"highway": "residential"} | ({"oneway": "yes"} if r % 4 == 1 else {})
        w.add_way(osmium.osm.mutable.Way(id=wid, nodes=[nid(r, c) for c in range(N)], tags=tags)); wid += 1
        for c in range(N - 1):
            travel.append((nid(r, c), nid(r, c + 1)))
            if "oneway" not in tags:
                travel.append((nid(r, c + 1), nid(r, c)))
    for c in range(N):
        w.add_way(osmium.osm.mutable.Way(id=wid, nodes=[nid(r, c) for r in range(N)],
                                         tags={"highway": "residential"})); wid += 1
        for r in range(N - 1):
            travel += [(nid(r, c), nid(r + 1, c)), (nid(r + 1, c), nid(r, c))]
    w.add_way(osmium.osm.mutable.Way(id=wid, nodes=[nid(0, 0), nid(N - 1, N - 1)], tags={"highway": "footway"}))
    w.close()
    return travel


def main():
    tmp = tempfile.mkdtemp()
    pbf, schools, out = (os.path.join(tmp, x) for x in ("t.osm.pbf", "schools.csv", "out"))
    travel = build_pbf(pbf)
    rng = np.random.default_rng(1)
    with open(schools, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["school_id", "lat", "lon"])
        for i in range(12):
            wr.writerow([f"S{i:03d}", LAT0 + rng.uniform(0, (N - 1) * STEP), LON0 + rng.uniform(0, (N - 1) * STEP)])
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "prepare_data.py"), "--pbf", pbf,
                    "--schools", schools, "--out", out], check=True, stdout=subprocess.DEVNULL)

    nodes = np.loadtxt(os.path.join(out, "nodes.csv"), delimiter=",", skiprows=1)
    key = {(round(lon, 5), round(lat, 5)): int(i) for i, lon, lat in nodes}
    final = {nid(r, c): key[(round(LON0 + c * STEP, 5), round(LAT0 + r * STEP, 5))]
             for r in range(N) for c in range(N)}
    n = len(nodes)
    lon, lat = nodes[:, 1], nodes[:, 2]
    src = np.array([final[a] for a, _ in travel]); dst = np.array([final[b] for _, b in travel])
    wt = np.maximum(1, np.rint(haversine(lon[src], lat[src], lon[dst], lat[dst])))
    smap = list(csv.DictReader(open(os.path.join(out, "schools_map.csv"))))
    snodes = [int(r["node"]) for r in smap]

    ok = True
    for alg, unweighted in (("dijkstra", False), ("bfs", True)):
        travel_adj = csr_matrix((np.ones(len(src)) if unweighted else wt, (src, dst)), shape=(n, n))
        d_all = dijkstra(travel_adj, directed=True, unweighted=unweighted)
        to_schools = d_all[:, snodes]
        want_d = to_schools.min(axis=1).astype(int)
        want_id = to_schools.argmin(axis=1)
        dump = os.path.join(tmp, f"{alg}.csv")
        subprocess.run([os.path.join(ROOT, "nsearch"), "--graph", os.path.join(out, "graph.bin"),
                        "--schools", os.path.join(out, "schools.bin"), "--alg", alg, "--strategy", "seq",
                        "--k", str(len(snodes)), "--q", str(n), "--trials", "1", "--dump", dump],
                       check=True, stdout=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(dump)))
        got_node = np.array([int(r["node"]) for r in rows])
        got_d = np.array([int(r["dist"]) for r in rows]); got_id = np.array([int(r["school_idx"]) for r in rows])
        bad_d = int((got_d != want_d[got_node]).sum()); bad_id = int((got_id != want_id[got_node]).sum())
        print(f"{alg}: origins={len(rows)} dist_mismatch={bad_d} id_mismatch={bad_id}")
        ok &= bad_d == 0 and bad_id == 0 and len(rows) == n
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
