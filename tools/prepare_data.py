#!/usr/bin/env python3
"""Build graph.bin, schools.bin, schools_map.csv and meta.json from an OSM extract
and a school list.

The stored graph is the TRANSPOSE of the travel graph: a forward search from the
schools then yields the road distance from every origin TO its nearest school.
"""
import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from graphio import write_graph, write_u32

DEFAULT_HIGHWAYS = "primary,secondary,tertiary,unclassified,residential"
ONEWAY_FWD = {"yes", "true", "1"}
EARTH_R = 6371000.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def read_ways(pbf, highways, ignore_oneway, bbox=None):
    import osmium

    allowed = set(highways)
    fp = osmium.FileProcessor(pbf).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
    ids, lon, lat, seg = {}, [], [], []

    def node_index(n):
        if bbox and not (bbox[0] <= n.lon <= bbox[2] and bbox[1] <= n.lat <= bbox[3]):
            return None
        idx = ids.get(n.ref)
        if idx is None:
            idx = ids[n.ref] = len(lon)
            lon.append(n.lon)
            lat.append(n.lat)
        return idx

    for obj in fp:
        if not obj.is_way():
            continue
        hw = obj.tags.get("highway")
        base = hw[: -len("_link")] if hw and hw.endswith("_link") else hw
        if base not in allowed:
            continue
        tag = obj.tags.get("oneway", "")
        rev = tag == "-1"
        fwd_only = tag in ONEWAY_FWD or obj.tags.get("junction") == "roundabout"
        if ignore_oneway:
            fwd_only = rev = False
        nodes = [node_index(n) for n in obj.nodes if n.location.valid()]
        for a, b in zip(nodes, nodes[1:]):
            if a is None or b is None or a == b:
                continue
            if rev:
                seg.append((b, a))
            else:
                seg.append((a, b))
                if not fwd_only:
                    seg.append((b, a))
    return np.array(lon), np.array(lat), np.array(seg, dtype=np.int64)


def largest_scc(n, src, dst):
    adj = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    _, lab = connected_components(adj, directed=True, connection="strong")
    return lab == np.bincount(lab).argmax()


def induced(keep, src, dst, w):
    new = np.cumsum(keep) - 1
    mask = keep[src] & keep[dst]
    return new[src[mask]], new[dst[mask]], w[mask], new


def morton_order(lon, lat):
    def spread(v):
        v = v.astype(np.uint64) & 0xFFFF
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        return (v | (v << 1)) & 0x55555555

    qx = ((lon - lon.min()) / max(np.ptp(lon), 1e-12) * 65535).astype(np.uint64)
    qy = ((lat - lat.min()) / max(np.ptp(lat), 1e-12) * 65535).astype(np.uint64)
    return np.argsort(spread(qx) | (spread(qy) << np.uint64(1)), kind="stable")


def load_schools(path, id_col, lat_col, lon_col):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [(r[id_col].strip(), float(r[lat_col]), float(r[lon_col]))
                for r in csv.DictReader(f) if r[lat_col].strip() and r[lon_col].strip()]
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--schools", required=True, help="CSV with school id, latitude, longitude")
    ap.add_argument("--school-id-col", default="school_id")
    ap.add_argument("--school-lat-col", default="lat")
    ap.add_argument("--school-lon-col", default="lon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--highways", default=DEFAULT_HIGHWAYS)
    ap.add_argument("--ignore-oneway", action="store_true")
    ap.add_argument("--centre", default="-26.2041,28.0473", help="lat,lon of crop centre")
    ap.add_argument("--target-nodes", type=int, default=0, help="0 keeps the whole network")
    ap.add_argument("--max-snap-m", type=float, default=1000.0)
    ap.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat: clip a larger extract before building")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lon, lat, seg = read_ways(
        a.pbf, a.highways.split(","), a.ignore_oneway,
        tuple(map(float, a.bbox.split(","))) if a.bbox else None)
    src, dst = seg[:, 0], seg[:, 1]
    w = np.maximum(1, np.rint(haversine(lon[src], lat[src], lon[dst], lat[dst]))).astype(np.int64)
    raw_nodes = len(lon)

    if a.target_nodes:
        clat, clon = map(float, a.centre.split(","))
        d = haversine(lon, lat, clon, clat)
        keep = d <= np.partition(d, min(a.target_nodes, len(d)) - 1)[min(a.target_nodes, len(d)) - 1]
        src, dst, w, _ = induced(keep, src, dst, w)
        lon, lat = lon[keep], lat[keep]
    keep = largest_scc(len(lon), src, dst)
    src, dst, w, _ = induced(keep, src, dst, w)
    lon, lat = lon[keep], lat[keep]

    order = morton_order(lon, lat)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    lon, lat, src, dst = lon[order], lat[order], rank[src], rank[dst]
    n = len(lon)
    write_graph(os.path.join(a.out, "graph.bin"), n, dst, src, w)  # transposed edges
    np.savetxt(os.path.join(a.out, "nodes.csv"), np.c_[np.arange(n), lon, lat],
               delimiter=",", header="node,lon,lat", comments="", fmt=["%d", "%.7f", "%.7f"])

    schools = load_schools(a.schools, a.school_id_col, a.school_lat_col, a.school_lon_col)
    scale = math.cos(math.radians(lat.mean()))
    tree = cKDTree(np.c_[lat, lon * scale])
    _, node = tree.query(np.array([[s[1], s[2] * scale] for s in schools]))
    snap = haversine(lon[node], lat[node], np.array([s[2] for s in schools]), np.array([s[1] for s in schools]))
    ok = snap <= a.max_snap_m
    pool = [(s, int(nd), float(sd)) for s, nd, sd, k in zip(schools, node, snap, ok) if k]
    write_u32(os.path.join(a.out, "schools.bin"), [p[1] for p in pool])
    with open(os.path.join(a.out, "schools_map.csv"), "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["pool_idx", "school_id", "lat", "lon", "node", "snap_m"])
        for i, (s, nd, sd) in enumerate(pool):
            out.writerow([i, s[0], s[1], s[2], nd, f"{sd:.1f}"])

    meta = {
        "nodes": n, "edges": int(len(src)), "raw_nodes_in_extract": raw_nodes,
        "schools_in_list": len(schools), "schools_in_pool": len(pool),
        "schools_dropped_outside_graph_or_snap": len(schools) - len(pool),
        "highways": a.highways, "ignore_oneway": a.ignore_oneway,
        "centre": a.centre, "bbox": a.bbox, "target_nodes": a.target_nodes, "max_snap_m": a.max_snap_m,
        "pbf_file": os.path.basename(a.pbf), "pbf_sha256": sha256(a.pbf),
        "schools_file": os.path.basename(a.schools), "schools_sha256": sha256(a.schools),
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
