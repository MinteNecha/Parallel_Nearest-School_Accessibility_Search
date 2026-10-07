"""Binary formats read by the C program (little endian, uint32 fields).

graph.bin:  "NSG1", n, m, xadj[n+1], adj[m], w[m]   (CSR of the SEARCH graph)
schools.bin: count, node[count]
"""
import struct

import numpy as np


def build_csr(n, src, dst, w):
    order = np.argsort(src, kind="stable")
    xadj = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum(np.bincount(src, minlength=n), out=xadj[1:])
    return xadj, dst[order].astype(np.uint32), w[order].astype(np.uint32)


def write_graph(path, n, src, dst, w):
    xadj, adj, wt = build_csr(n, np.asarray(src), np.asarray(dst), np.asarray(w))
    with open(path, "wb") as f:
        f.write(b"NSG1" + struct.pack("<II", n, len(adj)))
        for a in (xadj, adj, wt):
            f.write(a.astype("<u4").tobytes())


def write_u32(path, values):
    values = np.asarray(values, dtype="<u4")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(values)) + values.tobytes())
