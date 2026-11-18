import numpy as np
from scipy.spatial import cKDTree
from .io import point_seg_dist


class ArcIndex:
    def __init__(self, trip, spacing=20.0):
        a = trip.local_arcs
        x0, y0 = trip.nx[trip.arc_from[a]], trip.ny[trip.arc_from[a]]
        x1, y1 = trip.nx[trip.arc_to[a]], trip.ny[trip.arc_to[a]]
        n = np.maximum(1, np.ceil(trip.arc_len[a] / spacing).astype(int)) + 1
        rep = np.repeat(np.arange(len(a)), n)
        offs = np.concatenate([np.linspace(0, 1, k) for k in n])
        self.sx = x0[rep] + offs * (x1 - x0)[rep]
        self.sy = y0[rep] + offs * (y1 - y0)[rep]
        self.sarc = a[rep]
        self.spacing = spacing
        self.tree = cKDTree(np.c_[self.sx, self.sy])
        self.trip = trip


def generate_candidates(trip, index, radius=50.0, k=8, fallback_k=64):
    tr = trip
    r = np.hypot(radius, index.spacing / 2)
    hits = index.tree.query_ball_point(np.c_[tr.x, tr.y], r)
    cand, dist, proj = [], [], []
    for i, h in enumerate(hits):
        arcs = np.unique(index.sarc[h]) if len(h) else np.empty(0, dtype=np.int64)
        if len(arcs):
            d, t = point_seg_dist(tr.x[i], tr.y[i], tr.nx[tr.arc_from[arcs]], tr.ny[tr.arc_from[arcs]],
                                  tr.nx[tr.arc_to[arcs]], tr.ny[tr.arc_to[arcs]])
            m = d <= radius
            arcs, d, t = arcs[m], d[m], t[m]
        if len(arcs) == 0:
            _, nn = index.tree.query([tr.x[i], tr.y[i]], k=min(fallback_k, len(index.sx)))
            arcs = np.unique(index.sarc[np.atleast_1d(nn)])
            d, t = point_seg_dist(tr.x[i], tr.y[i], tr.nx[tr.arc_from[arcs]], tr.ny[tr.arc_from[arcs]],
                                  tr.nx[tr.arc_to[arcs]], tr.ny[tr.arc_to[arcs]])
        o = np.lexsort((arcs, d))[:k]
        cand.append(arcs[o]); dist.append(d[o]); proj.append(t[o])
    return cand, dist, proj
