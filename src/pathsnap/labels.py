import numpy as np
from .io import point_seg_dist


def derive_labels(trip, route):
    f, to = trip.arc_from[route], trip.arc_to[route]
    x0, y0, x1, y1 = trip.nx[f], trip.ny[f], trip.nx[to], trip.ny[to]
    N, M = trip.n, len(route)
    back = np.empty((N, M), dtype=np.int32)
    cost, _ = point_seg_dist(trip.x[0], trip.y[0], x0, y0, x1, y1)
    idx = np.arange(M)
    for i in range(1, N):
        d, _ = point_seg_dist(trip.x[i], trip.y[i], x0, y0, x1, y1)
        best = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost == best, idx, 0))
        back[i] = arg
        cost = d + best
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(cost))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    dist, _ = point_seg_dist(trip.x, trip.y, x0[j], y0[j], x1[j], y1[j])
    return route[j], dist, j


def route_connectivity(trip, route):
    if len(route) < 2:
        return 1.0
    return float(np.mean(trip.arc_to[route[:-1]] == trip.arc_from[route[1:]]))
