import numpy as np
from .graph import RoadGraph


def group_candidates(df, prob, n_points):
    d = df[["point", "arc", "proj_t"]].copy()
    d["p"] = prob
    d = d.sort_values(["point", "arc"])
    cands, projs, probs = [None] * n_points, [None] * n_points, [None] * n_points
    for pt, g in d.groupby("point", sort=True):
        cands[pt], projs[pt], probs[pt] = g["arc"].values, g["proj_t"].values, g["p"].values
    return cands, projs, probs


def transition_distances(trip, graph, cands, projs, cfg):
    n = trip.n
    tol = cfg["backward_tolerance"]
    out, gcs = [None] * n, np.zeros(n)
    for i in range(1, n):
        a, b = cands[i - 1], cands[i]
        la = trip.arc_len[a]
        pa, pb = projs[i - 1] * la, projs[i] * trip.arc_len[b]
        gc = float(np.hypot(trip.x[i] - trip.x[i - 1], trip.y[i] - trip.y[i - 1]))
        gcs[i] = gc
        bound = cfg["max_route_factor"] * gc + cfg["route_slack"]
        R = np.full((len(a), len(b)), np.inf)
        bf = trip.arc_from[b]
        for u, au in enumerate(a):
            dist, _ = graph.dijkstra(int(trip.arc_to[au]), bound)
            rem = la[u] - pa[u]
            for v, bv in enumerate(b):
                if bv == au:
                    diff = pb[v] - pa[u]
                    R[u, v] = max(diff, 0.0) if diff >= -tol else -diff + (-diff - tol)
                else:
                    d = dist.get(int(bf[v]))
                    if d is not None:
                        R[u, v] = rem + d + pb[v]
        out[i] = R
    return out, gcs


def viterbi(trip, probs, R, gcs, cfg):
    n = trip.n
    w, beta, gap_s = cfg["emission_weight"], cfg["beta"], cfg["gap_seconds"]
    emis = [w * np.log(np.clip((p + 1e-6) / (p + 1e-6).sum(), 1e-6, 1.0)) for p in probs]
    delta = emis[0].copy()
    back = [None] * n
    breaks = []
    for i in range(1, n):
        dt = trip.t[i] - trip.t[i - 1]
        b_eff = beta * max(1.0, dt / gap_s) ** 0.5 if dt > 1 else beta
        with np.errstate(invalid="ignore"):
            logT = -np.abs(R[i] - gcs[i]) / b_eff
        unreachable = ~np.isfinite(R[i])
        logT[unreachable] = -cfg["unreachable_penalty"]
        if unreachable.all():
            breaks.append(i)
        sc = delta[:, None] + logT
        back[i] = sc.argmax(0)
        delta = sc.max(0) + emis[i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = int(back[i][path[i]])
    return path, breaks


def stitch_route(trip, graph, arcs, max_bridge=3000.0):
    seq = [int(arcs[0])]
    for a in arcs[1:]:
        if int(a) != seq[-1]:
            seq.append(int(a))
    route, gaps = [seq[0]], 0
    for a in seq[1:]:
        prev = route[-1]
        if trip.arc_to[prev] != trip.arc_from[a]:
            p = graph.path_arcs(int(trip.arc_to[prev]), int(trip.arc_from[a]), max_bridge)
            if p is None:
                gaps += 1
            else:
                route.extend(p)
        route.append(a)
    return route, seq, gaps


def decode(trip, df, prob, cfg, graph=None, cache=None):
    graph = graph or RoadGraph(trip)
    cands, projs, probs = group_candidates(df, prob, trip.n)
    if cache is not None and "R" in cache:
        R, gcs = cache["R"], cache["gcs"]
    else:
        R, gcs = transition_distances(trip, graph, cands, projs, cfg)
        if cache is not None:
            cache.update(R=R, gcs=gcs, cands=cands)
    path, breaks = viterbi(trip, probs, R, gcs, cfg)
    arcs = np.array([cands[i][path[i]] for i in range(trip.n)])
    return arcs, breaks, graph
