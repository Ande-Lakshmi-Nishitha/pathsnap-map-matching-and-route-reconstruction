import numpy as np
import pandas as pd
from .io import point_seg_dist

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "proj_t", "log_arc_len", "angle_diff", "cos_angle", "moving", "speed",
    "win_dist_mean", "win_dist_max", "win_within_15",
    "prev_conn", "next_conn", "has_twin", "gap_prev", "out_degree", "in_degree",
]


def gps_heading(trip, w=2):
    n = trip.n
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    dx, dy = trip.x[hi] - trip.x[lo], trip.y[hi] - trip.y[lo]
    disp = np.hypot(dx, dy)
    dt = np.maximum(trip.t[hi] - trip.t[lo], 1e-6)
    return np.degrees(np.arctan2(dx, dy)) % 360, disp, disp / dt


def build_features(trip, cand, dist, proj, win=5):
    n = trip.n
    heading, disp, speed = gps_heading(trip)
    nearest = np.array([c[0] for c in cand])
    outdeg = np.bincount(trip.arc_from, minlength=len(trip.nx))
    indeg = np.bincount(trip.arc_to, minlength=len(trip.nx))
    gap = np.r_[0.0, np.diff(trip.t)]
    rows = []
    for i in range(n):
        a, d, t = cand[i], dist[i], proj[i]
        k = len(a)
        lo, hi = max(0, i - win), min(n, i + win + 1)
        px, py = trip.x[lo:hi, None], trip.y[lo:hi, None]
        wd, _ = point_seg_dist(px, py, trip.nx[trip.arc_from[a]][None], trip.ny[trip.arc_from[a]][None],
                               trip.nx[trip.arc_to[a]][None], trip.ny[trip.arc_to[a]][None])
        diff = np.abs((trip.arc_bearing[a] - heading[i] + 180) % 360 - 180)
        moving = disp[i] > 3.0
        if not moving:
            diff = np.full(k, 90.0)
        pv = nearest[i - 1] if i > 0 else -1
        nx_ = nearest[i + 1] if i < n - 1 else -1
        prev_conn = (a == pv) | ((trip.arc_from[a] == trip.arc_to[pv]) if pv >= 0 else False)
        next_conn = (a == nx_) | ((trip.arc_to[a] == trip.arc_from[nx_]) if nx_ >= 0 else False)
        pairs = set(zip(trip.arc_from[a].tolist(), trip.arc_to[a].tolist()))
        twin = np.array([(int(trip.arc_to[x]), int(trip.arc_from[x])) in pairs for x in a])
        rows.append(np.c_[
            np.full(k, i), a, d, np.arange(k), d - d.min(), d / (d.min() + 1.0), np.full(k, k),
            t, np.log1p(trip.arc_len[a]), diff, np.cos(np.radians(diff)), np.full(k, float(moving)),
            np.full(k, speed[i]), wd.mean(0), wd.max(0), (wd <= 15).mean(0),
            prev_conn.astype(float), next_conn.astype(float), twin.astype(float), np.full(k, gap[i]),
            outdeg[trip.arc_to[a]], indeg[trip.arc_from[a]],
        ])
    df = pd.DataFrame(np.vstack(rows), columns=["point", "arc"] + FEATURES)
    df["point"] = df["point"].astype(np.int64)
    df["arc"] = df["arc"].astype(np.int64)
    df["fnode"] = trip.arc_from[df["arc"].values]
    df["tnode"] = trip.arc_to[df["arc"].values]
    return df
