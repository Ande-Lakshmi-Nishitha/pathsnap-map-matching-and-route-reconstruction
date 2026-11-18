import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT, DATA_DIR, Trip, list_trips, load_route, trip_file
from pathsnap.labels import derive_labels, route_connectivity

OUT = os.path.join(ROOT, "results", "eda")
FIG = os.path.join(ROOT, "figures", "eda")


def inventory(partition):
    folder = os.path.join(DATA_DIR, partition)
    rows = []
    for t in list_trips(partition):
        raw = np.loadtxt(trip_file(folder, t, "track"), ndmin=2)
        tr = Trip(folder, t)
        route = load_route(folder, t)
        dt = np.diff(tr.t)
        step = np.hypot(np.diff(tr.x), np.diff(tr.y))
        n_nodes, n_arcs = len(tr.nx), len(tr.arc_from)
        g = coo_matrix((np.ones(n_arcs), (tr.arc_from, tr.arc_to)), shape=(n_nodes, n_nodes))
        n_weak, _ = connected_components(g, directed=True, connection="weak")
        n_strong, lab_s = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(tr.arc_from, minlength=n_nodes) + np.bincount(tr.arc_to, minlength=n_nodes)
        pairs = set(zip(tr.arc_from.tolist(), tr.arc_to.tolist()))
        lab_arc, lab_dist, _ = derive_labels(tr, route)
        sizes = {f: os.path.getsize(trip_file(folder, t, f)) for f in ["track", "nodes", "arcs", "route"]}
        rows.append(dict(
            partition=partition, trip=t, n_points=len(raw), n_points_clean=tr.n, dropped_points=tr.dropped_points,
            nan_values=int(np.isnan(raw).sum()), duration_s=float(tr.t[-1] - tr.t[0]),
            dt_median=float(np.median(dt)), dt_max=float(dt.max()), frac_dt_1s=float(np.mean(dt == 1)),
            gaps_over_10s=int((dt > 10).sum()), max_step_m=float(step.max()), speed_p99_kmh=float(np.percentile(step / np.maximum(dt, 1), 99) * 3.6),
            track_km=float(step.sum() / 1000), lon_min=float(tr.lon.min()), lon_max=float(tr.lon.max()),
            lat_min=float(tr.lat.min()), lat_max=float(tr.lat.max()), lon_c=float(np.median(tr.lon)), lat_c=float(np.median(tr.lat)),
            n_nodes=n_nodes, n_arcs=n_arcs, isolated_nodes=int((deg == 0).sum()), weak_components=n_weak,
            strong_components=n_strong, largest_scc_frac=float(np.bincount(lab_s).max() / n_nodes),
            twoway_frac=float(np.mean([(b, a) in pairs for a, b in zip(tr.arc_from.tolist(), tr.arc_to.tolist())])),
            self_loops=int((tr.arc_from == tr.arc_to).sum()), route_arcs=len(route), route_unique=len(set(route.tolist())),
            route_km=float(tr.arc_len[route].sum() / 1000), route_connectivity=route_connectivity(tr, route),
            label_dist_median=float(np.median(lab_dist)), label_dist_p95=float(np.percentile(lab_dist, 95)),
            frac_label_over_50m=float(np.mean(lab_dist > 50)), **{f"bytes_{k}": v for k, v in sizes.items()}))
        print(partition, t, flush=True)
    return pd.DataFrame(rows)


def plots(inv, pts):
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, g in inv.groupby("partition"):
        ax[0].hist(g["n_points_clean"], bins=25, alpha=0.6, label=p)
        ax[1].hist(g["track_km"], bins=25, alpha=0.6, label=p)
        ax[2].hist(np.log10(g["n_arcs"]), bins=25, alpha=0.6, label=p)
    ax[0].set(title="GPS points per trip", xlabel="points", ylabel="trips")
    ax[1].set(title="Trajectory length", xlabel="km")
    ax[2].set(title="Road-network size", xlabel="log10(#arcs)")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trip_size_distributions.png"), dpi=130); plt.close(fig)

    tr = pts[pts["partition"] == "train"]
    dt = tr.groupby("trip")["t"].diff().dropna()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    vc = dt.clip(upper=11).value_counts().sort_index()
    ax[0].bar(vc.index.astype(int).astype(str).str.replace("11", ">10"), vc.values)
    ax[0].set_yscale("log"); ax[0].set(title="Sampling interval (train)", xlabel="seconds", ylabel="count (log)")
    ax[1].hist(tr["label_dist"].clip(upper=200), bins=80)
    ax[1].axvline(50, color="r", ls="--", label="50 m validity threshold")
    ax[1].set_yscale("log"); ax[1].set(title="GPS-to-reference-arc distance (train)", xlabel="metres (clipped 200)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sampling_and_label_distance.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for p, g in inv.groupby("partition"):
        ax.scatter(g["lon_c"], g["lat_c"], s=30, label=p, alpha=0.8)
    ax.set(title="Trip locations (median GPS position)", xlabel="longitude", ylabel="latitude")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trip_locations.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    s = inv[inv["partition"] == "train"].sort_values("frac_label_over_50m", ascending=False).head(15)
    ax.bar(s["trip"], s["frac_label_over_50m"])
    ax.set(title="Share of GPS points > 50 m from reference route (worst train trips)", ylabel="fraction")
    plt.xticks(rotation=60); fig.tight_layout(); fig.savefig(os.path.join(FIG, "reference_route_coverage.png"), dpi=130); plt.close(fig)


def zoom_plot(partition, t, i, half=500.0):
    folder = os.path.join(DATA_DIR, partition)
    tr = Trip(folder, t)
    route = load_route(folder, t)
    cx, cy = tr.x[i], tr.y[i]
    near = lambda a: abs(tr.nx[tr.arc_from[a]] - cx) < half * 1.5 and abs(tr.ny[tr.arc_from[a]] - cy) < half * 1.5
    fig, ax = plt.subplots(figsize=(7, 7))
    for a in tr.local_arcs:
        if near(a):
            ax.plot([tr.nx[tr.arc_from[a]], tr.nx[tr.arc_to[a]]], [tr.ny[tr.arc_from[a]], tr.ny[tr.arc_to[a]]], color="0.7", lw=0.8)
    for a in route:
        if near(a):
            ax.plot([tr.nx[tr.arc_from[a]], tr.nx[tr.arc_to[a]]], [tr.ny[tr.arc_from[a]], tr.ny[tr.arc_to[a]]], color="g", lw=3)
    ax.scatter(tr.x, tr.y, s=6, c="b", zorder=3)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal")
    ax.set(title=f"{partition}/{t} zoom around point {i}", xlabel="UTM x (m)", ylabel="UTM y (m)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"zoom_{partition}_{t}_{i}.png"), dpi=130); plt.close(fig)


def example_plot(partition, t):
    folder = os.path.join(DATA_DIR, partition)
    tr = Trip(folder, t)
    route = load_route(folder, t)
    fig, ax = plt.subplots(figsize=(8, 8))
    for a in tr.local_arcs[:: max(1, len(tr.local_arcs) // 20000)]:
        ax.plot([tr.nx[tr.arc_from[a]], tr.nx[tr.arc_to[a]]], [tr.ny[tr.arc_from[a]], tr.ny[tr.arc_to[a]]], color="0.8", lw=0.5)
    for a in route:
        ax.plot([tr.nx[tr.arc_from[a]], tr.nx[tr.arc_to[a]]], [tr.ny[tr.arc_from[a]], tr.ny[tr.arc_to[a]]], color="g", lw=2)
    ax.scatter(tr.x, tr.y, s=3, c="b", zorder=3)
    ax.set_aspect("equal"); ax.set(title=f"{partition}/{t}: road network (grey), reference route (green), GPS (blue)", xlabel="UTM x (m)", ylabel="UTM y (m)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"trajectory_{partition}_{t}.png"), dpi=130); plt.close(fig)


def overview_map(inv):
    m = folium.Map(location=[30, 10], zoom_start=2)
    colors = {"train": "blue", "student_test": "red"}
    for _, r in inv.iterrows():
        folium.CircleMarker([r["lat_c"], r["lon_c"]], radius=5, color=colors[r["partition"]], fill=True,
                            tooltip=f"{r['partition']} {r['trip']}: {r['n_points_clean']} pts, {r['track_km']:.1f} km").add_to(m)
    os.makedirs(os.path.join(ROOT, "maps"), exist_ok=True)
    m.save(os.path.join(ROOT, "maps", "eda_trip_locations.html"))


def main():
    os.makedirs(OUT, exist_ok=True)
    inv = pd.concat([inventory("train"), inventory("student_test")], ignore_index=True)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    pts = []
    for p in ["train", "student_test"]:
        q = pd.read_pickle(os.path.join(ROOT, "results", "data", f"{p}_points.pkl"))
        q["partition"] = p
        pts.append(q)
    pts = pd.concat(pts, ignore_index=True)
    summary = inv.groupby("partition").agg(
        trips=("trip", "count"), points=("n_points", "sum"), points_mean=("n_points", "mean"), points_min=("n_points", "min"),
        points_max=("n_points", "max"), nan_values=("nan_values", "sum"), dropped_points=("dropped_points", "sum"),
        frac_dt_1s=("frac_dt_1s", "mean"), trips_with_gap_over_10s=("gaps_over_10s", lambda s: int((s > 0).sum())),
        km_total=("track_km", "sum"), nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"),
        isolated_nodes=("isolated_nodes", "sum"), weak_components_mean=("weak_components", "mean"),
        largest_scc_frac_mean=("largest_scc_frac", "mean"), twoway_frac_mean=("twoway_frac", "mean"),
        route_connectivity_mean=("route_connectivity", "mean"), label_dist_median=("label_dist_median", "median"),
        trips_with_partial_route=("frac_label_over_50m", lambda s: int((s > 0.2).sum())))
    summary.T.to_csv(os.path.join(OUT, "summary.csv"))
    miss = pts.groupby("partition").agg(points=("point", "size"), label_valid=("label_valid", "mean"),
                                         label_in_candidates=("label_in_cands", "mean"))
    miss.to_csv(os.path.join(OUT, "label_coverage.csv"))
    print(summary.T.to_string())
    plots(inv, pts)
    example_plot("train", "00000048")
    example_plot("train", "00000054")
    zoom_plot("train", "00000054", 6000)
    zoom_plot("student_test", "00000077", 1500)
    overview_map(inv)


if __name__ == "__main__":
    main()
