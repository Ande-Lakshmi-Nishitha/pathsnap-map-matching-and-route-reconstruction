import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import Trip, list_trips, load_route, DATA_DIR, ROOT
from pathsnap.labels import derive_labels
from pathsnap.candidates import ArcIndex, generate_candidates
from pathsnap.features import build_features
from pathsnap.config import CFG


def process(args):
    partition, trip_id = args
    t0 = time.time()
    folder = os.path.join(DATA_DIR, partition)
    trip = Trip(folder, trip_id)
    route = load_route(folder, trip_id)
    label_arc, label_dist, label_idx = derive_labels(trip, route)
    idx = ArcIndex(trip, CFG["candidates"]["spacing"])
    cand, dist, proj = generate_candidates(trip, idx, CFG["candidates"]["radius"], CFG["candidates"]["k"])
    df = build_features(trip, cand, dist, proj)
    df["y"] = (df["arc"].values == label_arc[df["point"].values]).astype(np.int8)
    df["trip"] = trip_id
    pts = pd.DataFrame({
        "trip": trip_id, "point": np.arange(trip.n), "t": trip.t, "lon": trip.lon, "lat": trip.lat,
        "label_arc": label_arc, "label_dist": label_dist, "route_pos": label_idx,
    })
    hit = df.groupby("point")["y"].max().reindex(pts["point"]).fillna(0).values
    pts["label_in_cands"] = hit.astype(np.int8)
    pts["label_valid"] = (label_dist <= CFG["labels"]["max_dist"]).astype(np.int8)
    print(f"{partition}/{trip_id}: {trip.n} pts, {len(df)} rows, miss={1 - hit.mean():.3f}, {time.time() - t0:.1f}s", flush=True)
    return df, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "data")
    os.makedirs(out, exist_ok=True)
    for p in a.partitions:
        jobs = [(p, t) for t in list_trips(p)]
        with Pool(a.workers) as pool:
            res = pool.map(process, jobs)
        pd.concat([r[0] for r in res], ignore_index=True).to_pickle(os.path.join(out, f"{p}_candidates.pkl"))
        pd.concat([r[1] for r in res], ignore_index=True).to_pickle(os.path.join(out, f"{p}_points.pkl"))


if __name__ == "__main__":
    main()
