import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT, DATA_DIR, Trip, load_route
from pathsnap.config import CFG
from pathsnap.data import load_partition, training_rows, xy
from pathsnap.hmm import decode, stitch_route
from pathsnap.metrics import continuity, route_prf
from pathsnap.models import registry

OUT = os.path.join(ROOT, "results", "hmm")
GRID = {"beta": [3.0, 5.0, 10.0, 20.0, 40.0], "emission_weight": [0.5, 1.0, 2.0], "unreachable_penalty": [15.0, 30.0]}


def classifier_only(df, prob, n):
    d = df[["point", "arc"]].assign(p=prob)
    return d.loc[d.groupby("point")["p"].idxmax()].set_index("point").reindex(range(n))["arc"].values


def score_trip(trip, pts, arcs, ref, graph):
    v = pts["label_valid"].values == 1
    acc = float((arcs[v] == pts["label_arc"].values[v]).mean()) if v.any() else np.nan
    route, seq, gaps = stitch_route(trip, graph, arcs)
    prec, rec, f1 = route_prf(route, ref, trip.arc_len)
    cont = continuity(trip.arc_from[arcs], trip.arc_to[arcs], arcs)
    stitched = 1.0 - gaps / max(1, len(seq) - 1)
    return dict(top1=acc, continuity=cont, stitched_continuity=stitched, route_precision=prec, route_recall=rec, route_f1=f1,
                n_valid=int(v.sum()), n_points=len(arcs))


def oof_probs(c_tr, p_tr, params, seed):
    tr = training_rows(c_tr, p_tr, CFG["training"]["point_stride"])
    trips = np.array(sorted(c_tr["trip"].unique()))
    prob = np.zeros(len(c_tr))
    for a, b in GroupKFold(n_splits=CFG["training"]["cv_folds"]).split(trips, groups=trips):
        fit_rows = tr[tr["trip"].isin(trips[a])]
        X, y = xy(fit_rows)
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(**params, subsample_freq=1, random_state=seed, n_jobs=2, verbose=-1).fit(X, y)
        mask = c_tr["trip"].isin(trips[b]).values
        prob[mask] = m.predict_proba(xy(c_tr[mask])[0])[:, 1]
    return prob


def run_partition(partition, c, p, probs_by_model, cfg, caches):
    folder = os.path.join(DATA_DIR, partition)
    rows = []
    for t in sorted(c["trip"].unique()):
        m = (c["trip"] == t).values
        df, pts = c[m], p[p["trip"] == t].sort_values("point")
        trip = Trip(folder, t)
        ref = load_route(folder, t)
        cache = caches.setdefault((partition, t), {})
        for name, prob in probs_by_model.items():
            pr = prob[m]
            t0 = time.time()
            arcs, breaks, graph = decode(trip, df, pr, cfg, graph=cache.get("graph"), cache=cache)
            cache["graph"] = graph
            dec_s = time.time() - t0
            base = classifier_only(df, pr, trip.n)
            for mode, a in [("classifier_only", base), ("classifier_hmm", arcs)]:
                r = score_trip(trip, pts, a, ref, graph)
                r.update(trip=t, model=name, mode=mode, decode_seconds=dec_s if mode == "classifier_hmm" else 0.0)
                rows.append(r)
    return pd.DataFrame(rows)


def summarise(df):
    w = df.assign(correct=df["top1"] * df["n_valid"])
    g = w.groupby(["model", "mode"]).agg(correct=("correct", "sum"), n_valid=("n_valid", "sum"),
                                         continuity=("continuity", "mean"), stitched_continuity=("stitched_continuity", "mean"),
                                         route_precision=("route_precision", "mean"), route_recall=("route_recall", "mean"),
                                         route_f1=("route_f1", "mean"), decode_seconds=("decode_seconds", "sum"))
    g["top1"] = g["correct"] / g["n_valid"]
    return g.drop(columns=["correct"]).reset_index()


def main():
    seed = CFG["seed"]
    os.makedirs(OUT, exist_ok=True)
    final = joblib.load(os.path.join(ROOT, "models", "final_model.joblib"))
    c_tr, p_tr = load_partition("train")
    c_te, p_te = load_partition("student_test")
    caches = {}

    print("OOF probabilities for decoder tuning", flush=True)
    oof = oof_probs(c_tr, p_tr, final["params"], seed)
    np.save(os.path.join(OUT, "train_oof_prob.npy"), oof)
    tune_trips = sorted(c_tr["trip"].unique())[::3]
    m = c_tr["trip"].isin(tune_trips).values
    ct, pt = c_tr[m].reset_index(drop=True), p_tr[p_tr["trip"].isin(tune_trips)]
    grid_rows = []
    for beta, w, pen in itertools.product(*GRID.values()):
        cfg = dict(CFG["hmm"], beta=beta, emission_weight=w, unreachable_penalty=pen)
        r = run_partition("train", ct, pt, {"final": oof[m]}, cfg, caches)
        s = summarise(r[r["mode"] == "classifier_hmm"]).iloc[0]
        grid_rows.append(dict(beta=beta, emission_weight=w, unreachable_penalty=pen, top1=s["top1"], route_f1=s["route_f1"]))
        print(grid_rows[-1], flush=True)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(OUT, "decoder_grid_train_oof.csv"), index=False)
    best = grid.sort_values(["top1", "route_f1"], ascending=False).iloc[0]
    tuned = dict(CFG["hmm"], beta=float(best["beta"]), emission_weight=float(best["emission_weight"]),
                 unreachable_penalty=float(best["unreachable_penalty"]))
    with open(os.path.join(ROOT, "configs", "hmm_tuned.json"), "w") as f:
        json.dump(tuned, f, indent=2)
    base_train = run_partition("train", ct, pt, {"final": oof[m]}, CFG["hmm"], caches)
    summarise(base_train).to_csv(os.path.join(OUT, "train_oof_default_decoder.csv"), index=False)

    probs = {name: np.load(os.path.join(ROOT, "results", "models", f"test_prob_{name}.npy")) for name in registry()}
    probs["lightgbm_tuned"] = np.load(os.path.join(ROOT, "results", "hpo", "test_prob_tuned.npy"))
    res = run_partition("student_test", c_te, p_te, probs, tuned, caches)
    res.to_csv(os.path.join(OUT, "student_test_per_trip.csv"), index=False)
    summ = summarise(res)
    summ.to_csv(os.path.join(OUT, "student_test_summary.csv"), index=False)
    print(summ.to_string(), flush=True)
    d = run_partition("student_test", c_te, p_te, {"lightgbm_tuned": probs["lightgbm_tuned"]}, CFG["hmm"], caches)
    summarise(d).to_csv(os.path.join(OUT, "student_test_default_decoder.csv"), index=False)


if __name__ == "__main__":
    main()
