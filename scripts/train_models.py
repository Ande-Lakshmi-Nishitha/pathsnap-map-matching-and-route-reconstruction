import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT
from pathsnap.config import CFG
from pathsnap.data import load_partition, training_rows, xy
from pathsnap.features import FEATURES
from pathsnap.models import registry
from pathsnap.metrics import evaluate, candidate_metrics, top1_arcs, observation_top1

OUT = os.path.join(ROOT, "results", "models")


def cap_rows(X, y, groups, max_rows, seed):
    if max_rows is None or len(y) <= max_rows:
        return X, y, groups
    idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
    return X[idx], y[idx], groups[idx]


def run_cv(name, spec, tr, pts, seed):
    X, y = xy(tr)
    groups = tr["trip"].values
    rows = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=CFG["training"]["cv_folds"]).split(X, y, groups)):
        Xa, ya, _ = cap_rows(X[a], y[a], groups[a], spec.get("max_rows"), seed + k)
        m = spec["make"]()
        t0 = time.time()
        m.fit(Xa, ya)
        fit_s = time.time() - t0
        prob = m.predict_proba(X[b])[:, 1]
        cm = candidate_metrics(y[b], prob)
        sel = top1_arcs(tr.iloc[b], prob)
        rows.append(dict(model=name, fold=k, n_train_rows=len(ya), n_val_rows=len(b), fit_seconds=fit_s,
                         f1_macro=cm["f1_macro"], precision_macro=cm["precision_macro"], recall_macro=cm["recall_macro"],
                         log_loss=cm["log_loss"], top1_accuracy=observation_top1(sel, pts)))
        print(f"  {name} fold {k}: f1={cm['f1_macro']:.4f} top1={rows[-1]['top1_accuracy']:.4f} ({fit_s:.1f}s)", flush=True)
    return rows


def fit_final(name, spec, tr, te, te_pts, seed):
    X, y = xy(tr)
    X, y, _ = cap_rows(X, y, tr["trip"].values, spec.get("max_rows"), seed)
    m = spec["make"]()
    t0 = time.time()
    m.fit(X, y)
    fit_s = time.time() - t0
    Xt, _ = xy(te)
    t0 = time.time()
    prob = m.predict_proba(Xt)[:, 1]
    inf_s = time.time() - t0
    res = evaluate(te, prob, te_pts)
    res["fit_seconds"] = fit_s
    res["inference_ms_per_1000_obs"] = 1000 * inf_s / te_pts.shape[0] * 1000
    return m, prob, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    seed = CFG["seed"]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    c_tr, p_tr = load_partition("train")
    c_te, p_te = load_partition("student_test")
    tr = training_rows(c_tr, p_tr, CFG["training"]["point_stride"])
    print(f"training rows {len(tr)} from {tr[['trip','point']].drop_duplicates().shape[0]} points", flush=True)
    reg = registry(seed)
    names = a.models or list(reg)
    for name in names:
        spec = reg[name]
        print(f"== {name}", flush=True)
        cv = run_cv(name, spec, tr, p_tr, seed)
        pd.DataFrame(cv).to_csv(os.path.join(OUT, f"cv_{name}.csv"), index=False)
        m, prob, res = fit_final(name, spec, tr, c_te, p_te, seed)
        cvdf = pd.DataFrame(cv)
        res.update(model=name, family=spec["family"], config=spec["config"],
                   cv_f1_macro_mean=cvdf["f1_macro"].mean(), cv_f1_macro_std=cvdf["f1_macro"].std(),
                   cv_top1_mean=cvdf["top1_accuracy"].mean(), cv_log_loss_mean=cvdf["log_loss"].mean())
        with open(os.path.join(OUT, f"test_{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        np.save(os.path.join(OUT, f"test_prob_{name}.npy"), prob)
        joblib.dump({"model": m, "features": FEATURES, "name": name}, os.path.join(ROOT, "models", f"{name}.joblib"), compress=3)
        print(f"  student-test top1={res['top1_accuracy']:.4f} f1={res['f1_macro']:.4f} ll={res['log_loss']:.4f} cont={res['route_continuity']:.3f}", flush=True)


if __name__ == "__main__":
    main()
