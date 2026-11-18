import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT
from pathsnap.config import CFG
from pathsnap.data import load_partition, training_rows, xy
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, top1_arcs, observation_top1, evaluate

OUT = os.path.join(ROOT, "results", "hpo")

SPACE = {
    "num_leaves": ("int_log", 15, 255),
    "learning_rate": ("float_log", 0.01, 0.2),
    "n_estimators": ("int", 100, 800),
    "min_child_samples": ("int_log", 10, 300),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "reg_lambda": ("float_log", 1e-3, 10.0),
    "reg_alpha": ("float_log", 1e-3, 10.0),
}


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SPACE.items():
        if kind == "int":
            p[k] = trial.suggest_int(k, lo, hi)
        elif kind == "int_log":
            p[k] = trial.suggest_int(k, lo, hi, log=True)
        elif kind == "float":
            p[k] = trial.suggest_float(k, lo, hi)
        else:
            p[k] = trial.suggest_float(k, lo, hi, log=True)
    return p


def make(params, seed):
    return LGBMClassifier(**params, subsample_freq=1, random_state=seed, n_jobs=2, verbose=-1)


def cv_score(params, tr, pts, seed, trial=None):
    X, y = xy(tr)
    g = tr["trip"].values
    top1s, f1s, lls = [], [], []
    for k, (a, b) in enumerate(GroupKFold(n_splits=CFG["training"]["cv_folds"]).split(X, y, g)):
        m = make(params, seed)
        m.fit(X[a], y[a])
        prob = m.predict_proba(X[b])[:, 1]
        cm = candidate_metrics(y[b], prob)
        top1s.append(observation_top1(top1_arcs(tr.iloc[b], prob), pts))
        f1s.append(cm["f1_macro"])
        lls.append(cm["log_loss"])
        if trial is not None:
            trial.report(float(np.mean(top1s)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(top1s)), float(np.mean(f1s)), float(np.mean(lls)), float(np.std(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    seed = CFG["seed"]
    os.makedirs(OUT, exist_ok=True)
    c_tr, p_tr = load_partition("train")
    c_te, p_te = load_partition("student_test")
    tr = training_rows(c_tr, p_tr, CFG["training"]["point_stride"])

    def objective(trial):
        params = suggest(trial)
        t0 = time.time()
        top1, f1, ll, f1sd = cv_score(params, tr, p_tr, seed, trial)
        trial.set_user_attr("cv_f1_macro", f1)
        trial.set_user_attr("cv_f1_std", f1sd)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1={top1:.4f} f1={f1:.4f} ll={ll:.4f} {params}", flush=True)
        return top1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                study_name="lightgbm_tpe")
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials)
    total = time.time() - t0
    study.trials_dataframe().to_csv(os.path.join(OUT, "optuna_trials.csv"), index=False)
    joblib.dump(study, os.path.join(OUT, "optuna_study.pkl"))

    best = study.best_params
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8)
    rows = {}
    for tag, params in [("untuned", base), ("tuned", best)]:
        top1, f1, ll, f1sd = cv_score(params, tr, p_tr, seed)
        X, y = xy(tr)
        m = make(params, seed)
        t1 = time.time()
        m.fit(X, y)
        fit_s = time.time() - t1
        Xt, _ = xy(c_te)
        t1 = time.time()
        prob = m.predict_proba(Xt)[:, 1]
        inf_s = time.time() - t1
        res = evaluate(c_te, prob, p_te)
        rows[tag] = dict(params=params, cv_top1=top1, cv_f1_macro=f1, cv_f1_std=f1sd, cv_log_loss=ll,
                         test_top1=res["top1_accuracy"], test_f1_macro=res["f1_macro"], test_log_loss=res["log_loss"],
                         test_precision_macro=res["precision_macro"], test_recall_macro=res["recall_macro"],
                         test_route_continuity=res["route_continuity"], fit_seconds=fit_s,
                         inference_ms_per_1000_obs=1e6 * inf_s / len(p_te), confusion=res["confusion"])
        if tag == "tuned":
            np.save(os.path.join(OUT, "test_prob_tuned.npy"), prob)
            joblib.dump({"model": m, "features": FEATURES, "name": "lightgbm_tuned", "params": params},
                        os.path.join(ROOT, "models", "final_model.joblib"), compress=3)
        print(tag, json.dumps({k: v for k, v in rows[tag].items() if k != "params"}), flush=True)
    rows["search"] = dict(space=SPACE, sampler="TPE (Optuna)", pruner="MedianPruner", objective="mean 5-fold GroupKFold observation Top-1 accuracy",
                          n_trials=len(study.trials), n_pruned=sum(t.state.name == "PRUNED" for t in study.trials),
                          total_seconds=total, best_value=study.best_value, best_trial=study.best_trial.number)
    with open(os.path.join(OUT, "tuned_vs_untuned.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)


if __name__ == "__main__":
    main()
