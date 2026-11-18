import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT
from pathsnap.config import CFG
from pathsnap.data import load_partition, training_rows, xy
from pathsnap.models import registry
from pathsnap.metrics import candidate_metrics, top1_arcs, observation_top1


def main():
    c, p = load_partition("train")
    tr = training_rows(c, p, CFG["training"]["point_stride"])
    X, y = xy(tr)
    rows = []
    for name in list(registry()) + ["final_model"]:
        b = joblib.load(os.path.join(ROOT, "models", f"{name}.joblib"))
        prob = b["model"].predict_proba(X)[:, 1]
        cm = candidate_metrics(y, prob)
        top1 = observation_top1(top1_arcs(tr, prob), p)
        rows.append(dict(model=name, train_f1_macro=cm["f1_macro"], train_top1=top1, train_log_loss=cm["log_loss"]))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    comp = pd.read_csv(os.path.join(ROOT, "results", "model_comparison.csv"))
    d = d.merge(comp[["model", "cv_f1_macro", "cv_top1", "cv_log_loss", "test_f1_macro", "test_top1", "test_log_loss"]], on="model", how="left")
    d["f1_gap_train_minus_cv"] = d["train_f1_macro"] - d["cv_f1_macro"]
    d.to_csv(os.path.join(ROOT, "results", "models", "fit_diagnostics.csv"), index=False)
    print(d.round(4).to_string())


if __name__ == "__main__":
    main()
