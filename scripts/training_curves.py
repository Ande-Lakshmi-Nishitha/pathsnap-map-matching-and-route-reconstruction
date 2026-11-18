import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT
from pathsnap.config import CFG
from pathsnap.data import load_partition, training_rows, xy

FIG = os.path.join(ROOT, "figures", "models")
OUT = os.path.join(ROOT, "results", "models")


def main():
    seed = CFG["seed"]
    os.makedirs(FIG, exist_ok=True)
    c, p = load_partition("train")
    tr = training_rows(c, p, CFG["training"]["point_stride"])
    X, y = xy(tr)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, tr["trip"].values))

    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=seed, n_jobs=2, verbose=-1)
    lgb.fit(X[a], y[a], eval_set=[(X[a], y[a]), (X[b], y[b])], eval_names=["train", "validation"], eval_metric="binary_logloss")
    h = lgb.evals_result_
    lg = pd.DataFrame({"iteration": np.arange(1, 401), "train_logloss": h["train"]["binary_logloss"],
                       "val_logloss": h["validation"]["binary_logloss"]})
    lg.to_csv(os.path.join(OUT, "history_lightgbm.csv"), index=False)

    sc = StandardScaler().fit(X[a])
    Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3, random_state=seed, batch_size=512)
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(1, 41):
        idx = rng.permutation(len(Xa))
        for s in range(0, len(idx), 512):
            mlp.partial_fit(Xa[idx[s:s + 512]], y[a][idx[s:s + 512]], classes=[0, 1])
        rows.append(dict(epoch=ep, train_logloss=log_loss(y[a], mlp.predict_proba(Xa)[:, 1]),
                         val_logloss=log_loss(y[b], mlp.predict_proba(Xb)[:, 1])))
        print(rows[-1], flush=True)
    ml = pd.DataFrame(rows)
    ml.to_csv(os.path.join(OUT, "history_mlp.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(lg["iteration"], lg["train_logloss"], label="train")
    ax[0].plot(lg["iteration"], lg["val_logloss"], label="validation (20% held-out trips)")
    ax[0].set(title="LightGBM: log loss per boosting iteration", xlabel="iteration", ylabel="binary log loss"); ax[0].legend()
    ax[1].plot(ml["epoch"], ml["train_logloss"], label="train")
    ax[1].plot(ml["epoch"], ml["val_logloss"], label="validation (20% held-out trips)")
    ax[1].set(title="MLP (64,32): log loss per epoch", xlabel="epoch", ylabel="binary log loss"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
