import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT
from pathsnap.models import registry

R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures", "models")


def model_table():
    rows = []
    for name in registry():
        with open(os.path.join(R, "models", f"test_{name}.json")) as f:
            r = json.load(f)
        cv = pd.read_csv(os.path.join(R, "models", f"cv_{name}.csv"))
        rows.append(dict(model=name, family=r["family"], config=r["config"],
                         cv_f1_macro=cv["f1_macro"].mean(), cv_f1_std=cv["f1_macro"].std(), cv_top1=cv["top1_accuracy"].mean(),
                         cv_log_loss=cv["log_loss"].mean(), test_top1=r["top1_accuracy"], test_f1_macro=r["f1_macro"],
                         test_precision_macro=r["precision_macro"], test_recall_macro=r["recall_macro"], test_log_loss=r["log_loss"],
                         test_route_continuity=r["route_continuity"], candidate_miss_rate=r["candidate_miss_rate"],
                         fit_seconds=r["fit_seconds"], inference_ms_per_1000_obs=r["inference_ms_per_1000_obs"],
                         tn=r["confusion"][0][0], fp=r["confusion"][0][1], fn=r["confusion"][1][0], tp=r["confusion"][1][1]))
    return pd.DataFrame(rows)


def bar(ax, names, vals, title, ylabel, err=None, lo=None):
    ax.bar(names, vals, yerr=err, capsize=4, color="#4C72B0")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(title=title, ylabel=ylabel)
    if lo is not None:
        ax.set_ylim(lo, None)
    ax.tick_params(axis="x", rotation=40)


def main():
    os.makedirs(FIG, exist_ok=True)
    t = model_table()
    t.to_csv(os.path.join(R, "model_comparison.csv"), index=False)
    print(t.round(4).to_string())
    n = t["model"].tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    bar(ax, n, t["cv_f1_macro"].values, "5-fold GroupKFold CV macro F1 (train, 80 trips)", "macro F1", t["cv_f1_std"].values, 0.7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_f1_comparison.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    bar(ax[0], n, t["test_top1"].values, "Student-test observation Top-1 accuracy", "accuracy", lo=0.7)
    bar(ax[1], n, t["test_f1_macro"].values, "Student-test candidate macro F1", "macro F1", lo=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "student_test_top1_f1.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    bar(ax[0], n, t["test_log_loss"].values, "Student-test log loss (lower is better)", "log loss")
    bar(ax[1], n, t["inference_ms_per_1000_obs"].values, "Inference time", "ms per 1000 GPS observations")
    ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "logloss_inference.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (_, r) in zip(axes.ravel(), t.iterrows()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["wrong arc", "true arc"]); ax.set_yticks([0, 1], ["wrong arc", "true arc"])
        ax.set(title=r["model"], xlabel="predicted", ylabel="actual")
    fig.suptitle("Student-test candidate confusion matrices (threshold 0.5)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_matrices.png"), dpi=130); plt.close(fig)

    cvs = pd.concat([pd.read_csv(os.path.join(R, "models", f"cv_{m}.csv")) for m in n])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([cvs[cvs["model"] == m]["top1_accuracy"].values for m in n], tick_labels=n)
    ax.set(title="CV fold stability: observation Top-1 accuracy per fold", ylabel="Top-1 accuracy")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_fold_stability.png"), dpi=130); plt.close(fig)

    hp = os.path.join(R, "hpo", "optuna_trials.csv")
    if os.path.exists(hp):
        tr = pd.read_csv(hp)
        done = tr[tr["state"] == "COMPLETE"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].scatter(done["number"], done["value"], label="completed trial")
        ax[0].plot(done["number"], done["value"].cummax(), color="r", label="best so far")
        ax[0].set(title="Optuna TPE optimisation history (LightGBM)", xlabel="trial", ylabel="CV Top-1 accuracy"); ax[0].legend()
        ax[1].scatter(done["params_learning_rate"], done["value"], c=done["params_num_leaves"], cmap="viridis")
        ax[1].set_xscale("log"); ax[1].set(title="Trials: learning rate vs score (colour = num_leaves)", xlabel="learning_rate", ylabel="CV Top-1")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "bayes_opt_history.png"), dpi=130); plt.close(fig)
        with open(os.path.join(R, "hpo", "tuned_vs_untuned.json")) as f:
            tv = json.load(f)
        keys = ["cv_f1_macro", "test_top1", "test_f1_macro", "test_log_loss"]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [tv["untuned"][k] for k in keys], 0.4, label="untuned")
        ax.bar(x + 0.2, [tv["tuned"][k] for k in keys], 0.4, label="tuned (TPE)")
        ax.set_xticks(x, keys); ax.legend(); ax.set(title="LightGBM untuned vs Bayesian-tuned")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "tuned_vs_untuned.png"), dpi=130); plt.close(fig)

    hs = os.path.join(R, "hmm", "student_test_summary.csv")
    if os.path.exists(hs):
        s = pd.read_csv(hs)
        piv = s.pivot(index="model", columns="mode", values="top1").loc[n + ["lightgbm_tuned"]]
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        x = np.arange(len(piv))
        ax[0].bar(x - 0.2, piv["classifier_only"], 0.4, label="classifier only")
        ax[0].bar(x + 0.2, piv["classifier_hmm"], 0.4, label="classifier + HMM/Viterbi")
        ax[0].set_xticks(x, piv.index, rotation=40); ax[0].set_ylim(0.7, 1); ax[0].legend(loc="lower right")
        ax[0].set(title="Student-test Top-1 accuracy", ylabel="accuracy")
        pc = s.pivot(index="model", columns="mode", values="continuity").loc[piv.index]
        ax[1].bar(x - 0.2, pc["classifier_only"], 0.4, label="classifier only")
        ax[1].bar(x + 0.2, pc["classifier_hmm"], 0.4, label="classifier + HMM/Viterbi")
        ax[1].set_xticks(x, piv.index, rotation=40); ax[1].set_ylim(0.5, 1); ax[1].legend(loc="lower right")
        ax[1].set(title="Route continuity (directly connected consecutive arcs)", ylabel="fraction")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "hmm_vs_classifier.png"), dpi=130); plt.close(fig)
        pt = pd.read_csv(os.path.join(R, "hmm", "student_test_per_trip.csv"))
        pt = pt[pt["model"] == "lightgbm_tuned"].pivot(index="trip", columns="mode", values="top1")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(pt))
        ax.bar(x - 0.2, pt["classifier_only"], 0.4, label="classifier only")
        ax.bar(x + 0.2, pt["classifier_hmm"], 0.4, label="classifier + HMM")
        ax.set_xticks(x, pt.index, rotation=40); ax.set_ylim(0.6, 1); ax.legend()
        ax.set(title="Per-trip Top-1 accuracy, final model (student test)", ylabel="accuracy")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "hmm_per_trip.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
