import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, confusion_matrix


def top1_arcs(df, prob):
    d = df[["trip", "point", "arc", "fnode", "tnode"]].copy()
    d["p"] = prob
    idx = d.groupby(["trip", "point"], sort=False)["p"].idxmax()
    return d.loc[idx].sort_values(["trip", "point"]).reset_index(drop=True)


def continuity(fnode, tnode, arcs):
    keep = np.r_[True, arcs[1:] != arcs[:-1]]
    f, t = fnode[keep], tnode[keep]
    if len(f) < 2:
        return 1.0
    return float(np.mean(t[:-1] == f[1:]))


def route_continuity(sel):
    vals = [continuity(g["fnode"].values, g["tnode"].values, g["arc"].values) for _, g in sel.groupby("trip")]
    return float(np.mean(vals))


def observation_top1(sel, pts):
    m = sel.merge(pts[["trip", "point", "label_arc", "label_valid"]], on=["trip", "point"])
    m = m[m["label_valid"] == 1]
    return float((m["arc"] == m["label_arc"]).mean())


def candidate_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def evaluate(df, prob, pts):
    valid = df.merge(pts[["trip", "point", "label_valid"]], on=["trip", "point"], how="left")["label_valid"].values == 1
    out = candidate_metrics(df["y"].values[valid], prob[valid])
    sel = top1_arcs(df, prob)
    out["top1_accuracy"] = observation_top1(sel, pts)
    out["route_continuity"] = route_continuity(sel)
    v = pts[pts["label_valid"] == 1]
    out["candidate_miss_rate"] = float(1 - v["label_in_cands"].mean())
    return out


def route_prf(pred_arcs, ref_arcs, arc_len):
    p, r = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    inter = sum(arc_len[a] for a in p & r)
    lp, lr = sum(arc_len[a] for a in p), sum(arc_len[a] for a in r)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
