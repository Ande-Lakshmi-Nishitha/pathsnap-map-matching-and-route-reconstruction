import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT, DATA_DIR, list_trips, load_route
from pathsnap.config import CFG
from pathsnap.labels import derive_labels
from pathsnap.pipeline import load_artifacts, predict_trip


def kind(trip, pred, lab, pos, ref):
    if pred == lab:
        return "correct"
    if trip.arc_from[pred] == trip.arc_to[lab] and trip.arc_to[pred] == trip.arc_from[lab]:
        return "reverse direction of true arc"
    if pred in set(ref[max(0, pos - 3):pos + 4].tolist()):
        return "neighbouring arc on true route (junction boundary)"
    if trip.arc_from[pred] in (trip.arc_from[lab], trip.arc_to[lab]) or trip.arc_to[pred] in (trip.arc_from[lab], trip.arc_to[lab]):
        return "side road sharing a node"
    return "other road (parallel / off route)"


def main():
    bundle, hmm_cfg = load_artifacts()
    folder = os.path.join(DATA_DIR, "student_test")
    rows = []
    for t in list_trips("student_test"):
        pred = predict_trip(folder, t, bundle, hmm_cfg)
        trip = pred["trip"]
        ref = load_route(folder, t)
        lab, dist, pos = derive_labels(trip, ref)
        valid = dist <= CFG["labels"]["max_dist"]
        for mode, arcs in [("classifier_only", pred["classifier_arcs"]), ("classifier_hmm", pred["point_arcs"])]:
            for i in np.flatnonzero(valid):
                rows.append(dict(trip=t, mode=mode, kind=kind(trip, int(arcs[i]), int(lab[i]), int(pos[i]), ref)))
        print(t, flush=True)
    d = pd.DataFrame(rows)
    tab = d.groupby(["mode", "kind"]).size().unstack(0).fillna(0).astype(int)
    tab = tab.div(tab.sum(0), axis=1).round(4).join(tab, rsuffix="_count")
    tab.to_csv(os.path.join(ROOT, "results", "hmm", "error_taxonomy.csv"))
    print(tab.to_string())


if __name__ == "__main__":
    main()
