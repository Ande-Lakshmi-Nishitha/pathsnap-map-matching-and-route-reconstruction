import os
import json
import time
import numpy as np
import joblib
from .io import ROOT, Trip, load_route
from .config import CFG
from .candidates import ArcIndex, generate_candidates
from .features import build_features
from .hmm import decode, stitch_route
from .labels import derive_labels
from .metrics import continuity, route_prf


def load_artifacts(model_path=None, hmm_path=None):
    bundle = joblib.load(model_path or os.environ.get("PATHSNAP_MODEL") or os.path.join(ROOT, "models", "final_model.joblib"))
    hp = hmm_path or os.path.join(ROOT, "configs", "hmm_tuned.json")
    hmm_cfg = json.load(open(hp)) if os.path.exists(hp) else CFG["hmm"]
    return bundle, hmm_cfg


def predict_trip(folder, trip_id, bundle, hmm_cfg):
    timings = {}
    t0 = time.time()
    trip = Trip(folder, trip_id)
    idx = ArcIndex(trip, CFG["candidates"]["spacing"])
    cand, dist, proj = generate_candidates(trip, idx, CFG["candidates"]["radius"], CFG["candidates"]["k"])
    df = build_features(trip, cand, dist, proj)
    timings["preprocess_s"] = time.time() - t0
    t0 = time.time()
    prob = bundle["model"].predict_proba(df[bundle["features"]].values.astype(np.float32))[:, 1]
    timings["classifier_s"] = time.time() - t0
    d = df[["point", "arc"]].assign(p=prob)
    clf_arcs = d.loc[d.groupby("point")["p"].idxmax()].set_index("point").reindex(range(trip.n))["arc"].values
    t0 = time.time()
    arcs, breaks, graph = decode(trip, df, prob, hmm_cfg)
    route, seq, gaps = stitch_route(trip, graph, arcs)
    timings["hmm_s"] = time.time() - t0
    return dict(trip=trip, point_arcs=arcs, classifier_arcs=clf_arcs, route=route, decoded_sequence=seq,
                route_gaps=gaps, hmm_breaks=len(breaks), timings=timings)


def score_prediction(folder, trip_id, pred):
    trip = pred["trip"]
    ref = load_route(folder, trip_id)
    if ref is None:
        return None
    lab, ldist, _ = derive_labels(trip, ref)
    valid = ldist <= CFG["labels"]["max_dist"]
    arcs, clf = pred["point_arcs"], pred["classifier_arcs"]
    p, r, f1 = route_prf(pred["route"], ref, trip.arc_len)
    return dict(
        observation_top1_accuracy=float((arcs[valid] == lab[valid]).mean()),
        classifier_only_top1_accuracy=float((clf[valid] == lab[valid]).mean()),
        route_precision=p, route_recall=r, route_f1=f1,
        route_continuity=continuity(trip.arc_from[arcs], trip.arc_to[arcs], arcs),
        classifier_only_continuity=continuity(trip.arc_from[clf], trip.arc_to[clf], clf),
        scored_points=int(valid.sum()), total_points=int(trip.n))
