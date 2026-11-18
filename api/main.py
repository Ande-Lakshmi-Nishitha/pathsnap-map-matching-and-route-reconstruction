import os
import sys
import time
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT, list_trips
from pathsnap.pipeline import load_artifacts, predict_trip, score_prediction

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "evaluator_test"))
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(ROOT, "logs", "api.log"))])
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap Map-Matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BUNDLE, HMM_CFG = load_artifacts()
log.info(f"loaded model={BUNDLE['name']} features={len(BUNDLE['features'])} hmm={HMM_CFG} eval_dir={EVAL_DIR}")


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    parent, name = os.path.split(EVAL_DIR.rstrip("/"))
    return list_trips(name, parent) if os.path.isdir(EVAL_DIR) else []


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info(f"{request.client.host if request.client else '-'} {request.method} {request.url.path} -> {resp.status_code} ({1000 * (time.time() - t0):.0f} ms)")
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": BUNDLE["name"], "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info(f"PREDICT request trip={req.trip_id}")
    pred = predict_trip(EVAL_DIR, req.trip_id, BUNDLE, HMM_CFG)
    score = score_prediction(EVAL_DIR, req.trip_id, pred)
    trip = pred["trip"]
    step = max(1, trip.n // 2000)
    out = {
        "trip_id": req.trip_id,
        "model": BUNDLE["name"],
        "n_points": int(trip.n),
        "gps_points": [[float(trip.lat[i]), float(trip.lon[i])] for i in range(0, trip.n, step)],
        "predicted_route": [trip.arc_lonlat(int(a)) for a in pred["route"]],
        "predicted_route_arc_ids": [int(a) for a in pred["route"]],
        "point_arc_ids": [int(a) for a in pred["point_arcs"]],
        "metrics": score,
        "decoder": {"route_gaps": pred["route_gaps"], "hmm_breaks": pred["hmm_breaks"]},
        "timings": {**{k: round(v, 3) for k, v in pred["timings"].items()}, "total_s": round(time.time() - t0, 3)},
    }
    m = score or {}
    log.info(f"PREDICT done trip={req.trip_id} points={trip.n} route_arcs={len(pred['route'])} "
             f"top1={m.get('observation_top1_accuracy', float('nan')):.4f} route_f1={m.get('route_f1', float('nan')):.4f} "
             f"total={out['timings']['total_s']}s")
    return out
