import os
import sys
import argparse
import numpy as np
import folium

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.io import ROOT, DATA_DIR, load_route
from pathsnap.pipeline import load_artifacts, predict_trip, score_prediction

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;border:1px solid #888;font-size:13px">
<b>{title}</b><br><span style="color:#1f77b4">&#9679;</span> raw GPS observations<br>
<span style="color:#999">&#9473;</span> road network (near track)<br><span style="color:#2ca02c">&#9473;</span> reference route<br>
<span style="color:#ff7f0e">&#9473;</span> classifier-only top-1 arcs<br><span style="color:#d62728">&#9473;</span> predicted route (classifier + HMM)<br>{stats}</div>"""


def lines(trip, arcs):
    return [trip.arc_lonlat(int(a)) for a in arcs]


def build_map(partition, trip_id, bundle, hmm_cfg):
    folder = os.path.join(DATA_DIR, partition)
    pred = predict_trip(folder, trip_id, bundle, hmm_cfg)
    score = score_prediction(folder, trip_id, pred)
    trip = pred["trip"]
    m = folium.Map(location=[float(np.median(trip.lat)), float(np.median(trip.lon))], zoom_start=14, control_scale=True)
    net = folium.FeatureGroup("Road network (within 150 m)", show=True)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[trip.x, trip.y])
    mid = np.c_[(trip.nx[trip.arc_from[trip.local_arcs]] + trip.nx[trip.arc_to[trip.local_arcs]]) / 2,
                (trip.ny[trip.arc_from[trip.local_arcs]] + trip.ny[trip.arc_to[trip.local_arcs]]) / 2]
    d, _ = tree.query(mid)
    for a in trip.local_arcs[d < 150]:
        folium.PolyLine(trip.arc_lonlat(int(a)), color="#999999", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = load_route(folder, trip_id)
    g = folium.FeatureGroup("Reference route")
    folium.PolyLine(lines(trip, ref), color="#2ca02c", weight=7, opacity=0.6, tooltip="reference route").add_to(g)
    g.add_to(m)
    c = folium.FeatureGroup("Classifier-only top-1", show=False)
    seq = [a for i, a in enumerate(pred["classifier_arcs"]) if i == 0 or a != pred["classifier_arcs"][i - 1]]
    folium.PolyLine(lines(trip, seq), color="#ff7f0e", weight=3, opacity=0.9, tooltip="classifier only").add_to(c)
    c.add_to(m)
    h = folium.FeatureGroup("Predicted route (HMM)")
    folium.PolyLine(lines(trip, pred["route"]), color="#d62728", weight=3, opacity=0.9, tooltip="predicted route").add_to(h)
    h.add_to(m)
    gp = folium.FeatureGroup("Raw GPS")
    step = max(1, trip.n // 1500)
    for i in range(0, trip.n, step):
        folium.CircleMarker([trip.lat[i], trip.lon[i]], radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=0.8,
                            tooltip=f"pt {i}, t={trip.t[i]:.0f}").add_to(gp)
    gp.add_to(m)
    folium.Marker([trip.lat[0], trip.lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([trip.lat[-1], trip.lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[trip.lat.min(), trip.lon.min()], [trip.lat.max(), trip.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    stats = (f"Top-1 HMM: {score['observation_top1_accuracy']:.3f} | classifier: {score['classifier_only_top1_accuracy']:.3f}<br>"
             f"route F1: {score['route_f1']:.3f}")
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=f"{partition} / {trip_id}", stats=stats)))
    out = os.path.join(ROOT, "maps", f"map_{partition}_{trip_id}.html")
    m.save(out)
    print(out, score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["student_test/00000077", "student_test/00000040", "student_test/00000065", "train/00000048"])
    a = ap.parse_args()
    bundle, hmm_cfg = load_artifacts()
    os.makedirs(os.path.join(ROOT, "maps"), exist_ok=True)
    for s in a.trips:
        p, t = s.split("/")
        build_map(p, t, bundle, hmm_cfg)


if __name__ == "__main__":
    main()
