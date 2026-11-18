import os
import numpy as np
from pyproj import CRS, Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))


def list_trips(partition, data_dir=None):
    root = os.path.join(data_dir or DATA_DIR, partition)
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def trip_file(folder, trip, ext):
    return os.path.join(folder, trip, f"{trip}.{ext}")


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32700 if lat < 0 else 32600) + zone)


class Trip:
    def __init__(self, folder, trip, margin_m=400.0):
        self.folder, self.trip_id = folder, trip
        tr = np.loadtxt(trip_file(folder, trip, "track"), ndmin=2)
        nodes = np.loadtxt(trip_file(folder, trip, "nodes"), ndmin=2)
        arcs = np.loadtxt(trip_file(folder, trip, "arcs"), dtype=np.int64, ndmin=2)
        ok = np.isfinite(tr).all(1) & (np.abs(tr[:, 0]) <= 180) & (np.abs(tr[:, 1]) <= 90)
        self.raw_track_len = len(tr)
        tr = tr[ok]
        tr = tr[np.argsort(tr[:, 2], kind="stable")]
        keep = np.r_[True, np.diff(tr[:, 2]) > 0]
        self.dropped_points = int((~ok).sum() + (~keep).sum())
        tr = tr[keep]
        self.lon, self.lat, self.t = tr[:, 0], tr[:, 1], tr[:, 2]
        self.node_lon, self.node_lat = nodes[:, 0], nodes[:, 1]
        self.arc_from, self.arc_to = arcs[:, 0], arcs[:, 1]
        self.crs = utm_crs(float(np.median(self.lon)), float(np.median(self.lat)))
        self.fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self.x, self.y = self.fwd.transform(self.lon, self.lat)
        self.nx, self.ny = self.fwd.transform(self.node_lon, self.node_lat)
        ax0, ay0 = self.nx[self.arc_from], self.ny[self.arc_from]
        ax1, ay1 = self.nx[self.arc_to], self.ny[self.arc_to]
        self.arc_len = np.hypot(ax1 - ax0, ay1 - ay0)
        self.arc_bearing = np.degrees(np.arctan2(ax1 - ax0, ay1 - ay0)) % 360
        xmin, xmax = self.x.min() - margin_m, self.x.max() + margin_m
        ymin, ymax = self.y.min() - margin_m, self.y.max() + margin_m
        inbox = lambda X, Y: (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
        self.local_arcs = np.where(inbox(ax0, ay0) | inbox(ax1, ay1))[0]

    @property
    def n(self):
        return len(self.t)

    def arc_lonlat(self, arc):
        return [[float(self.node_lat[self.arc_from[arc]]), float(self.node_lon[self.arc_from[arc]])],
                [float(self.node_lat[self.arc_to[arc]]), float(self.node_lon[self.arc_to[arc]])]]


def load_route(folder, trip):
    p = trip_file(folder, trip, "route")
    if not os.path.exists(p):
        return None
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


def point_seg_dist(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - x0) * dx + (py - y0) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)), t
