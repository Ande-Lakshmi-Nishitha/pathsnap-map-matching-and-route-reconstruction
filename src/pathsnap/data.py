import os
import numpy as np
import pandas as pd
from .io import ROOT
from .features import FEATURES


def load_partition(partition):
    d = os.path.join(ROOT, "results", "data")
    c = pd.read_pickle(os.path.join(d, f"{partition}_candidates.pkl"))
    p = pd.read_pickle(os.path.join(d, f"{partition}_points.pkl"))
    return c, p


def training_rows(c, p, stride=1):
    keep = p[(p["label_valid"] == 1) & (p["point"] % stride == 0)][["trip", "point"]]
    return c.merge(keep, on=["trip", "point"]).reset_index(drop=True)


def xy(c):
    return c[FEATURES].values.astype(np.float32), c["y"].values.astype(int)
