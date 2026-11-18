import os
import yaml
from .io import ROOT

with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
    CFG = yaml.safe_load(f)
