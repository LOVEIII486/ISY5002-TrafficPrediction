"""One synthetic forward/backward per model; no benchmark metrics produced."""
import sys
from pathlib import Path
import time
import csv
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.forecast import compat, datasets
from src.forecast.models import registry
from src.forecast.protocols import default_protocol

torch.set_num_threads(4)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--models", nargs="+", choices=registry.available(), default=registry.available())
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="JSON-parsed config override")
options = parser.parse_args()
import json
overrides = {key: json.loads(value) for key, _, value in (item.partition("=") for item in options.set)}
compat.activate(999001)
from libcity.config import ConfigParser
from libcity.utils import StandardScaler

for name in options.models:
    start = time.monotonic()
    spec = registry.resolve(name)
    args = datasets.build_other_args(datasets.PEMSD8, default_protocol("traffic_flow"), exp_id=999001, seed=0, gpu=False)
    args.update(spec.default_args)
    args.update(overrides)
    with compat.libcity_cwd():
        config = ConfigParser("traffic_state_pred", "FNN" if spec.source == "ours" else name, "PEMSD8", other_args=args)
    config["model"] = name
    config["device"] = torch.device("mps")
    channels = 1 + (int(bool(config.get("add_time_in_day"))) + 7*int(bool(config.get("add_day_in_week"))) if config.get("load_external") else 0)
    adj = np.zeros((170,170),dtype=np.float32)
    with (ROOT / "datasets/libcity/PEMSD8/PEMSD8.rel").open(newline="") as stream:
        for edge in csv.DictReader(stream):
            a,b = int(edge["origin_id"]),int(edge["destination_id"])
            adj[a,b] = 1
            if config.get("bidir_adj_mx"):
                adj[b,a] = 1
    features = {"num_nodes": 170, "feature_dim": channels, "output_dim": 1, "ext_dim": channels-1,
                "adj_mx": adj, "scaler": StandardScaler(230.,145.), "num_batches": 196}
    model = registry.build(name, config, features).to("mps")
    torch.manual_seed(0)
    batch = {"X": torch.randn(options.batch_size,12,170,channels,device="mps"), "y": torch.randn(options.batch_size,12,170,channels,device="mps")}
    value = model.calculate_loss(batch, batches_seen=0) if name == "MTGNN" else model.calculate_loss(batch)
    if not torch.isfinite(value):
        raise RuntimeError(f"{name}: invalid loss")
    value.backward()
    model.eval()
    with torch.no_grad():
        pred = model.predict(batch)
    assert pred.shape == (options.batch_size,12,170,1) and torch.isfinite(pred).all(), name
    print(name, "PASS", channels, "channels", round(time.monotonic()-start,2), "seconds", flush=True)
    del model,batch,pred,value
    torch.mps.empty_cache()
