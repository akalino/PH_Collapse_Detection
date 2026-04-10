import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path

def load_config(_path):
    with open(_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_root = Path(cfg["run"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    return cfg


def resolve_output(cfg: dict, relative_path: str) -> str:
    return str(Path(cfg["run"]["output_root"]) / relative_path)


def stable_seed(*parts, mod=2**31 - 1) -> int:
    s = "::".join(map(str, parts))
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % mod


def resolve_master_tau_map(cfg):
    stage = cfg["tau_parallel"]
    if "master_tau_map_path" in stage and stage["master_tau_map_path"]:
        return stage["master_tau_map_path"]
    return resolve_output(cfg, stage["out_path"])


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()