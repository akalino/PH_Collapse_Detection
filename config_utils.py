import json
from pathlib import Path

def load_config(_path):
    with open(_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_root = Path(cfg["run"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    return cfg

def resolve_output(cfg: dict, relative_path: str) -> str:
    return str(Path(cfg["run"]["output_root"]) / relative_path)
