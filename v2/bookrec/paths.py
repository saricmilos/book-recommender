"""Where things are, relative to this file, so v2 runs from any checkout."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
V1 = REPO / "v1"
V1_API_MODEL = V1 / "models" / "topk_item_sim.npz"
REPORTS = REPO / "v2" / "reports"
DATA = REPO / "v2" / "data"
