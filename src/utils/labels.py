from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COCO_YAML_PATH = PROJECT_ROOT / "models" / "yolov5" / "coco.yaml"


def load_coco_labels():

    if not COCO_YAML_PATH.exists():
        raise FileNotFoundError(f"COCO labels file not found: {COCO_YAML_PATH}")

    with open(COCO_YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    elif isinstance(names, list):
        return names
    else:
        raise ValueError("Unsupported COCO labels format")
