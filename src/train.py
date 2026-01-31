import argparse
import yaml
from ultralytics import YOLO


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

ALLOWED_TRAIN_KEYS = {
    "data", "epochs", "imgsz", "batch", "device", "project", "name",
    "patience", "workers", "seed", "deterministic", "exist_ok",

    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr",

    "label_smoothing", "cos_lr", "close_mosaic", "freeze",

    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr",
    "mosaic", "mixup", "copy_paste",
    "auto_augment", "erasing",

    "amp", "val", "save", "save_period",
    "rect", "cache",
    "plots",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/*.yaml")
    args = ap.parse_args()

    print("🚀 train.py başladı")
    print(f"📄 args: {args}")

    cfg = load_yaml(args.config)

    model_path = cfg["model"]
    model = YOLO(model_path)

    train_kwargs = {
        "data": cfg["data"],
        "epochs": cfg.get("epochs", 30),
        "imgsz": cfg.get("imgsz", 640),
        "batch": cfg.get("batch", 8),
        "device": cfg.get("device", 0),
        "project": cfg.get("project", "runs"),
        "name": cfg.get("name", "exp"),
        "patience": cfg.get("patience", 30),
        "workers": cfg.get("workers", 4),
        "exist_ok": True,
        "seed": cfg.get("seed", 0),
        "deterministic": cfg.get("deterministic", True),
    }

    for k, v in cfg.items():
        if k in ALLOWED_TRAIN_KEYS and v is not None:
            train_kwargs[k] = v

    print("\n🔧 Effective train kwargs:")
    for k in sorted(train_kwargs.keys()):
        print(f"  - {k}: {train_kwargs[k]}")

    results = model.train(**train_kwargs)

    print("\n✅ Training finished.")
    print(results)


if __name__ == "__main__":
    main()
