import argparse
from pathlib import Path
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt veya yolov8n.pt")
    ap.add_argument("--source", required=True, help="image folder path")
    ap.add_argument("--out", default="outputs/preds_test", help="output folder")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"Image bulunamadı: {src}")

    for p in imgs:
        res = model.predict(source=str(p), conf=args.conf, save=True, project=str(out), name=".", exist_ok=True)
    print(f"✅ Saved predictions into: {out}")

if __name__ == "__main__":
    main()
