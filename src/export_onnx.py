import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--outdir", default="outputs/onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.imgsz, opset=12)

    print("✅ ONNX export done. (Ultralytics runs/ altında da export klasörü oluşabilir.)")

if __name__ == "__main__":
    main()
