import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.pt or .onnx)")
    parser.add_argument("--source", required=True, help="RTSP URL (or video file path)")
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--device", default=0, help="0 for GPU, 'cpu' for CPU")
    parser.add_argument("--save", action="store_true", help="Save outputs")
    parser.add_argument("--project", default="outputs/rtsp", help="Base output folder")
    parser.add_argument("--name", default="live_run", help="Run folder name under project")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    model = YOLO(str(model_path))



    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=True,      
        stream=True,    
        save=args.save,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    for _ in results:
        pass


if __name__ == "__main__":
    main()
