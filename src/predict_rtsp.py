import os
import time
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

def main():
    load_dotenv()
    rtsp_url = os.getenv("RTSP_URL")
    if not rtsp_url:
        raise ValueError("RTSP_URL .env içinde tanımlı değil.")

    weights = os.getenv("WEIGHTS", "yolov8n.pt")  
    conf = float(os.getenv("CONF", "0.25"))

    model = YOLO(weights)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError("RTSP stream açılamadı. RTSP_URL doğru mu?")

    save = os.getenv("SAVE_VIDEO", "0") == "1"
    writer = None

    print("✅ RTSP inference başladı. Çıkmak için q.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()  

        if save:
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join("outputs", "preds_realworld", "rtsp_out.mp4")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
            writer.write(annotated)

        cv2.imshow("YOLOv8 RTSP", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
