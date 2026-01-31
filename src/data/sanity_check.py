from pathlib import Path
import random
import cv2
import yaml
import argparse

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def yolo_line_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls, cx, cy, bw, bh = parts
    cls = int(float(cls))
    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return cls, x1, y1, x2, y2

def draw_samples(images_dir: Path, labels_dir: Path, out_dir: Path, n: int = 12):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"Görüntü bulunamadı: {images_dir}")

    picks = random.sample(imgs, k=min(n, len(imgs)))
    for img_path in picks:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for ln in lines:
                if not ln.strip():
                    continue
                out = yolo_line_to_xyxy(ln, w, h)
                if out is None:
                    continue
                cls, x1, y1, x2, y2 = out
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    str(cls),
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        save_path = out_dir / f"{img_path.stem}_sample.jpg"
        cv2.imwrite(str(save_path), img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", default="data/coco6/data.yaml", help="Kontrol edilecek data.yaml yolu")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Örnek alacağımız split")
    ap.add_argument("--n", type=int, default=12, help="Kaç örnek görsel kaydedilsin")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml}")

    cfg = read_yaml(data_yaml)

    names = cfg.get("names", [])
    print(f"✅ data.yaml: {data_yaml}")
    print(f"✅ class count: {len(names)}")
    print(f"✅ classes: {names}")

    base = data_yaml.parent
    images_dir = base / args.split / "images"
    labels_dir = base / args.split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"{args.split}/images yok: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"{args.split}/labels yok: {labels_dir}")

    out_dir = Path("outputs/aug_samples") / f"{base.name}_{args.split}"
    draw_samples(images_dir, labels_dir, out_dir, n=args.n)
    print(f"✅ {args.n} örnek bbox çizimi {out_dir} altına kaydedildi.")

if __name__ == "__main__":
    main()
