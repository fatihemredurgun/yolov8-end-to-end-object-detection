from __future__ import annotations
from pathlib import Path
import shutil
import yaml

TARGET_NAMES = ["person", "cat", "cell phone", "sports ball", "bottle", "chair"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: dict, p: Path):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    rest = parts[1:]
    return cls, rest

def main():
    src_root = Path("data/coco_subset") 
    src_yaml = src_root / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(f"data.yaml bulunamadı: {src_yaml}")

    cfg = load_yaml(src_yaml)
    names = cfg["names"]  

    old_idx_to_name = {i: n for i, n in enumerate(names)}

    target_old_indices = {}
    for tn in TARGET_NAMES:
        try:
            old_i = names.index(tn)
        except ValueError:
            raise ValueError(f"names içinde bulunamadı: {tn}")
        target_old_indices[old_i] = tn

    name_to_new = {n: i for i, n in enumerate(TARGET_NAMES)}
    old_to_new = {old_i: name_to_new[nm] for old_i, nm in target_old_indices.items()}

    out_root = Path("data/coco6")
    if out_root.exists():
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    for split in ["train", "val"]:
        ensure_dir(out_root / split / "images")
        ensure_dir(out_root / split / "labels")

    def process_split(split: str):
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(f"Split klasörü eksik: {img_dir} veya {lbl_dir}")

        kept_images = 0
        kept_labels = 0

        for lbl_path in lbl_dir.glob("*.txt"):
            lines = lbl_path.read_text(encoding="utf-8").splitlines()
            new_lines = []
            for ln in lines:
                if not ln.strip():
                    continue
                parsed = parse_label_line(ln)
                if parsed is None:
                    continue
                old_cls, rest = parsed
                if old_cls in old_to_new:
                    new_cls = old_to_new[old_cls]
                    new_lines.append(" ".join([str(new_cls)] + rest))

            if not new_lines:
                continue  

            stem = lbl_path.stem
            img_path = None
            for ext in IMG_EXTS:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue

            shutil.copy2(img_path, out_root / split / "images" / img_path.name)
            (out_root / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(new_lines) + "\n", encoding="utf-8"
            )
            kept_images += 1
            kept_labels += len(new_lines)

        print(f"✅ {split}: kept_images={kept_images}, kept_boxes={kept_labels}")

    process_split("train")
    process_split("val")

    out_yaml = {
        "path": "data/coco6",
        "train": "train/images",
        "val": "val/images",
        "nc": len(TARGET_NAMES),
        "names": TARGET_NAMES,
    }
    save_yaml(out_yaml, out_root / "data.yaml")

    print("✅ Oluşturuldu:", out_root)
    print("✅ Yeni data.yaml:", out_root / "data.yaml")
    print("✅ Sınıflar:", TARGET_NAMES)

if __name__ == "__main__":
    main()
