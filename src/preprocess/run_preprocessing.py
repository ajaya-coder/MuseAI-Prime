"""
Preprocessing script for MuseAI-Prime.

What this does:
  - STYLE:
      data/style_raw/{picasso, rembrandt}/*.jpg
        -> center-crop to square, resize to 512x512
        -> save to data/style/{artist}/
        -> write metadata/style_catalog.csv

  - CONTENT FACES:
      data/content/faces/raw/*.jpg
        -> detect face with MTCNN
        -> expand bbox, square-crop, resize to 512x512
        -> split into train/val/test
        -> save to data/content/faces/{train,val,test}/
        -> write metadata/content_catalog.csv

Run from repo root:
    python src/preprocess/run_preprocessing.py
"""

import csv
import random
import warnings
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN

warnings.filterwarnings("ignore")

# -------------------------------
# GLOBAL CONFIG
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # MuseAI-Prime/
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_ROOT = PROJECT_ROOT / "metadata"

# STYLE
STYLE_RAW_ROOT = DATA_ROOT / "style_raw"
STYLE_PROC_ROOT = DATA_ROOT / "style"

# CONTENT FACES
CONTENT_RAW_ROOT = DATA_ROOT / "content" / "faces" / "raw"
CONTENT_PROC_ROOT = DATA_ROOT / "content" / "faces"

IMAGE_SIZE = 512

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FACE_EXPAND_FACTOR = 1.4  # how much bigger than face bbox
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# UTILS
# -------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    if w > h:
        left = (w - h) // 2
        right = left + h
        top = 0
        bottom = h
    else:
        top = (h - w) // 2
        bottom = top + w
        left = 0
        right = w
    return img.crop((left, top, right, bottom))


# -------------------------------
# STYLE PREPROCESSING
# -------------------------------

def preprocess_styles():
    """
    Style images: simple center-crop + resize.
    """
    print("\n[STYLE] Preprocessing style images...")

    ensure_dir(STYLE_PROC_ROOT)
    ensure_dir(METADATA_ROOT)

    metadata_path = METADATA_ROOT / "style_catalog.csv"
    rows = []

    artist_dirs = [d for d in STYLE_RAW_ROOT.iterdir() if d.is_dir()]
    if not artist_dirs:
        print(f"  ⚠️ No artist folders found in {STYLE_RAW_ROOT}")
        return

    for artist_dir in artist_dirs:
        artist_name = artist_dir.name
        out_dir = STYLE_PROC_ROOT / artist_name
        ensure_dir(out_dir)

        img_files = [p for p in artist_dir.rglob("*") if p.is_file() and is_image_file(p)]
        if not img_files:
            print(f"  ⚠️ No images found for artist '{artist_name}' in {artist_dir}")
            continue

        print(f"  ▶ {artist_name}: {len(img_files)} images")

        for img_path in tqdm(img_files, desc=f"    {artist_name}", leave=False):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"    ⚠️ Skipping {img_path} (open error: {e})")
                continue

            img = center_crop_to_square(img)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

            out_path = out_dir / img_path.name
            try:
                img.save(out_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"    ⚠️ Failed to save {out_path}: {e}")
                continue

            rows.append({
                "artist": artist_name,
                "filename": img_path.name,
                "original_path": str(img_path.relative_to(PROJECT_ROOT)),
                "processed_path": str(out_path.relative_to(PROJECT_ROOT)),
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            })

    if rows:
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["artist", "filename", "original_path", "processed_path", "width", "height"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✅ Style metadata saved to {metadata_path}")
    else:
        print("  ⚠️ No style images processed.")


# -------------------------------
# FACE CROP WITH MTCNN
# -------------------------------

def preprocess_face_image(image_path: Path, mtcnn: MTCNN) -> Image.Image | None:
    """
    Detect face and crop with expansion + squaring.

    Returns processed PIL.Image or None on failure.
    """
    img = Image.open(image_path).convert("RGB")

    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        # fallback: center-crop
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return img

    # most confident face
    box = boxes[0]
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    expand_w = w * (FACE_EXPAND_FACTOR - 1) / 2.0
    expand_h = h * (FACE_EXPAND_FACTOR - 1) / 2.0

    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(img.width, x2 + expand_w)
    y2 = min(img.height, y2 + expand_h)

    # make square by expanding the shorter side
    bw = x2 - x1
    bh = y2 - y1

    if bw > bh:
        diff = bw - bh
        y1 = max(0, y1 - diff / 2.0)
        y2 = min(img.height, y2 + diff / 2.0)
    else:
        diff = bh - bw
        x1 = max(0, x1 - diff / 2.0)
        x2 = min(img.width, x2 + diff / 2.0)

    face_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
    face_img = face_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    return face_img


def preprocess_faces():
    """
    Process celeb faces using MTCNN and split into train/val/test.
    """
    print("\n[CONTENT] Preprocessing faces with MTCNN...")
    ensure_dir(CONTENT_PROC_ROOT)
    ensure_dir(METADATA_ROOT)

    metadata_path = METADATA_ROOT / "content_catalog.csv"

    if not CONTENT_RAW_ROOT.exists():
        print(f"  ⚠️ Raw face dir not found: {CONTENT_RAW_ROOT}")
        return

    img_files = [p for p in CONTENT_RAW_ROOT.rglob("*") if p.is_file() and is_image_file(p)]
    if not img_files:
        print(f"  ⚠️ No face images found in {CONTENT_RAW_ROOT}")
        return

    print(f"  Found {len(img_files)} raw face images.")

    random.seed(RANDOM_SEED)
    random.shuffle(img_files)

    n = len(img_files)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_test = n - n_train - n_val

    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train + n_val]
    test_files = img_files[n_train + n_val:]

    print(f"  Split:")
    print(f"    Train: {len(train_files)}")
    print(f"    Val:   {len(val_files)}")
    print(f"    Test:  {len(test_files)}")

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=0,
        device=DEVICE,
        post_process=False
    )

    rows = []
    failed_count = 0

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in splits.items():
        out_split_dir = CONTENT_PROC_ROOT / split_name
        ensure_dir(out_split_dir)

        print(f"  ▶ Processing {split_name} ({len(files)} images)")
        for img_path in tqdm(files, desc=f"    faces/{split_name}", leave=False):
            try:
                processed_img = preprocess_face_image(img_path, mtcnn)

                if processed_img is None:
                    failed_count += 1
                    continue

                out_path = out_split_dir / img_path.name
                processed_img.save(out_path, format="JPEG", quality=95)

                rows.append({
                    "split": split_name,
                    "filename": img_path.name,
                    "original_path": str(img_path.relative_to(PROJECT_ROOT)),
                    "processed_path": str(out_path.relative_to(PROJECT_ROOT)),
                    "width": IMAGE_SIZE,
                    "height": IMAGE_SIZE,
                })

            except Exception as e:
                print(f"    ⚠️ Error processing {img_path.name}: {e}")
                failed_count += 1

    if failed_count > 0:
        print(f"  ⚠️ Failed / fallback processed: {failed_count} images.")

    if rows:
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "filename", "original_path", "processed_path", "width", "height"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✅ Content metadata saved to {metadata_path}")
    else:
        print("  ⚠️ No content images processed.")


# -------------------------------
# MAIN
# -------------------------------

def main():
    print("=" * 70)
    print("MUSEAI-PRIME - PREPROCESSING")
    print("=" * 70)

    print(f"\nProject root:   {PROJECT_ROOT}")
    print(f"Style raw:      {STYLE_RAW_ROOT}")
    print(f"Faces raw:      {CONTENT_RAW_ROOT}")
    print(f"Output size:    {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Device:         {DEVICE}")

    preprocess_styles()
    preprocess_faces()

    print("\n✅ Preprocessing complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
