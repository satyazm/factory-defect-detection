"""Batch-resize images (and their YOLO txt labels, which are already
normalized so they don't need changes) to a fixed square size."""
import argparse
from pathlib import Path

import cv2


def resize_dir(src_dir: Path, dst_dir: Path, size: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip (unreadable): {path}")
            continue
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_dir / path.name), resized)
    print(f"Resized {len(paths)} images from {src_dir} -> {dst_dir} at {size}x{size}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="source image directory")
    parser.add_argument("--dst", required=True, type=Path, help="destination image directory")
    parser.add_argument("--size", type=int, default=640, help="output size (square)")
    args = parser.parse_args()
    resize_dir(args.src, args.dst, args.size)


if __name__ == "__main__":
    main()
