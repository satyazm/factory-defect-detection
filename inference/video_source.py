"""
Video source resolution shared by realtime.py, combined_pipeline.py, and
the dashboard: turns a --source string into something with a
cv2.VideoCapture-compatible isOpened()/read()/release() interface.

Supports three kinds of source:
  - webcam index, e.g. "0"
  - a video file or RTSP URL, e.g. "video.mp4" or "rtsp://..."
  - a folder of images, e.g. "dataset/test/images" — cycles through
    them indefinitely, sorted by filename, one frame every
    `delay_seconds`. Useful for demoing continuous detection without a
    real camera or a pre-made video file.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ImageFolderCapture:
    """Drop-in cv2.VideoCapture replacement that loops over a folder of images."""

    def __init__(self, folder: str | Path, delay_seconds: float = 1.5):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.delay_seconds = delay_seconds
        self._index = 0
        self._last_read_time = 0.0
        self.last_path: Path | None = None

    def isOpened(self) -> bool:
        return len(self.paths) > 0

    def read(self):
        if not self.paths:
            return False, None

        elapsed = time.time() - self._last_read_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

        path = self.paths[self._index]
        self._index = (self._index + 1) % len(self.paths)
        self._last_read_time = time.time()
        self.last_path = path

        frame = cv2.imread(str(path))
        return frame is not None, frame

    def release(self) -> None:
        pass


def open_capture(source: str, image_folder_delay_seconds: float = 1.5):
    if Path(source).is_dir():
        return ImageFolderCapture(source, delay_seconds=image_folder_delay_seconds)
    parsed = int(source) if source.isdigit() else source
    return cv2.VideoCapture(parsed)


def guess_ground_truth_label(path: Path | None, candidates: list[str]) -> str | None:
    """
    Best-effort ground truth from a demo image's filename, e.g.
    "scratches_241.jpg" -> "scratches", "broken_large_003.png" ->
    "broken_large". Only meaningful for this repo's own demo folders
    (preprocessing/voc_to_yolo.py's NEU-DET output, or the flattened
    MVTec bottle test split) — returns None for anything else rather
    than guessing wrong.
    """
    if path is None:
        return None
    stem = path.stem
    for candidate in sorted(candidates, key=len, reverse=True):
        if stem.startswith(candidate):
            return candidate
    return None
