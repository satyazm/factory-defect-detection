"""
Pixel normalization helpers.

Note: if you train with ultralytics YOLO, normalization (0-1 scaling,
letterboxing) is handled internally by the training pipeline — you don't
need to pre-normalize images on disk. These helpers are for the inference
side, or for feeding a custom classifier/anomaly model where you need
explicit control over the tensor values.
"""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_unit_range(image: np.ndarray) -> np.ndarray:
    """uint8 BGR/RGB image -> float32 in [0, 1]."""
    return image.astype(np.float32) / 255.0


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    """image already scaled to [0, 1], RGB channel order -> standardized."""
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_for_display(image: np.ndarray) -> np.ndarray:
    """Inverse of imagenet_normalize, clipped back to a displayable uint8 image."""
    restored = image * IMAGENET_STD + IMAGENET_MEAN
    restored = np.clip(restored * 255.0, 0, 255)
    return restored.astype(np.uint8)
