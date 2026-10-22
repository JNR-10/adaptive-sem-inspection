"""Current-observation image-quality feature definitions for the router."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


class QualityMetricError(RuntimeError):
    """Raised when the locked OpenCV dependency is unavailable."""


def quality_metrics(observation: np.ndarray) -> dict[str, float]:
    """Compute all six non-learned quality scalars on unnormalized grayscale."""

    if observation.shape != (448, 448):
        raise ValueError("Quality metrics require a 448×448 observation")
    x = np.asarray(observation, dtype=np.float32)
    if np.nanmin(x) < 0 or np.nanmax(x) > 1:
        raise ValueError("Quality metric input must be in [0, 1]")
    try:
        import cv2
    except ImportError as exc:
        raise QualityMetricError("opencv-python-headless is required for the locked quality metrics") from exc
    laplacian = cv2.Laplacian(x, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    sobel_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    sobel_y = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    highpass = x.astype(np.float64) - gaussian_filter(x.astype(np.float64), sigma=1.0, mode="reflect", truncate=4.0)
    highpass_median = float(np.median(highpass))
    return {
        "intensity_mean": float(np.mean(x)),
        "intensity_std": float(np.std(x, ddof=0)),
        "robust_dynamic_range": float(np.percentile(x, 95, method="linear") - np.percentile(x, 5, method="linear")),
        "laplacian_variance": float(np.var(laplacian, ddof=0)),
        "tenengrad": float(np.mean(sobel_x**2 + sobel_y**2)),
        "highpass_mad": float(np.median(np.abs(highpass - highpass_median)) / 0.67448975),
    }
