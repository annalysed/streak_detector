"""
Image normalization utilities for display and preprocessing.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.defaults import NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH


def normalize_for_display(
    data: np.ndarray,
    p_low: float = NORM_PERCENTILE_LOW,
    p_high: float = NORM_PERCENTILE_HIGH,
) -> np.ndarray:
    """
    Percentile-clip and scale to uint8 [0, 255].

    Parameters
    ----------
    data : float ndarray, shape (H, W)
    p_low, p_high : percentile bounds for clipping

    Returns
    -------
    uint8 ndarray, shape (H, W)
    """
    lo, hi = np.percentile(data, (p_low, p_high))
    clipped = np.clip(data, lo, hi)
    if hi > lo:
        scaled = (clipped - lo) / (hi - lo) * 255.0
    else:
        scaled = np.zeros_like(clipped)
    return scaled.astype(np.uint8)


def normalize(
    data: np.ndarray,
    img_log,
) -> np.ndarray:
    """
    By subtracting the median and dividing by the standard deviation, the background sky centers around ~0.0, and bright streaks become positive outliers. 
    This prevents the cross-correlation algorithm from heavily weighting bright, noisy sky backgrounds.

    Parameters
    ----------
    data : float ndarray, shape (H, W)

    Returns
    -------
    ndarray, shape (H, W)
    """
    image = ((data - np.median(data))
                           / np.std(data))
    #img_log.info("image data:")
    #img_log.info(image)
    img_log.info("FITS image normalized.")
    #np.savetxt("image.txt", image, fmt="%.3f")
    return image