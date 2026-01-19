from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi


# 4-connected (same as GEE eightConnected:false)
STRUCTURE_4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

# 8-connected (sometimes useful)
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass
class MorphologyConfig:
    """
    Tunable parameters for mask consolidation / cleaning.

    Parameters
    ----------
    pixel_size_m : float
        Pixel size in meters (e.g. 10 for Sentinel-2 exports at 10 m).
    merge_dist_m : float
        Consolidation distance in meters. Implemented as morphological closing
        (dilation->erosion) with disk radius ~= merge_dist_m/2.
    min_area_m2 : float
        Remove objects smaller than this area (m^2) after consolidation.
        Requires pixel_area_m2 to be provided to `clean_mask`.
    fill_holes : bool
        Fill holes inside blobs (often useful for water bodies).
    connectivity : int
        4 or 8 connectivity for labeling and small-object removal.
    """
    pixel_size_m: float = 10.0
    merge_dist_m: float = 0.0
    min_area_m2: float = 0.0
    fill_holes: bool = True
    connectivity: int = 4  # 4 or 8


def structure_from_connectivity(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return STRUCTURE_4
    if connectivity == 8:
        return STRUCTURE_8
    raise ValueError("connectivity must be 4 or 8")


def disk_kernel(radius_px: int) -> np.ndarray:
    """
    Create a circular (disk) structuring element for morphology.
    radius_px=0 returns a single pixel kernel.
    """
    r = int(max(radius_px, 0))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def consolidate_mask_close(mask01: np.ndarray, merge_dist_m: float, pixel_size_m: float) -> np.ndarray:
    """
    Merge fragments within ~merge_dist_m using morphological closing.

    Uses disk radius = merge_dist_m/2.
    """
    if merge_dist_m <= 0:
        return mask01.astype(np.uint8)

    radius_m = merge_dist_m / 2.0
    radius_px = int(np.ceil(radius_m / pixel_size_m))
    if radius_px <= 0:
        return mask01.astype(np.uint8)

    se = disk_kernel(radius_px)
    m = mask01.astype(bool)

    # Closing = dilation -> erosion
    m = ndi.binary_dilation(m, structure=se)
    m = ndi.binary_erosion(m, structure=se)

    return m.astype(np.uint8)


def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """Fill holes inside connected components."""
    return ndi.binary_fill_holes(mask01.astype(bool)).astype(np.uint8)


def remove_small_objects(
    mask01: np.ndarray,
    min_area_m2: float,
    pixel_area_m2: float,
    connectivity: int = 4
) -> np.ndarray:
    """
    Remove connected components smaller than min_area_m2.

    Parameters
    ----------
    mask01 : np.ndarray
        Binary mask (0/1).
    min_area_m2 : float
        Minimum area to keep (m^2).
    pixel_area_m2 : float
        Area per pixel (m^2). For 10 m pixels, this is 100.
    connectivity : int
        4 or 8.
    """
    if min_area_m2 <= 0:
        return mask01.astype(np.uint8)

    structure = structure_from_connectivity(connectivity)
    lab, n = ndi.label(mask01.astype(bool), structure=structure)
    if n == 0:
        return mask01.astype(np.uint8)

    labels = np.arange(1, n + 1)
    counts = ndi.sum(np.ones_like(lab, dtype=np.uint8), lab, labels).astype(float)
    area_m2 = counts * float(pixel_area_m2)

    keep = labels[area_m2 >= float(min_area_m2)]
    out = np.isin(lab, keep).astype(np.uint8)
    return out


def clean_mask(
    mask01: np.ndarray,
    cfg: MorphologyConfig,
    pixel_area_m2: Optional[float] = None,
    *,
    return_intermediate: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    One-stop cleaning pipeline:
      1) consolidate fragments (closing)
      2) fill holes (optional)
      3) remove small objects (optional; needs pixel_area_m2)

    Parameters
    ----------
    mask01 : np.ndarray
        Binary mask (0/1).
    cfg : MorphologyConfig
        Tunable params.
    pixel_area_m2 : float or None
        Required if cfg.min_area_m2 > 0.
    return_intermediate : bool
        If True, returns (final_mask, steps_dict) where steps_dict contains:
          - 'raw'
          - 'after_closing'
          - 'after_fill_holes'
          - 'final'

    Returns
    -------
    final_mask : np.ndarray
        Cleaned mask (0/1 uint8).
    steps_dict : dict (optional)
        Intermediate masks for debugging/visualisation.
    """
    steps: Dict[str, np.ndarray] = {}

    m0 = mask01.astype(np.uint8)
    steps["raw"] = m0

    m1 = consolidate_mask_close(m0, merge_dist_m=cfg.merge_dist_m, pixel_size_m=cfg.pixel_size_m)
    steps["after_closing"] = m1

    if cfg.fill_holes:
        m2 = fill_holes(m1)
    else:
        m2 = m1
    steps["after_fill_holes"] = m2

    if cfg.min_area_m2 > 0:
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 is required when cfg.min_area_m2 > 0")
        m3 = remove_small_objects(
            m2,
            min_area_m2=cfg.min_area_m2,
            pixel_area_m2=float(pixel_area_m2),
            connectivity=cfg.connectivity
        )
    else:
        m3 = m2

    steps["final"] = m3.astype(np.uint8)

    if return_intermediate:
        return steps["final"], steps
    return steps["final"]
