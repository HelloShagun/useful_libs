from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .morphology import structure_from_connectivity


@dataclass
class MatchConfig:
    """
    Tunable parameters for object-to-mask matching.

    An object in A is considered 'matched' to mask B if:
      overlap_frac >= overlap_frac_tol  OR  min_distance_m <= dist_tol_m
    """
    pixel_size_m: float = 10.0
    dist_tol_m: float = 30.0
    overlap_frac_tol: float = 0.10
    connectivity: int = 4  # used only if you want to label inside helper


def match_objects_labels_to_mask(
    labA: np.ndarray,
    maskB01: np.ndarray,
    cfg: MatchConfig,
) -> pd.DataFrame:
    """
    For each labeled object in labA, compute:
      - overlap_frac with B
      - min_dist_m to B
      - matched boolean

    Returns
    -------
    DataFrame with columns:
      obj_id, matched, overlap_frac, min_dist_m, px_count
    """
    A_ids = np.unique(labA)
    A_ids = A_ids[A_ids != 0]
    if A_ids.size == 0:
        return pd.DataFrame(columns=["obj_id", "matched", "overlap_frac", "min_dist_m", "px_count"])

    b = maskB01.astype(bool)
    # distance (in pixels) from each pixel to nearest True in B
    dist_to_B = ndi.distance_transform_edt(~b)
    dist_tol_px = float(cfg.dist_tol_m) / float(cfg.pixel_size_m)

    rows = []
    for oid in A_ids:
        obj = (labA == oid)
        px = int(obj.sum())
        if px == 0:
            continue

        inter = int((obj & b).sum())
        frac = float(inter / (px + 1e-9))
        md_px = float(dist_to_B[obj].min()) if obj.any() else np.inf
        md_m = float(md_px * cfg.pixel_size_m)

        matched = (frac >= cfg.overlap_frac_tol) or (md_m <= cfg.dist_tol_m)

        rows.append((int(oid), bool(matched), float(frac), float(md_m), px))

    return pd.DataFrame(rows, columns=["obj_id", "matched", "overlap_frac", "min_dist_m", "px_count"])


def label_mask(mask01: np.ndarray, connectivity: int = 4) -> Tuple[np.ndarray, int]:
    """Convenience labeling."""
    structure = structure_from_connectivity(connectivity)
    lab, n = ndi.label(mask01.astype(bool), structure=structure)
    return lab.astype(np.int32), int(n)


def compare_object_set_to_mask(
    *,
    source_labels: np.ndarray,
    reference_mask: np.ndarray,
    source_stats: Optional[pd.DataFrame] = None,
    cfg: MatchConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare objects (from source_labels) against a reference mask.

    An object is considered 'supported' if:
      - overlap fraction >= overlap_frac_tol OR
      - minimum distance to reference mask <= dist_tol_m

    Parameters
    ----------
    source_labels : np.ndarray
        Labeled object image (0 = background).
    reference_mask : np.ndarray
        Binary mask (0/1) to compare against.
    source_stats : DataFrame, optional
        Per-object stats (obj_id, area_ha, area_m2, px_count).
    cfg : MatchConfig
        Matching thresholds.

    Returns
    -------
    per_object_df : DataFrame
        One row per object with:
          obj_id, matched, overlap_frac, min_dist_m, px_count (+ stats if provided)
    summary : dict
        Headline numbers (counts, fractions, areas if available).
    """
    mdf = match_objects_labels_to_mask(source_labels, reference_mask, cfg)

    if source_stats is not None and not source_stats.empty:
        stats = source_stats.copy()
        if "px_count" in stats.columns:
            stats = stats.rename(columns={"px_count": "px_count_stats"})
        mdf = mdf.merge(stats, on="obj_id", how="left")

    unsupported = mdf[~mdf["matched"]].copy()

    summary = {
        "n_objects_source": int(len(mdf)),
        "n_objects_supported": int(mdf["matched"].sum()),
        "n_objects_unsupported": int(len(unsupported)),
        "fraction_unsupported": float(len(unsupported) / len(mdf)) if len(mdf) else np.nan,
    }

    if "area_ha" in mdf.columns:
        summary["unsupported_area_ha_total"] = float(unsupported["area_ha"].sum())
        summary["unsupported_area_ha_median"] = float(unsupported["area_ha"].median()) if len(unsupported) else np.nan

    return mdf, summary


def union_masks(masks: Dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Boolean union of multiple 0/1 masks."""
    out = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for n in names:
        out |= masks[n].astype(bool)
    return out.astype(np.uint8)
