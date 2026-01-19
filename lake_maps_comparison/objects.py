from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .morphology import structure_from_connectivity


@dataclass
class ObjectConfig:
    """
    Tunable parameters for object extraction/statistics.
    """
    connectivity: int = 4  # 4 or 8
    pixel_area_m2: float = 100.0  # 10m pixels -> 100 m^2


def label_objects(mask01: np.ndarray, connectivity: int = 4) -> Tuple[np.ndarray, int]:
    """
    Connected-component labeling.

    Returns
    -------
    labeled : np.ndarray
        Label image (0=background, 1..n = objects)
    n : int
        number of objects
    """
    structure = structure_from_connectivity(connectivity)
    lab, n = ndi.label(mask01.astype(bool), structure=structure)
    return lab.astype(np.int32), int(n)


def object_pixel_counts_from_labels(lab: np.ndarray) -> pd.DataFrame:
    """
    Per-object pixel counts from a label image.
    """
    ids = np.unique(lab)
    ids = ids[ids != 0]
    if ids.size == 0:
        return pd.DataFrame(columns=["obj_id", "px_count"])

    counts = ndi.sum(np.ones_like(lab, dtype=np.uint8), lab, ids).astype(int)
    return pd.DataFrame({"obj_id": ids.astype(int), "px_count": counts})


def object_stats_from_labels(
    lab: np.ndarray,
    pixel_area_m2: float,
    extra_properties: Optional[Dict[int, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Per-object stats: px_count, area_m2, area_ha.
    Optionally merges extra_properties by obj_id.
    """
    df = object_pixel_counts_from_labels(lab)
    if df.empty:
        return pd.DataFrame(columns=["obj_id", "px_count", "area_m2", "area_ha"])

    df["area_m2"] = df["px_count"].astype(float) * float(pixel_area_m2)
    df["area_ha"] = df["area_m2"] / 1e4

    if extra_properties:
        extra_df = (
            pd.DataFrame.from_dict(extra_properties, orient="index")
            .reset_index()
            .rename(columns={"index": "obj_id"})
        )
        df = df.merge(extra_df, on="obj_id", how="left")

    return df


def object_areas_ha(mask01: np.ndarray, pixel_area_m2: float, connectivity: int = 4) -> np.ndarray:
    """
    Convenience: mask -> label -> per-object areas in hectares.
    """
    lab, n = label_objects(mask01, connectivity=connectivity)
    if n == 0:
        return np.array([], dtype=float)

    ids = np.arange(1, n + 1)
    counts = ndi.sum(np.ones_like(lab, dtype=np.uint8), lab, ids).astype(float)
    return (counts * float(pixel_area_m2)) / 1e4


def ecdf_xy(x: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return sorted x and ECDF y. If x empty, returns (None, None).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def summarize_object_counts(lab: np.ndarray) -> Dict[str, float]:
    """
    Quick summary stats for a label image.
    """
    ids = np.unique(lab)
    ids = ids[ids != 0]
    return {"n_objects": float(ids.size)}


def label_and_stats(
    mask01: np.ndarray,
    cfg: ObjectConfig,
    *,
    return_intermediate: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, object]]]:
    """
    One call: mask -> labels -> stats dataframe.

    If return_intermediate=True, returns:
      (stats_df, {"labels": lab, "n": n})
    """
    lab, n = label_objects(mask01, connectivity=cfg.connectivity)
    df = object_stats_from_labels(lab, pixel_area_m2=cfg.pixel_area_m2)

    if return_intermediate:
        return df, {"labels": lab, "n": n}
    return df
