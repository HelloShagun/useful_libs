from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio


@dataclass(frozen=True)
class StackMeta:
    tif_path: str
    band_desc: List[str]
    transform: rasterio.Affine
    crs: object
    nodata: Optional[float]
    pixel_size_m: Optional[float]
    pixel_area_m2: Optional[float]


def read_stack(tif_path: str) -> Tuple[np.ndarray, StackMeta]:
    """
    Read multiband GeoTIFF stack.

    Returns
    -------
    arr : np.ndarray
        Shape (bands, H, W)
    meta : StackMeta
        Band descriptions + georeferencing metadata.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read()
        band_desc = list(src.descriptions) if src.descriptions is not None else [None] * arr.shape[0]
        band_desc = [("" if b is None else str(b)) for b in band_desc]

        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # Pixel size (meters) only if projected CRS in meters; otherwise None
        # Still useful pixel area from affine regardless, but interpret carefully for geographic CRS.
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        pixel_area_m2 = px_w * px_h

        # If CRS is geographic (degrees), px_w/px_h are degrees not meters → keep pixel_size_m=None
        pixel_size_m = None
        try:
            if crs is not None and hasattr(crs, "is_projected") and crs.is_projected:
                # assumes CRS linear unit is meters (common for UTM etc.)
                pixel_size_m = float(np.mean([px_w, px_h]))
        except Exception:
            pixel_size_m = None

    meta = StackMeta(
        tif_path=tif_path,
        band_desc=band_desc,
        transform=transform,
        crs=crs,
        nodata=nodata,
        pixel_size_m=pixel_size_m,
        pixel_area_m2=float(pixel_area_m2) if pixel_area_m2 is not None else None,
    )
    return arr, meta


def set_band_desc(meta: StackMeta, band_desc: Sequence[str]) -> StackMeta:
    """Return a copy of meta with band_desc replaced."""
    if len(band_desc) != len(meta.band_desc):
        raise ValueError(f"band_desc length {len(band_desc)} != number of bands {len(meta.band_desc)}")
    return StackMeta(
        tif_path=meta.tif_path,
        band_desc=list(band_desc),
        transform=meta.transform,
        crs=meta.crs,
        nodata=meta.nodata,
        pixel_size_m=meta.pixel_size_m,
        pixel_area_m2=meta.pixel_area_m2,
    )


def index_of_band(band_desc: Sequence[str], name: str, aliases: Optional[Dict[str, Sequence[str]]] = None) -> int:
    """
    Find band index by exact match to name, or by aliases.

    aliases example:
      {"DW_Water": ["DW", "DynamicWorld", "DW_WATER"]}
    """
    if name in band_desc:
        return list(band_desc).index(name)

    if aliases and name in aliases:
        for alt in aliases[name]:
            if alt in band_desc:
                return list(band_desc).index(alt)

    raise KeyError(f"Band '{name}' not found. Available: {list(band_desc)}")


def get_band(
    arr: np.ndarray,
    band_desc: Sequence[str],
    name: str,
    aliases: Optional[Dict[str, Sequence[str]]] = None,
) -> np.ndarray:
    """Return raw band array (H, W) for band name."""
    idx = index_of_band(band_desc, name, aliases=aliases)
    return arr[idx]


def get_band_mask(
    arr: np.ndarray,
    band_desc: Sequence[str],
    name: str,
    *,
    threshold: float = 0.0,
    invert: bool = False,
    nodata_value: Optional[float] = None,
    aliases: Optional[Dict[str, Sequence[str]]] = None,
) -> np.ndarray:
    """
    Return a binary mask (uint8 0/1) from a band.

    Parameters
    ----------
    threshold : float
        mask = (band > threshold) by default.
        For probability bands, set threshold e.g. 0.5.
    invert : bool
        If True, mask = (band <= threshold).
        Useful if water coded as -1 etc.
    nodata_value : float or None
        If provided, nodata pixels are forced to 0.
    """
    b = get_band(arr, band_desc, name, aliases=aliases).astype(np.float32)

    if nodata_value is not None:
        b = np.where(b == nodata_value, np.nan, b)

    if invert:
        m = (b <= threshold)
    else:
        m = (b > threshold)

    m = np.where(np.isfinite(b), m, False)
    return m.astype(np.uint8)


def require_bands(band_desc: Sequence[str], required: Sequence[str], aliases: Optional[Dict[str, Sequence[str]]] = None) -> None:
    """Raise a helpful error if required bands are missing."""
    missing = []
    for r in required:
        try:
            index_of_band(band_desc, r, aliases=aliases)
        except KeyError:
            missing.append(r)
    if missing:
        raise KeyError(f"Missing required bands: {missing}. Available: {list(band_desc)}")
