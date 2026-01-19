from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape  # <-- key fix

from rasterio.features import shapes
from rasterio.warp import transform_geom


def labels_to_geodataframe(
    *,
    label_image: np.ndarray,
    obj_stats: Optional[pd.DataFrame],
    transform,
    crs,
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert a labeled raster (connected components) into a GeoDataFrame.

    Parameters
    ----------
    label_image : np.ndarray
        Label image (0 = background, 1..N = objects)
    obj_stats : DataFrame or None
        Per-object stats with column 'obj_id'. Merged if provided.
    transform : rasterio.Affine
        Affine transform of the raster.
    crs : rasterio.CRS or str
        CRS of the raster.
    target_crs : str
        Output CRS (default EPSG:4326 for web maps).

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        One row per object with geometry + attributes.
    """
    features = []

    src_crs = crs
    dst_crs = target_crs

    for geom, val in shapes(
        label_image.astype(np.int32),
        mask=(label_image > 0),
        transform=transform
    ):
        oid = int(val)
        if oid == 0:
            continue

        # geom is a GeoJSON-like dict -> optionally reproject -> still dict
        if src_crs and dst_crs and str(src_crs) != str(dst_crs):
            geom = transform_geom(src_crs, dst_crs, geom, precision=6)

        # Convert GeoJSON dict -> shapely geometry (THIS fixes your error)
        geom_shp = shape(geom)

        features.append({
            "obj_id": oid,
            "geometry": geom_shp
        })

    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs=dst_crs)

    if obj_stats is not None and (not obj_stats.empty):
        gdf = gdf.merge(obj_stats, on="obj_id", how="left")

    return gdf
