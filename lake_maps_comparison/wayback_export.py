
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None


# ------------------------------------------------------------
# Wayback snapshots (edit/add freely)
# ------------------------------------------------------------
WAYBACK_DATE_TO_TIMEID = {
    "2016-01-13": 3515,
    "2017-01-11": 577,
    "2018-01-08": 13161,
    "2019-01-09": 6036,
    "2020-01-08": 23001,
    "2021-01-13": 1049,
    "2022-01-12": 42663,
    "2023-01-11": 11475,
}

def wayback_xyz(timeid: int) -> str:
    return (
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
        "world_imagery/wmts/1.0.0/default028mm/mapserver/tile/"
        f"{timeid}" + "/{z}/{y}/{x}"
    )


# ------------------------------------------------------------
# Cartography helpers (scalebar, north, lon/lat)
# ------------------------------------------------------------
def add_scalebar_arcgis(ax, length_m: Optional[float] = None, location="lower left",
                        pad=0.02, height_frac=0.015, linewidth=0.8):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w = x1 - x0; h = y1 - y0

    if length_m is None:
        target = w * 0.22
        nice = np.array([25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
                         10000, 20000, 50000], float)
        length_m = float(nice[np.argmin(np.abs(nice - target))])

    if "lower" in location:
        y = y0 + pad*h; text_y = y + 0.02*h; va = "bottom"
    else:
        y = y1 - pad*h; text_y = y - 0.03*h; va = "top"

    if "left" in location:
        x = x0 + pad*w
    else:
        x = x1 - pad*w - length_m

    bar_h = height_frac*h
    seg = length_m / 4.0

    for i in range(4):
        ax.add_patch(plt.Rectangle((x + i*seg, y), seg, bar_h,
                                   facecolor=("black" if i % 2 == 0 else "white"),
                                   edgecolor="black", linewidth=linewidth, zorder=10))
    ax.add_patch(plt.Rectangle((x, y), length_m, bar_h,
                               facecolor="none", edgecolor="black",
                               linewidth=linewidth, zorder=11))

    label = f"{length_m/1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m"
    ax.text(x, text_y, label, fontsize=10, ha="left", va=va,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
            zorder=12)

def add_north_arrow(ax, location="upper right", size=0.08, pad=0.04):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w = x1 - x0; h = y1 - y0

    if "upper" in location:
        y = y1 - pad*h
        dy = -size*h
        va = "top"
    else:
        y = y0 + pad*h
        dy = size*h
        va = "bottom"

    if "right" in location:
        x = x1 - pad*w
    else:
        x = x0 + pad*w

    ax.annotate(
        "",
        xy=(x, y),
        xytext=(x, y + dy),
        arrowprops=dict(facecolor="black", edgecolor="black",
                        width=4, headwidth=12, headlength=12),
        zorder=15,
    )
    ax.text(
        x, y + dy*1.15, "N",
        ha="center", va=va, fontsize=12, fontweight="bold", color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
        zorder=16,
    )

def add_lonlat_edges(ax, n_ticks=4):
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()

    xt = np.linspace(x0, x1, n_ticks)
    yt = np.linspace(y0, y1, n_ticks)

    ax.set_xticks(xt)
    ax.set_yticks(yt)

    lons = [to_wgs84.transform(x, (y0+y1)/2)[0] for x in xt]
    lats = [to_wgs84.transform((x0+x1)/2, y)[1] for y in yt]

    ax.set_xticklabels([f"{lon:.3f}°" for lon in lons], fontsize=9)
    ax.set_yticklabels([f"{lat:.3f}°" for lat in lats], fontsize=9, rotation=90, va="center")

    ax.tick_params(axis="both", direction="out", length=3)
    ax.grid(alpha=0.25)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
@dataclass
class ExportStyle:
    figsize: tuple[float, float] = (8.5, 8.5)
    dpi: int = 400
    pad_m: float = 250.0
    line_width: float = 2.0
    draw_scalebar: bool = True
    draw_north_arrow: bool = True
    draw_coords: bool = True
    n_coord_ticks: int = 4
    scalebar_location: str = "lower left"
    north_location: str = "upper right"


def export_wayback_panels(
    polygons_path: str,
    out_dir: str,
    wayback_date: str = "2021-01-13",
    mode: str = "individual",            # "individual" or "grouped"
    group_eps_m: float = 350.0,          # used only if grouped
    group_min_samples: int = 1,
    id_field: Optional[str] = None,      # if provided, filenames use this
    style: Optional[ExportStyle] = None,
    assume_crs: str = "EPSG:4326",
    max_panels: Optional[int] = None,
) -> list[str]:
    """
    Export PNG map panels for polygons over Esri Wayback imagery.

    Parameters
    ----------
    polygons_path : str
        GeoJSON/Shapefile/GPKG path.
    out_dir : str
        Folder to write PNGs.
    wayback_date : str
        Must be a key in WAYBACK_DATE_TO_TIMEID.
    mode : str
        "individual" -> one PNG per polygon feature
        "grouped" -> cluster by centroid distance, one PNG per group
    group_eps_m : float
        Clustering distance (meters) for grouped mode (DBSCAN eps).
    id_field : str or None
        If set, uses this attribute as filename stem (safe-sanitized).
    style : ExportStyle
        Styling options (scalebar, coords, north arrow, padding, dpi, etc.)
    assume_crs : str
        Used if polygons have missing CRS metadata.
    max_panels : int or None
        Limit number of exported panels (useful for quick tests).

    Returns
    -------
    list[str]
        Paths of saved PNGs.
    """
    if style is None:
        style = ExportStyle()

    if wayback_date not in WAYBACK_DATE_TO_TIMEID:
        raise ValueError(f"wayback_date='{wayback_date}' not in WAYBACK_DATE_TO_TIMEID keys: {list(WAYBACK_DATE_TO_TIMEID)}")
    timeid = WAYBACK_DATE_TO_TIMEID[wayback_date]
    tile_url = wayback_xyz(timeid)

    os.makedirs(out_dir, exist_ok=True)

    gdf = gpd.read_file(polygons_path)
    if gdf.empty:
        return []

    if gdf.crs is None:
        gdf = gdf.set_crs(assume_crs)

    gdf3857 = gdf.to_crs(epsg=3857).copy()

    def _safe_name(x: str) -> str:
        s = str(x)
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)[:80]

    saved: list[str] = []

    if mode == "individual":
        n = len(gdf3857)
        if max_panels is not None:
            n = min(n, max_panels)

        for i in range(n):
            sub = gdf3857.iloc[[i]].copy()

            if id_field and id_field in sub.columns:
                stem = _safe_name(sub.iloc[0][id_field])
            else:
                stem = f"panel_{i:03d}"

            out_png = os.path.join(out_dir, f"{stem}_wayback_{wayback_date}.png")
            _export_one_panel(sub, tile_url, out_png, title=stem, style=style)
            saved.append(out_png)

    elif mode == "grouped":
        if DBSCAN is None:
            raise RuntimeError("Grouped mode requires scikit-learn. `pip install scikit-learn`")

        cent = np.vstack([gdf3857.geometry.centroid.x.values,
                          gdf3857.geometry.centroid.y.values]).T
        labels = DBSCAN(eps=group_eps_m, min_samples=group_min_samples).fit_predict(cent)
        gdf3857["group_id"] = labels

        groups = sorted(gdf3857["group_id"].unique())
        if max_panels is not None:
            groups = groups[:max_panels]

        for gid in groups:
            sub = gdf3857[gdf3857["group_id"] == gid].copy()
            stem = f"group_{gid:03d}"
            out_png = os.path.join(out_dir, f"{stem}_wayback_{wayback_date}.png")
            title = f"{stem} ({len(sub)} polys)"
            _export_one_panel(sub, tile_url, out_png, title=title, style=style)
            saved.append(out_png)

    else:
        raise ValueError("mode must be 'individual' or 'grouped'")

    return saved


def _export_one_panel(gdf3857: gpd.GeoDataFrame, tile_url: str, out_png: str, title: str, style: ExportStyle):
    xmin, ymin, xmax, ymax = gdf3857.total_bounds
    xmin -= style.pad_m; ymin -= style.pad_m; xmax += style.pad_m; ymax += style.pad_m

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ctx.add_basemap(ax, source=tile_url, crs=gdf3857.crs)

    gdf3857.boundary.plot(ax=ax, linewidth=style.line_width)

    if style.draw_scalebar:
        add_scalebar_arcgis(ax, location=style.scalebar_location)
    if style.draw_north_arrow:
        add_north_arrow(ax, location=style.north_location)
    if style.draw_coords:
        add_lonlat_edges(ax, n_ticks=style.n_coord_ticks)
    else:
        ax.set_axis_off()

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
