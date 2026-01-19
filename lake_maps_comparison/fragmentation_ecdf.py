from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io_stack import read_stack, get_band_mask
from .morphology import MorphologyConfig, clean_mask
from .objects import ObjectConfig, object_areas_ha, ecdf_xy


@dataclass
class FragmentationECDFConfig:
    """
    Config for plotting ECDF raw vs consolidated.

    Notes
    -----
    - Uses ObjectConfig for connectivity + pixel_area_m2.
    - Uses MorphologyConfig for merge_dist_m / fill_holes / min_area_m2.
    """
    xscale: str = "log"        # "log" or "linear"
    legend_ncol: int = 2
    legend_fontsize: int = 9
    figsize: Tuple[float, float] = (7.6, 4.8)
    grid_alpha: float = 0.25
    title: Optional[str] = None

def _plot_mask_grid(mask_dict, title="", show_diff=True):
    import matplotlib.pyplot as plt
    keys = list(mask_dict.keys())
    n = len(keys)
    rows = 3 if show_diff else 2

    fig, axes = plt.subplots(rows, n, figsize=(3.2*n, 3.2*rows))
    if n == 1:
        axes = np.array(axes).reshape(rows, 1)

    for j, k in enumerate(keys):
        raw = mask_dict[k]["raw"]
        con = mask_dict[k]["consolidated"]

        axes[0, j].imshow(raw, cmap="gray")
        axes[0, j].set_title(f"{k}\nraw")
        axes[0, j].axis("off")

        axes[1, j].imshow(con, cmap="gray")
        axes[1, j].set_title(f"{k}\nmerged")
        axes[1, j].axis("off")

        if show_diff:
            diff = (con.astype(int) - raw.astype(int))
            axes[2, j].imshow(diff, cmap="gray")
            axes[2, j].set_title(f"{k}\nmerged-raw")
            axes[2, j].axis("off")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig



def plot_ecdf_raw_vs_consolidated(
    *,
    tif_path: Optional[str] = None,
    arr: Optional[np.ndarray] = None,
    band_desc: Optional[Sequence[str]] = None,
    sources: Sequence[str],
    morph_cfg: MorphologyConfig,
    obj_cfg: ObjectConfig,
    colors: Optional[Dict[str, str]] = None,
    site: str = "",
    out_path_no_ext: Optional[str] = None,
    return_fig: bool = False,
    return_tables: bool = False,
    show_masks: bool = False,
    max_sources_to_show: int = 6,
    show_diff: bool = True,
) -> Union[None, Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes, pd.DataFrame]]:
    """
    Plot ECDF of object sizes (ha) for each source:
      - solid = raw
      - dashed = consolidated (merge_dist_m applied)

    Inputs
    ------
    Provide either:
      (A) tif_path
      OR
      (B) arr + band_desc

    Parameters
    ----------
    sources : list[str]
        Band names in stack (e.g., ["DW_Water", "AE_ge_70", ...])
    morph_cfg : MorphologyConfig
        Controls consolidation/cleaning (merge_dist_m, fill_holes, min_area_m2, etc.)
    obj_cfg : ObjectConfig
        Controls object connectivity + pixel_area_m2
    colors : dict[str, str]
        Optional color mapping per source.
    out_path_no_ext : str or None
        If provided, saves .png and .pdf to this path (without extension)
    return_fig : bool
        If True, returns (fig, ax)
    return_tables : bool
        If True, also returns a summary DataFrame with object counts (raw vs consolidated)

    Returns
    -------
    None (default) OR (fig, ax) OR (fig, ax, summary_df)
    """
    if tif_path is None and (arr is None or band_desc is None):
        raise ValueError("Provide either tif_path or (arr and band_desc).")

    if tif_path is not None:
        arr, meta = read_stack(tif_path)
        band_desc = meta.band_desc
    else:
        arr = np.asarray(arr)
        band_desc = list(band_desc)  # type: ignore

    # Prepare plot
    plt.style.use("seaborn-v0_8-paper")
    cfg = FragmentationECDFConfig()
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Summary table rows
    rows: List[dict] = []
    mask_debug = {}

    for s in sources:
        raw = get_band_mask(arr, band_desc, s)

        # consolidated = clean_mask (closing + holes + small-object removal)
        cons = clean_mask(raw, morph_cfg, pixel_area_m2=obj_cfg.pixel_area_m2, return_intermediate=False)
        if show_masks and len(mask_debug) < max_sources_to_show:
          mask_debug[s] = {"raw": raw, "consolidated": cons}

        areas_raw = object_areas_ha(raw, pixel_area_m2=obj_cfg.pixel_area_m2, connectivity=obj_cfg.connectivity)
        areas_con = object_areas_ha(cons, pixel_area_m2=obj_cfg.pixel_area_m2, connectivity=obj_cfg.connectivity)

        xr, yr = ecdf_xy(areas_raw)
        xc, yc = ecdf_xy(areas_con)

        color = (colors or {}).get(s, None)

        if xr is not None:
            ax.plot(xr, yr, linestyle="-", linewidth=2, color=color, label=f"{s} (raw)")
        if xc is not None:
            ax.plot(xc, yc, linestyle="--", linewidth=2, color=color,
                    label=f"{s} (merged {morph_cfg.merge_dist_m:.0f} m)")

        rows.append({
            "source": s,
            "merge_dist_m": float(morph_cfg.merge_dist_m),
            "n_objects_raw": int(len(areas_raw)),
            "n_objects_consolidated": int(len(areas_con)),
            "median_ha_raw": float(np.median(areas_raw)) if len(areas_raw) else np.nan,
            "median_ha_consolidated": float(np.median(areas_con)) if len(areas_con) else np.nan,
        })

    
    if show_masks and mask_debug:
        _plot_mask_grid(
            mask_debug,
            title=f"{site} — raw vs consolidated masks (merge={morph_cfg.merge_dist_m:.0f} m)",
            show_diff=show_diff
        )
    
    # Ax formatting
    if cfg.xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel("Object area (ha)")
    ax.set_ylabel("ECDF")

    title = cfg.title or (f"{site} — Object-size ECDF (raw vs consolidated)" if site else "Object-size ECDF (raw vs consolidated)")
    ax.set_title(title)

    ax.grid(alpha=cfg.grid_alpha)
    ax.legend(frameon=False, ncol=cfg.legend_ncol, fontsize=cfg.legend_fontsize)

    fig.tight_layout()

    # Save if requested
    if out_path_no_ext:
        fig.savefig(out_path_no_ext + ".pdf")
        fig.savefig(out_path_no_ext + ".png", dpi=600)

    summary_df = pd.DataFrame(rows)

    # Return options
    if return_tables and return_fig:
        return fig, ax, summary_df
    if return_fig:
        return fig, ax
    if return_tables:
        # still return fig? No, keep simple:
        return fig, ax, summary_df  # type: ignore

    plt.close(fig)
    return None
