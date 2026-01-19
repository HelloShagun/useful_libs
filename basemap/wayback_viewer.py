from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from ipyleaflet import Map, TileLayer, WidgetControl, basemaps, LayersControl
from ipywidgets import SelectionSlider, Dropdown, HBox, VBox, Text, Button, Label


# -------------------------
# 1) Your FIXED release list
# -------------------------
# Keep this static. Add/remove dates whenever you want.
DATE_TIMEID_MAPPING: List[Tuple[str, int]] = [
    ("2016-01-13", 3515),
    ("2017-01-11", 577),
    ("2018-01-08", 13161),
    ("2019-01-09", 6036),
    ("2020-01-08", 23001),
    ("2021-01-13", 1049),
    ("2022-01-12", 42663),
    ("2023-01-11", 11475),
]


# -------------------------
# 2) URL builder (Wayback)
# -------------------------
def make_wayback_url(time_id: int) -> str:
    return (
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
        "world_imagery/wmts/1.0.0/default028mm/mapserver/tile/"
        f"{time_id}" + "/{z}/{y}/{x}"
    )


# -------------------------
# 3) Basemap options
# -------------------------
BASEMAPS = {
    "OpenStreetMap (Mapnik)": basemaps.OpenStreetMap.Mapnik,  # OSM ✅
    "CartoDB Positron": basemaps.CartoDB.Positron,
    "CartoDB DarkMatter": basemaps.CartoDB.DarkMatter,
    "Esri WorldImagery": basemaps.Esri.WorldImagery,
    "Esri WorldTopoMap": basemaps.Esri.WorldTopoMap,
    "Esri WorldStreetMap": basemaps.Esri.WorldStreetMap,
}


def _closest_timeid_for_year_month(
    year: int,
    month: Optional[int],
    mapping: List[Tuple[str, int]],
) -> int:
    """
    Pick the closest available date in your fixed list to (year, month).
    If month is None, pick first date within that year if available, else closest year.
    """
    # Parse mapping dates into (y,m,d,timeid)
    parsed = []
    for date_str, tid in mapping:
        y, m, d = [int(x) for x in date_str.split("-")]
        parsed.append((y, m, d, tid, date_str))

    # If user gave a month, minimize absolute difference in (year, month)
    if month is not None:
        target = year * 12 + (month - 1)
        best = min(parsed, key=lambda r: abs((r[0] * 12 + (r[1] - 1)) - target))
        return best[3]

    # If month not given: try exact year first
    in_year = [r for r in parsed if r[0] == year]
    if in_year:
        # choose earliest in that year
        in_year_sorted = sorted(in_year, key=lambda r: (r[1], r[2]))
        return in_year_sorted[0][3]

    # else closest year overall
    best = min(parsed, key=lambda r: abs(r[0] - year))
    return best[3]


class WaybackViewer:
    """
    Reusable ipyleaflet viewer:
      - fixed date<->timeId list (no auto updates)
      - basemap dropdown (includes OSM)
      - slider for Wayback date
      - optional 'Go to year/month' prompt
    """

    def __init__(
        self,
        center: Tuple[float, float] = (52.2053, 0.1218),
        zoom: int = 14,
        mapping: List[Tuple[str, int]] = DATE_TIMEID_MAPPING,
        default_basemap_name: str = "Esri WorldImagery",
    ):
        self.mapping = mapping

        # Map
        self.m = Map(center=center, zoom=zoom, basemap=BASEMAPS[default_basemap_name], scroll_wheel_zoom=True)
        self.m.add_control(LayersControl(position="bottomright"))

        # Slider (shows dates, returns timeId)
        self.slider = SelectionSlider(
            options=self.mapping,              # list of (label, value)
            value=self.mapping[0][1],
            description="Wayback",
            continuous_update=True,
            orientation="horizontal",
            readout=True,
        )

        # Basemap dropdown
        self.basemap_dd = Dropdown(
            options=list(BASEMAPS.keys()),
            value=default_basemap_name,
            description="Basemap",
        )

        # Wayback layer
        self.wayback = TileLayer(url=make_wayback_url(self.slider.value), name="Esri Wayback")
        self.m.add_layer(self.wayback)

        # Wiring
        self.slider.observe(self._on_date_change, names="value")
        self.basemap_dd.observe(self._on_basemap_change, names="value")

        # “Go to year/month” mini prompt (year required, month optional)
        self.year_box = Text(value="2022", description="Year", placeholder="e.g., 2022")
        self.month_box = Text(value="", description="Month", placeholder="1-12 (optional)")
        self.go_btn = Button(description="Go", button_style="primary")
        self.status = Label(value="Tip: type Year (+ optional Month) and click Go.")

        self.go_btn.on_click(self._on_go_clicked)

        # Put controls on map
        controls_ui = VBox([
            HBox([self.basemap_dd]),
            self.slider,
            HBox([self.year_box, self.month_box, self.go_btn]),
            self.status
        ])
        self.m.add_control(WidgetControl(widget=controls_ui, position="topright"))

    def _on_date_change(self, change):
        self.wayback.url = make_wayback_url(change["new"])
        self.wayback.redraw()

        # update status with selected date label
        tid = change["new"]
        date = next((d for d, t in self.mapping if t == tid), str(tid))
        self.status.value = f"Showing Wayback date: {date}"

    def _on_basemap_change(self, change):
        self.m.basemap = BASEMAPS[change["new"]]

    def _on_go_clicked(self, _):
        # Parse year/month from text boxes
        try:
            year = int(self.year_box.value.strip())
        except Exception:
            self.status.value = "Year must be an integer (e.g., 2022)."
            return

        month_txt = self.month_box.value.strip()
        month = None
        if month_txt:
            try:
                month = int(month_txt)
                if not (1 <= month <= 12):
                    raise ValueError()
            except Exception:
                self.status.value = "Month must be 1–12 (or leave blank)."
                return

        tid = _closest_timeid_for_year_month(year, month, self.mapping)
        self.slider.value = tid  # triggers redraw via observer

    def show(self):
        """Return the ipyleaflet map widget."""
        return self.m
