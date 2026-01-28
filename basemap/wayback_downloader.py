import os
import requests
import math
from PIL import Image
from io import BytesIO
from ipyleaflet import DrawControl
from ipywidgets import Button, VBox, Text, Label, HBox

# Assuming you are in the root of your repo
from useful_libs.basemap.wayback_viewer import make_wayback_url

class WaybackDownloader:
    def __init__(self, viewer_instance):
        self.viewer = viewer_instance
        
        # 1. Add Draw Control for visual selection
        self.draw_control = DrawControl(
            rectangle={'shapeOptions': {'color': '#27ae60'}}, # Green for download
            circlemarker={}, polyline={}, polygon={}
        )
        self.viewer.m.add_control(self.draw_control)
        
        # 2. Add UI Elements for Directory and Download
        self.dir_input = Text(
            value='my_wayback_series',
            placeholder='folder_name',
            description='Save to:',
            style={'description_width': 'initial'}
        )
        
        self.btn_download = Button(
            description="Download Stitched Series", 
            button_style="success", # Green button
            icon='download',
            layout={'width': 'max-content'}
        )
        self.btn_download.on_click(self._on_download_click)
        
        # 3. Inject UI into the existing viewer's control panel
        download_ui = VBox([
            Label("--- Download Tool ---"),
            self.dir_input,
            self.btn_download
        ])
        
        for control in self.viewer.m.controls:
            if hasattr(control, 'widget') and isinstance(control.widget, VBox):
                control.widget.children += (download_ui,)

    @staticmethod
    def latlon_to_tile(lat, lon, z):
        lat_rad = math.radians(lat)
        n = 2.0 ** z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x, y

def run_download(self, bbox, zoom, folder_name):
    """
    bbox: (lat_min, lon_min, lat_max, lon_max)
    zoom: Set this to 19 or 20 for high res (0.3m/pixel - 0.6m/pixel)
    """
    os.makedirs(folder_name, exist_ok=True)
    lat_min, lon_min, lat_max, lon_max = bbox
    
    # 1. Calculate tile indices
    # lat_max is North (top), lat_min is South (bottom)
    x_start, y_start = self.latlon_to_tile(lat_max, lon_min, zoom)
    x_end, y_end = self.latlon_to_tile(lat_min, lon_max, zoom)
    
    # 2. Correct the ranges to ensure North -> South, West -> East
    x_min, x_max = min(x_start, x_end), max(x_start, x_end)
    y_min, y_max = min(y_start, y_end), max(y_start, y_end)
    
    grid_w = x_max - x_min + 1
    grid_h = y_max - y_min + 1
    
    # 3. High-Res Safety Check 
    # At zoom 20, a small neighborhood can easily exceed 25 tiles.
    # Increase this limit if you are sure your RAM can handle it.
    if grid_w * grid_h > 100: 
        print(f"Warning: Downloading {grid_w * grid_h} tiles. This may take a while.")

    for date_str, time_id in self.viewer.mapping:
        # Create a blank high-res canvas
        canvas = Image.new('RGB', (grid_w * 256, grid_h * 256))
        
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                url = make_wayback_url(time_id).format(z=zoom, x=x, y=y)
                try:
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        tile = Image.open(BytesIO(r.content))
                        # Paste using the corrected offsets
                        canvas.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))
                except Exception as e:
                    print(f"Failed at {x},{y} for {date_str}: {e}")
        
        save_path = os.path.join(folder_name, f"{date_str}.jpg")
        canvas.save(save_path, quality=95) # High quality JPEG

    def _on_download_click(self, _):
        if not self.draw_control.last_draw.get('geometry'):
            self.viewer.status.value = "Please draw a box on the map first!"
            return
        
        # Extract bbox from the drawing
        geom = self.draw_control.last_draw['geometry']['coordinates'][0]
        lons = [c[0] for c in geom]
        lats = [c[1] for c in geom]
        bbox = (min(lats), min(lons), max(lats), max(lons))
        
        # Execute using the directory name from the text box
        self.run_download(bbox, int(self.viewer.m.zoom), self.dir_input.value)
