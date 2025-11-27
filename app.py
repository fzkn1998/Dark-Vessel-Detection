#!/usr/bin/env python3
# Flask app: Dark Vessel Reporter (Pixel Scene + World Map)
from pathlib import Path
import io, base64, warnings
from flask import Flask, render_template, abort
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as rio_transform
from pyproj import CRS, Transformer
from PIL import Image
import plotly.graph_objects as go

app = Flask(__name__)

# ==================== CONFIG (edit these paths) ====================
DETECTIONS_CSV   = Path(r"C:\Users\fayya\Desktop\Dark_Vessel_App\data\f9fd0c2000d1a632p.csv")   # predictions
AIS_CSV          = Path(r"C:\Users\fayya\Desktop\Dark_Vessel_App\data\public.csv")              # AIS data
GEOTIFF_ROOT     = Path(r"C:\Users\fayya\Desktop\Dark_Vessel_App")
GEOTIFF_FILENAME = Path(r"C:\Users\fayya\Desktop\Dark_Vessel_App\data\VV_dB.tif")  # abs/rel ok

DEFAULT_FLIPLR   = False
DEFAULT_FLIPUD   = False
MAX_DIST_M       = 50.0
SHOW_SCENE_RASTER = True
RASTER_MAX_SIZE   = 1024

# thresholds
MIN_OBJECTNESS_P = 0.30
MIN_IS_VESSEL_P  = 0.50
MIN_IS_FISHING_P = None  # e.g. 0.5 for fishing-only

# boxes
SHOW_BOXES_DEFAULT = True
MAX_BOXES_DEFAULT  = 200
MIN_BOX_PX         = 6

# ==================== helpers ====================
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def _is_abs(p: str) -> bool:
    try: return Path(p).is_absolute()
    except: return False

def get_tif_path(scene_id: str) -> Path:
    if GEOTIFF_FILENAME and _is_abs(str(GEOTIFF_FILENAME)):
        return Path(GEOTIFF_FILENAME)
    return Path(GEOTIFF_ROOT) / scene_id / GEOTIFF_FILENAME

# -------- lon/lat <-> pixels ----------
def ensure_lonlat_from_pixels(df, tif_path: Path, fliplr=False, flipud=False, who="detections"):
    if {"lon","lat"}.issubset(df.columns): return df
    row_col = find_col(df, ["detect_scene_row","row","y"])
    col_col = find_col(df, ["detect_scene_column","col","x"])
    if row_col is None or col_col is None:
        raise ValueError(f"{who} need lon/lat or detect_scene_row/column.")
    if not tif_path or not tif_path.exists():
        raise ValueError(f"{who} lon/lat missing and GeoTIFF not found: {tif_path}")
    with rasterio.open(tif_path) as ds:
        w, h = ds.width, ds.height
        rows = df[row_col].astype(float).to_numpy()
        cols = df[col_col].astype(float).to_numpy()
        if fliplr: cols = (w - 1) - cols
        if flipud: rows = (h - 1) - rows
        xs, ys = zip(*[xy(ds.transform, r, c, offset="center") for r, c in zip(rows, cols)])
        if ds.crs is None: raise ValueError("GeoTIFF has no CRS.")
        if CRS.from_user_input(ds.crs).is_geographic:
            lon, lat = np.array(xs), np.array(ys)
        else:
            to84 = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            lon, lat = to84.transform(np.array(xs), np.array(ys))
    out = df.copy(); out["lon"], out["lat"] = lon, lat
    return out

def ensure_ais_lonlat(ais_df: pd.DataFrame, tif_path: Path|None):
    lon_col = find_col(ais_df, ["lon","longitude","x","lon_deg","LONGITUDE","Lon","Lon_dd"])
    lat_col = find_col(ais_df, ["lat","latitude","y","lat_deg","LATITUDE","Lat","Lat_dd"])
    if lon_col and lat_col:
        if "lon" not in ais_df.columns: ais_df = ais_df.rename(columns={lon_col:"lon"})
        if "lat" not in ais_df.columns: ais_df = ais_df.rename(columns={lat_col:"lat"})
        return ais_df
    if tif_path: return ensure_lonlat_from_pixels(ais_df, tif_path, who="AIS")
    raise ValueError("AIS data lacks lon/lat and no GeoTIFF to compute from pixels.")

def ensure_pixels_from_lonlat(df, tif_path: Path):
    if not tif_path.exists(): raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")
    if not {"lon","lat"}.issubset(df.columns):
        raise ValueError("need lon/lat to compute pixel positions")
    with rasterio.open(tif_path) as ds:
        if ds.crs is None: raise ValueError("GeoTIFF has no CRS")
        if CRS.from_user_input(ds.crs).to_epsg() == 4326:
            xs, ys = df["lon"].to_numpy(), df["lat"].to_numpy()
        else:
            xs, ys = rio_transform("EPSG:4326", ds.crs, df["lon"].to_numpy(), df["lat"].to_numpy())
        rc = [ds.index(x, y) for x, y in zip(xs, ys)]
        rows = np.array([r for r, _ in rc], float)
        cols = np.array([c for _, c in rc], float)
    out = df.copy(); out["px"], out["py"] = cols, rows
    return out

# ---------- spatial ----------
def utm_crs_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)

def nearest_match(dets_gdf, ais_gdf, max_dist_m):
    if dets_gdf.empty:
        return pd.Series([], dtype=bool), pd.Series([], dtype=float)
    cx, cy = dets_gdf.geometry.x.mean(), dets_gdf.geometry.y.mean()
    local = utm_crs_for_lonlat(cx, cy)
    D, A = dets_gdf.to_crs(local), (ais_gdf.to_crs(local) if not ais_gdf.empty else ais_gdf)
    if A.empty:
        return pd.Series([False]*len(D), bool), pd.Series([np.nan]*len(D), float)
    joined = gpd.sjoin_nearest(D, A, how="left", distance_col="dist_m")
    matched = joined["dist_m"].notna() & (joined["dist_m"] <= max_dist_m)
    return matched.reset_index(drop=True), joined["dist_m"].reset_index(drop=True)

# ---------- raster -> PNG ----------
def raster_to_png_and_extent(tif_path: Path, max_size=2048):
    with rasterio.open(tif_path) as ds:
        arr = ds.read()  # (bands,H,W)
        H_orig, W_orig = arr.shape[1], arr.shape[2]
        try: m = ds.read_masks(1)
        except: m = None
        if m is not None:
            arr = arr.astype(np.float32)
            for b in range(arr.shape[0]):
                band = arr[b]; band[m == 0] = np.nan; arr[b] = band
        # resample (nearest by sub-sampling) to max_size
        H, W = H_orig, W_orig
        if max(H, W) > max_size:
            scale = max_size / float(max(H, W))
            new_h, new_w = max(1, int(H*scale)), max(1, int(W*scale))
            yy = (np.linspace(0, H-1, new_h)).astype(int)
            xx = (np.linspace(0, W-1, new_w)).astype(int)
            arr = arr[:, yy][:, :, xx]; H, W = new_h, new_w
        # contrast
        if arr.shape[0] == 1:
            band = arr[0]; finite = np.isfinite(band)
            if not finite.any(): band[:] = 0; finite = np.ones_like(band, bool)
            p2, p98 = np.nanpercentile(band[finite], [2, 98])
            if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                p2, p98 = float(np.nanmin(band[finite])), float(np.nanmax(band[finite]))
                if p98 <= p2: p2, p98 = 0.0, 1.0
            band = np.clip((band - p2)/(p98 - p2 + 1e-9), 0, 1); band[~finite] = 0.0
            img = (band*255).astype(np.uint8)
        else:
            rgb = []
            for i in range(3):
                b = arr[i]; finite = np.isfinite(b)
                if not finite.any(): b[:] = 0; finite = np.ones_like(b, bool)
                p2, p98 = np.nanpercentile(b[finite], [2, 98])
                if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                    p2, p98 = float(np.nanmin(b[finite])), float(np.nanmax(b[finite]))
                    if p98 <= p2: p2, p98 = 0.0, 1.0
                b = np.clip((b - p2)/(p98 - p2 + 1e-9), 0, 1); b[~finite] = 0.0
                rgb.append((b*255).astype(np.uint8))
            img = np.dstack(rgb)
        # encode PNG
        buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG", optimize=True)
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # bounds in WGS84 (unused in raster plot, kept for completeness)
        left, bottom, right, top = ds.bounds
        if ds.crs is None: raise ValueError("GeoTIFF has no CRS")
        if CRS.from_user_input(ds.crs).to_epsg() == 4326:
            xmin, ymin, xmax, ymax = left, bottom, right, top
        else:
            (xmin, ymin), (xmax, ymax) = rio_transform(ds.crs, "EPSG:4326", [left, right], [bottom, top])
    return png_b64, (xmin, ymin, xmax, ymax), (W, H), (W_orig, H_orig)

def meters_per_pixel(tif_path: Path):
    with rasterio.open(tif_path) as ds:
        H, W = ds.height, ds.width
        r = H/2.0; c = W/2.0
        x0, y0 = xy(ds.transform, r, c, offset='center')
        x1, y1 = xy(ds.transform, r, c+1, offset='center')
        x2, y2 = xy(ds.transform, r+1, c, offset='center')
        if ds.crs is None: raise ValueError("GeoTIFF has no CRS")
        crs = CRS.from_user_input(ds.crs)
        if crs.is_projected:
            mx = abs(x1 - x0); my = abs(y2 - y0)
            return float(mx), float(my)
        else:
            to84 = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            lon0, lat0 = to84.transform(x0, y0)
            lon1, lat1 = to84.transform(x1, y1)
            lon2, lat2 = to84.transform(x2, y2)
            utm = utm_crs_for_lonlat(lon0, lat0)
            to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
            X0, Y0 = to_utm.transform(lon0, lat0)
            X1, Y1 = to_utm.transform(lon1, lat1)
            X2, Y2 = to_utm.transform(lon2, lat2)
            mx = abs(X1 - X0); my = abs(Y2 - Y0)
            return float(mx), float(my)

# ---------- boxes ----------
def _detect_bbox_columns(df: pd.DataFrame):
    xyxy_candidates = [
        ("xmin","ymin","xmax","ymax"),
        ("bbox_xmin","bbox_ymin","bbox_xmax","bbox_ymax"),
        ("x1","y1","x2","y2"),
    ]
    for cols in xyxy_candidates:
        if all(c in df.columns for c in cols):
            return ("xyxy", cols)
    center_wh_candidates = [
        ("px","py","w","h"),
        ("px","py","width","height"),
        ("detect_scene_column","detect_scene_row","w","h"),
        ("x","y","w","h"),
    ]
    for cols in center_wh_candidates:
        if all(c in df.columns for c in cols):
            return ("cwh", cols)
    return (None, None)

def _square_box_from_length(px, py, length_m, mx_per_px, my_per_px):
    if not np.isfinite(length_m):
        side_x = side_y = MIN_BOX_PX
    else:
        px_per_m_x = 1.0/max(mx_per_px, 1e-6)
        px_per_m_y = 1.0/max(my_per_px, 1e-6)
        side_x = max(MIN_BOX_PX, float(length_m) * px_per_m_x)
        side_y = max(MIN_BOX_PX, float(length_m) * px_per_m_y)
    x0 = float(px) - side_x/2.0; x1 = float(px) + side_x/2.0
    y0 = float(py) - side_y/2.0; y1 = float(py) + side_y/2.0
    return x0, y0, x1, y1

def _boxes_from_df(df: pd.DataFrame, mx_per_px, my_per_px):
    mode, cols = _detect_bbox_columns(df)
    if mode == "xyxy":
        x0c,y0c,x1c,y1c = cols
        for _, r in df.iterrows():
            yield float(r[x0c]), float(r[y0c]), float(r[x1c]), float(r[y1c])
    elif mode == "cwh":
        cx, cy, wc, hc = cols
        for _, r in df.iterrows():
            w = float(r[wc]); h = float(r[hc])
            x0 = float(r[cx]) - w/2.0; x1 = float(r[cx]) + w/2.0
            y0 = float(r[cy]) - h/2.0; y1 = float(r[cy]) + h/2.0
            yield x0, y0, x1, y1
    else:
        length_col = find_col(df, ["vessel_length_m","length_m","len_m","length"])
        for _, r in df.iterrows():
            length_m = float(r[length_col]) if (length_col and pd.notna(r.get(length_col))) else np.nan
            px = r.get("px", r.get("detect_scene_column", r.get("x", np.nan)))
            py = r.get("py", r.get("detect_scene_row", r.get("y", np.nan)))
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            yield _square_box_from_length(px, py, length_m, mx_per_px, my_per_px)

def _scale_box_to_display(box, sx, sy):
    x0,y0,x1,y1 = box
    return (x0*sx, y0*sy, x1*sx, y1*sy)

# ---------- plots ----------
_PLOT_CONFIG = dict(displaylogo=False, scrollZoom=True, showTips=False)

def make_map_raster(det_df, ais_df, tif_path: Path, title, draw_boxes=True, max_boxes=500):
    png_b64, _, (w_disp, h_disp), (w_orig, h_orig) = raster_to_png_and_extent(tif_path, RASTER_MAX_SIZE)
    sx = (w_disp-1)/max(w_orig-1, 1) if w_orig > 1 else 1.0
    sy = (h_disp-1)/max(h_orig-1, 1) if h_orig > 1 else 1.0
    try: mx_per_px, my_per_px = meters_per_pixel(tif_path)
    except Exception: mx_per_px = my_per_px = np.nan

    fig = go.Figure()
    fig.update_layout(
        images=[dict(source="data:image/png;base64,"+png_b64, xref="x", yref="y",
                     x=0, y=0, sizex=w_disp, sizey=h_disp, sizing="stretch", layer="below", opacity=1.0)],
        title=dict(text=title, font=dict(size=18, family="Inter, SF Pro Display, -apple-system, sans-serif", color="#1e293b", weight="bold"), x=0.5, xanchor="center"),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=15, weight="bold"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1),
        margin=dict(l=10, r=10, t=45, b=10), dragmode="zoom",
        hoverlabel=dict(bgcolor="#0f172a", font_size=16, font_color="#fff")
    )
    fig.update_xaxes(range=[0, w_disp], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[h_disp, 0], showgrid=False, zeroline=False, visible=False)

    # remap to display if px/py present
    det_df = det_df.copy()
    if "px" in det_df.columns and "py" in det_df.columns:
        det_df["px_disp"] = det_df["px"] * sx
        det_df["py_disp"] = det_df["py"] * sy
    ais_df = ais_df.copy()
    if len(ais_df) and ("px" in ais_df.columns and "py" in ais_df.columns):
        ais_df["px_disp"] = ais_df["px"] * sx
        ais_df["py_disp"] = ais_df["py"] * sy

    det_m = det_df[det_df["matched"] == True]
    det_d = det_df[det_df["matched"] != True]
    det_hover = ("<b>🚢 %{name}</b><br>"
                 "<b>📍 Location:</b> %{customdata[8]:.6f}°E, %{customdata[9]:.6f}°N<br>"
                 "<b>🖼️ Scene:</b> %{customdata[7]} (pixel: %{x:.0f}, %{y:.0f})<br>"
                 "<b>📏 Vessel Length:</b> %{customdata[2]:.1f}m<br>"
                 "<b>🎯 Detection Confidence:</b><br>"
                 "• Objectness: %{customdata[3]:.1%}<br>"
                 "• Is Vessel: %{customdata[4]:.1%}<br>"
                 "• Is Fishing: %{customdata[5]:.1%}<br>"
                 "<b>📡 AIS Status:</b> %{customdata[6]:.1f}m from nearest AIS vessel<extra></extra>")

    if len(det_d):
        fig.add_trace(go.Scatter(
            x=det_d.get("px_disp", det_d["px"]), y=det_d.get("py_disp", det_d["py"]),
            mode="markers", name="🚢 Dark Vessel (No AIS)",
            visible="legendonly",
            marker=dict(
                symbol="diamond", 
                size=12, 
                color="#000000",
                line=dict(width=3, color="#ffffff")
            ),
            customdata=np.c_[det_d.get("detect_scene_row", pd.Series([""]*len(det_d))),
                             det_d.get("detect_scene_column", pd.Series([""]*len(det_d))),
                             det_d.get("vessel_length_m", pd.Series([np.nan]*len(det_d))),
                             det_d.get("objectness_p", pd.Series([np.nan]*len(det_d))),
                             det_d.get("is_vessel_p", pd.Series([np.nan]*len(det_d))),
                             det_d.get("is_fishing_p", pd.Series([np.nan]*len(det_d))),
                             det_d.get("match_dist_m", pd.Series([np.nan]*len(det_d))),
                             det_d.get("scene_id", pd.Series([""]*len(det_d))),
                             det_d["lon"], det_d["lat"]],
            hovertemplate=det_hover
        ))
    if len(det_m):
        fig.add_trace(go.Scatter(
            x=det_m.get("px_disp", det_m["px"]), y=det_m.get("py_disp", det_m["py"]),
            mode="markers", name="✅ Vessel (AIS Matched)",
            visible="legendonly",
            marker=dict(
                symbol="circle", 
                size=8, 
                color="#2563eb",
                line=dict(width=2, color="#ffffff")
            ),
            customdata=np.c_[det_m.get("detect_scene_row", pd.Series([""]*len(det_m))),
                             det_m.get("detect_scene_column", pd.Series([""]*len(det_m))),
                             det_m.get("vessel_length_m", pd.Series([np.nan]*len(det_m))),
                             det_m.get("objectness_p", pd.Series([np.nan]*len(det_m))),
                             det_m.get("is_vessel_p", pd.Series([np.nan]*len(det_m))),
                             det_m.get("is_fishing_p", pd.Series([np.nan]*len(det_m))),
                             det_m.get("match_dist_m", pd.Series([np.nan]*len(det_m))),
                             det_m.get("scene_id", pd.Series([""]*len(det_m))),
                             det_m["lon"], det_m["lat"]],
            hovertemplate=det_hover
        ))
    if len(ais_df):
        ais_row  = ais_df.get("detect_scene_row", pd.Series([""]*len(ais_df)))
        ais_col  = ais_df.get("detect_scene_column", pd.Series([""]*len(ais_df)))
        ais_len  = ais_df.get("vessel_length_m", pd.Series([np.nan]*len(ais_df)))
        ais_obj  = ais_df.get("objectness_p", pd.Series([np.nan]*len(ais_df)))
        # prefer boolean flags if present; otherwise fall back to probability columns
        vcol = find_col(ais_df, ["is_vessel","is_vessel_flag","isVessel","is_vessel_p"])
        fcol = find_col(ais_df, ["is_fishing","isFishing","is_fishing_flag","is_fishing_p"])
        ais_ivp  = ais_df.get(vcol, pd.Series([np.nan]*len(ais_df))) if vcol else pd.Series([np.nan]*len(ais_df))
        ais_ifp  = ais_df.get(fcol, pd.Series([np.nan]*len(ais_df))) if fcol else pd.Series([np.nan]*len(ais_df))
        ais_lab  = ais_df.get("label", ais_df.get("type", pd.Series([""]*len(ais_df))))
        ais_sid  = ais_df.get("scene_id", pd.Series([""]*len(ais_df)))
        ais_hover = ("<b>📡 AIS Tracked Vessel</b><br>"
                     "<b>📍 Location:</b> %{customdata[8]:.6f}°E, %{customdata[9]:.6f}°N<br>"
                     "<b>🖼️ Scene:</b> %{customdata[7]} (pixel: %{x:.0f}, %{y:.0f})<br>"
                     "<b>📏 Vessel Length:</b> %{customdata[2]:.1f}m<br>"
                     "<b>Is Vessel:</b> %{customdata[4]}<br>"
                     "<b>Is Fishing:</b> %{customdata[5]}<br>"
                     "<b>✅ Status:</b> Tracked by AIS system<extra></extra>")
        fig.add_trace(go.Scatter(
            x=ais_df.get("px_disp", ais_df["px"]), y=ais_df.get("py_disp", ais_df["py"]),
            mode="markers", name="📡 AIS Tracked Vessels",
            visible="legendonly",
            marker=dict(
                symbol="x", 
                size=12, 
                color="#dc2626", 
                line=dict(width=3, color="#ffffff")
            ),
            customdata=np.c_[ais_row, ais_col, ais_len, ais_obj, ais_ivp, ais_ifp, ais_lab, ais_sid, ais_df["lon"], ais_df["lat"]],
            hovertemplate=ais_hover
        ))

    # boxes
    if draw_boxes:
        shapes = []
        try: mx_per_px, my_per_px = meters_per_pixel(tif_path)
        except Exception: mx_per_px = my_per_px = np.nan
        if len(det_d):
            count = 0
            for box in _boxes_from_df(det_d, mx_per_px, my_per_px):
                x0,y0,x1,y1 = _scale_box_to_display(box, sx, sy)
                shapes.append(dict(type="rect", xref="x", yref="y",
                                   x0=x0, y0=y0, x1=x1, y1=y1,
                                   line=dict(color="black", width=1)))
                count += 1
                if count >= max_boxes: break
        if len(det_m):
            count = 0
            for box in _boxes_from_df(det_m, mx_per_px, my_per_px):
                x0,y0,x1,y1 = _scale_box_to_display(box, sx, sy)
                shapes.append(dict(type="rect", xref="x", yref="y",
                                   x0=x0, y0=y0, x1=x1, y1=y1,
                                   line=dict(color="royalblue", width=1)))
                count += 1
                if count >= max_boxes: break
        if len(ais_df):
            count = 0
            for box in _boxes_from_df(ais_df, mx_per_px, my_per_px):
                x0,y0,x1,y1 = _scale_box_to_display(box, sx, sy)
                shapes.append(dict(type="rect", xref="x", yref="y",
                                   x0=x0, y0=y0, x1=x1, y1=y1,
                                   line=dict(color="red", width=2)))
                count += 1
                if count >= max_boxes: break
        fig.update_layout(shapes=shapes)

    # inline plotly JS to avoid CDN dependency
    return fig.to_html(full_html=False, include_plotlyjs="inline", config=_PLOT_CONFIG)

def make_map_world(det_df, ais_df, title):
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top", font=dict(size=18, family="Inter, SF Pro Display, -apple-system, sans-serif", color="#1e293b", weight="bold"), x=0.5, xanchor="center"), 
                      geo=dict(projection_type="natural earth"),
                      legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="right", x=0.99, font=dict(size=15, weight="bold"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#e5e7eb", borderwidth=1),
                      margin=dict(l=0, r=0, t=35, b=0), dragmode="zoom",
                      hoverlabel=dict(bgcolor="#0f172a", font_size=16, font_color="#fff"))
    fig.update_geos(fitbounds="locations", visible=True, projection_scale=2.2)
    det_m = det_df[det_df["matched"] == True]
    det_d = det_df[det_df["matched"] != True]
    det_hover = ("<b>🚢 %{name}</b><br>"
                 "<b>📍 Location:</b> %{lon:.6f}°E, %{lat:.6f}°N<br>"
                 "<b>🖼️ Scene:</b> %{customdata[7]} (pixel: %{customdata[1]}, %{customdata[0]})<br>"
                 "<b>📏 Vessel Length:</b> %{customdata[2]:.1f}m<br>"
                 "<b>🎯 Detection Confidence:</b><br>"
                 "• Objectness: %{customdata[3]:.1%}<br>"
                 "• Is Vessel: %{customdata[4]:.1%}<br>"
                 "• Is Fishing: %{customdata[5]:.1%}<br>"
                 "<b>📡 AIS Status:</b> %{customdata[6]:.1f}m from nearest AIS vessel<extra></extra>")
    if len(det_d):
        fig.add_trace(go.Scattergeo(lon=det_d["lon"], lat=det_d["lat"], mode="markers", name="🚢 Dark Vessel (No AIS)",
                                    visible="legendonly",
                                    marker=dict(
                                        symbol="diamond", 
                                        size=12, 
                                        color="#000000",
                                        line=dict(width=3, color="#ffffff")
                                    ),
                                    customdata=np.c_[det_d.get("detect_scene_row", pd.Series([""]*len(det_d))),
                                                     det_d.get("detect_scene_column", pd.Series([""]*len(det_d))),
                                                     det_d.get("vessel_length_m", pd.Series([np.nan]*len(det_d))),
                                                     det_d.get("objectness_p", pd.Series([np.nan]*len(det_d))),
                                                     det_d.get("is_vessel_p", pd.Series([np.nan]*len(det_d))),
                                                     det_d.get("is_fishing_p", pd.Series([np.nan]*len(det_d))),
                                                     det_d.get("match_dist_m", pd.Series([np.nan]*len(det_d))),
                                                     det_d.get("scene_id", pd.Series([""]*len(det_d)))] ,
                                    hovertemplate=det_hover))
    if len(det_m):
        fig.add_trace(go.Scattergeo(lon=det_m["lon"], lat=det_m["lat"], mode="markers", name="✅ Vessel (AIS Matched)",
                                    visible="legendonly",
                                    marker=dict(
                                        symbol="circle", 
                                        size=8, 
                                        color="#2563eb",
                                        line=dict(width=2, color="#ffffff")
                                    ),
                                    customdata=np.c_[det_m.get("detect_scene_row", pd.Series([""]*len(det_m))),
                                                     det_m.get("detect_scene_column", pd.Series([""]*len(det_m))),
                                                     det_m.get("vessel_length_m", pd.Series([np.nan]*len(det_m))),
                                                     det_m.get("objectness_p", pd.Series([np.nan]*len(det_m))),
                                                     det_m.get("is_vessel_p", pd.Series([np.nan]*len(det_m))),
                                                     det_m.get("is_fishing_p", pd.Series([np.nan]*len(det_m))),
                                                     det_m.get("match_dist_m", pd.Series([np.nan]*len(det_m))),
                                                     det_m.get("scene_id", pd.Series([""]*len(det_m)))] ,
                                    hovertemplate=det_hover))
    if len(ais_df):
        ais_row  = ais_df.get("detect_scene_row", pd.Series([""]*len(ais_df)))
        ais_col  = ais_df.get("detect_scene_column", pd.Series([""]*len(ais_df)))
        ais_len  = ais_df.get("vessel_length_m", pd.Series([np.nan]*len(ais_df)))
        ais_obj  = ais_df.get("objectness_p", pd.Series([np.nan]*len(ais_df)))
        vcol = find_col(ais_df, ["is_vessel","is_vessel_flag","isVessel","is_vessel_p"])
        fcol = find_col(ais_df, ["is_fishing","isFishing","is_fishing_flag","is_fishing_p"])
        ais_ivp  = ais_df.get(vcol, pd.Series([np.nan]*len(ais_df))) if vcol else pd.Series([np.nan]*len(ais_df))
        ais_ifp  = ais_df.get(fcol, pd.Series([np.nan]*len(ais_df))) if fcol else pd.Series([np.nan]*len(ais_df))
        ais_lab  = ais_df.get("label", ais_df.get("type", pd.Series([""]*len(ais_df))))
        ais_sid  = ais_df.get("scene_id", pd.Series([""]*len(ais_df)))
        ais_custom = np.c_[ais_row, ais_col, ais_len, ais_obj, ais_ivp, ais_ifp, ais_lab, ais_sid]
        ais_hover = ("<b>📡 AIS Tracked Vessel</b><br>"
                     "<b>📍 Location:</b> %{lon:.6f}°E, %{lat:.6f}°N<br>"
                     "<b>🖼️ Scene:</b> %{customdata[7]} (pixel: %{customdata[1]}, %{customdata[0]})<br>"
                     "<b>📏 Vessel Length:</b> %{customdata[2]:.1f}m<br>"
                     "<b>Is Vessel:</b> %{customdata[4]}<br>"
                     "<b>Is Fishing:</b> %{customdata[5]}<br>"
                     "<b>✅ Status:</b> Tracked by AIS system<extra></extra>")
        fig.add_trace(go.Scattergeo(lon=ais_df["lon"], lat=ais_df["lat"], mode="markers", name="📡 AIS Tracked Vessels",
                                    visible="legendonly",
                                    marker=dict(
                                        symbol="x", 
                                        size=12, 
                                        color="#dc2626", 
                                        line=dict(width=3, color="#ffffff")
                                    ),
                                    customdata=ais_custom, hovertemplate=ais_hover))
    return fig.to_html(full_html=False, include_plotlyjs="inline", config=_PLOT_CONFIG)

# ---------- filters & wrappers ----------
def apply_thresholds(df: pd.DataFrame, thr_obj, thr_vessel, thr_fishing):
    out = df.copy()
    if "objectness_p" in out.columns and thr_obj is not None:
        out = out[out["objectness_p"].astype(float) >= float(thr_obj)]
    if "is_vessel_p" in out.columns and thr_vessel is not None:
        out = out[out["is_vessel_p"].astype(float) >= float(thr_vessel)]
    if thr_fishing is not None and "is_fishing_p" in out.columns:
        out = out[out["is_fishing_p"].astype(float) >= float(thr_fishing)]
    return out.reset_index(drop=True)

def _load_dataframes():
    det = pd.read_csv(DETECTIONS_CSV)
    ais = pd.read_csv(AIS_CSV)
    if "scene_id" not in det.columns:
        det["scene_id"] = "scene_000"
    return det, ais

def list_summaries():
    det, ais = _load_dataframes()
    scenes = list(det["scene_id"].dropna().astype(str).unique())
    rows = []
    for sid in scenes:
        det_s = det[det["scene_id"] == sid].copy()
        det_s = apply_thresholds(det_s, MIN_OBJECTNESS_P, MIN_IS_VESSEL_P, MIN_IS_FISHING_P)
        tif_path = get_tif_path(sid)
        try:
            det_s = ensure_lonlat_from_pixels(det_s, tif_path, DEFAULT_FLIPLR, DEFAULT_FLIPUD, f"detections[{sid}]")
        except Exception:
            if not {"lon","lat"}.issubset(det_s.columns): raise
        ais_s = ais[ais["scene_id"] == sid].copy() if ("scene_id" in ais.columns) else ais.copy()
        try:
            ais_s = ensure_ais_lonlat(ais_s, tif_path if len(ais_s) else None)
        except Exception:
            ais_s = pd.DataFrame(columns=["lon","lat","scene_id"])
        det_gdf = gpd.GeoDataFrame(det_s.copy(), geometry=gpd.points_from_xy(det_s["lon"], det_s["lat"]), crs="EPSG:4326")
        ais_gdf  = gpd.GeoDataFrame(ais_s.copy(), geometry=gpd.points_from_xy(ais_s["lon"], ais_s["lat"]) if len(ais_s) else [], crs="EPSG:4326") if len(ais_s) else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        matched, dists = nearest_match(det_gdf, ais_gdf, MAX_DIST_M)
        det_s["matched"], det_s["match_dist_m"] = matched, dists
        rows.append({
            "scene_id": sid,
            "pred_total": len(det_s),
            "matched": int(det_s["matched"].sum()),
            "dark": int((det_s["matched"] != True).sum()),
            "ais_plotted": len(ais_s),
        })
    return pd.DataFrame(rows).sort_values("scene_id")

def build_scene_context(scene_id: str):
    det, ais = _load_dataframes()
    det_s = det[det["scene_id"] == scene_id].copy()
    if det_s.empty:
        raise FileNotFoundError(f"Scene '{scene_id}' not found in detections CSV.")
    det_s = apply_thresholds(det_s, MIN_OBJECTNESS_P, MIN_IS_VESSEL_P, MIN_IS_FISHING_P)
    tif_path = get_tif_path(scene_id)

    try:
        det_s = ensure_lonlat_from_pixels(det_s, tif_path, DEFAULT_FLIPLR, DEFAULT_FLIPUD, f"detections[{scene_id}]")
    except Exception:
        if not {"lon","lat"}.issubset(det_s.columns): raise

    ais_s = ais[ais["scene_id"] == scene_id].copy() if ("scene_id" in ais.columns) else ais.copy()
    try:
        ais_s = ensure_ais_lonlat(ais_s, tif_path if len(ais_s) else None)
    except Exception:
        warnings.warn(f"AIS data for scene {scene_id} could not get lon/lat; plotting without AIS data.")
        ais_s = pd.DataFrame(columns=["lon","lat","scene_id"])

    det_gdf = gpd.GeoDataFrame(det_s.copy(), geometry=gpd.points_from_xy(det_s["lon"], det_s["lat"]), crs="EPSG:4326")
    ais_gdf  = (gpd.GeoDataFrame(ais_s.copy(),
              geometry=gpd.points_from_xy(ais_s["lon"], ais_s["lat"]) if len(ais_s) else [],
              crs="EPSG:4326") if len(ais_s) else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326"))
    matched, dists = nearest_match(det_gdf, ais_gdf, MAX_DIST_M)
    det_s["matched"], det_s["match_dist_m"] = matched, dists

    if tif_path.exists() and SHOW_SCENE_RASTER:
        det_s = ensure_pixels_from_lonlat(det_s, tif_path)
        if len(ais_s): ais_s = ensure_pixels_from_lonlat(ais_s, tif_path)
    else:
        det_s["px"] = det_s.get("detect_scene_column", det_s.get("x"))
        det_s["py"] = det_s.get("detect_scene_row", det_s.get("y"))
        if len(ais_s):
            ais_s["px"] = ais_s.get("detect_scene_column", ais_s.get("x"))
            ais_s["py"] = ais_s.get("detect_scene_row", ais_s.get("y"))

    title = "Dark Vessels vs AIS Data"
    scene_map = make_map_raster(det_s, ais_s, tif_path, title, draw_boxes=SHOW_BOXES_DEFAULT, max_boxes=MAX_BOXES_DEFAULT) if (SHOW_SCENE_RASTER and tif_path.exists()) else make_map_world(det_s, ais_s, title)
    world_map = make_map_world(det_s, ais_s, "World Map View")

    pred_cols = ["scene_id","lon","lat","detect_scene_row","detect_scene_column","vessel_length_m","objectness_p","is_vessel_p","is_fishing_p","matched","match_dist_m"]
    dark_cols = ["scene_id","lon","lat","detect_scene_row","detect_scene_column","vessel_length_m","objectness_p","is_vessel_p","is_fishing_p","match_dist_m"]
    ais_candidates = ["scene_id","lon","lat","detect_scene_row","detect_scene_column","row","col","x","y","vessel_length_m","objectness_p","is_vessel_p","is_fishing_p","label","type","confidence"]
    ais_cols = [c for c in ais_candidates if c in ais_s.columns] or list(ais_s.columns)

    ctx = {
        "scene_id": scene_id,
        "scene_map_html": scene_map,
        "world_map_html": world_map,
        "predictions_table": det_s[pred_cols].to_html(index=False) if len(det_s) else "",
        "dark_table": det_s[det_s["matched"] != True][dark_cols].to_html(index=False) if len(det_s) else "",
        "gt_table": ais_s[ais_cols].to_html(index=False) if len(ais_s) else "",
    }
    return ctx

# ==================== ROUTES ====================
@app.route("/")
def index():
    summary_df = list_summaries()
    scenes = list(summary_df["scene_id"]) if len(summary_df) else []
    first_scene = scenes[0] if len(scenes) else None
    stats = {
        "num_scenes": int(len(summary_df)) if len(summary_df) else 0,
        "total_pred": int(summary_df["pred_total"].sum()) if len(summary_df) else 0,
        "total_matched": int(summary_df["matched"].sum()) if len(summary_df) else 0,
        "total_dark": int(summary_df["dark"].sum()) if len(summary_df) else 0,
    }
    return render_template(
        "index.html",
        scenes=scenes,
        summary_html=summary_df.to_html(index=False) if len(summary_df) else "<p>No data.</p>",
        stats=stats,
        first_scene=first_scene,
    )

@app.route("/scene/<scene_id>/")
def scene(scene_id):
    try:
        ctx = build_scene_context(scene_id)
    except Exception as e:
        abort(404, description=str(e))
    return render_template("scene.html", **ctx)

# ==================== MAIN ====================
# if __name__ == "__main__":
#     # Tip: use Flask dev server
#     app.run(host="127.0.0.1", port=8000, debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)