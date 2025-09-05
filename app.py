# app.py
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.graph_objects as go
from pyproj import CRS, Transformer

FT_PER_M = 3.280839895  # meters -> feet

# -------------------------------
# Soil color map + legend order
# -------------------------------
SOIL_COLOR_MAP = {
    "Topsoil": "#ffffcb",
    "SM": "#76d7c4",
    "SC-SM": "#fff59d",
    "CL": "#c5cae9",
    "PWR": "#808080",
    "RF": "#929591",
    "ML": "#ef5350",
    "CL-ML": "#ef9a9a",
    "CH": "#64b5f6",
    "MH": "#ffb74d",
    "GM": "#aed581",
    "SC": "#81c784",
    "Rock": "#f8bbd0",
    "SM-SC": "#e1bee7",
    "SP": "#ce93d8",
    "SW": "#ba68c8",
    "GW": "#c8e6c9",
    "SM-ML": "#dcedc8",
    "CL-CH": "#fff176",
    "SC-CL": "#ffee58",
}
ORDERED_SOIL_TYPES = [
    "Topsoil", "SM", "SC-SM", "CL", "PWR", "RF", "ML", "CL-ML", "CH", "MH", "GM",
    "SC", "Rock", "SM-SC", "SP", "SW", "GW", "SM-ML", "CL-CH", "SC-CL"
]

# -------------------------------
# Column normalization (MAIN)
# -------------------------------
RENAME_MAP = {
    'Bore Log': 'Borehole',
    'Borehole': 'Borehole',
    'Elevation From': 'Elevation_From',
    'Elevation To': 'Elevation_To',
    'Soil Layer Description': 'Soil_Type',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'SPT N-Value': 'SPT',
}

def normalize_main_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    # rename if present
    for k, v in RENAME_MAP.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    need = ['Borehole', 'Elevation_From', 'Elevation_To', 'Soil_Type', 'Latitude', 'Longitude']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in MAIN Excel: {missing}")

    # extract USCS in parentheses or 'Topsoil'
    soil = df['Soil_Type'].astype(str)
    # try "(SM)" etc.
    extracted = soil.str.extract(r'\(([^)]+)\)')[0]
    # fallbacks
    extracted = extracted.fillna(
        soil.str.replace(r'^.*top\s*soil.*$', 'Topsoil', case=False, regex=True)
    )
    df['Soil_Type'] = extracted.fillna(soil).str.strip()

    # numeric coords
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # coerce elevations to numeric
    df['Elevation_From'] = pd.to_numeric(df['Elevation_From'], errors='coerce')
    df['Elevation_To'] = pd.to_numeric(df['Elevation_To'], errors='coerce')

    # drop rows missing coords
    df = df.dropna(subset=['Latitude', 'Longitude']).copy()

    # ensure y0<y1 semantics are consistent (top > bottom)
    # Elevation_From is the top of the layer; Elevation_To is the bottom.
    # Nothing to swap unless the sheet is inverted; add safety:
    swap_mask = df['Elevation_From'] < df['Elevation_To']
    df.loc[swap_mask, ['Elevation_From', 'Elevation_To']] = df.loc[
        swap_mask, ['Elevation_To', 'Elevation_From']
    ].values

    # keep a clean borehole order by name
    df['Borehole'] = df['Borehole'].astype(str)

    return df

# -------------------------------
# Proposed-only loader (lat/lon/name)
# -------------------------------
def normalize_cols_general(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    if "latitude" in lower: ren[lower["latitude"]] = "Latitude"
    elif "lat" in lower:    ren[lower["lat"]] = "Latitude"
    if "longitude" in lower: ren[lower["longitude"]] = "Longitude"
    elif "lon" in lower:     ren[lower["lon"]] = "Longitude"
    elif "long" in lower:    ren[lower["long"]] = "Longitude"
    if "name" in lower:      ren[lower["name"]] = "Name"
    elif "id" in lower:      ren[lower["id"]] = "Name"
    df = df.rename(columns=ren)
    if "Name" not in df.columns:
        df["Name"] = [f"Proposed-{i+1}" for i in range(len(df))]
    df["Latitude"]  = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna(subset=["Latitude","Longitude"])[["Latitude","Longitude","Name"]]

def try_read_proposed(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["Latitude","Longitude","Name"])
    try:
        pdf = pd.read_excel(file)
        return normalize_cols_general(pdf)
    except Exception:
        return pd.DataFrame(columns=["Latitude","Longitude","Name"])

# -------------------------------
# Geometry helpers (no shapely)
# -------------------------------
def get_transformer(lat, lon):
    zone = int((lon + 180) // 6) + 1
    crs_target = CRS.from_string(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs_target, always_xy=True)

def project_chainage_to_polyline(XY_m, poly_xy_m):
    if len(poly_xy_m) < 2:
        return (np.zeros(len(XY_m)), np.full(len(XY_m), np.inf), 0.0)

    seg_vecs = poly_xy_m[1:] - poly_xy_m[:-1]
    seg_len_m = np.linalg.norm(seg_vecs, axis=1)
    seg_len_m[seg_len_m == 0] = 1e-9
    cum_m = np.concatenate([[0.0], np.cumsum(seg_len_m)])

    chain_m = np.zeros(len(XY_m))
    dist_m = np.full(len(XY_m), np.inf)

    for i, P in enumerate(XY_m):
        best_d = np.inf
        best_ch = 0.0
        for k in range(len(seg_vecs)):
            A = poly_xy_m[k]
            v = seg_vecs[k]
            L2 = np.dot(v, v)
            t = np.clip(np.dot(P - A, v) / L2, 0.0, 1.0)
            Q = A + t * v
            d = np.linalg.norm(P - Q)
            if d < best_d:
                best_d = d
                best_ch = cum_m[k] + t * seg_len_m[k]
        chain_m[i] = best_ch
        dist_m[i] = best_d

    return chain_m * FT_PER_M, dist_m * FT_PER_M, cum_m[-1] * FT_PER_M

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Soil Profile (Disconnected Columns)", layout="wide")
st.title("📍 Borehole Visualization Tool")
st.caption("Upload MAIN layered Excel, optionally PROPOSED points. Draw a section on the map, then generate a soil profile. All units in feet.")

left, right = st.columns([1,1])
with left:
    main_file = st.file_uploader("Upload MAIN Excel (required)", type=["xls","xlsx"], key="main")
with right:
    prop_file = st.file_uploader("Upload PROPOSED Excel (optional: Latitude/Longitude/Name)", type=["xls","xlsx"], key="prop")

if not main_file:
    st.stop()

# Load
try:
    df_main = normalize_main_excel(main_file)
except Exception as e:
    st.error(f"Failed to read MAIN Excel: {e}")
    st.stop()

df_prop = try_read_proposed(prop_file)

# One coordinate per borehole for map/chainage
bh_coords = df_main.groupby('Borehole')[['Latitude','Longitude']].first().reset_index()

# Map center/bounds
center_lat, center_lon = float(bh_coords['Latitude'].mean()), float(bh_coords['Longitude'].mean())

# ------------------- Map & draw -------------------
st.header("🗺️ Map — Draw your section line")
with st.expander("Tip", expanded=True):
    st.write("Use the **polyline tool** on the map to draw your section. **Click the small Finish button** on the toolbar to end the line.")

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)
# Base layers
folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, USDA, USGS, AeroGRID, IGN, GIS User Community",
    name="Satellite", overlay=False, control=True
).add_to(m)

# Existing boreholes (blue)
for _, r in bh_coords.iterrows():
    folium.CircleMarker(
        location=[float(r['Latitude']), float(r['Longitude'])],
        radius=6, color="blue", fill=True, fill_opacity=0.9,
        tooltip=str(r['Borehole'])
    ).add_to(m)
    folium.Marker(
        [float(r['Latitude']), float(r['Longitude'])],
        icon=folium.DivIcon(html=f"<div style='font-size:10pt; color:black; transform: translateY(12px);'>{r['Borehole']}</div>")
    ).add_to(m)

# Proposed (red)
if not df_prop.empty:
    for _, r in df_prop.iterrows():
        folium.CircleMarker(
            location=[float(r['Latitude']), float(r['Longitude'])],
            radius=6, color="red", fill=True, fill_opacity=0.9,
            tooltip=str(r.get('Name','Proposed'))
        ).add_to(m)

Draw(
    export=False,
    draw_options={"polyline": True, "polygon": False, "rectangle": False,
                  "circle": False, "circlemarker": False, "marker": False},
    edit_options={"edit": True, "remove": True},
).add_to(m)
folium.LayerControl().add_to(m)

map_state = st_folium(
    m, height=600,
    returned_objects=["all_drawings"],
    use_container_width=True
)

# ------------------- Profile controls -------------------
st.header("📈 Section / Profile (ft) — Disconnected soil columns")
colA, colB, colC = st.columns([1,1,1])
with colA:
    corridor_ft = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)
with colB:
    half_w = st.slider("Column half-width (ft)", 5, 100, 30, step=5)
with colC:
    title_txt = st.text_input("Title", "Soil Profile")

# Find last drawn polyline
polyline_coords = None
drawings = (map_state or {}).get("all_drawings") or []
for g in reversed(drawings):
    if g and g.get("type") == "Feature":
        geom = g.get("geometry", {})
        if geom.get("type") == "LineString":
            polyline_coords = geom.get("coordinates")  # [[lon, lat], ...]
            break

if polyline_coords is None:
    st.info("Draw a polyline on the map and press **Finish** to create the section line.")
    st.stop()

# ------------------- Chainage & selection -------------------
transformer = get_transformer(center_lat, center_lon)
XY_m = np.array([transformer.transform(lon, lat)
                 for lat, lon in zip(bh_coords["Latitude"], bh_coords["Longitude"])])
poly_xy_m = np.array([transformer.transform(lon, lat)
                      for lon, lat in np.array(polyline_coords)])
chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(XY_m, poly_xy_m)

bh_pos = bh_coords.copy()
bh_pos["Chainage_ft"] = chain_ft
bh_pos["Offset_ft"] = dist_ft
bh_pos = bh_pos[bh_pos["Offset_ft"] <= corridor_ft].sort_values("Chainage_ft").reset_index(drop=True)

if bh_pos.empty:
    st.warning("No boreholes within the corridor. Widen the corridor or redraw.")
    st.stop()

# ------------------- Build profile (disconnected rectangles) -------------------
# Subset MAIN to only the selected boreholes, keep layers, sort by elevation
sel_bhs = set(bh_pos["Borehole"].tolist())
df_layers = df_main[df_main["Borehole"].isin(sel_bhs)].copy()
df_layers.sort_values(["Borehole", "Elevation_From"], ascending=[True, False], inplace=True)

# Chainage per borehole
ch_map = dict(zip(bh_pos["Borehole"], bh_pos["Chainage_ft"]))

# compute y limits suggestion
ymax = float(df_layers["Elevation_From"].max())
ymin = float(df_layers["Elevation_To"].min())

# Create figure
fig = go.Figure()

# Legend entries only for used types
used_types = set()

# Rectangles per layer (not connected between boreholes)
shapes = []
for bh, group in df_layers.groupby("Borehole"):
    if bh not in ch_map:
        continue
    x0 = ch_map[bh] - half_w
    x1 = ch_map[bh] + half_w
    # add label at the top of the topmost layer
    top_el = float(group["Elevation_From"].max())
    fig.add_annotation(x=ch_map[bh], y=top_el, text=str(bh),
                       showarrow=True, arrowhead=1, arrowsize=1, ax=0, ay=-25)

    for _, r in group.iterrows():
        soil = str(r["Soil_Type"]).strip()
        color = SOIL_COLOR_MAP.get(soil, "#cccccc")
        used_types.add(soil)
        y_top = float(r["Elevation_From"])
        y_bot = float(r["Elevation_To"])
        shapes.append(dict(
            type="rect",
            x0=x0, x1=x1, y0=y_bot, y1=y_top,
            line=dict(color="black", width=1),
            fillcolor=color,
            layer="below"
        ))

fig.update_layout(shapes=shapes)

# Add optional outlines for better reading
x_order = bh_pos["Chainage_ft"].to_numpy()
# top and bottom outlines (thin)
tops = []
bots = []
for bh, group in df_layers.groupby("Borehole"):
    if bh in ch_map:
        tops.append((ch_map[bh], float(group["Elevation_From"].max())))
        bots.append((ch_map[bh], float(group["Elevation_To"].min())))
if tops:
    tops = np.array(sorted(tops, key=lambda t: t[0]))
    bots = np.array(sorted(bots, key=lambda t: t[0]))
    fig.add_trace(go.Scatter(x=tops[:,0], y=tops[:,1], mode="lines+markers",
                             line=dict(color="black", width=1), marker=dict(size=5, color="black"),
                             name="Top EL (ft)", hovertemplate="Top EL (ft): %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bots[:,0], y=bots[:,1], mode="lines",
                             line=dict(color="black", width=1),
                             name="Bottom EL (ft)", hovertemplate="Bottom EL (ft): %{y:.2f}<extra></extra>"))

# Legend dummies for used soil types (in your fixed order)
for stype in [s for s in ORDERED_SOIL_TYPES if s in used_types]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name=stype,
        marker=dict(size=12, color=SOIL_COLOR_MAP.get(stype, "#cccccc"),
                    symbol="square", line=dict(color="black", width=1))
    ))

fig.update_layout(
    title=f"{title_txt} — Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor_ft} ft",
    xaxis_title="Chainage (ft)",
    yaxis_title="Elevation (ft)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h"),
)
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                 spikethickness=1, spikedash="dot")

st.plotly_chart(fig, use_container_width=True)
