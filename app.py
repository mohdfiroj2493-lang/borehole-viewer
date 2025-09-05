# app.py
import io
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
# Soil color map + legend order (used in "Soil types" mode)
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

# Acceptable labels that mean “PWR”
PWR_ALIASES = {"PWR", "WEATHERED ROCK", "WEATHERED-ROCK", "PARTIALLY WEATHERED ROCK", "WR"}

# -------------------------------
# File normalization
# -------------------------------
RENAME_MAP = {
    'Bore Log': 'Borehole',
    'Borehole': 'Borehole',
    'Elevation From': 'Elevation_From',
    'Elevation To': 'Elevation_To',
    'Soil Layer Description': 'Soil_Type',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
}

def normalize_main_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    for k, v in RENAME_MAP.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    need = ['Borehole','Elevation_From','Elevation_To','Soil_Type','Latitude','Longitude']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # soil type -> USCS code in parentheses; “top soil” → Topsoil; otherwise keep text
    soil = df['Soil_Type'].astype(str)
    extracted = soil.str.extract(r'\(([^)]+)\)')[0]
    extracted = extracted.fillna(
        soil.str.replace(r'^.*top\s*soil.*$', 'Topsoil', case=False, regex=True)
    )
    df['Soil_Type'] = extracted.fillna(soil).str.strip()

    df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Elevation_From'] = pd.to_numeric(df['Elevation_From'], errors='coerce')
    df['Elevation_To']   = pd.to_numeric(df['Elevation_To'],   errors='coerce')
    df = df.dropna(subset=['Latitude','Longitude']).copy()

    # ensure Elevation_From (top) >= Elevation_To (bottom)
    swap = df['Elevation_From'] < df['Elevation_To']
    df.loc[swap, ['Elevation_From','Elevation_To']] = df.loc[
        swap, ['Elevation_To','Elevation_From']
    ].values

    df['Borehole'] = df['Borehole'].astype(str)
    return df

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
    dist_m  = np.full(len(XY_m), np.inf)

    for i, P in enumerate(XY_m):
        best_d = np.inf; best_ch = 0.0
        for k in range(len(seg_vecs)):
            A = poly_xy_m[k]; v = seg_vecs[k]
            L2 = np.dot(v, v)
            t  = np.clip(np.dot(P - A, v) / L2, 0.0, 1.0)
            Q  = A + t * v
            d  = np.linalg.norm(P - Q)
            if d < best_d:
                best_d = d
                best_ch = cum_m[k] + t * seg_len_m[k]
        chain_m[i] = best_ch
        dist_m[i]  = best_d

    return chain_m * FT_PER_M, dist_m * FT_PER_M, cum_m[-1] * FT_PER_M

def add_band(fig, x_arr, y_upper, y_lower, fill_rgba, name, showlegend=True, legendgroup=None):
    if len(x_arr) < 2:
        return
    x_closed = np.concatenate([x_arr, x_arr[::-1]])
    y_closed = np.concatenate([y_upper, y_lower[::-1]])
    fig.add_trace(go.Scatter(
        x=x_closed, y=y_closed,
        mode="lines", line=dict(color="black", width=1),
        fill="toself", fillcolor=fill_rgba,
        name=name, showlegend=showlegend, legendgroup=legendgroup,
        hoverinfo="skip"
    ))

# -------------------------------
# App
# -------------------------------
st.set_page_config(page_title="Soil Profile", layout="wide")
st.title("📍 Borehole Visualization Tool")
st.caption("Upload Excel, draw a section, then plot either **Soil & PWR** bands (screenshot style) or **Soil types** as disconnected columns. All units in feet.")

main_file = st.file_uploader("Upload MAIN Excel (required)", type=["xls","xlsx"])
if not main_file:
    st.stop()

try:
    df_main = normalize_main_excel(main_file)
except Exception as e:
    st.error(f"Failed to read MAIN Excel: {e}")
    st.stop()

# 1 point per BH for map + chainage
bh_coords = df_main.groupby('Borehole')[['Latitude','Longitude']].first().reset_index()
center_lat, center_lon = float(bh_coords['Latitude'].mean()), float(bh_coords['Longitude'].mean())

# --- Map & draw ---
st.header("🗺️ Map — Draw your section line")
with st.expander("Tip", expanded=True):
    st.write("Use the **polyline tool** to draw the section. Click the small **Finish** button (on the map toolbar) to end the line.")

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)
folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, USDA, USGS, AeroGRID, IGN, GIS User Community",
    name="Satellite", overlay=False, control=True
).add_to(m)

for _, r in bh_coords.iterrows():
    folium.CircleMarker(
        [float(r['Latitude']), float(r['Longitude'])],
        radius=6, color="blue", fill=True, fill_opacity=0.9,
        tooltip=str(r['Borehole'])
    ).add_to(m)
    folium.Marker(
        [float(r['Latitude']), float(r['Longitude'])],
        icon=folium.DivIcon(html=f"<div style='font-size:10pt; color:black; transform: translateY(12px);'>{r['Borehole']}</div>")
    ).add_to(m)

Draw(
    export=False,
    draw_options={"polyline": True, "polygon": False, "rectangle": False,
                  "circle": False, "circlemarker": False, "marker": False},
    edit_options={"edit": True, "remove": True},
).add_to(m)
folium.LayerControl().add_to(m)

map_state = st_folium(m, height=600, returned_objects=["all_drawings"], use_container_width=True)

# --- Profile controls ---
st.header("📈 Section / Profile (ft)")
cA, cB, cC = st.columns([1,1,1])
with cA:
    corridor_ft = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)
with cB:
    mode = st.radio("Profile style", ["Soil & PWR bands (screenshot style)", "Soil types (disconnected columns)"], index=0)
with cC:
    title_txt = st.text_input("Title", "Soil Profile")

# pick last polyline
polyline_coords = None
drawings = (map_state or {}).get("all_drawings") or []
for g in reversed(drawings):
    if g and g.get("type") == "Feature":
        geom = g.get("geometry", {})
        if geom.get("type") == "LineString":
            polyline_coords = geom.get("coordinates")
            break

if polyline_coords is None:
    st.info("Draw a polyline on the map and press **Finish** to create the section line.")
    st.stop()

# chainage
transformer = get_transformer(center_lat, center_lon)
XY_m = np.array([transformer.transform(lon, lat)
                 for lat, lon in zip(bh_coords["Latitude"], bh_coords["Longitude"])])
poly_xy_m = np.array([transformer.transform(lon, lat)
                      for lon, lat in np.array(polyline_coords)])
chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(XY_m, poly_xy_m)

bh_pos = bh_coords.copy()
bh_pos["Chainage_ft"] = chain_ft
bh_pos["Offset_ft"]   = dist_ft
bh_pos = bh_pos[bh_pos["Offset_ft"] <= corridor_ft].sort_values("Chainage_ft").reset_index(drop=True)
if bh_pos.empty:
    st.warning("No boreholes within the corridor. Widen the corridor or redraw.")
    st.stop()

sel_bhs = set(bh_pos["Borehole"])
layers  = df_main[df_main["Borehole"].isin(sel_bhs)].copy()
layers.sort_values(["Borehole","Elevation_From"], ascending=[True, False], inplace=True)
ch_map = dict(zip(bh_pos["Borehole"], bh_pos["Chainage_ft"]))

# ---------- MODE 1: Soil & PWR (your screenshot style) ----------
if mode.startswith("Soil & PWR"):
    # aggregate per BH
    ag = []
    for bh, g in layers.groupby("Borehole"):
        top = float(g["Elevation_From"].max())
        bot = float(g["Elevation_To"].min())
        # find PWR layer top (if any)
        pwr_el = np.nan
        for _, r in g.iterrows():
            tag = str(r["Soil_Type"]).upper().strip()
            if tag in PWR_ALIASES:
                pwr_el = float(r["Elevation_From"])
                break
        ag.append((bh, ch_map[bh], top, bot, pwr_el))
    sec = pd.DataFrame(ag, columns=["Borehole","Chainage_ft","Top_EL","Bottom_EL","PWR_EL"]).sort_values("Chainage_ft")

    x   = sec["Chainage_ft"].to_numpy()
    top = sec["Top_EL"].to_numpy()
    bot = sec["Bottom_EL"].to_numpy()
    pwr = sec["PWR_EL"].to_numpy()
    mask = ~np.isnan(pwr)

    fig = go.Figure()

    # Soil band (green): Top -> (PWR or Bottom)
    lower_soil = np.where(mask, pwr, bot)
    add_band(fig, x, top, lower_soil, "rgba(34,197,94,0.55)", "Soil", True, "soil")

    # PWR band (maroon): PWR -> Bottom
    if mask.any():
        idx = np.where(mask)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        segments = np.split(idx, splits + 1)
        first = True
        for seg in segments:
            xs  = x[seg]; y_up = pwr[seg]; y_lo = bot[seg]
            add_band(fig, xs, y_up, y_lo, "rgba(127,29,29,0.70)", "PWR", first, "pwrband")
            first = False

    # Posts
    for xi, ytop, ybot in zip(x, top, bot):
        fig.add_trace(go.Scatter(x=[xi, xi], y=[ybot, ytop], mode="lines",
                                 line=dict(color="black", width=2),
                                 showlegend=False, hoverinfo="skip"))
    # Outlines and dashed PWR line + markers
    fig.add_trace(go.Scatter(x=x, y=top, mode="lines+markers",
                             line=dict(color="black", width=1),
                             marker=dict(size=5, color="black"),
                             name="Top EL (ft)", hovertemplate="Top EL (ft): %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=bot, mode="lines",
                             line=dict(color="black", width=1),
                             name="Bottom EL (ft)", hovertemplate="Bottom EL (ft): %{y:.2f}<extra></extra>"))
    if mask.any():
        first = True
        idx = np.where(mask)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        segments = np.split(idx, splits + 1)
        for seg in segments:
            xs = x[seg]; ys = pwr[seg]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color="black", width=1, dash="dot"),
                                     name="PWR EL (ft)", showlegend=first,
                                     hoverinfo="skip", legendgroup="pwr"))
            first = False
        fig.add_trace(go.Scatter(x=x[mask], y=pwr[mask], mode="markers",
                                 marker=dict(size=4, color="black"),
                                 name="PWR EL (ft)", legendgroup="pwr",
                                 showlegend=False,
                                 hovertemplate="PWR EL (ft): %{y:.2f}<extra></extra>"))

    # Labels
    for xi, yi, label in zip(x, top, sec["Borehole"]):
        fig.add_annotation(x=xi, y=yi, text=str(label),
                           showarrow=True, arrowhead=1, arrowsize=1, ax=0, ay=-25)

    fig.update_layout(
        title=f"Section along drawn line (Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor_ft} ft)",
        xaxis_title="Chainage (ft)", yaxis_title="Elevation (ft)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot")
    st.plotly_chart(fig, use_container_width=True)

# ---------- MODE 2: Disconnected soil-type columns ----------
else:
    half_w = st.slider("Column half-width (ft)", 5, 100, 30, step=5, key="halfw")
    fig = go.Figure()
    shapes = []
    used_types = set()

    for bh, grp in layers.groupby("Borehole"):
        if bh not in ch_map:  # outside corridor
            continue
        x0 = ch_map[bh] - half_w
        x1 = ch_map[bh] + half_w
        top_el = float(grp["Elevation_From"].max())
        fig.add_annotation(x=ch_map[bh], y=top_el, text=str(bh),
                           showarrow=True, arrowhead=1, arrowsize=1, ax=0, ay=-25)
        for _, r in grp.iterrows():
            soil = str(r["Soil_Type"]).strip()
            used_types.add(soil)
            color = SOIL_COLOR_MAP.get(soil, "#cccccc")
            y_top = float(r["Elevation_From"]); y_bot = float(r["Elevation_To"])
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y_bot, y1=y_top,
                               line=dict(color="black", width=1), fillcolor=color, layer="below"))
    fig.update_layout(shapes=shapes)

    # top/bottom outlines
    tops, bots = [], []
    for bh, grp in layers.groupby("Borehole"):
        if bh in ch_map:
            tops.append((ch_map[bh], float(grp["Elevation_From"].max())))
            bots.append((ch_map[bh], float(grp["Elevation_To"].min())))
    if tops:
        tops = np.array(sorted(tops, key=lambda t: t[0]))
        bots = np.array(sorted(bots, key=lambda t: t[0]))
        fig.add_trace(go.Scatter(x=tops[:,0], y=tops[:,1], mode="lines+markers",
                                 line=dict(color="black", width=1),
                                 marker=dict(size=5, color="black"),
                                 name="Top EL (ft)", hovertemplate="Top EL (ft): %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=bots[:,0], y=bots[:,1], mode="lines",
                                 line=dict(color="black", width=1),
                                 name="Bottom EL (ft)", hovertemplate="Bottom EL (ft): %{y:.2f}<extra></extra>"))

    # legend dummies for only-used soil types (ordered)
    for stype in [s for s in ORDERED_SOIL_TYPES if s in used_types]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=stype,
                                 marker=dict(size=12, color=SOIL_COLOR_MAP.get(stype,"#ccc"),
                                             symbol="square", line=dict(color="black", width=1))))

    fig.update_layout(
        title=f"{title_txt} — Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor_ft} ft",
        xaxis_title="Chainage (ft)", yaxis_title="Elevation (ft)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot")
    st.plotly_chart(fig, use_container_width=True)
