# Borehole Visualization Tool — PWR bridging, AR/BR styled at feature elevation, Proposed at TOP
# Units: FEET

import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
import plotly.graph_objects as go
from pyproj import CRS, Transformer
import streamlit as st
from streamlit_folium import st_folium

FT_PER_M = 3.280839895  # meters -> feet


# -------------------------------
# Helpers
# -------------------------------
def pick(colnames, *aliases):
    for a in aliases:
        a = a.lower()
        if a in colnames:
            return a
    return None


def get_transformer(lat, lon):
    """Pick a local UTM projection for accurate XY distances (meters)."""
    zone = int((lon + 180) // 6) + 1
    crs_target = CRS.from_string(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs_target, always_xy=True)


def project_chainage_to_polyline(XY_m, poly_xy_m):
    """
    XY_m: Nx2 in meters (points)
    poly_xy_m: Mx2 in meters (polyline vertices)
    Returns (in feet): chain_ft, dist_ft, total_len_ft
    """
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


def add_band(fig, x_arr, y_upper, y_lower, fill_rgba, name, showlegend=True, legendgroup=None):
    """Add a filled band polygon between y_upper and y_lower along x_arr."""
    if len(x_arr) < 2:
        return
    x_closed = np.concatenate([x_arr, x_arr[::-1]])
    y_closed = np.concatenate([y_upper, y_lower[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_closed, y=y_closed,
            mode="lines",
            line=dict(color="black", width=1),
            fill="toself",
            fillcolor=fill_rgba,
            name=name,
            showlegend=showlegend,
            legendgroup=legendgroup,
            hoverinfo="skip"
        )
    )


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Borehole Viewer", layout="wide")
st.title("📍 Borehole Visualization Tool")
st.caption("Upload your Excel, draw a section on the map, then generate a filled profile and 3D view. All units shown in **feet**.")

# Two uploaders visible immediately
c_left, c_right = st.columns(2)
with c_left:
    uploaded_file = st.file_uploader("Upload Main Borehole Excel", type=["xls", "xlsx"], key="main")
with c_right:
    proposed_file = st.file_uploader("Upload Proposed Bore Logs Excel (optional)", type=["xls", "xlsx"], key="proposed")

# Parse main data (if provided)
data = None
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Columns (case-insensitive aliases)
    c_name   = pick(df.columns, 'name', 'id', 'boring id', 'hole id')
    c_lat    = pick(df.columns, 'latitude', 'lat')
    c_lon    = pick(df.columns, 'longitude', 'lon', 'long')
    c_top    = pick(df.columns, 'boring elevation', 'ground elevation', 'top el', 'top elevation')
    c_depth  = pick(df.columns, 'depth', 'total depth', 'hole depth')

    # Weathered rock (PWR)
    c_pwr_el = pick(df.columns, 'pwr el', 'pwr elevation', 'weathered rock el', 'weathered rock elevation')
    c_pwr_d  = pick(df.columns, 'pwr depth', 'weathered rock depth')

    # Bedrock (BR/BT) and Auger Refusal (AR)
    c_br_el  = pick(df.columns, 'br el', 'bt el', 'bedrock el', 'rock el', 'br elevation', 'bedrock elevation', 'rock elevation')
    c_br_d   = pick(df.columns, 'br depth', 'bt depth', 'bedrock depth', 'rock depth')
    c_ar_el  = pick(df.columns, 'ar el', 'ar elevation', 'auger refusal el', 'refusal el')
    c_ar_d   = pick(df.columns, 'ar depth', 'auger refusal depth', 'refusal depth')

    # BT/AR flag column (as in your sheet)
    c_bt_ar  = pick(df.columns, 'bt/ar', 'br/ar')

    # Build dataframe (feet)
    data = pd.DataFrame({
        'Name': df[c_name],
        'Latitude': df[c_lat],
        'Longitude': df[c_lon],
        'Top_EL': pd.to_numeric(df[c_top], errors="coerce") if c_top else np.nan,
        'Depth':  pd.to_numeric(df[c_depth], errors="coerce") if c_depth else np.nan,
    })
    data['Bottom_EL'] = data['Top_EL'] - data['Depth']

    # PWR elevation
    data['PWR_EL'] = np.nan
    if c_pwr_el:
        data['PWR_EL'] = pd.to_numeric(df[c_pwr_el], errors="coerce")
    if c_pwr_d:
        pwr_d = pd.to_numeric(df[c_pwr_d], errors="coerce")
        data['PWR_EL'] = data['PWR_EL'].fillna(data['Top_EL'] - pwr_d)

    # BR elevation
    data['BR_EL'] = np.nan
    if c_br_el:
        data['BR_EL'] = pd.to_numeric(df[c_br_el], errors="coerce")
    if c_br_d:
        br_d = pd.to_numeric(df[c_br_d], errors="coerce")
        data['BR_EL'] = data['BR_EL'].fillna(data['Top_EL'] - br_d)

    # AR elevation
    data['AR_EL'] = np.nan
    if c_ar_el:
        data['AR_EL'] = pd.to_numeric(df[c_ar_el], errors="coerce")
    if c_ar_d:
        ar_d = pd.to_numeric(df[c_ar_d], errors="coerce")
        data['AR_EL'] = data['AR_EL'].fillna(data['Top_EL'] - ar_d)

    # BT/AR flag
    if c_bt_ar:
        flag = df[c_bt_ar].astype(str).str.strip()
        data['BT_AR_Flag'] = flag.replace({'nan': '', 'NaN': '', 'None': '', 'NONE': ''})

    # Clean
    data = data.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if data.empty:
        st.warning("Main file: No valid rows with Latitude/Longitude found.")
        data = None

# Parse proposed (if provided) — Name, Latitude, Longitude only
proposed = None
if proposed_file:
    pdf = pd.read_excel(proposed_file)
    pdf.columns = [c.strip().lower() for c in pdf.columns]
    p_name = pick(pdf.columns, 'name', 'id', 'boring id', 'hole id')
    p_lat  = pick(pdf.columns, 'latitude', 'lat')
    p_lon  = pick(pdf.columns, 'longitude', 'lon', 'long')

    if not (p_name and p_lat and p_lon):
        st.error("Proposed bore logs must include columns for Name, Latitude, and Longitude.")
    else:
        proposed = pd.DataFrame({
            "Name": pdf[p_name].astype(str),
            "Latitude": pd.to_numeric(pdf[p_lat], errors="coerce"),
            "Longitude": pd.to_numeric(pdf[p_lon], errors="coerce"),
        }).dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        if proposed.empty:
            st.warning("Proposed file: No valid rows with Latitude/Longitude found.")
            proposed = None

# If nothing uploaded, guide the user
if (data is None) and (proposed is None):
    st.info("Upload a main borehole file, a proposed bore log file, or both to see them on the map.")
    st.stop()

# -------------------
# Map + Drawing
# -------------------
st.header("🌍 Map — Draw your section line")
with st.expander("Tip", expanded=True):
    st.write("Use the **polyline tool** on the map to draw your section. Double-click to finish. Popups/labels are in **ft**.")
    st.caption("Blue = existing borings (main file). Red = proposed borings (optional upload).")

# Center map
lat_series = pd.Series(dtype=float)
lon_series = pd.Series(dtype=float)
if data is not None and not data.empty:
    lat_series = pd.concat([lat_series, data["Latitude"]], ignore_index=True)
    lon_series = pd.concat([lon_series, data["Longitude"]], ignore_index=True)
if proposed is not None and not proposed.empty:
    lat_series = pd.concat([lat_series, proposed["Latitude"]], ignore_index=True)
    lon_series = pd.concat([lon_series, proposed["Longitude"]], ignore_index=True)

center_lat, center_lon = lat_series.mean(), lon_series.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

# Basemaps
folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
folium.TileLayer(
    tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
    name="Terrain",
    attr="Map tiles by Stamen Design (CC BY 3.0). Data © OSM contributors (ODbL)."
).add_to(m)
folium.TileLayer(
    tiles="https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
    name="Black & White",
    attr="Map tiles by Stamen Design (CC BY 3.0). Data © OSM contributors (ODbL)."
).add_to(m)
folium.TileLayer(
    tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}{r}.png",
    name="Light",
    attr="©OpenStreetMap, ©CartoDB"
).add_to(m)
folium.TileLayer(
    tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}{r}.png",
    name="Dark",
    attr="©OpenStreetMap, ©CartoDB"
).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri | Sources: Esri, Maxar, USGS, etc.",
    name="Satellite", overlay=False, control=True
).add_to(m)

# Feature groups
fg_existing = folium.FeatureGroup(name="Existing Boreholes", show=True)
fg_proposed = folium.FeatureGroup(name="Proposed Boreholes", show=True)

# Existing markers
if data is not None and not data.empty:
    for _, r in data.iterrows():
        popup = f"<b>{r['Name']}</b>"
        if 'BT_AR_Flag' in data.columns and str(r.get('BT_AR_Flag','')).strip():
            popup += f" &nbsp; <i>({r['BT_AR_Flag']})</i>"
        if pd.notna(r['Top_EL']):    popup += f"<br>Top EL: {r['Top_EL']:.2f} ft"
        if pd.notna(r.get('PWR_EL', np.nan)): popup += f"<br>PWR EL: {r['PWR_EL']:.2f} ft"
        if pd.notna(r.get('BR_EL', np.nan)):  popup += f"<br>BR EL: {r['BR_EL']:.2f} ft"
        if pd.notna(r.get('AR_EL', np.nan)):  popup += f"<br>AR EL: {r['AR_EL']:.2f} ft"
        if pd.notna(r['Bottom_EL']): popup += f"<br>Bottom EL: {r['Bottom_EL']:.2f} ft"

        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=6, color="blue", fill=True, fill_opacity=0.7,
            popup=popup, tooltip=r["Name"]
        ).add_to(fg_existing)
        folium.Marker(
            [r["Latitude"], r["Longitude"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="font-size: 10pt; color: black; white-space: nowrap;
                           text-align: center; transform: translateY(12px);">
                    {r['Name']}
                </div>
                """
            )
        ).add_to(fg_existing)
    fg_existing.add_to(m)

# Proposed markers
if proposed is not None and not proposed.empty:
    for _, r in proposed.iterrows():
        popup = f"<b>{r['Name']}</b><br>Lat: {r['Latitude']:.6f}, Lon: {r['Longitude']:.6f}"
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=6, color="red", fill=True, fill_opacity=0.8,
            popup=popup, tooltip=f"(Proposed) {r['Name']}"
        ).add_to(fg_proposed)
        folium.Marker(
            [r["Latitude"], r["Longitude"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="font-size: 10pt; color: red; white-space: nowrap;
                           text-align: center; transform: translateY(12px);">
                    {r['Name']}
                </div>
                """
            )
        ).add_to(fg_proposed)
    fg_proposed.add_to(m)

Draw(
    export=False,
    draw_options={"polyline": True, "polygon": False, "rectangle": False,
                  "circle": False, "circlemarker": False, "marker": False},
    edit_options={"edit": True, "remove": True},
).add_to(m)
folium.LayerControl().add_to(m)

map_state = st_folium(
    m, height=600,
    returned_objects=["last_active_drawing", "all_drawings"],
    use_container_width=True
)

# -------------------
# Section/Profile
# -------------------
if data is None or data.empty:
    st.info("Upload the **Main Borehole** file to enable the Section/Profile and 3D views.")
    st.stop()

st.header("📈 Section / Profile (ft) — PWR bridging; Names at top; AR/BR styled; Proposed at top")

corridor_ft = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)

# AR/BR label styling controls
cA, cB, cC = st.columns([1,1,1])
with cA:
    ar_color = st.color_picker("AR label color", value="#7F1D1D")   # maroon-ish
with cB:
    br_color = st.color_picker("BR label color", value="#7C3AED")   # purple-ish (match screenshot feel)
with cC:
    flag_font_size = st.slider("AR/BR font size", 8, 22, 12, step=1)
flag_yshift_px = st.slider("AR/BR vertical offset (px)", -20, 40, -6, step=1,
                           help="Negative = above the line; positive = below")
flag_font_family = "Arial Black, Arial, sans-serif"

# get last drawn polyline
polyline_coords = None
drawings = map_state.get("all_drawings") or []
if drawings:
    for g in reversed(drawings):
        if g and g.get("type") == "Feature":
            geom = g.get("geometry", {})
            if geom.get("type") == "LineString":
                polyline_coords = geom.get("coordinates")
                break

sec = None
total_len_ft = 0.0

if polyline_coords is None:
    st.info("Draw a polyline on the map to define the section line.")
else:
    transformer = get_transformer(center_lat, center_lon)
    XY_m = np.array([transformer.transform(lon, lat)
                    for lat, lon in zip(data["Latitude"], data["Longitude"])])
    poly_xy_m = np.array([transformer.transform(lon, lat)
                          for lon, lat in np.array(polyline_coords)])
    chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(XY_m, poly_xy_m)

    keep = dist_ft <= corridor_ft
    sec = data.loc[keep].copy()
    sec["Chainage_ft"] = chain_ft[keep]
    sec = sec.sort_values("Chainage_ft").reset_index(drop=True)

    if sec.empty:
        st.warning("No borings fall within the selected corridor width. Widen the corridor or redraw the line.")
    else:
        x   = sec["Chainage_ft"].to_numpy()
        top = sec["Top_EL"].to_numpy()
        bot = sec["Bottom_EL"].to_numpy()
        pwr = sec["PWR_EL"].to_numpy()
        br  = sec["BR_EL"].to_numpy() if "BR_EL" in sec.columns else np.full_like(top, np.nan)
        ar  = sec["AR_EL"].to_numpy() if "AR_EL" in sec.columns else np.full_like(top, np.nan)

        fig = go.Figure()

        # Soil band: Top -> (PWR if present else Bottom)
        lower_soil = np.where(np.isnan(pwr), bot, pwr)
        add_band(fig, x, top, lower_soil, "rgba(34,197,94,0.55)", "Soil", True, "soil")

        # --- PWR filled band to Bottom (bridged across neighbors without PWR) ---
        mask_pwr = ~np.isnan(pwr)
        n = len(x)
        first_pwr_band = True

        mask_dilated = np.zeros(n, dtype=bool)
        idx_true = np.where(mask_pwr)[0]
        for i in idx_true:
            mask_dilated[i] = True
            if i > 0:     mask_dilated[i-1] = True
            if i < n - 1: mask_dilated[i+1] = True

        if mask_dilated.any():
            idx = np.where(mask_dilated)[0]
            splits = np.where(np.diff(idx) > 1)[0]
            segments = np.split(idx, splits + 1)
            for seg in segments:
                xs  = x[seg]
                yup = np.where(mask_pwr[seg], pwr[seg], bot[seg])  # bridge
                ylo = bot[seg]
                add_band(fig, xs, yup, ylo, "rgba(127,29,29,0.70)", "PWR",
                         showlegend=first_pwr_band, legendgroup="pwrband")
                first_pwr_band = False

        # Posts
        for xi, ytop, ybot in zip(x, top, bot):
            fig.add_trace(go.Scatter(
                x=[xi, xi], y=[ybot, ytop],
                mode="lines", line=dict(color="black", width=2),
                showlegend=False, hoverinfo="skip"
            ))

        # Top & Bottom outlines
        fig.add_trace(go.Scatter(
            x=x, y=top, mode="lines+markers",
            line=dict(color="black", width=1),
            marker=dict(size=5, color="black"),
            name="Top EL (ft)",
            hovertemplate="Top EL (ft): %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=bot, mode="lines",
            line=dict(color="black", width=1),
            name="Bottom EL (ft)",
            hovertemplate="Bottom EL (ft): %{y:.2f}<extra></extra>"
        ))

        # PWR line & markers only where PWR exists
        if mask_pwr.any():
            first_pwr_line = True
            idx = np.where(mask_pwr)[0]
            splits = np.where(np.diff(idx) > 1)[0]
            segments = np.split(idx, splits + 1)
            for seg in segments:
                xs = x[seg]; ys = pwr[seg]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="black", width=1, dash="dot"),
                    name="PWR EL (ft)",
                    showlegend=first_pwr_line,
                    hoverinfo="skip",
                    legendgroup="pwr"
                ))
                first_pwr_line = False
            fig.add_trace(go.Scatter(
                x=x[mask_pwr], y=pwr[mask_pwr], mode="markers",
                marker=dict(size=4, color="black"),
                name="PWR EL (ft)",
                legendgroup="pwr",
                showlegend=False,
                hovertemplate="PWR EL (ft): %{y:.2f}<extra></extra>"
            ))

        # BR line + diamond markers
        mask_br = ~np.isnan(br)
        if mask_br.any():
            fig.add_trace(go.Scatter(
                x=x[mask_br], y=br[mask_br], mode="lines+markers",
                line=dict(color="black", width=1, dash="dash"),
                marker=dict(symbol="diamond", size=7),
                name="BR EL (ft)",
                hovertemplate="BR EL (ft): %{y:.2f}<extra></extra>"
            ))

        # AR line + cross markers
        mask_ar = ~np.isnan(ar)
        if mask_ar.any():
            fig.add_trace(go.Scatter(
                x=x[mask_ar], y=ar[mask_ar], mode="lines+markers",
                line=dict(color="black", width=1, dash="dashdot"),
                marker=dict(symbol="x", size=8),
                name="AR EL (ft)",
                hovertemplate="AR EL (ft): %{y:.2f}<extra></extra>"
            ))

        # ---------- Labels ----------
        # Names at top
        for xi, yt, label in zip(x, top, sec["Name"]):
            fig.add_annotation(
                x=xi, y=yt,
                text=str(label),
                showarrow=True, arrowhead=1, arrowsize=1,
                ax=0, ay=-25  # above top
            )

        # AR/BR label at feature elevation (fallback to bottom)
        flags = sec["BT_AR_Flag"].astype(str).str.upper().str.strip() if "BT_AR_Flag" in sec.columns else pd.Series([""]*len(sec))
        for xi, btm, br_el, ar_el, flag in zip(x, bot, br, ar, flags):
            if not flag or flag == "NAN":
                continue
            if flag == "BT":  # normalize
                flag = "BR"

            if flag == "AR" and not np.isnan(ar_el):
                y_anchor = ar_el
                color = ar_color
            elif flag == "BR" and not np.isnan(br_el):
                y_anchor = br_el
                color = br_color
            else:
                y_anchor = btm
                color = ar_color if flag == "AR" else br_color

            fig.add_annotation(
                x=xi, y=y_anchor,
                text=flag,
                showarrow=False,
                yshift=flag_yshift_px,
                font=dict(size=flag_font_size, color=color, family=flag_font_family),
                align="center"
            )
        # ---------------------------

        # ---------- Proposed positions: names + triangle markers at TOP ----------
        if proposed is not None and not proposed.empty:
            XYp_m = np.array([transformer.transform(lon, lat)
                              for lat, lon in zip(proposed["Latitude"], proposed["Longitude"])])
            chain_p_ft, dist_p_ft, _ = project_chainage_to_polyline(XYp_m, poly_xy_m)
            keep_p = dist_p_ft <= corridor_ft
            prop_sec = proposed.loc[keep_p].copy()
            if not prop_sec.empty:
                prop_sec["Chainage_ft"] = chain_p_ft[keep_p]
                xprop = prop_sec["Chainage_ft"].to_numpy()

                ymin = float(np.nanmin(bot))
                ymax = float(np.nanmax(top))
                yrng = max(ymax - ymin, 1.0)

                y_prop_top = ymax + max(0.03 * yrng, 8.0)
                fig.add_trace(go.Scatter(
                    x=xprop,
                    y=[y_prop_top] * len(xprop),
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=10, color="red"),
                    text=prop_sec["Name"].astype(str),
                    textposition="top center",
                    name="Proposed (location)",
                    hovertemplate="<b>%{text}</b><br>Chainage: %{x:.1f} ft<extra></extra>"
                ))

                # Expand y-range for top labels
                pad_top = max(0.08 * yrng, 12.0)
                pad_bot = max(0.04 * yrng, 8.0)
                fig.update_yaxes(range=[ymin - pad_bot, ymax + pad_top])
        else:
            ymin = float(np.nanmin(bot))
            ymax = float(np.nanmax(top))
            pad_top = max(0.06 * (ymax - ymin), 10.0)
            pad_bot = max(0.04 * (ymax - ymin), 8.0)
            fig.update_yaxes(range=[ymin - pad_bot, ymax + pad_top])
        # ------------------------------------------------------------------------

        # Layout
        fig.update_layout(
            title=f"Section along drawn line (Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor_ft} ft)",
            xaxis_title="Chainage (ft)",
            yaxis_title="Elevation (ft)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h"),
        )
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                         spikethickness=1, spikedash="dot")

        st.plotly_chart(fig, use_container_width=True)

# -------------------
# 3D Borehole View (ft)
# -------------------
st.header("🌀 3D Borehole View (ft, Plan Coordinates)")
c1, c2 = st.columns([1,1])
with c1:
    limit3d = st.checkbox("Limit to section corridor", value=True)
with c2:
    ve = st.slider("Vertical exaggeration (display only)", 1.0, 6.0, 2.0, step=0.5)

data3d = data
if limit3d and 'sec' in locals() and sec is not None and not sec.empty:
    data3d = sec

# Project to local UTM -> feet
transformer = get_transformer(center_lat, center_lon)
XY_m = np.array([transformer.transform(lon, lat)
                 for lat, lon in zip(data3d["Latitude"], data3d["Longitude"])])
X_ft = XY_m[:, 0] * FT_PER_M
Y_ft = XY_m[:, 1] * FT_PER_M

z_top = data3d["Top_EL"].to_numpy()
z_bot = data3d["Bottom_EL"].to_numpy()
z_pwr = data3d["PWR_EL"].to_numpy()
names = data3d["Name"].astype(str).to_numpy()

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(x=X_ft, y=Y_ft, z=z_top, mode="markers",
    marker=dict(size=5, color="rgb(135,206,250)"), name="Top EL (ft)",
    text=names,
    hovertemplate="<b>%{text}</b><br>Top EL: %{z:.2f} ft<br>E: %{x:.1f} ft, N: %{y:.1f} ft<extra></extra>"
))
fig3d.add_trace(go.Scatter3d(x=X_ft, y=Y_ft, z=z_bot, mode="markers",
    marker=dict(size=4, color="rgb(90,90,90)"), name="Bottom EL (ft)",
    text=names,
    hovertemplate="<b>%{text}</b><br>Bottom EL: %{z:.2f} ft<br>E: %{x:.1f} ft, N: %{y:.1f} ft<extra></extra>"
))
mask = ~np.isnan(z_pwr)
if mask.any():
    fig3d.add_trace(go.Scatter3d(x=X_ft[mask], y=Y_ft[mask], z=z_pwr[mask], mode="markers",
        marker=dict(size=4, color="red"), name="PWR EL (ft)",
        text=names[mask],
        hovertemplate="<b>%{text}</b><br>PWR EL: %{z:.2f} ft<br>E: %{x:.1f} ft, N: %{y:.1f} ft<extra></extra>"
    ))

fig3d.update_layout(
    height=650,
    scene=dict(
        xaxis_title="Easting (ft)",
        yaxis_title="Northing (ft)",
        zaxis_title=f"Elevation (ft) — {ve}×",
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=ve),
    ),
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, b=0, t=10),
    scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))
)

st.plotly_chart(fig3d, use_container_width=True)
