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
            hoverinfo="skip"  # bands don't clutter hover
        )
    )


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Borehole Viewer", layout="wide")
st.title("📍 Borehole Visualization Tool")
st.caption("Upload your Excel, draw a section on the map, then generate a filled profile and 3D view. All units shown in **feet**.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Columns (case-insensitive aliases)
    c_name   = pick(df.columns, 'name', 'id', 'boring id', 'hole id')
    c_lat    = pick(df.columns, 'latitude', 'lat')
    c_lon    = pick(df.columns, 'longitude', 'lon', 'long')
    c_top    = pick(df.columns, 'boring elevation', 'ground elevation', 'top el', 'top elevation')
    c_depth  = pick(df.columns, 'depth', 'total depth', 'hole depth')
    c_pwr_d  = pick(df.columns, 'pwr depth', 'weathered rock depth')
    c_pwr_el = pick(df.columns, 'pwr el', 'pwr elevation', 'weathered rock elevation')

    # Build dataframe (all elevations/depths assumed feet from Excel)
    data = pd.DataFrame({
        'Name': df[c_name],
        'Latitude': df[c_lat],
        'Longitude': df[c_lon],
        'Top_EL': pd.to_numeric(df[c_top], errors="coerce"),
        'Depth':  pd.to_numeric(df[c_depth], errors="coerce"),
    })
    data['Bottom_EL'] = data['Top_EL'] - data['Depth']
    data['PWR_EL'] = np.nan
    if c_pwr_el:
        data['PWR_EL'] = pd.to_numeric(df[c_pwr_el], errors="coerce")
    if c_pwr_d:
        pwr_d = pd.to_numeric(df[c_pwr_d], errors="coerce")
        data['PWR_EL'] = data['PWR_EL'].fillna(data['Top_EL'] - pwr_d)

    # Clean
    data = data.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if data.empty:
        st.error("No valid rows with Latitude/Longitude found.")
        st.stop()

    # -------------------
    # Map + Drawing
    # -------------------
    st.header("🌍 Map — Draw your section line")
    with st.expander("Tip", expanded=True):
        st.write("Use the **polyline tool** on the map to draw your section. Double-click to finish. Popups/labels are in **ft**.")

    center_lat, center_lon = data["Latitude"].mean(), data["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # Basemaps (with attributions)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
        name="Terrain",
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
    ).add_to(m)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
        name="Black & White",
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
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
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, USDA, USGS, AeroGRID, IGN, GIS User Community",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # Borehole markers + labels
    for _, r in data.iterrows():
        popup = (
            f"<b>{r['Name']}</b><br>"
            f"Top EL: {r['Top_EL']:.2f} ft<br>"
            f"PWR EL: {('%.2f ft' % r['PWR_EL']) if pd.notna(r['PWR_EL']) else 'N/A'}<br>"
            f"Bottom EL: {r['Bottom_EL']:.2f} ft"
        )
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=6, color="blue", fill=True, fill_opacity=0.7,
            popup=popup, tooltip=r["Name"]
        ).add_to(m)
        folium.Marker(
            [r["Latitude"], r["Longitude"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 10pt; color: black; white-space: nowrap;
                    text-align: center; transform: translateY(12px);
                ">{r['Name']}</div>
                """
            )
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
        returned_objects=["last_active_drawing", "all_drawings"],
        use_container_width=True
    )

    # -------------------
    # Section from drawn line (feet)
    # -------------------
    st.header("📈 Section / Profile (ft) — Soil & PWR")
    corridor_ft = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)

    # get last drawn polyline
    polyline_coords = None
    drawings = map_state.get("all_drawings") or []
    if drawings:
        for g in reversed(drawings):
            if g and g.get("type") == "Feature":
                geom = g.get("geometry", {})
                if geom.get("type") == "LineString":
                    polyline_coords = geom.get("coordinates")  # [[lon, lat], ...]
                    break

    sec = None
    total_len_ft = 0.0

    if polyline_coords is None:
        st.info("Draw a polyline on the map to define the section line.")
    else:
        # project to meters then convert to feet for outputs
        transformer = get_transformer(center_lat, center_lon)
        XY_m = np.array([transformer.transform(lon, lat)
                        for lat, lon in zip(data["Latitude"], data["Longitude"])])
        poly_xy_m = np.array([transformer.transform(lon, lat)
                              for lon, lat in np.array(polyline_coords)])
        chain_ft, dist_ft, total_len_ft = project_chainage_to_polyline(XY_m, poly_xy_m)

        keep = dist_ft <= corridor_ft
        sec = data.loc[keep].copy()
        sec["Chainage_ft"] = chain_ft[keep]
        sec = sec.sort_values("Chainage_ft")

        if sec.empty:
            st.warning("No borings fall within the selected corridor width. Widen the corridor or redraw the line.")
        else:
            # ---- Filled-band profile (single PWR legend; unified hover) ----
            x   = sec["Chainage_ft"].to_numpy()
            top = sec["Top_EL"].to_numpy()
            bot = sec["Bottom_EL"].to_numpy()
            pwr = sec["PWR_EL"].to_numpy()  # may include NaNs

            fig = go.Figure()

            # Soil (green): Top -> (PWR if present else Bottom)
            lower_soil = np.where(np.isnan(pwr), bot, pwr)
            add_band(fig, x, top, lower_soil, "rgba(34,197,94,0.55)", "Soil", True, "soil")

            # PWR (maroon): PWR -> Bottom  [single legend entry even if many segments]
            mask = ~np.isnan(pwr)
            first_pwr_band = True
            if mask.any():
                idx = np.where(mask)[0]
                splits = np.where(np.diff(idx) > 1)[0]
                segments = np.split(idx, splits + 1)
                for seg in segments:
                    xs  = x[seg]
                    y_up = pwr[seg]
                    y_lo = bot[seg]
                    add_band(fig, xs, y_up, y_lo, "rgba(127,29,29,0.70)",
                             "PWR", showlegend=first_pwr_band, legendgroup="pwrband")
                    first_pwr_band = False

            # Vertical posts at each boring
            for xi, ytop, ybot in zip(x, top, bot):
                fig.add_trace(go.Scatter(
                    x=[xi, xi], y=[ybot, ytop],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip"
                ))

            # Top & Bottom outlines (clean hover text)
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

            # PWR line (dashed) & markers ONLY where PWR exists
            if mask.any():
                first_pwr_line = True
                idx = np.where(mask)[0]
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
                    x=x[mask], y=pwr[mask], mode="markers",
                    marker=dict(size=4, color="black"),
                    name="PWR EL (ft)",
                    legendgroup="pwr",
                    showlegend=False,
                    hovertemplate="PWR EL (ft): %{y:.2f}<extra></extra>"
                ))

            # Borehole labels
            for xi, yi, label in zip(x, top, sec["Name"]):
                fig.add_annotation(
                    x=xi, y=yi, text=str(label),
                    showarrow=True, arrowhead=1, arrowsize=1, ax=0, ay=-25
                )

            # Unified vertical tooltip + spike
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
# 3D Borehole View (points only, plan coordinates in feet) with lower vertical exaggeration
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

# Project lon/lat to local UTM → meters → feet (plan coordinates)
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
        aspectratio=dict(x=1, y=1, z=ve),   # lower exaggeration
    ),
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, b=0, t=10),
    scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))
)

st.plotly_chart(fig3d, use_container_width=True)
