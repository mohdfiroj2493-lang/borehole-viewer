import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
import plotly.graph_objects as go
from pyproj import CRS, Transformer
import streamlit as st
from streamlit_folium import st_folium


# -------------------------------
# Helper functions
# -------------------------------
def pick(colnames, *aliases):
    for a in aliases:
        a = a.lower()
        if a in colnames:
            return a
    return None


def get_transformer(lat, lon):
    zone = int((lon + 180) // 6) + 1
    crs_target = CRS.from_string(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs_target, always_xy=True)


def project_chainage_to_polyline(XY, poly_xy):
    """
    For each point in XY (Nx2), compute:
      - shortest distance to a polyline (piecewise linear),
      - chainage along the polyline at the closest projection point.
    Returns arrays: chainage_m (N,), distance_m (N,)
    """
    # cumulative lengths along polyline
    seg_vecs = poly_xy[1:] - poly_xy[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    seg_len[seg_len == 0] = 1e-9
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    chain = np.zeros(len(XY))
    dist = np.full(len(XY), np.inf)

    for i, P in enumerate(XY):
        best_d = np.inf
        best_ch = 0.0
        for k in range(len(seg_vecs)):
            A = poly_xy[k]
            v = seg_vecs[k]
            L2 = np.dot(v, v)
            t = np.clip(np.dot(P - A, v) / L2, 0.0, 1.0)
            Q = A + t * v
            d = np.linalg.norm(P - Q)
            if d < best_d:
                best_d = d
                best_ch = cum[k] + t * seg_len[k]
        chain[i] = best_ch
        dist[i] = best_d
    return chain, dist, cum[-1]  # last is total polyline length


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Borehole Viewer", layout="wide")

st.title("📍 Borehole Visualization Tool")
st.write("Upload your Excel bore log data, **draw a section line on the map**, and generate the profile & 3D view.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # detect required cols
    c_name = pick(df.columns, 'name', 'id', 'boring id')
    c_lat = pick(df.columns, 'latitude', 'lat')
    c_lon = pick(df.columns, 'longitude', 'lon')
    c_top = pick(df.columns, 'boring elevation', 'ground elevation', 'top el')
    c_depth = pick(df.columns, 'depth', 'total depth')
    c_pwr_d = pick(df.columns, 'pwr depth')
    c_pwr_el = pick(df.columns, 'pwr el')

    data = pd.DataFrame({
        'Name': df[c_name],
        'Latitude': df[c_lat],
        'Longitude': df[c_lon],
        'Top_EL': df[c_top].astype(float),
        'Depth': df[c_depth].astype(float)
    })

    data['Bottom_EL'] = data['Top_EL'] - data['Depth']
    data['PWR_EL'] = np.nan

    if c_pwr_el:
        data['PWR_EL'] = pd.to_numeric(df[c_pwr_el], errors="coerce")
    if c_pwr_d:
        pwr_d = pd.to_numeric(df[c_pwr_d], errors="coerce")
        data['PWR_EL'] = data['PWR_EL'].fillna(data['Top_EL'] - pwr_d)

    # Drop rows without coordinates
    data = data.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if data.empty:
        st.error("No valid rows with Latitude/Longitude found in the uploaded file.")
        st.stop()

    # -------------------
    # 1) Map with drawing tools
    # -------------------
    st.header("🌍 Map — Draw your section line")
    with st.expander("Tip", expanded=True):
        st.write("Use the **polyline tool** (icon with a line) to draw your section on the map. Finish with a double-click.")

    center_lat, center_lon = data["Latitude"].mean(), data["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # base layers with attribution
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

    # show borings (circle + label)
    for _, r in data.iterrows():
        popup = f"<b>{r['Name']}</b><br>Top EL: {r['Top_EL']:.2f}<br>PWR EL: {r['PWR_EL'] if pd.notna(r['PWR_EL']) else 'N/A'}<br>Bottom EL: {r['Bottom_EL']:.2f}"
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=6,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=popup,
            tooltip=r["Name"]
        ).add_to(m)
        folium.Marker(
            [r["Latitude"], r["Longitude"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 10pt;
                    color: black;
                    white-space: nowrap;
                    text-align: center;
                    transform: translateY(12px);
                ">
                {r['Name']}
                </div>
                """
            )
        ).add_to(m)

    # drawing controls (polyline only)
    Draw(
        export=False,
        draw_options={
            "polyline": True,
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # render map and capture drawings
    map_state = st_folium(
        m,
        width=None,
        height=600,
        returned_objects=["last_active_drawing", "all_drawings"],
        use_container_width=True,
    )

    # -------------------
    # 2) Generate Section from drawn line
    # -------------------
    st.header("📈 Section / Profile (from drawn line)")

    corridor = st.slider("Corridor width around section line (meters)", 10, 200, 60, step=5)

    # Parse the latest drawn polyline (if any)
    polyline_coords = None
    drawings = map_state.get("all_drawings") or []
    if drawings:
        # find the last polyline geometry in the drawings list
        for g in reversed(drawings):
            if g and g.get("type") == "Feature":
                geom = g.get("geometry", {})
                if geom.get("type") == "LineString":
                    polyline_coords = geom.get("coordinates")  # [[lon, lat], ...]
                    break

    if polyline_coords is None:
        st.info("Draw a polyline on the map to define the section, then adjust the corridor and scroll down.")
    else:
        # Convert to numpy and project to a local metric CRS for accurate distances
        poly_lonlat = np.array(polyline_coords)  # (M, 2) (lon, lat)
        center_latlon = [data["Latitude"].mean(), data["Longitude"].mean()]
        transformer = get_transformer(center_latlon[0], center_latlon[1])

        # project borings and polyline to meters
        XY = np.array([transformer.transform(lon, lat) for lat, lon in zip(data["Latitude"], data["Longitude"])])
        poly_xy = np.array([transformer.transform(lon, lat) for lon, lat in poly_lonlat])

        # compute chainage & distance to polyline
        chain, dist, total_len = project_chainage_to_polyline(XY, poly_xy)

        # filter by corridor
        keep = dist <= corridor
        sec = data.loc[keep].copy()
        sec["Chainage_m"] = chain[keep]
        sec = sec.sort_values("Chainage_m")

        if sec.empty:
            st.warning("No borings fall within the selected corridor width. Try widening the corridor or adjusting the line.")
        else:
            # Plot section
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sec["Chainage_m"], y=sec["Top_EL"], mode="lines+markers", name="Top EL"))
            if sec["PWR_EL"].notna().any():
                fig.add_trace(go.Scatter(x=sec["Chainage_m"], y=sec["PWR_EL"], mode="lines+markers", name="PWR EL"))
            fig.add_trace(go.Scatter(x=sec["Chainage_m"], y=sec["Bottom_EL"], mode="lines+markers", name="Bottom EL"))

            # annotate names
            for x, y, n in zip(sec["Chainage_m"], sec["Top_EL"], sec["Name"]):
                fig.add_annotation(x=x, y=y, text=str(n), showarrow=True, arrowhead=1, yshift=6)

            fig.update_layout(
                title=f"Section along drawn line (Length ≈ {total_len:.0f} m, corridor ±{corridor} m)",
                xaxis_title="Chainage (m)",
                yaxis_title="Elevation",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # 3) 3D View (unchanged; shows all borings)
    # -------------------
    st.header("🌀 3D Borehole View")

    lines_x, lines_y, lines_z = [], [], []
    pwr_x, pwr_y, pwr_z = [], [], []

    for _, r in data.iterrows():
        x, y = r["Longitude"], r["Latitude"]
        lines_x += [x, x, None]
        lines_y += [y, y, None]
        lines_z += [r["Bottom_EL"], r["Top_EL"], None]
        if not pd.isna(r["PWR_EL"]):
            pwr_x.append(x); pwr_y.append(y); pwr_z.append(r["PWR_EL"])

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode="lines", name="Boring"))
    fig3d.add_trace(go.Scatter3d(x=data["Longitude"], y=data["Latitude"], z=data["Top_EL"], mode="markers", name="Top EL"))
    if pwr_x:
        fig3d.add_trace(go.Scatter3d(x=pwr_x, y=pwr_y, z=pwr_z, mode="markers", name="PWR EL"))

    fig3d.update_layout(height=600, scene=dict(zaxis_title="Elevation"))
    st.plotly_chart(fig3d, use_container_width=True)
