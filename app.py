import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
from pyproj import CRS, Transformer
import streamlit as st


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


def project_chainage(xy_pts, A, B):
    v = B - A
    L2 = np.dot(v, v)
    if L2 == 0:
        return np.zeros(len(xy_pts)), 0.0
    t = ((xy_pts - A) @ v) / L2
    s = np.clip(t, 0, 1)
    chain = s * np.sqrt(L2)
    return chain, np.sqrt(L2)


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Borehole Viewer", layout="wide")

st.title("📍 Borehole Visualization Tool")
st.write("Upload your Excel bore log data to visualize borings on a map, create profiles, and view in 3D.")

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

    # 🚨 Drop rows without coordinates
    data = data.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    if data.empty:
        st.error("No valid rows with Latitude/Longitude found in the uploaded file.")
        st.stop()

    # -------------------
    # 1. Folium Map
    # -------------------
    st.header("🌍 Interactive Map")

    center_lat, center_lon = data["Latitude"].mean(), data["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # ✅ Add tile layers with attribution
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

    # Add borehole markers
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

        folium.map.Marker(
            [r["Latitude"], r["Longitude"]],
            icon=folium.DivIcon(
                html=f"""<div style="font-size: 10pt; color: black">{r['Name']}</div>"""
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)

    st.components.v1.html(m._repr_html_(), height=600)

    # -------------------
    # 2. Section/Profile
    # -------------------
    st.header("📈 Section / Profile")

    start = st.selectbox("Start Boring", data["Name"].tolist())
    end = st.selectbox("End Boring", data["Name"].tolist())

    if st.button("Plot Section"):
        transformer = get_transformer(center_lat, center_lon)
        xy = np.array([transformer.transform(lon, lat) for lat, lon in zip(data["Latitude"], data["Longitude"])])
        data["X_m"], data["Y_m"] = xy[:, 0], xy[:, 1]

        A = data.loc[data["Name"] == start, ["X_m", "Y_m"]].values[0]
        B = data.loc[data["Name"] == end, ["X_m", "Y_m"]].values[0]

        chain, seg_len = project_chainage(xy, A, B)
        data["Chainage_m"] = chain

        sec = data.sort_values("Chainage_m")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sec["Chainage_m"], y=sec["Top_EL"], mode="lines+markers", name="Top EL"))
        if sec["PWR_EL"].notna().any():
            fig.add_trace(go.Scatter(x=sec["Chainage_m"], y=sec["PWR_EL"], mode="lines+markers", name="PWR EL"))
        fig.add_trace(go.Scatter(x=sec["Chainage_m"], y=sec["Bottom_EL"], mode="lines+markers", name="Bottom EL"))

        fig.update_layout(
            title=f"Section: {start} → {end} (≈{seg_len:.0f} m)",
            xaxis_title="Chainage (m)",
            yaxis_title="Elevation",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # 3. 3D View
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
            pwr_x.append(x)
            pwr_y.append(y)
            pwr_z.append(r["PWR_EL"])

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode="lines", name="Boring"))
    fig3d.add_trace(go.Scatter3d(x=data["Longitude"], y=data["Latitude"], z=data["Top_EL"], mode="markers", name="Top EL"))
    if pwr_x:
        fig3d.add_trace(go.Scatter3d(x=pwr_x, y=pwr_y, z=pwr_z, mode="markers", name="PWR EL"))

    fig3d.update_layout(height=600, scene=dict(zaxis_title="Elevation"))
    st.plotly_chart(fig3d, use_container_width=True)
