import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
import plotly.graph_objects as go
from pyproj import CRS, Transformer
import streamlit as st
from streamlit_folium import st_folium

FT_PER_M = 3.280839895  # exact conversion

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
    # Use a local UTM (meters) then convert results to feet
    zone = int((lon + 180) // 6) + 1
    crs_target = CRS.from_string(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs_target, always_xy=True)

def project_chainage_to_polyline(XY_m, poly_xy_m):
    """
    XY_m: Nx2 in meters
    poly_xy_m: Mx2 in meters
    Returns: chain_ft, dist_ft, total_len_ft
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

    # Convert outputs to feet
    return chain_m * FT_PER_M, dist_m * FT_PER_M, cum_m[-1] * FT_PER_M


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Borehole Viewer", layout="wide")
st.title("📍 Borehole Visualization Tool")
st.caption("Upload your Excel bore log data, draw a section on the map, and generate the profile & 3D view. All units shown in feet.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Columns
    c_name   = pick(df.columns, 'name', 'id', 'boring id', 'hole id')
    c_lat    = pick(df.columns, 'latitude', 'lat')
    c_lon    = pick(df.columns, 'longitude', 'lon', 'long')
    c_top    = pick(df.columns, 'boring elevation', 'ground elevation', 'top el', 'top elevation')
    c_depth  = pick(df.columns, 'depth', 'total depth', 'hole depth')
    c_pwr_d  = pick(df.columns, 'pwr depth', 'weathered rock depth')
    c_pwr_el = pick(df.columns, 'pwr el', 'pwr elevation', 'weathered rock elevation')

    data = pd.DataFrame({
        'Name': df[c_name],
        'Latitude': df[c_lat],
        'Longitude': df[c_lon],
        'Top_EL': pd.to_numeric(df[c_top], errors="coerce"),
        'Depth':  pd.to_numeric(df[c_depth], errors="coerce"),
    })

    data['Bottom_EL'] = data['Top_EL'] - data['Depth']  # feet - feet
    data['PWR_EL'] = np.nan

    if c_pwr_el:
        data['PWR_EL'] = pd.to_numeric(df[c_pwr_el], errors="coerce")
    if c_pwr_d:
        pwr_d = pd.to_numeric(df[c_pwr_d], errors="coerce")
        data['PWR_EL'] = data['PWR_EL'].fillna(data['Top_EL'] - pwr_d)  # compute in feet

    # Remove rows without coords
    data = data.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if data.empty:
        st.error("No valid rows with Latitude/Longitude found.")
        st.stop()

    # -------------------
    # Map + Drawing
    # -------------------
    st.header("🌍 Map — Draw your section line")
    with st.expander("Tip", expanded=True):
        st.write("Use the **polyline** tool to draw your section. Double-click to finish. Elevations are in **ft**. Corridor width slider is **ft**.")

    center_lat, center_lon = data["Latitude"].mean(), data["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # Base layers with attribution
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

    # Boreholes (feet in popup labels)
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
        m, height=600, returned_objects=["last_active_drawing", "all_drawings"],
        use_container_width=True
    )

    # -------------------
    # Section from drawn line (feet)
    # -------------------
    st.header("📈 Section / Profile (ft)")
    corridor_ft = st.slider("Corridor width (ft)", 25, 1000, 200, step=25)

    polyline_coords = None
    drawings = map_state.get("all_drawings") or []
    if drawings:
        for g in reversed(drawings):
            if g and g.get("type") == "Feature":
                geom = g.get("geometry", {})
                if geom.get("type") == "LineString":
                    polyline_coords = geom.get("coordinates")  # [[lon, lat], ...]
                    break

    if polyline_coords is None:
        st.info("Draw a polyline on the map to define the section line.")
    else:
        # Project to meters (then convert to ft for outputs)
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
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sec["Chainage_ft"], y=sec["Top_EL"],
                                     mode="lines+markers", name="Top EL (ft)"))
            if sec["PWR_EL"].notna().any():
                fig.add_trace(go.Scatter(x=sec["Chainage_ft"], y=sec["PWR_EL"],
                                         mode="lines+markers", name="PWR EL (ft)"))
            fig.add_trace(go.Scatter(x=sec["Chainage_ft"], y=sec["Bottom_EL"],
                                     mode="lines+markers", name="Bottom EL (ft)"))

            for x, y, n in zip(sec["Chainage_ft"], sec["Top_EL"], sec["Name"]):
                fig.add_annotation(x=x, y=y, text=str(n), showarrow=True, arrowhead=1, yshift=6)

            fig.update_layout(
                title=f"Section along drawn line (Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor_ft} ft)",
                xaxis_title="Chainage (ft)",
                yaxis_title="Elevation (ft)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # 3D View (Z in feet)
    # -------------------
    st.header("🌀 3D Borehole View (ft)")
    lines_x, lines_y, lines_z = [], [], []
    pwr_x, pwr_y, pwr_z = [], [], []

    for _, r in data.iterrows():
        x, y = r["Longitude"], r["Latitude"]
        lines_x += [x, x, None]
        lines_y += [y, y, None]
        lines_z += [r["Bottom_EL"], r["Top_EL"], None]  # feet
        if not pd.isna(r["PWR_EL"]):
            pwr_x.append(x); pwr_y.append(y); pwr_z.append(r["PWR_EL"])

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode="lines", name="Boring"))
    fig3d.add_trace(go.Scatter3d(x=data["Longitude"], y=data["Latitude"], z=data["Top_EL"], mode="markers", name="Top EL"))
    if pwr_x:
        fig3d.add_trace(go.Scatter3d(x=pwr_x, y=pwr_y, z=pwr_z, mode="markers", name="PWR EL"))

    fig3d.update_layout(height=600, scene=dict(zaxis_title="Elevation (ft)"))
    st.plotly_chart(fig3d, use_container_width=True)
