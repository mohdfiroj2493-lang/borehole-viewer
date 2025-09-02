# -------------------
# 1. Folium Map
# -------------------
st.header("🌍 Interactive Map")

center_lat, center_lon = data["Latitude"].mean(), data["Longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

# 🔄 Add different tile layers
folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
folium.TileLayer("Stamen Terrain", name="Terrain").add_to(m)
folium.TileLayer("Stamen Toner", name="Black & White").add_to(m)
folium.TileLayer("CartoDB positron", name="Light").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
    overlay=False,
    control=True
).add_to(m)

# Add borehole markers
for _, r in data.iterrows():
    popup = f"<b>{r['Name']}</b><br>Top EL: {r['Top_EL']:.2f}<br>PWR EL: {r['PWR_EL'] if pd.notna(r['PWR_EL']) else 'N/A'}<br>Bottom EL: {r['Bottom_EL']:.2f}"
    
    # Circle marker with tooltip
    folium.CircleMarker(
        location=[r["Latitude"], r["Longitude"]],
        radius=6,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=popup,
        tooltip=r["Name"]
    ).add_to(m)

    # Always visible text label
    folium.map.Marker(
        [r["Latitude"], r["Longitude"]],
        icon=folium.DivIcon(
            html=f"""<div style="font-size: 10pt; color: black">{r['Name']}</div>"""
        )
    ).add_to(m)

# 🔄 Add layer control
folium.LayerControl().add_to(m)

# Show map in Streamlit
st.components.v1.html(m._repr_html_(), height=600)
