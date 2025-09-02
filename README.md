# Borehole Viewer Web App

A Python + Streamlit app to visualize borehole data:
- 🌍 Interactive map (Folium)
- 📈 Section/Profile (Plotly)
- 🌀 3D borehole view (Plotly 3D)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io/ and link your repo.
3. Set main file to `app.py`.

## Deploy to Heroku
```bash
heroku login
heroku create borehole-viewer
git push heroku main
heroku open
```
