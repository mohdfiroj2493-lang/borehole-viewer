mkdir -p ~/.streamlit/
echo "\
[server]\nheadless = true\nenableCORS=false\nport = $PORT\n\n" > ~/.streamlit/config.toml
