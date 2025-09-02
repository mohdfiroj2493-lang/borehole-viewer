# ---- Filled-band profile (no duplicate legend entries; no PWR interpolation) ----
x   = sec["Chainage_ft"].to_numpy()
top = sec["Top_EL"].to_numpy()
bot = sec["Bottom_EL"].to_numpy()
pwr = sec["PWR_EL"].to_numpy()  # may include NaNs

fig = go.Figure()

def add_band(fig, x_arr, y_upper, y_lower, fill_rgba, name, showlegend=True, legendgroup=None):
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

# 1) Overburden (one legend item)
lower_overburden = np.where(np.isnan(pwr), bot, pwr)
add_band(
    fig, x, top, lower_overburden,
    "rgba(34,197,94,0.55)",  # green
    "Overburden", showlegend=True, legendgroup="overburden"
)

# 2) Rock: draw only where PWR exists, but show **one** legend item total
mask = ~np.isnan(pwr)
first_rock = True
if mask.any():
    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, splits + 1)
    for seg in segments:
        xs  = x[seg]
        y_up = pwr[seg]
        y_lo = bot[seg]
        add_band(
            fig, xs, y_up, y_lo,
            "rgba(127,29,29,0.70)",  # maroon
            "Rock",
            showlegend=first_rock,
            legendgroup="rock"
        )
        first_rock = False

# 3) Vertical posts at each boring
for xi, ytop, ybot in zip(x, top, bot):
    fig.add_trace(
        go.Scatter(
            x=[xi, xi], y=[ybot, ytop],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip"
        )
    )

# 4) Top & Bottom outlines
fig.add_trace(go.Scatter(
    x=x, y=top, mode="lines+markers",
    line=dict(color="black", width=1),
    marker=dict(size=5, color="black"),
    name="Top EL (ft)", legendgroup="top"
))
fig.add_trace(go.Scatter(
    x=x, y=bot, mode="lines",
    line=dict(color="black", width=1),
    name="Bottom EL (ft)", legendgroup="bottom"
))

# 5) PWR line/points ONLY where PWR exists (no interpolation, one legend item)
if mask.any():
    # dashed line segments (no hover, no extra legend entries)
    first_pwr = True
    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, splits + 1)
    for seg in segments:
        xs = x[seg]; ys = pwr[seg]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            name="PWR EL (ft)",
            legendgroup="pwr",
            showlegend=first_pwr,   # only once
            hoverinfo="skip"        # prevent “nearby” values in hover
        ))
        first_pwr = False
    # markers with hover **only at real PWR points**
    fig.add_trace(go.Scatter(
        x=x[mask], y=pwr[mask], mode="markers",
        marker=dict(size=4, color="black"),
        name="PWR EL (ft)",
        legendgroup="pwr",
        showlegend=False,          # keep legend tidy
        hovertemplate="PWR EL (ft): %{y:.2f}<extra></extra>"
    ))

# 6) Borehole labels
for xi, yi, label in zip(x, top, sec["Name"]):
    fig.add_annotation(
        x=xi, y=yi, text=str(label),
        showarrow=True, arrowhead=1, arrowsize=1, ax=0, ay=-25
    )

fig.update_layout(
    title=f"Section along drawn line (Length ≈ {total_len_ft:.0f} ft, corridor ±{corridor_ft} ft)",
    xaxis_title="Chainage (ft)",
    yaxis_title="Elevation (ft)",
    template="plotly_white",
    hovermode="closest",          # <- no unified interpolation
    legend=dict(orientation="h")
)
st.plotly_chart(fig, use_container_width=True)
