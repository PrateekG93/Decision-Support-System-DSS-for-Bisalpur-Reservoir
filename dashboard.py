"""
DASHBOARD — Bisalpur Reservoir DSS (Final)
============================================
Run: streamlit run dashboard.py

Architecture:
  - Weather fetched ONCE via cache (15-min TTL)
  - Sidebar sliders rebuild forecast live using cached API rain values
  - Model 1 applied on every forecast rebuild for local calibration
  - Alert system has two layers: current-state + trajectory-based
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weather_api import (
    get_full_weather_data, fetch_15day_forecast,
    build_forecast_dataframe, BISALPUR_LAT, BISALPUR_LON,
)
from risk_engine import (
    generate_approaches, get_alerts,
    CAPACITY_MCM, FLOOD_THRESHOLD, DROUGHT_THRESHOLD, MIN_STORAGE_PCT,
)
from storage_fetcher import get_current_storage

st.set_page_config(page_title="Bisalpur DSS", page_icon="💧", layout="wide")

RANK_COLORS = {"OPTIMAL":"#1D9E75","SAFE_ALTERNATIVE":"#378ADD",
               "ACCEPTABLE":"#BA7517","HIGH_RISK":"#E24B4A","INFEASIBLE":"#B4B2A9"}
RANK_ICONS  = {"OPTIMAL":"✅","SAFE_ALTERNATIVE":"🔵",
               "ACCEPTABLE":"🟡","HIGH_RISK":"🔴","INFEASIBLE":"⛔"}


# ── Cached API fetchers ────────────────────────────────────────────────────────
# These hit the network. Cached so sliders don't re-fetch.

@st.cache_data(ttl=900, show_spinner=False)
def _get_api_weather():
    """Fetch current temp + humidity from Open-Meteo. Cache 15 min."""
    from weather_api import fetch_current_weather
    t, h = fetch_current_weather()
    return (float(t) if t else 28.0), (float(h) if h else 55.0)

@st.cache_data(ttl=900, show_spinner=False)
def _get_api_rain():
    """Fetch 15-day raw rain values from Open-Meteo. Cache 15 min."""
    rain_mm, rain_prob = fetch_15day_forecast()
    return rain_mm, rain_prob

@st.cache_data(ttl=1800, show_spinner=False)
def _get_storage():
    """
    Auto-load current storage. Cache 30 min.
    Tries: India WRIS live API → local CSV → default 65%.
    """
    return get_current_storage()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💧 Bisalpur DSS")
    st.caption("Bisalpur Reservoir · Ganga Basin, Rajasthan")
    st.divider()

    st.subheader("Reservoir settings")
    capacity_mcm = st.number_input(
        "Dam capacity (MCM)", value=1076, min_value=100, max_value=50000, step=10,
    )
    min_floor = st.slider("Min storage floor (%)", 10, 40, 25,
        help="Hard minimum — never release below this (drinking water reserve)")

    st.divider()
    st.subheader("Current conditions")

    if st.button("📡 Auto-fetch weather"):
        st.cache_data.clear()
        st.rerun()

    # ── Auto-load current storage ─────────────────────────────────────────────
    # Tries India WRIS API → local CSV → default 65%
    auto_pct, auto_source, auto_date, auto_days_old, auto_warning = _get_storage()

    # Show where the value came from
    source_color = "🟢" if auto_days_old == 0 else ("🟡" if auto_days_old <= 30 else "🔴")
    st.caption(f"{source_color} Storage source: {auto_source}")
    if auto_date != "unknown":
        st.caption(f"   Reading date: {auto_date}")
    if auto_warning:
        st.warning(f"⚠️ {auto_warning}", icon=None)

    # Operator can override auto value if they have a fresher reading
    current_storage = st.slider(
        "Current storage (%) — adjust if you have a fresher reading",
        min_value=0, max_value=100,
        value=int(round(auto_pct)),
        help=(
            f"Auto-loaded from {auto_source} ({auto_date}). "
            "Drag to override if you have a more recent manual reading."
        ),
    )

    # Get live weather for temperature/humidity defaults
    live_temp_default, live_hum_default = _get_api_weather()
    current_temp     = st.slider("Temperature (°C)", 5, 50,
                                 value=int(live_temp_default))
    current_humidity = st.slider("Humidity (%)", 5, 100,
                                 value=int(live_hum_default))

    st.divider()
    st.caption(f"Storage: {current_storage/100*capacity_mcm:,.0f} MCM of {capacity_mcm:,}")
    st.caption(f"Min floor: {min_floor}% = {min_floor/100*capacity_mcm:,.0f} MCM")
    st.caption(f"Flood alert:   > {FLOOD_THRESHOLD}%")
    st.caption(f"Drought alert: < {DROUGHT_THRESHOLD}%")


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Bisalpur Reservoir — Decision Support System")
st.caption("Open-Meteo forecast + Model 1 local calibration · Staged release strategies · Trajectory-aware risk alerts")

if not os.path.exists("models/storage_model.pkl"):
    st.error(
        "**Models not trained yet.**\n\n"
        "```\npython step1_prepare_data.py\npython step2_train_models.py\n```"
    )
    st.stop()


# ── Build forecast (API rain + Model 1 calibration) ───────────────────────────
# Fetch raw API values (cached — no re-fetch on slider change)
with st.spinner("Fetching 15-day forecast from Open-Meteo…"):
    api_rain_mm, api_rain_prob = _get_api_rain()

# Rebuild the calibrated forecast using sidebar conditions
# This runs on every slider change, but uses cached API values
# so no network call happens — just Model 1 + math
rain_forecast = build_forecast_dataframe(
    rain_mm_list=api_rain_mm,
    rain_prob_list=api_rain_prob,
    current_temp=float(current_temp),
    current_humidity=float(current_humidity),
    current_month=datetime.today().month,
)

# Compute approaches (cached by storage + rain_df + month)
@st.cache_data(ttl=300, show_spinner=False)
def _get_approaches(storage: float, rain_df: pd.DataFrame, month: int):
    return generate_approaches(float(storage), rain_df, int(month))

approaches, base_risk = _get_approaches(
    float(current_storage), rain_forecast, datetime.today().month
)
primary_risk, alerts = get_alerts(float(current_storage), rain_forecast, approaches)
optimal  = next((a for a in approaches if a.rank == "OPTIMAL"), None)
today    = datetime.today()
date_lbl = [(today + timedelta(days=i+1)).strftime("%b %d") for i in range(15)]

# Refresh button
if st.button("🔄 Refresh forecast (clear cache)"):
    st.cache_data.clear()
    st.rerun()


# ── KPI row ────────────────────────────────────────────────────────────────────
risk_emoji = {"CRITICAL FLOOD":"🔴","FLOOD":"🔴","ELEVATED":"🟡",
              "NORMAL":"🟢","DRY":"🟠","DROUGHT":"🔴"}

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Current storage", f"{current_storage}%",
              delta=f"{current_storage/100*capacity_mcm:,.0f} MCM")
    freshness = "live" if auto_days_old == 0 else (f"{auto_days_old}d old" if auto_days_old < 9999 else "manual")
    st.caption(f"{risk_emoji.get(primary_risk,'🟢')} {primary_risk} · {freshness}")

with c2:
    # Use p50_mm sum and count from the actual calibrated forecast
    total_rain = float(rain_forecast["p50_mm"].sum())
    # Count days where p50 > 2mm OR where rain_prob > 40% (covers both cases)
    rain_days  = int(
        ((rain_forecast["p50_mm"] > 2) | (rain_forecast["rain_prob"] > 0.40)).sum()
    )
    st.metric("15-day rainfall (P50)", f"{total_rain:.1f} mm")
    st.caption(f"{rain_days} rainy days expected (>2mm or >40% probability)")

with c3:
    if optimal:
        st.metric("Recommended release", f"{optimal.release_pct_per_day}% / day")
        st.caption(f"{optimal.release_mcm_per_day} MCM/day · {optimal.active_days} active days")
    else:
        st.metric("Recommended release", "—")

with c4:
    if optimal:
        fi = "🔴" if optimal.flood_risk=="HIGH"   else ("🟡" if optimal.flood_risk=="MEDIUM"   else "🟢")
        di = "🔴" if optimal.drought_risk=="HIGH" else ("🟡" if optimal.drought_risk=="MEDIUM" else "🟢")
        st.metric("Best strategy", optimal.name)
        st.caption(f"Flood {fi} {optimal.flood_risk}   Drought {di} {optimal.drought_risk}")


# ── Live weather strip ─────────────────────────────────────────────────────────
with st.container(border=True):
    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("Temperature",  f"{current_temp}°C")
    wc2.metric("Humidity",     f"{current_humidity}%")
    wc3.metric("Forecast source", "Open-Meteo + Model 1")
    wc4.metric("Last updated", today.strftime("%H:%M"))


# ── Alerts ─────────────────────────────────────────────────────────────────────
for alert in alerts:
    lvl = alert["level"]
    msg = alert["msg"]
    if   lvl == "CRITICAL":                 st.error(f"🚨 {msg}")
    elif lvl in ("HIGH", "WARNING"):        st.warning(f"⚠️ {msg}")
    elif lvl == "MEDIUM":                   st.info(f"ℹ️ {msg}")
    # LOW alerts are shown silently (no banner) — keeps dashboard clean

st.divider()


# ── Rainfall forecast chart ────────────────────────────────────────────────────
st.subheader("15-Day Rainfall Forecast")
st.caption(
    "Base forecast: Open-Meteo (GFS + ECMWF ensemble). "
    "P10/P90 bands: Model 1 local calibration blended with API uncertainty. "
    "Confidence decays from 92% (Day 1) to 40% (Day 15)."
)

p50v  = rain_forecast["p50_mm"].tolist()
p90v  = rain_forecast["p90_mm"].tolist()
p10v  = rain_forecast["p10_mm"].tolist()
confv = rain_forecast["confidence"].tolist()
probv = rain_forecast["rain_prob"].tolist()

fig_rain = go.Figure()
fig_rain.add_trace(go.Bar(
    x=date_lbl, y=p50v, name="P50 forecast (mm/day)",
    marker_color=["#534AB7" if v > 10 else "#B5D4F4" for v in p50v],
))
fig_rain.add_trace(go.Scatter(
    x=date_lbl, y=p90v, mode="lines", name="P90 — heavy scenario",
    line=dict(color="#E24B4A", width=1.5, dash="dot"),
))
fig_rain.add_trace(go.Scatter(
    x=date_lbl, y=p10v, mode="lines", name="P10 — light scenario",
    line=dict(color="#1D9E75", width=1.5, dash="dot"),
))
fig_rain.add_trace(go.Scatter(
    x=date_lbl, y=confv, mode="lines+markers", name="Forecast confidence",
    line=dict(color="#BA7517", width=1.5, dash="dash"), marker=dict(size=5),
    yaxis="y2",
))
fig_rain.add_trace(go.Bar(
    x=date_lbl, y=probv, name="Rain probability",
    marker_color="rgba(83,74,183,0.15)", yaxis="y2",
    hovertemplate="%{y:.0%} chance of rain",
))
fig_rain.update_layout(
    yaxis =dict(title="Rainfall (mm/day)", rangemode="tozero"),
    yaxis2=dict(title="Probability / Confidence", overlaying="y", side="right",
                range=[0, 1.1], showgrid=False, tickformat=".0%"),
    legend=dict(orientation="h", yanchor="bottom", y=-0.40, font=dict(size=11)),
    height=340, margin=dict(l=0, r=0, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.15,
    barmode="overlay",
)
st.plotly_chart(fig_rain, use_container_width=True)
st.caption(
    "Dark bars = heavy rain (>10mm). "
    "Amber dashed = forecast confidence (right axis). "
    "Faint bars = rain probability (right axis). "
    "P10/P90 dotted lines = Model 1 uncertainty bands."
)

st.divider()


# ── Storage trajectory chart ───────────────────────────────────────────────────
st.subheader("Storage Trajectory — 4 Release Approaches")
st.caption(
    "Simulated 15-day storage under each approach using P50 rainfall. "
    "Shaded band = P10–P90 uncertainty range for the OPTIMAL strategy."
)

day_lbl_traj = ["Today"] + date_lbl
fig_traj = go.Figure()

for ap in approaches:
    if not ap.feasible:
        continue
    color = RANK_COLORS.get(ap.rank, "#888780")
    width = 3.0 if ap.rank == "OPTIMAL" else 1.5
    dash  = "solid" if ap.rank == "OPTIMAL" else "dot"
    icon  = RANK_ICONS.get(ap.rank, "")
    fig_traj.add_trace(go.Scatter(
        x=day_lbl_traj[:len(ap.trajectory_p50)], y=ap.trajectory_p50,
        mode="lines", name=f"{icon} {ap.name} ({ap.release_pct_per_day}%/day)",
        line=dict(color=color, width=width, dash=dash),
    ))

if optimal:
    fig_traj.add_trace(go.Scatter(
        x=day_lbl_traj[:len(optimal.trajectory_p90)], y=optimal.trajectory_p90,
        mode="lines", name="Optimal — P90 (heavy rain)",
        line=dict(color="#1D9E75", width=1.0, dash="dot"), showlegend=True,
    ))
    fig_traj.add_trace(go.Scatter(
        x=day_lbl_traj[:len(optimal.trajectory_p10)], y=optimal.trajectory_p10,
        mode="lines", name="Optimal — P10 (light rain)",
        line=dict(color="#1D9E75", width=1.0, dash="dot"),
        fill="tonexty", fillcolor="rgba(29,158,117,0.08)", showlegend=True,
    ))

fig_traj.add_hline(y=FLOOD_THRESHOLD,   line_dash="dash", line_color="#E24B4A",
                   annotation_text=f"Flood threshold ({FLOOD_THRESHOLD}%)")
fig_traj.add_hline(y=DROUGHT_THRESHOLD, line_dash="dash", line_color="#BA7517",
                   annotation_text=f"Drought threshold ({DROUGHT_THRESHOLD}%)")
fig_traj.add_hline(y=min_floor,         line_dash="dash", line_color="#888780",
                   annotation_text=f"Min floor ({min_floor}%)")
fig_traj.update_layout(
    yaxis=dict(title="Storage (%)", range=[0, 108]),
    legend=dict(orientation="h", yanchor="bottom", y=-0.38, font=dict(size=11)),
    height=420, margin=dict(l=0, r=0, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_traj, use_container_width=True)

st.divider()


# ── Approach cards ─────────────────────────────────────────────────────────────
st.subheader("Release Approaches — Ranked")
st.caption(
    "Release rates (3/5/8/12%/day) are fixed named protocols. "
    "Ranking, active days, feasibility, and risk scores are recalculated every time."
)

sorted_ap = sorted(approaches, key=lambda x: (0 if x.feasible else 1, x.composite_score))
for ap in sorted_ap:
    icon = RANK_ICONS.get(ap.rank, "")
    with st.container(border=True):
        ca, cb, cc = st.columns([3, 2, 2])
        with ca:
            st.markdown(f"**{icon} Approach {ap.id}: {ap.name}**")
            st.caption(ap.description)
            if not ap.feasible:
                st.error(f"⛔ {ap.infeasible_reason}")
            else:
                st.caption(f"*When to use: {ap.recommended_when}*")
        with cb:
            st.caption("**Release plan**")
            st.markdown(f"**{ap.release_pct_per_day}%** per day × {ap.active_days} days")
            st.caption(f"= {ap.release_mcm_per_day} MCM/day")
            st.caption(f"Total: {ap.total_release_pct}% = {ap.total_release_pct/100*capacity_mcm:,.0f} MCM")
        with cc:
            st.caption("**Risk assessment**")
            fi = "🔴" if ap.flood_risk=="HIGH"   else ("🟡" if ap.flood_risk=="MEDIUM"   else "🟢")
            di = "🔴" if ap.drought_risk=="HIGH" else ("🟡" if ap.drought_risk=="MEDIUM" else "🟢")
            st.caption(f"Flood:   {fi} {ap.flood_risk}")
            st.caption(f"Drought: {di} {ap.drought_risk}")
            st.caption(f"Final storage P50:    {ap.final_storage_p50}%")
            st.caption(f"Worst case P10 rain:  {ap.final_storage_p10}%")

st.divider()


# ── Day-by-day schedule ────────────────────────────────────────────────────────
if optimal:
    st.subheader(f"Day-by-Day Schedule — {optimal.name}")
    st.caption("Recalculate every morning as actual rainfall arrives.")

    rows = []
    for i in range(15):
        rel_pct      = optimal.schedule_pct[i]
        rain_p50     = float(rain_forecast.iloc[i]["p50_mm"])
        rain_prob_val= float(rain_forecast.iloc[i]["rain_prob"]) * 100
        conf         = float(rain_forecast.iloc[i]["confidence"])
        proj_storage = (optimal.trajectory_p50[i+1]
                        if (i+1) < len(optimal.trajectory_p50)
                        else optimal.trajectory_p50[-1])
        rows.append({
            "Day":              i + 1,
            "Date":             (today + timedelta(days=i+1)).strftime("%b %d"),
            "Release (%/day)":  rel_pct,
            "Release (MCM)":    round(rel_pct / 100 * capacity_mcm, 1),
            "Rain P50 (mm)":    round(rain_p50, 1),
            "Rain probability": f"{rain_prob_val:.0f}%",
            "Confidence":       f"{conf*100:.0f}%",
            "Proj. storage (%)":round(proj_storage, 1),
            "Action":           "RELEASE" if rel_pct > 0 else "HOLD",
        })

    sched_df = pd.DataFrame(rows)
    styled   = sched_df.style.map(
        lambda v: ("background-color:rgba(29,158,117,0.15)" if v == "RELEASE"
                   else "background-color:rgba(186,117,23,0.10)"),
        subset=["Action"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.info(
        f"**Key principle:** Each morning, check actual rainfall vs forecast. "
        f"If it rained less than expected, reduce tomorrow's release. "
        f"If more, increase it. The {optimal.active_days}-day plan adapts daily."
    )

st.divider()


# ── Historical chart ───────────────────────────────────────────────────────────
st.subheader("Historical Storage — Last 3 Years")
hist_path = "data/merged_clean.csv"
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path, parse_dates=["date"])
    hist = hist[hist["date"] >= hist["date"].max() - pd.DateOffset(years=3)]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist["date"], y=hist["storage_pct"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(29,158,117,0.15)",
        line=dict(color="#1D9E75", width=1.5),
        name="Storage",
    ))
    fig_hist.add_hline(y=FLOOD_THRESHOLD,   line_dash="dash", line_color="#E24B4A",
                       annotation_text="Flood threshold")
    fig_hist.add_hline(y=DROUGHT_THRESHOLD, line_dash="dash", line_color="#BA7517",
                       annotation_text="Drought threshold")
    fig_hist.add_hline(y=current_storage,   line_dash="dot",  line_color="#534AB7",
                       annotation_text=f"Today ({current_storage}%)")
    fig_hist.update_layout(
        yaxis=dict(title="Storage (%)", range=[0, 108]),
        height=280, margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("Purple dotted line = today's storage. Compare with same month in previous years.")
else:
    st.caption("Historical chart unavailable — run step1_prepare_data.py to generate data/merged_clean.csv")

st.divider()


# ── Expanders ──────────────────────────────────────────────────────────────────
with st.expander("Model accuracy metrics"):
    mc1, mc2 = st.columns(2)
    with mc1:
        st.caption("**Model 1 — Rainfall (historical pattern accuracy)**")
        if os.path.exists("models/rainfall_meta.json"):
            with open("models/rainfall_meta.json") as f: m1 = json.load(f)
            a = m1.get("accuracy", {})
            st.caption(f"Overall MAE:          {a.get('mae_overall','—')} mm/day")
            st.caption(f"R² score:             {a.get('r2','—')}  (0.2–0.4 is normal for rainfall)")
            st.caption(f"P10–P90 coverage:     {a.get('coverage_pct','—')}%  (target ~80%)")
            st.caption(f"Heavy rain detection: {a.get('heavy_rain_detection_pct','—')}%  (>20mm days)")
            st.caption(f"Dry-day specificity:  {a.get('dry_day_specificity_pct','—')}%")
            st.caption(f"Monsoon MAE:          {a.get('mae_monsoon','—')} mm/day")
    with mc2:
        st.caption("**Model 2 — Storage predictor (7-day ahead)**")
        if os.path.exists("models/storage_meta.json"):
            with open("models/storage_meta.json") as f: m2 = json.load(f)
            a2 = m2.get("accuracy", {})
            st.caption(f"MAE (7-day ahead):    {a2.get('mae_overall','—')}%")
            st.caption(f"R² score:             {a2.get('r2','—')}  (>0.85 expected for storage)")
            st.caption(f"P10–P90 coverage:     {a2.get('coverage_pct','—')}%")
            st.caption(f"Flood zone recall:    {a2.get('flood_recall_pct','—')}%  (P90 catches storage>85%)")
            st.caption(f"Drought zone recall:  {a2.get('drought_recall_pct','—')}%  (P10 catches storage<25%)")
            st.caption(f"Monsoon MAE:          {a2.get('mae_monsoon','—')}%")

with st.expander("Raw data — last 30 days"):
    if os.path.exists(hist_path):
        raw  = pd.read_csv(hist_path, parse_dates=["date"])
        cols = [c for c in ["date","storage_pct","rainfall_mm","temp_c","humidity_pct"] if c in raw.columns]
        st.dataframe(raw[cols].tail(30).round(2), use_container_width=True)

with st.expander("About this system"):
    st.markdown("""
**Data sources**
- Reservoir storage: India WRIS / data.gov.in (Bisalpur, daily 2010–2025)
- Historical rainfall + temperature: NASA POWER API (15 years)
- Live 15-day forecast: Open-Meteo GFS+ECMWF ensemble (free, no API key, updated every 6h)

**Forecast pipeline**
1. Open-Meteo returns daily rainfall (mm) + rain probability for next 15 days
2. Model 1 (LightGBM, trained on 15yr Bisalpur data) produces locally-calibrated P10/P90 bands
3. Blending: API dominates Day 1–7 (90→70% weight), Model 1 dominates Day 8–15 (60→85% weight)
4. P50 = API forecast (authoritative). P10/P90 = blended bands (uncertainty)

**Alert system — two layers**
- Layer 1: Current-state threshold alerts (storage now vs fixed thresholds)
- Layer 2: Trajectory alerts (will storage cross a threshold in next N days under optimal strategy?)

**Water balance simulation**
S(t+1) = S(t) + Inflow(t) − Release(t) − Evaporation(t)

Flood risk scored on P90 trajectory. Drought risk scored on P10 trajectory.
Hard constraints: min 25% storage, max 12%/day release — never overridden by ML.

**Research prototype** — Bisalpur Reservoir, Ganga Basin, Rajasthan
    """)

st.caption("Bisalpur Reservoir DSS · Open-Meteo + LightGBM + water balance simulation · Research prototype")
