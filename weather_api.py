"""
weather_api.py — Open-Meteo + Model 1 Local Calibration
=========================================================
Flow:
  1. Fetch 15-day rainfall forecast from Open-Meteo API
     (physics-based GFS+ECMWF, same as AccuWeather/IMD)
  2. Fetch current temperature + humidity from same API
  3. Load trained Model 1 (LightGBM, trained on 15yr Bisalpur data)
  4. For each forecast day:
       - API gives the base P50 (most likely mm)
       - Model 1 gives locally-calibrated P10/P90 bands
       - Blend: API dominates Day 1–7 (high skill), Model 1 adds
         local knowledge Day 8–15 (API skill decays)
  5. Return unified DataFrame with p10/p50/p90/confidence/rain_prob

Why Model 1 matters:
  Open-Meteo predicts for a 28km × 28km grid cell. Bisalpur sits in
  specific terrain in Rajasthan. 15 years of local data teaches the
  model how much the global forecast over/under-predicts here. This
  technique is called Model Output Statistics (MOS) and is standard
  practice at every national weather service.
"""

import requests
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime

BISALPUR_LAT = 25.9246
BISALPUR_LON = 75.4538


# ── API calls ──────────────────────────────────────────────────────────────────

def fetch_current_weather(lat=BISALPUR_LAT, lon=BISALPUR_LON):
    """Get current temperature (°C) and humidity (%)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m"
            f"&forecast_days=1"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        c = r.json()["current"]
        return float(c["temperature_2m"]), float(c["relative_humidity_2m"])
    except Exception as e:
        print(f"  Weather API unavailable: {e}")
        return None, None


def fetch_15day_forecast(lat=BISALPUR_LAT, lon=BISALPUR_LON):
    """
    Get 15-day daily rainfall forecast from Open-Meteo.
    Returns (rain_mm_list, rain_prob_list).
    Falls back to zeros if unavailable.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,precipitation_probability_max"
            f"&forecast_days=15"
            f"&timezone=Asia%2FKolkata"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        daily    = r.json()["daily"]
        rain_raw = daily.get("precipitation_sum", [0.0]*15)
        prob_raw = daily.get("precipitation_probability_max", [50]*15)
        rain_mm  = [float(x) if x is not None else 0.0 for x in rain_raw[:15]]
        rain_prob= [float(x)/100.0 if x is not None else 0.5 for x in prob_raw[:15]]
        return rain_mm, rain_prob
    except Exception as e:
        print(f"  Forecast API unavailable: {e}")
        return [0.0]*15, [0.0]*15


# ── Model 1 loading ────────────────────────────────────────────────────────────

def _load_model1():
    """Load trained Model 1 from disk. Returns (models_dict, feature_cols, use_log) or Nones."""
    path = "models/rainfall_model.pkl"
    if not os.path.exists(path):
        return None, None, False
    try:
        saved = joblib.load(path)
        return saved["models"], saved["feature_cols"], saved.get("log_target", True)
    except Exception as e:
        print(f"  Could not load Model 1: {e}")
        return None, None, False


def _build_feature_row(rain_history, temp, humidity, month, day_offset):
    """
    Build one feature row for Model 1 prediction.
    Mirrors step2_train_models.py build_rainfall_features() exactly.

    rain_history: list of recent past rainfall values (most recent last)
    day_offset:   1-based index of the forecast day (1 = tomorrow)
    """
    hist = rain_history  # recent history, most recent = hist[-1]
    row  = {}

    # Rainfall lags
    for lag in [1, 2, 3, 5, 7, 10, 14]:
        row[f"rain_lag{lag}d"] = float(hist[-lag]) if lag <= len(hist) else 0.0

    # Rolling stats
    for w in [3, 7, 14, 30]:
        h = hist[-w:] if w <= len(hist) else hist
        row[f"rain_roll{w}d"] = float(np.mean(h)) if h else 0.0
        row[f"rain_sum{w}d"]  = float(np.sum(h))  if h else 0.0
        row[f"rain_max{w}d"]  = float(np.max(h))  if h else 0.0

    # Wet/heavy day indicators
    row["is_wet"]         = 1 if (hist[-1] > 1.0)  else 0
    row["is_heavy"]       = 1 if (hist[-1] > 10.0) else 0
    row["wet_streak_7"]   = sum(1 for x in hist[-7:] if x > 1.0)
    row["heavy_streak_7"] = sum(1 for x in hist[-7:] if x > 10.0)

    # Soil moisture proxy
    row["soil_moisture"] = (0.5 * row["rain_sum7d"] +
                            0.3 * row["rain_sum14d"] +
                            0.2 * row["rain_sum30d"])

    # Humidity
    row["hum_lag1d"]   = float(humidity)
    row["hum_lag2d"]   = float(humidity)
    row["hum_lag3d"]   = float(humidity)
    row["hum_lag5d"]   = float(humidity)
    row["hum_roll7d"]  = float(humidity)
    row["hum_trend3d"] = 0.0
    row["hum_above70"] = 1 if humidity > 70 else 0
    row["hum70_3d"]    = 1 if humidity > 70 else 0

    # Temperature
    row["temp_lag1d"]  = float(temp)
    row["temp_lag2d"]  = float(temp)
    row["temp_lag3d"]  = float(temp)
    row["temp_roll7d"] = float(temp)
    row["temp_drop3d"] = 0.0

    # Seasonal — advance month if needed
    fmonth = ((month - 1 + (day_offset // 30)) % 12) + 1
    row["month"]          = fmonth
    row["month_sin"]      = float(np.sin(2 * np.pi * fmonth / 12))
    row["month_cos"]      = float(np.cos(2 * np.pi * fmonth / 12))
    row["is_monsoon"]     = 1 if 6 <= fmonth <= 9 else 0
    row["is_pre_monsoon"] = 1 if fmonth in (4, 5) else 0
    doy = min(365, 30 * (fmonth - 1) + 15 + day_offset)
    row["doy_sin"] = float(np.sin(2 * np.pi * doy / 365))
    row["doy_cos"] = float(np.cos(2 * np.pi * doy / 365))

    # Interaction features
    row["hum_x_monsoon"]  = row["hum_lag1d"]    * row["is_monsoon"]
    row["soil_x_monsoon"] = row["soil_moisture"] * row["is_monsoon"]

    return row


def _predict_model1(models, feature_cols, use_log, feat_row):
    """Run Model 1 P10/P50/P90 on one feature row. Returns (p10, p50, p90)."""
    X         = pd.DataFrame([feat_row])
    available = [c for c in feature_cols if c in X.columns]
    if use_log:
        p10 = max(0.0, float(np.expm1(models[0.1].predict(X[available])[0])))
        p50 = max(0.0, float(np.expm1(models[0.5].predict(X[available])[0])))
        p90 = max(0.0, float(np.expm1(models[0.9].predict(X[available])[0])))
    else:
        p10 = max(0.0, float(models[0.1].predict(X[available])[0]))
        p50 = max(0.0, float(models[0.5].predict(X[available])[0]))
        p90 = max(0.0, float(models[0.9].predict(X[available])[0]))
    return p10, p50, p90


def _math_bands(api_rain, confidence):
    """Simple uncertainty bands when Model 1 not available."""
    if api_rain < 1.0:
        return 0.0, api_rain, max(0.2, api_rain * 1.5 + (1 - confidence) * 3)
    uf  = 1 + (1 - confidence) * 1.5
    p10 = max(0.0, api_rain * (1 - (1 - confidence) * 0.8))
    p90 = api_rain * uf
    return p10, api_rain, p90


# ── Main forecast builder ──────────────────────────────────────────────────────

def build_forecast_dataframe(
    rain_mm_list,
    rain_prob_list,
    current_temp=28.0,
    current_humidity=55.0,
    current_month=None,
    recent_rain_history=None,
):
    """
    Build 15-row forecast DataFrame: [day, p10_mm, p50_mm, p90_mm, confidence, rain_prob]

    Blending strategy:
      Day 1–3:  90% API weight + 10% Model 1  (API very reliable)
      Day 4–7:  70% API weight + 30% Model 1
      Day 8–11: 40% API weight + 60% Model 1  (API skill decaying)
      Day 12–15:15% API weight + 85% Model 1  (local patterns dominate)

    The API P50 is always used as the central estimate (p50_mm = api value).
    Model 1 shapes the P10/P90 uncertainty bands around that central estimate,
    biased toward its local knowledge of how extreme events behave at Bisalpur.
    """
    if current_month is None:
        current_month = datetime.today().month
    if recent_rain_history is None:
        recent_rain_history = [0.0] * 30

    models, feature_cols, use_log = _load_model1()
    model_available = models is not None
    if model_available:
        print("  Model 1 loaded — applying local calibration")
    else:
        print("  Model 1 not found — using math bands (train models first)")

    rain_history = list(recent_rain_history[-30:])
    rows = []

    for i in range(15):
        api_rain = float(rain_mm_list[i])
        api_prob = float(rain_prob_list[i])

        # Confidence curve (known property of NWP forecast skill)
        if   i < 3:  confidence = 0.92
        elif i < 7:  confidence = 0.82 - (i - 3) * 0.02
        elif i < 10: confidence = 0.72 - (i - 7) * 0.03
        else:        confidence = max(0.40, 0.63 - (i - 10) * 0.04)

        # API blend weight (how much we trust the API vs local model)
        if   i < 3:  api_w = 0.90
        elif i < 7:  api_w = 0.70
        elif i < 11: api_w = 0.40
        else:        api_w = 0.15
        m1_w = 1.0 - api_w

        if model_available:
            try:
                feat_row = _build_feature_row(
                    rain_history, current_temp, current_humidity, current_month, i + 1
                )
                m1_p10, m1_p50, m1_p90 = _predict_model1(
                    models, feature_cols, use_log, feat_row
                )
                # P50: API is authoritative (it has today's weather data)
                p50 = api_rain

                # P10/P90: blend API math bands with Model 1 output
                api_p10, _, api_p90 = _math_bands(api_rain, confidence)
                p10 = api_w * api_p10 + m1_w * m1_p10
                p90 = api_w * api_p90 + m1_w * m1_p90

                # Sanity: p10 <= p50 <= p90
                p10 = min(p10, p50)
                p90 = max(p90, p50)

            except Exception as e:
                print(f"  Day {i+1} model prediction failed: {e} — using math bands")
                p10, p50, p90 = _math_bands(api_rain, confidence)
        else:
            p10, p50, p90 = _math_bands(api_rain, confidence)

        rows.append({
            "day":        i + 1,
            "p10_mm":     round(max(0.0, p10), 1),
            "p50_mm":     round(max(0.0, p50), 1),
            "p90_mm":     round(max(0.0, p90), 1),
            "confidence": round(confidence, 2),
            "rain_prob":  round(api_prob, 2),
        })

        # Advance history: use p50 as the assumed "realised" value for next day's features
        rain_history.append(rows[-1]["p50_mm"])

    return pd.DataFrame(rows)


# ── Dashboard entry point ──────────────────────────────────────────────────────

def get_full_weather_data(lat=BISALPUR_LAT, lon=BISALPUR_LON):
    """
    Called once by the dashboard cache.
    Returns: (temp_c, humidity_pct, forecast_df)
    """
    print("Fetching weather from Open-Meteo...")
    temp, humidity = fetch_current_weather(lat, lon)
    if temp is None:
        temp, humidity = 28.0, 55.0

    rain_mm, rain_prob = fetch_15day_forecast(lat, lon)

    forecast_df = build_forecast_dataframe(
        rain_mm_list=rain_mm,
        rain_prob_list=rain_prob,
        current_temp=float(temp),
        current_humidity=float(humidity),
        current_month=datetime.today().month,
    )

    total_rain = float(forecast_df["p50_mm"].sum())
    rain_days  = int((forecast_df["p50_mm"] > 2).sum())
    print(f"  Temp: {temp:.1f}°C  Humidity: {humidity:.0f}%")
    print(f"  15-day P50 total: {total_rain:.0f}mm  Rain days: {rain_days}")

    return float(temp), float(humidity), forecast_df
