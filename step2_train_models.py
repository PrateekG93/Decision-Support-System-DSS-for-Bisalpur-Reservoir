"""
STEP 2 — Train Both Models
============================
Run AFTER step1_prepare_data.py.

MODEL 1 — Rainfall Bias Corrector
  New role: learns how much the Open-Meteo global forecast
  over/under-predicts for Bisalpur specifically.
  Input:  Open-Meteo forecast rain + season features
  Output: Calibrated P10/P50/P90 for this location
  Why:    Global weather models miss local terrain effects.
          15 years of local data corrects for this.

MODEL 2 — Storage Predictor
  Unchanged: predicts 7-day-ahead storage from current
  storage + recent rainfall + season features.
  Produces P10/P50/P90 for risk scoring.

Usage:
    python step2_train_models.py
"""

import pandas as pd
import numpy as np
import os, json, joblib, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False
    print("Install: pip install lightgbm")

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

DATA_FILE = "data/merged_clean.csv"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: RAINFALL BIAS CORRECTOR
# ─────────────────────────────────────────────────────────────────────────────

def build_rainfall_features(df):
    """
    Features for Model 1 (bias corrector).
    The key insight: we train on historical patterns so the model learns
    'given conditions like this, how much does the Bisalpur location
    typically deviate from a global forecast?'
    The actual Open-Meteo forecast is fed in at prediction time.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Past rainfall (autocorrelation — monsoon systems persist)
    for lag in [1, 2, 3, 5, 7, 10, 14]:
        df[f"rain_lag{lag}d"] = df["rainfall_mm"].shift(lag)
    for w in [3, 7, 14, 30]:
        df[f"rain_roll{w}d"]  = df["rainfall_mm"].shift(1).rolling(w, min_periods=max(2,w//2)).mean()
        df[f"rain_sum{w}d"]   = df["rainfall_mm"].shift(1).rolling(w, min_periods=max(2,w//2)).sum()
        df[f"rain_max{w}d"]   = df["rainfall_mm"].shift(1).rolling(w, min_periods=max(2,w//2)).max()

    # Wet day streak (active monsoon signal)
    df["is_wet"]         = (df["rainfall_mm"] > 1).astype(int)
    df["is_heavy"]       = (df["rainfall_mm"] > 10).astype(int)
    df["wet_streak_7"]   = df["is_wet"].shift(1).rolling(7, min_periods=3).sum()
    df["heavy_streak_7"] = df["is_heavy"].shift(1).rolling(7, min_periods=3).sum()

    # Soil moisture proxy
    df["soil_moisture"] = 0.5*df["rain_sum7d"] + 0.3*df["rain_sum14d"] + 0.2*df["rain_sum30d"]

    # Humidity (direct rain precursor)
    for lag in [1, 2, 3, 5]:
        df[f"hum_lag{lag}d"] = df["humidity_pct"].shift(lag)
    df["hum_roll7d"]   = df["humidity_pct"].shift(1).rolling(7, min_periods=3).mean()
    df["hum_trend3d"]  = df["humidity_pct"].shift(1) - df["humidity_pct"].shift(4)
    df["hum_above70"]  = (df["humidity_pct"] > 70).astype(int)
    df["hum70_3d"]     = df["hum_above70"].shift(1).rolling(3, min_periods=2).sum()

    # Temperature
    for lag in [1, 2, 3]:
        df[f"temp_lag{lag}d"] = df["temp_c"].shift(lag)
    df["temp_roll7d"]  = df["temp_c"].shift(1).rolling(7, min_periods=3).mean()
    df["temp_drop3d"]  = df["temp_c"].shift(4) - df["temp_c"].shift(1)  # cooling → rain

    # Seasonal
    df["month"]          = df["date"].dt.month
    df["month_sin"]      = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]      = np.cos(2*np.pi*df["month"]/12)
    df["is_monsoon"]     = df["month"].between(6, 9).astype(int)
    df["is_pre_monsoon"] = df["month"].between(4, 5).astype(int)
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365)
    df["doy_cos"] = np.cos(2*np.pi*doy/365)

    # Interactions
    df["hum_x_monsoon"]   = df["hum_lag1d"] * df["is_monsoon"]
    df["soil_x_monsoon"]  = df["soil_moisture"] * df["is_monsoon"]

    # Targets (log1p for better extreme event prediction)
    df["target_rain_raw"]   = df["rainfall_mm"].shift(-1)
    df["target_rain_log1p"] = np.log1p(df["target_rain_raw"])

    return df.dropna().reset_index(drop=True)


RAIN_FEATURES = [
    "rain_lag1d","rain_lag2d","rain_lag3d","rain_lag5d","rain_lag7d","rain_lag10d","rain_lag14d",
    "rain_roll3d","rain_roll7d","rain_roll14d","rain_roll30d",
    "rain_sum3d","rain_sum7d","rain_sum14d","rain_sum30d",
    "rain_max7d","rain_max14d","rain_max30d",
    "is_wet","is_heavy","wet_streak_7","heavy_streak_7","soil_moisture",
    "hum_lag1d","hum_lag2d","hum_lag3d","hum_lag5d",
    "hum_roll7d","hum_trend3d","hum_above70","hum70_3d",
    "temp_lag1d","temp_lag2d","temp_lag3d","temp_roll7d","temp_drop3d",
    "hum_x_monsoon","soil_x_monsoon",
    "month","month_sin","month_cos","is_monsoon","is_pre_monsoon","doy_sin","doy_cos",
]


def train_rainfall_model(df):
    print("\n" + "="*55)
    print("TRAINING MODEL 1 — Rainfall Bias Corrector")
    print("="*55)

    feature_cols = [c for c in RAIN_FEATURES if c in df.columns]
    cutoff   = df["date"].max() - pd.DateOffset(years=3)
    train_df = df[df["date"] <= cutoff]
    test_df  = df[df["date"] > cutoff]

    X_train = train_df[feature_cols]; y_train = train_df["target_rain_log1p"]
    X_test  = test_df[feature_cols];  y_test_raw = test_df["target_rain_raw"].values

    val_cut = int(len(X_train)*0.85)
    X_tr, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_tr, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

    print(f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} ({len(train_df)} rows)")
    print(f"Test:  {test_df['date'].min().date()} → {test_df['date'].max().date()} ({len(test_df)} rows)")

    models    = {}
    all_preds = {}
    for q, qname in [(0.1,"P10"),(0.5,"P50"),(0.9,"P90")]:
        print(f"\n  Training {qname}...")
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=1000, learning_rate=0.04,
            max_depth=6, num_leaves=50,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=15, reg_alpha=0.05, reg_lambda=0.1,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(400)])
        preds_raw = np.maximum(np.expm1(m.predict(X_test)), 0)
        all_preds[q] = preds_raw
        mae = np.mean(np.abs(y_test_raw - preds_raw))
        r2  = 1 - np.sum((y_test_raw-preds_raw)**2) / np.sum((y_test_raw-y_test_raw.mean())**2)
        print(f"    MAE: {mae:.2f} mm/day  |  R²: {r2:.4f}  |  Iters: {m.best_iteration_}")
        models[q] = m

    # Accuracy report
    actual  = y_test_raw
    p10, p50, p90 = all_preds[0.1], all_preds[0.5], all_preds[0.9]
    coverage     = np.mean((actual>=p10) & (actual<=p90)) * 100
    wet_mask     = actual > 1
    heavy_mask   = actual > 20
    wet_det      = np.mean((p50>1)[wet_mask])  * 100 if wet_mask.sum()  > 0 else 0
    dry_spec     = np.mean((p50<=1)[~wet_mask]) * 100 if (~wet_mask).sum() > 0 else 0
    heavy_det    = np.mean((p50>1)[heavy_mask]) * 100 if heavy_mask.sum() > 0 else 0
    mon_mask     = test_df["month"].between(6,9).values
    mae_mon      = np.mean(np.abs(actual[mon_mask]-p50[mon_mask])) if mon_mask.sum() > 0 else 0

    print("\n── Model 1 Accuracy Report ──")
    print(f"  MAE (overall):          {np.mean(np.abs(actual-p50)):.2f} mm/day")
    print(f"  R²:                     {r2:.4f}")
    print(f"  P10–P90 coverage:       {coverage:.1f}%  (target ~80%)")
    print(f"  Heavy rain detection:   {heavy_det:.1f}%  (>20mm days)")
    print(f"  Dry-day specificity:    {dry_spec:.1f}%")
    print(f"  Monsoon MAE (Jun–Sep):  {mae_mon:.2f} mm/day")

    accuracy = {
        "mae_overall": round(float(np.mean(np.abs(actual-p50))),3),
        "r2": round(float(r2),4),
        "coverage_pct": round(float(coverage),1),
        "wet_day_detection_pct": round(float(wet_det),1),
        "dry_day_specificity_pct": round(float(dry_spec),1),
        "heavy_rain_detection_pct": round(float(heavy_det),1),
        "mae_monsoon": round(float(mae_mon),3),
    }

    _plot_rainfall_eval(all_preds, test_df, actual)
    joblib.dump({"models":models,"feature_cols":feature_cols,"log_target":True}, "models/rainfall_model.pkl")
    with open("models/rainfall_meta.json","w") as f:
        json.dump({"feature_cols":feature_cols,"accuracy":accuracy}, f, indent=2)
    print("  Saved: models/rainfall_model.pkl")
    return models, feature_cols


def _plot_rainfall_eval(all_preds, test_df, actual):
    p10, p50, p90 = all_preds[0.1], all_preds[0.5], all_preds[0.9]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mon_mask = test_df["month"].between(6,9).values
    ax = axes[0]
    ax.fill_between(range(mon_mask.sum()), p10[mon_mask], p90[mon_mask], alpha=0.3, color="#1D9E75", label="P10–P90")
    ax.plot(p50[mon_mask], color="#1D9E75", lw=1.5, label="P50 predicted")
    ax.bar(range(mon_mask.sum()), actual[mon_mask], color="#534AB7", alpha=0.5, width=0.9, label="Actual")
    cov = np.mean((actual[mon_mask]>=p10[mon_mask]) & (actual[mon_mask]<=p90[mon_mask])) * 100
    ax.set_title(f"Rainfall — monsoon months (test set)\nBand coverage: {cov:.0f}%")
    ax.set_ylabel("mm/day"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.scatter(actual, p50, alpha=0.2, s=5, color="#378ADD")
    lim = max(float(actual.max()), float(p50.max())) * 1.05
    ax2.plot([0,lim],[0,lim],"r--",lw=1.2)
    mae = np.mean(np.abs(actual-p50))
    r2  = 1 - np.sum((actual-p50)**2)/np.sum((actual-actual.mean())**2)
    ax2.set_title(f"Actual vs Predicted\nMAE={mae:.2f} mm   R²={r2:.3f}")
    ax2.set_xlabel("Actual (mm/day)"); ax2.set_ylabel("Predicted P50 (mm/day)")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig("plots/rainfall_model_eval.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: plots/rainfall_model_eval.png")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: STORAGE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

def build_storage_features(df):
    df = df.copy().sort_values("date").reset_index(drop=True)

    for lag in [1,2,3,5,7,14,30]:
        df[f"storage_lag{lag}d"] = df["storage_pct"].shift(lag)
    df["storage_trend7d"]  = df["storage_pct"].shift(1) - df["storage_pct"].shift(8)
    df["storage_trend14d"] = df["storage_pct"].shift(1) - df["storage_pct"].shift(15)
    df["storage_roll7d"]   = df["storage_pct"].shift(1).rolling(7, min_periods=4).mean()
    df["storage_roll30d"]  = df["storage_pct"].shift(1).rolling(30, min_periods=15).mean()

    for lag in [1,2,3,5,7,10,14]:
        df[f"rain_lag{lag}d"] = df["rainfall_mm"].shift(lag)
    for w in [3,7,14,21,30]:
        df[f"rain_sum{w}d"] = df["rainfall_mm"].shift(1).rolling(w, min_periods=max(2,w//2)).sum()
        df[f"rain_max{w}d"] = df["rainfall_mm"].shift(1).rolling(w, min_periods=max(2,w//2)).max()

    df["temp_lag1d"] = df["temp_c"].shift(1)
    df["hum_lag1d"]  = df["humidity_pct"].shift(1)
    df["hum_roll7d"] = df["humidity_pct"].shift(1).rolling(7, min_periods=3).mean()

    df["month"]      = df["date"].dt.month
    df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
    df["is_monsoon"] = df["month"].between(6,9).astype(int)
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365)
    df["doy_cos"] = np.cos(2*np.pi*doy/365)

    df["target_storage_7d"] = df["storage_pct"].shift(-7)

    return df.dropna().reset_index(drop=True)


STORAGE_FEATURES = [
    "storage_lag1d","storage_lag2d","storage_lag3d","storage_lag5d",
    "storage_lag7d","storage_lag14d","storage_lag30d",
    "storage_trend7d","storage_trend14d","storage_roll7d","storage_roll30d",
    "rain_lag1d","rain_lag2d","rain_lag3d","rain_lag5d",
    "rain_lag7d","rain_lag10d","rain_lag14d",
    "rain_sum3d","rain_sum7d","rain_sum14d","rain_sum21d","rain_sum30d",
    "rain_max7d","rain_max14d",
    "temp_lag1d","hum_lag1d","hum_roll7d",
    "month","month_sin","month_cos","is_monsoon","doy_sin","doy_cos",
]


def train_storage_model(df):
    print("\n" + "="*55)
    print("TRAINING MODEL 2 — Storage Predictor")
    print("="*55)

    feature_cols = [c for c in STORAGE_FEATURES if c in df.columns]
    cutoff   = df["date"].max() - pd.DateOffset(years=3)
    train_df = df[df["date"] <= cutoff]
    test_df  = df[df["date"] > cutoff]

    X_train = train_df[feature_cols]; y_train = train_df["target_storage_7d"]
    X_test  = test_df[feature_cols];  y_test  = test_df["target_storage_7d"]

    val_cut = int(len(X_train)*0.85)
    X_tr, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_tr, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

    print(f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} ({len(train_df)} rows)")
    print(f"Test:  {test_df['date'].min().date()} → {test_df['date'].max().date()} ({len(test_df)} rows)")

    models    = {}
    all_preds = {}
    for q, qname in [(0.1,"P10"),(0.5,"P50"),(0.9,"P90")]:
        print(f"\n  Training {qname}...")
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=1000, learning_rate=0.04,
            max_depth=6, num_leaves=40,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(400)])
        preds = np.clip(m.predict(X_test), 0, 100)
        all_preds[q] = preds
        mae = float(np.mean(np.abs(y_test-preds)))
        r2  = float(1-np.sum((y_test-preds)**2)/np.sum((y_test-y_test.mean())**2))
        print(f"    MAE: {mae:.2f}%  |  R²: {r2:.4f}  |  Iters: {m.best_iteration_}")
        models[q] = m

    actual = y_test.values
    p10, p50, p90 = all_preds[0.1], all_preds[0.5], all_preds[0.9]
    coverage = float(np.mean((actual>=p10) & (actual<=p90)) * 100)
    mon_mask = test_df["month"].between(6,9).values
    mae_mon  = float(np.mean(np.abs(actual[mon_mask]-p50[mon_mask]))) if mon_mask.sum()>0 else 0

    true_flood   = actual > 85
    pred_flood   = p90 > 85
    flood_recall = float(np.mean(pred_flood[true_flood])*100) if true_flood.sum()>0 else 0
    true_drought = actual < 25
    pred_drought = p10 < 25
    drought_recall = float(np.mean(pred_drought[true_drought])*100) if true_drought.sum()>0 else 0

    overall_mae = float(np.mean(np.abs(actual-p50)))
    overall_r2  = float(1-np.sum((actual-p50)**2)/np.sum((actual-actual.mean())**2))

    print("\n── Model 2 Accuracy Report ──")
    print(f"  MAE (7-day ahead):      {overall_mae:.2f}%")
    print(f"  R²:                     {overall_r2:.4f}")
    print(f"  P10–P90 coverage:       {coverage:.1f}%  (target ~80%)")
    print(f"  Flood zone recall:      {flood_recall:.1f}%  (P90 catches storage>85%)")
    print(f"  Drought zone recall:    {drought_recall:.1f}%  (P10 catches storage<25%)")
    print(f"  Monsoon MAE (Jun–Sep):  {mae_mon:.2f}%")

    accuracy = {
        "mae_overall": round(overall_mae,3),
        "r2": round(overall_r2,4),
        "coverage_pct": round(coverage,1),
        "mae_monsoon": round(mae_mon,3),
        "flood_recall_pct": round(flood_recall,1),
        "drought_recall_pct": round(drought_recall,1),
    }

    _plot_storage_eval(all_preds, test_df, actual, feature_cols, models)
    joblib.dump({"models":models,"feature_cols":feature_cols}, "models/storage_model.pkl")
    with open("models/storage_meta.json","w") as f:
        json.dump({"feature_cols":feature_cols,"accuracy":accuracy}, f, indent=2)
    print("  Saved: models/storage_model.pkl")
    return models, feature_cols


def _plot_storage_eval(all_preds, test_df, actual, feature_cols, models):
    p10, p50, p90 = all_preds[0.1], all_preds[0.5], all_preds[0.9]
    dates = pd.to_datetime(test_df["date"].values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.fill_between(dates, p10, p90, alpha=0.25, color="#1D9E75", label="P10–P90")
    ax.plot(dates, p50, color="#1D9E75", lw=1.5, label="P50 predicted")
    ax.plot(dates, actual, color="#D85A30", lw=0.8, alpha=0.8, label="Actual")
    ax.axhline(85, color="#E24B4A", ls="--", lw=1, alpha=0.7)
    ax.axhline(25, color="#BA7517", ls="--", lw=1, alpha=0.7)
    cov = np.mean((actual>=p10) & (actual<=p90)) * 100
    ax.set_title(f"Storage prediction — test set\nBand coverage: {cov:.0f}%")
    ax.set_ylabel("Storage (%)"); ax.set_ylim(0, 108)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.scatter(actual, p50, alpha=0.2, s=5, color="#378ADD")
    ax2.plot([0,100],[0,100],"r--",lw=1.2)
    mae = np.mean(np.abs(actual-p50))
    r2  = 1 - np.sum((actual-p50)**2)/np.sum((actual-actual.mean())**2)
    ax2.set_title(f"Actual vs Predicted\nMAE={mae:.2f}%   R²={r2:.4f}")
    ax2.set_xlabel("Actual storage (%)"); ax2.set_ylabel("Predicted P50 (%)")
    ax2.set_xlim(0,102); ax2.set_ylim(0,102); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig("plots/storage_model_eval.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: plots/storage_model_eval.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not LGBM_OK:
        print("Install lightgbm: pip install lightgbm")
        exit(1)

    df_raw = pd.read_csv(DATA_FILE, parse_dates=["date"])
    print(f"Loaded {len(df_raw)} rows from {DATA_FILE}")

    df_rain = build_rainfall_features(df_raw)
    train_rainfall_model(df_rain)

    df_stor = build_storage_features(df_raw)
    train_storage_model(df_stor)

    print("\n" + "="*55)
    print("BOTH MODELS TRAINED SUCCESSFULLY")
    print("="*55)
    print("Next: streamlit run dashboard.py")
