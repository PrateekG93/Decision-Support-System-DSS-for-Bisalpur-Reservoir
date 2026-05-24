"""
STEP 1 — Load, Merge, and Clean All Data
==========================================
Run this FIRST.

Reads your 3 CSV files, fixes units, merges by date, cleans missing values.
Saves: data/merged_clean.csv

Usage:
    python step1_prepare_data.py
"""

import pandas as pd
import numpy as np
import os

RAW_DIR    = "data/raw"
OUTPUT     = "data/merged_clean.csv"

os.makedirs("data", exist_ok=True)


def nasa_skip(path):
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip().startswith("-END HEADER-"):
                return i + 1
    raise ValueError(f"No -END HEADER- in {path}")


def load_rainfall():
    path = os.path.join(RAW_DIR, "Rainfall_Daily.csv")
    df = pd.read_csv(path, skiprows=nasa_skip(path))
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["DOY"].astype(str), format="%Y-%j"
    )
    df = df.rename(columns={"PRECTOTCORR": "rainfall_mm"})[["date", "rainfall_mm"]]
    df["rainfall_mm"] = df["rainfall_mm"].replace(-999, np.nan).clip(lower=0)
    return df


def load_temp_humidity():
    path = os.path.join(RAW_DIR, "Temp_and_Humidity_Daily.csv")
    df = pd.read_csv(path, skiprows=nasa_skip(path))
    df["date"] = pd.to_datetime(
        df[["YEAR","MO","DY"]].rename(columns={"YEAR":"year","MO":"month","DY":"day"})
    )
    # NASA T2M is Fahrenheit — convert to Celsius
    df["temp_c"] = (df["T2M"] - 32) * 5 / 9
    df = df.rename(columns={"RH2M": "humidity_pct"})[["date", "temp_c", "humidity_pct"]]
    return df


def load_storage():
    path = os.path.join(RAW_DIR, "Yearwise_Storage_data.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    capacity = df["Live_capacity_FRL"].iloc[0]  # 1.076 BCM
    df["storage_pct"] = (df["Storage"] / capacity * 100).clip(0, 100)
    df["storage_mcm"] = df["Storage"] * 1000    # BCM → MCM
    df["delta_storage_mcm"] = df["storage_mcm"].diff()
    print(f"  Reservoir: {df['Reservoir_name'].iloc[0]}")
    print(f"  Capacity:  {capacity} BCM = {capacity*1000:.0f} MCM")
    return df[["date", "storage_pct", "storage_mcm", "delta_storage_mcm"]]


def merge_and_clean(rain, th, st):
    df = rain.merge(th, on="date").merge(st, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Before cleaning: {len(df)} rows")

    # Interpolate short gaps (≤7 consecutive days)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].interpolate(method="linear", limit=7)

    df = df.dropna().reset_index(drop=True)
    print(f"  After cleaning:  {len(df)} rows")
    print(f"  Date range:      {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def main():
    print("=" * 55)
    print("STEP 1: DATA PREPARATION")
    print("=" * 55)

    print("\nLoading rainfall...")
    rain = load_rainfall()
    print(f"  {len(rain)} rows | {rain['date'].min().date()} → {rain['date'].max().date()}")

    print("\nLoading temperature + humidity...")
    th = load_temp_humidity()
    print(f"  Temp range: {th['temp_c'].min():.1f}°C → {th['temp_c'].max():.1f}°C")

    print("\nLoading reservoir storage...")
    st = load_storage()
    print(f"  Storage range: {st['storage_pct'].min():.1f}% → {st['storage_pct'].max():.1f}%")

    print("\nMerging and cleaning...")
    df = merge_and_clean(rain, th, st)

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")
    print("Next: python step2_train_models.py")


if __name__ == "__main__":
    main()
