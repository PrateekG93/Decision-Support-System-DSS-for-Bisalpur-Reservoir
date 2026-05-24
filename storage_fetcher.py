"""
storage_fetcher.py — Current Reservoir Storage
================================================
Gets the most current storage % for Bisalpur by trying 3 sources
in order of preference:

  Source 1 — India WRIS web scrape (live, if available)
    Bisalpur page: https://indiawris.gov.in/wris/#/Reservoirs
    Tries to fetch current level from the WRIS API endpoint.
    Returns today's value if the site is up.

  Source 2 — Local CSV fallback (always available)
    Reads the most recent non-null row from
    data/raw/Yearwise_Storage_data.csv.
    This file was last updated when you downloaded it.
    Returns the latest date + storage %.

  Source 3 — Manual entry (always works)
    If both above fail or the CSV is old, the operator
    enters the value directly in the dashboard sidebar.

Usage:
    from storage_fetcher import get_current_storage
    storage_pct, source, date_str = get_current_storage()
"""

import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

CSV_PATH    = "data/raw/Yearwise_Storage_data.csv"
CAPACITY_BCM= 1.076   # Bisalpur full capacity


def _try_wris_api():
    """
    Attempt to fetch current Bisalpur storage from India WRIS.

    WRIS exposes a data API used by their own web app. We try the
    known endpoint pattern. Returns (storage_pct, date_str) or (None, None).

    Note: This API is unofficial (not publicly documented). It may
    change or become unavailable. The CSV fallback always works.
    """
    try:
        # WRIS uses a REST API for their dashboard — reservoir ID for Bisalpur
        # The endpoint returns JSON with current storage levels
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (research project)",
            "Referer": "https://indiawris.gov.in/wris/",
        }
        # Primary endpoint: reservoir daily data
        url = (
            "https://indiawris.gov.in/wris/api/reservoirs/daily"
            "?reservoirName=Bisalpur&days=1"
        )
        r = requests.get(url, headers=headers, timeout=6)
        r.raise_for_status()
        data = r.json()

        # Parse response — structure varies by WRIS version
        if isinstance(data, list) and len(data) > 0:
            row = data[0]
            storage_bcm = float(row.get("storage", row.get("liveStorage", 0)))
            storage_pct = round(storage_bcm / CAPACITY_BCM * 100, 2)
            date_str    = str(row.get("date", datetime.today().strftime("%Y-%m-%d")))
            if 0 < storage_pct <= 100:
                return storage_pct, date_str

        elif isinstance(data, dict):
            storage_bcm = float(data.get("storage", data.get("currentStorage", 0)))
            storage_pct = round(storage_bcm / CAPACITY_BCM * 100, 2)
            date_str    = str(data.get("date", datetime.today().strftime("%Y-%m-%d")))
            if 0 < storage_pct <= 100:
                return storage_pct, date_str

    except Exception:
        pass  # Silently fall through to CSV

    # Try alternate WRIS endpoint
    try:
        url2 = (
            "https://indiawris.gov.in/wris/api/reservoir"
            "?name=Bisalpur&format=json"
        )
        r2 = requests.get(url2, headers=headers, timeout=6)
        r2.raise_for_status()
        data2 = r2.json()
        if data2:
            entry = data2[0] if isinstance(data2, list) else data2
            bcm  = float(entry.get("storage", entry.get("live_storage", 0)))
            pct  = round(bcm / CAPACITY_BCM * 100, 2)
            dt   = str(entry.get("date", ""))
            if 0 < pct <= 100:
                return pct, dt
    except Exception:
        pass

    return None, None


def _from_csv():
    """
    Read the most recent non-null storage value from the local CSV.
    Returns (storage_pct, date_str) or (None, None).
    """
    if not os.path.exists(CSV_PATH):
        return None, None
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=["Storage"])
        if df.empty:
            return None, None

        latest      = df.iloc[-1]
        storage_bcm = float(latest["Storage"])
        capacity    = float(latest.get("Live_capacity_FRL", CAPACITY_BCM))
        storage_pct = round(storage_bcm / capacity * 100, 2)
        date_str    = str(latest["Date"].date())

        # Check how old the data is
        latest_date = latest["Date"]
        days_old    = (datetime.today() - latest_date).days

        return storage_pct, date_str, days_old
    except Exception:
        return None, None, None


def get_current_storage():
    """
    Main function — returns best available storage reading.

    Returns:
        storage_pct  : float — storage as % of full capacity
        source       : str   — where the value came from
        date_str     : str   — date of the reading
        days_old     : int   — how many days old the reading is
        warning      : str   — any warning message to show the user
    """
    warning = ""

    # ── Source 1: Try WRIS live API ───────────────────────────────────────────
    wris_pct, wris_date = _try_wris_api()
    if wris_pct is not None:
        return wris_pct, "India WRIS (live)", wris_date, 0, ""

    # ── Source 2: Local CSV ───────────────────────────────────────────────────
    result = _from_csv()
    if result and result[0] is not None:
        csv_pct, csv_date, days_old = result
        days_old = days_old or 0

        if days_old == 0:
            source  = "Local CSV (today)"
            warning = ""
        elif days_old <= 3:
            source  = f"Local CSV ({csv_date})"
            warning = ""
        elif days_old <= 30:
            source  = f"Local CSV ({csv_date})"
            warning = (
                f"Storage reading is {days_old} days old ({csv_date}). "
                "Download fresh data from India WRIS or enter manually."
            )
        else:
            source  = f"Local CSV ({csv_date})"
            warning = (
                f"Storage data is {days_old} days old. "
                "Update data/raw/Yearwise_Storage_data.csv from India WRIS."
            )

        return csv_pct, source, csv_date, days_old, warning

    # ── Source 3: No data available ───────────────────────────────────────────
    return (
        65.0,   # sensible fallback
        "Default (no data)",
        "unknown",
        9999,
        "Could not read storage data. Enter current value manually using the slider.",
    )


def get_storage_summary():
    """
    Returns a dict with all info for the dashboard to display.
    Suitable for st.session_state storage.
    """
    pct, source, date_str, days_old, warning = get_current_storage()
    return {
        "storage_pct": pct,
        "source":      source,
        "date":        date_str,
        "days_old":    days_old,
        "warning":     warning,
        "auto_loaded": (source != "Default (no data)"),
    }
