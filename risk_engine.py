"""
risk_engine.py — Release Strategy + Risk Scoring Engine
=========================================================
Rules are AUTHORITY. ML is ADVISORY.

Water balance: S(t+1) = S(t) + Inflow(t) - Release(t) - Evaporation(t)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

# ── Bisalpur constants ────────────────────────────────────────────────────────
CAPACITY_MCM      = 1076.0
MIN_STORAGE_PCT   = 25.0
MAX_GATE_PCT_DAY  = 12.0
FLOOD_THRESHOLD   = 85.0
DROUGHT_THRESHOLD = 25.0
CATCHMENT_KM2     = 4800
RUNOFF_COEFF      = 0.25


@dataclass
class Approach:
    id:                  int
    name:                str
    description:         str
    recommended_when:    str
    release_pct_per_day: float
    release_mcm_per_day: float
    active_days:         int
    total_release_pct:   float
    schedule_pct:        List[float]
    trajectory_p50:      List[float]
    trajectory_p10:      List[float]
    trajectory_p90:      List[float]
    final_storage_p50:   float
    final_storage_p10:   float
    final_storage_p90:   float
    flood_risk:          str
    drought_risk:        str
    flood_score:         float
    drought_score:       float
    composite_score:     float
    rank:                str
    feasible:            bool = True
    infeasible_reason:   str  = ""


def _evap_mcm_day(storage_pct, month):
    area_km2 = 85 * (max(storage_pct, 5) / 100) ** 0.67
    rate_mm  = {1:2.0,2:2.5,3:3.5,4:5.0,5:6.0,6:5.0,
                7:3.5,8:3.0,9:3.5,10:3.0,11:2.5,12:2.0}.get(int(month), 3.0)
    return rate_mm / 1000 * area_km2


def _simulate(current_storage_pct, rain_forecast_df, release_schedule, scenario, month):
    rain_col = {"p10":"p10_mm","p50":"p50_mm","p90":"p90_mm"}[scenario]
    storage_mcm = current_storage_pct / 100 * CAPACITY_MCM
    trajectory   = [round(current_storage_pct, 2)]
    flood_days   = 0
    drought_days = 0

    for i in range(min(15, len(rain_forecast_df))):
        rain_mm     = max(0.0, float(rain_forecast_df.iloc[i][rain_col]))
        release_mcm = np.clip(
            release_schedule[i] / 100 * CAPACITY_MCM, 0,
            MAX_GATE_PCT_DAY / 100 * CAPACITY_MCM
        )
        inflow_mcm = rain_mm * CATCHMENT_KM2 * RUNOFF_COEFF / 1000
        evap_mcm   = _evap_mcm_day(storage_mcm / CAPACITY_MCM * 100, month)
        storage_mcm= np.clip(storage_mcm + inflow_mcm - release_mcm - evap_mcm, 0, CAPACITY_MCM)
        pct = storage_mcm / CAPACITY_MCM * 100
        trajectory.append(round(pct, 2))
        if pct > FLOOD_THRESHOLD:   flood_days   += 1
        if pct < DROUGHT_THRESHOLD: drought_days += 1

    return trajectory, flood_days, drought_days


def generate_approaches(current_storage_pct, rain_forecast_df, current_month):
    high_conf_days = max(1, int((rain_forecast_df["confidence"] > 0.70).sum()))
    total_p90      = float(rain_forecast_df["p90_mm"].sum())
    total_p50      = float(rain_forecast_df["p50_mm"].sum())

    definitions = [
        {
            "id": 1, "release_pct": 3.0, "active_days": high_conf_days,
            "name": "Conservative release",
            "description": (
                f"Release 3%/day for {high_conf_days} high-confidence forecast days, then stop. "
                "If actual rainfall is less than forecast, you haven't over-released."
            ),
            "recommended_when": "Storage < 75%  OR  forecast confidence low",
        },
        {
            "id": 2, "release_pct": 5.0, "active_days": 7,
            "name": "Moderate release",
            "description": (
                "Release 5%/day for 7 days. If rain stops at Day 4, only 20% total released — recoverable. "
                "Best balance between flood prevention and conservation."
            ),
            "recommended_when": "Storage 75–88%",
        },
        {
            "id": 3, "release_pct": 8.0, "active_days": 5,
            "name": "Aggressive release",
            "description": (
                "Release 8%/day for 5 days. Creates larger buffer quickly. "
                "Use when confidence in heavy rain is high."
            ),
            "recommended_when": "Storage > 88%  AND  high-confidence heavy rain",
        },
        {
            "id": 4, "release_pct": 12.0, "active_days": 4,
            "name": "Emergency protocol",
            "description": (
                "Release 12%/day. ONLY for critical situations. "
                "Issue downstream flood warning before activating."
            ),
            "recommended_when": "Storage > 95%  ONLY",
        },
    ]

    approaches = []
    for d in definitions:
        schedule = [d["release_pct"] if i < d["active_days"] else 0.0 for i in range(15)]
        traj_p50, f50, dr50 = _simulate(current_storage_pct, rain_forecast_df, schedule, "p50", current_month)
        traj_p10, f10, dr10 = _simulate(current_storage_pct, rain_forecast_df, schedule, "p10", current_month)
        traj_p90, f90, dr90 = _simulate(current_storage_pct, rain_forecast_df, schedule, "p90", current_month)

        flood_score   = f90  / 15
        drought_score = dr10 / 15
        composite     = 0.6 * flood_score + 0.4 * drought_score
        total_released= sum(schedule)
        final_p10     = traj_p10[-1]

        feasible, reason = True, ""
        if final_p10 < MIN_STORAGE_PCT:
            feasible = False
            reason   = (
                f"Worst-case (P10 rain): storage drops to {final_p10:.1f}% "
                f"— below {MIN_STORAGE_PCT}% drinking water minimum."
            )
        if d["release_pct"] > MAX_GATE_PCT_DAY:
            feasible = False
            reason   = f"Release {d['release_pct']}%/day exceeds structural gate limit ({MAX_GATE_PCT_DAY}%)."

        approaches.append(Approach(
            id=d["id"], name=d["name"], description=d["description"],
            recommended_when=d["recommended_when"],
            release_pct_per_day=d["release_pct"],
            release_mcm_per_day=round(d["release_pct"]/100*CAPACITY_MCM, 1),
            active_days=d["active_days"], total_release_pct=round(total_released, 1),
            schedule_pct=[round(s, 1) for s in schedule],
            trajectory_p50=traj_p50, trajectory_p10=traj_p10, trajectory_p90=traj_p90,
            final_storage_p50=round(traj_p50[-1],1), final_storage_p10=round(traj_p10[-1],1),
            final_storage_p90=round(traj_p90[-1],1),
            flood_risk  ="HIGH" if flood_score>=0.5 else ("MEDIUM" if flood_score>=0.2 else "LOW"),
            drought_risk="HIGH" if drought_score>=0.5 else ("MEDIUM" if drought_score>=0.2 else "LOW"),
            flood_score=round(flood_score,3), drought_score=round(drought_score,3),
            composite_score=round(composite,3), rank="UNRANKED",
            feasible=feasible, infeasible_reason=reason,
        ))

    feasible_sorted = sorted([a for a in approaches if a.feasible], key=lambda x: x.composite_score)
    for i, a in enumerate(feasible_sorted):
        a.rank = ("OPTIMAL" if i==0 else "SAFE_ALTERNATIVE" if i==1
                  else "ACCEPTABLE" if a.composite_score<0.5 else "HIGH_RISK")
    for a in approaches:
        if not a.feasible:
            a.rank = "INFEASIBLE"
        elif a.rank == "UNRANKED":
            a.rank = "HIGH_RISK"

    if   current_storage_pct > 95:                               base_risk = "CRITICAL"
    elif current_storage_pct > FLOOD_THRESHOLD and total_p90>50: base_risk = "FLOOD"
    elif current_storage_pct > 70 and (rain_forecast_df["p50_mm"]>2).sum()>8: base_risk = "ELEVATED"
    elif current_storage_pct < DROUGHT_THRESHOLD:                base_risk = "DROUGHT"
    else:                                                         base_risk = "NORMAL"

    return approaches, base_risk


def get_alerts(current_storage_pct, rain_forecast_df, approaches):
    """
    Alert system — TWO layers:
      Layer 1: Current-state threshold alerts (storage now vs fixed thresholds)
      Layer 2: Trajectory-based alerts (will storage cross a threshold in the next N days?)
    Both layers run independently and both appear on the dashboard.
    """
    total_p90  = float(rain_forecast_df["p90_mm"].sum())
    total_p50  = float(rain_forecast_df["p50_mm"].sum())
    rain_days  = int((rain_forecast_df["p50_mm"] > 2).sum())
    optimal    = next((a for a in approaches if a.rank == "OPTIMAL"), None)

    alerts       = []
    primary_risk = "NORMAL"

    # ── LAYER 1: Current-state threshold alerts ───────────────────────────────
    if current_storage_pct > 95:
        primary_risk = "CRITICAL FLOOD"
        alerts.append({"level": "CRITICAL",
            "msg": f"Storage at {current_storage_pct:.1f}% — near overflow. Emergency release immediately."})
    elif current_storage_pct > FLOOD_THRESHOLD and total_p90 > 40:
        primary_risk = "FLOOD"
        alerts.append({"level": "HIGH",
            "msg": f"Storage {current_storage_pct:.1f}% above flood threshold. "
                   f"{total_p90:.0f}mm P90 forecast over 15 days. Controlled release recommended."})
    elif current_storage_pct > 70 and rain_days > 8:
        primary_risk = "ELEVATED"
        alerts.append({"level": "MEDIUM",
            "msg": f"Storage {current_storage_pct:.1f}% with {rain_days} rain days forecast. "
                   "Consider proactive release to create buffer capacity."})
    elif current_storage_pct < DROUGHT_THRESHOLD:
        primary_risk = "DROUGHT"
        alerts.append({"level": "HIGH",
            "msg": f"Storage critically low at {current_storage_pct:.1f}%. Minimise all releases."})
    elif current_storage_pct < 35 and total_p50 < 10:
        primary_risk = "DRY"
        alerts.append({"level": "MEDIUM",
            "msg": f"Low storage ({current_storage_pct:.1f}%) with minimal rain forecast. Conservation mode."})
    else:
        primary_risk = "NORMAL"
        alerts.append({"level": "LOW",
            "msg": "Storage and forecast within normal operating range."})

    # ── LAYER 2: Trajectory-based alerts (the new smarter alerts) ────────────
    # Use the OPTIMAL approach's P90 trajectory (worst case flood scenario)
    # and P10 trajectory (worst case drought scenario) to detect future crossings
    if optimal:
        p90_traj = optimal.trajectory_p90   # storage trajectory under P90 rainfall
        p10_traj = optimal.trajectory_p10   # storage trajectory under P10 rainfall

        # Check if P90 trajectory will cross flood threshold
        flood_crossing_day = None
        for day_idx, storage_val in enumerate(p90_traj[1:], start=1):  # skip day 0 (today)
            if storage_val > FLOOD_THRESHOLD and current_storage_pct <= FLOOD_THRESHOLD:
                flood_crossing_day = day_idx
                break

        # Check if P10 trajectory will cross drought threshold
        drought_crossing_day = None
        for day_idx, storage_val in enumerate(p10_traj[1:], start=1):
            if storage_val < DROUGHT_THRESHOLD and current_storage_pct >= DROUGHT_THRESHOLD:
                drought_crossing_day = day_idx
                break

        # Check if P50 trajectory crosses flood threshold (more certain warning)
        p50_flood_crossing = None
        p50_traj = optimal.trajectory_p50
        for day_idx, storage_val in enumerate(p50_traj[1:], start=1):
            if storage_val > FLOOD_THRESHOLD and current_storage_pct <= FLOOD_THRESHOLD:
                p50_flood_crossing = day_idx
                break

        # Fire trajectory alerts (only if not already covered by Layer 1)
        if p50_flood_crossing is not None and primary_risk not in ("CRITICAL FLOOD", "FLOOD"):
            alerts.append({"level": "HIGH",
                "msg": f"TRAJECTORY WARNING: Even with the recommended release, "
                       f"storage is projected to cross the flood threshold (85%) "
                       f"in {p50_flood_crossing} day{'s' if p50_flood_crossing > 1 else ''} "
                       f"(P50 median scenario). Consider a more aggressive release strategy."})

        elif flood_crossing_day is not None and primary_risk not in ("CRITICAL FLOOD", "FLOOD"):
            alerts.append({"level": "MEDIUM",
                "msg": f"TRAJECTORY CAUTION: In the heavy rain scenario (P90), "
                       f"storage could cross the flood threshold (85%) in "
                       f"{flood_crossing_day} day{'s' if flood_crossing_day > 1 else ''}. "
                       f"Monitor closely. Current storage: {current_storage_pct:.1f}%."})

        if drought_crossing_day is not None and primary_risk not in ("DROUGHT",):
            alerts.append({"level": "MEDIUM",
                "msg": f"TRAJECTORY CAUTION: In the dry scenario (P10), "
                       f"storage could fall below the drought threshold (25%) in "
                       f"{drought_crossing_day} day{'s' if drought_crossing_day > 1 else ''}. "
                       f"Avoid large releases. Current storage: {current_storage_pct:.1f}%."})

    # ── Downstream release warning ────────────────────────────────────────────
    if optimal and optimal.release_pct_per_day >= 8.0:
        alerts.append({"level": "WARNING",
            "msg": f"Recommended release ({optimal.release_pct_per_day}%/day = "
                   f"{optimal.release_mcm_per_day} MCM/day) requires downstream flood warning notification."})

    return primary_risk, alerts
