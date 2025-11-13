#!/usr/bin/env python3
"""
Data & Pipeline Diagnostics (CSV setup, no utils.py)

What it checks:
  1) Loads config (paths, model_path, batch_size, first_timestamp, etc.)
  2) Loads TRAIN and PROD (real_time) CSVs
     - rows, columns, datetime min/max, inferred cadence
     - missingness % by column
  3) Compares TRAIN vs PROD scale for the target column (mean/std/min/max)
  4) Sanity-checks config:
     - first_timestamp exists in PROD
     - num_inference_steps <= PROD rows (warn if not)
  5) Validates that base columns required to build features exist in PROD:
     - taxi_pickups for rolling features
     - weather columns listed in config["feature_engineering"]["weather_deltas"]
  6) Loads trained model, prints expected feature names (feature_names_in_)
  7) If predictions.csv exists, merges with PROD and prints MAE/RMSE + tail
     NEW: with --int-preds (or config inference.output_integer: true),
          predictions are rounded for display/metrics.

Run:
  python tools/check_pipeline.py --config config/config.yaml --actual-col taxi_pickups
  python tools/check_pipeline.py --int-preds
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import yaml
import joblib
import numpy as np
import pandas as pd


# ------------- helpers -------------

def read_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def find_datetime_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["datetime", "timestamp", "time", "date", "ds"]
    for c in df.columns:
        if c.lower() in candidates:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                pass
    # fallback: try parsing each column, pick best
    best, best_nonnull = None, -1
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            nonnull = int(parsed.notna().sum())
            if nonnull > best_nonnull and nonnull > 0:
                best, best_nonnull = c, nonnull
        except Exception:
            continue
    return best

def load_csv(path: Path, parse_dt: bool = True) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if not parse_dt:
        return pd.read_csv(path)
    try:
        # try parse 'datetime' by default
        return pd.read_csv(path, parse_dates=["datetime"])
    except Exception:
        return pd.read_csv(path)

def describe_cadence(dts: pd.Series) -> str:
    if dts.empty:
        return "n/a"
    srt = dts.sort_values()
    if len(srt) < 2:
        return "n/a"
    diffs = srt.diff().dropna()
    if diffs.empty:
        return "n/a"
    mode = diffs.mode()
    return str(mode.iloc[0]) if not mode.empty else "n/a"

def pct_missing(df: pd.DataFrame) -> pd.DataFrame:
    return (df.isna().mean().sort_values(ascending=False) * 100.0).rename("pct_missing").to_frame()

def print_header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def print_kv(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}: {v}")

def numeric_summary(s: pd.Series) -> dict:
    return {
        "count": int(s.count()),
        "mean": float(s.mean()) if s.count() else np.nan,
        "std": float(s.std()) if s.count() else np.nan,
        "min": float(s.min()) if s.count() else np.nan,
        "p50": float(s.median()) if s.count() else np.nan,
        "max": float(s.max()) if s.count() else np.nan,
    }

def metrics_from_merge(merged: pd.DataFrame, actual_col: str) -> dict:
    err = merged["prediction"] - merged[actual_col]
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}


# ------------- main -------------

def main():
    parser = argparse.ArgumentParser(description="Diagnostics for CSV-based training & inference pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--actual-col", type=str, default="taxi_pickups", help="Ground-truth column in production DB")
    parser.add_argument("--tail", type=int, default=10, help="Rows to show at tail for merged predictions")
    parser.add_argument("--int-preds", action="store_true", help="Round predictions to integers for display/metrics.")
    args = parser.parse_args()

    ROOT = Path.cwd()
    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        print(f"ERROR: config file not found at {cfg_path}")
        sys.exit(1)

    cfg = read_config(cfg_path)

    # --- paths from config ---
    dm_cfg = cfg["data_manager"]
    train_csv = ROOT / dm_cfg["csv_path"]
    prod_csv  = ROOT / dm_cfg["real_time_data_prod_path"]
    pred_csv  = ROOT / dm_cfg["predictions_path"]

    model_path = ROOT / cfg["pipeline_runner"]["model_path"]
    batch_size = int(cfg["pipeline_runner"]["batch_size"])
    first_ts   = pd.to_datetime(cfg["pipeline_runner"]["first_timestamp"])
    num_steps  = int(cfg["pipeline_runner"]["num_inference_steps"])
    time_inc   = cfg["pipeline_runner"]["time_increment"]

    fe_cfg = cfg.get("feature_engineering", {})
    weather_deltas: List[str] = list(fe_cfg.get("weather_deltas", []))
    source_col = cfg["training"]["target"]["source_col"]
    actual_col = args.actual_col

    # --- load CSVs ---
    print_header("1) Loading CSVs")
    train = load_csv(train_csv)
    prod  = load_csv(prod_csv)

    if train.empty:
        print("WARNING: Training CSV is empty or missing:", train_csv)
    if prod.empty:
        print("WARNING: Production (stream) CSV is empty or missing:", prod_csv)

    print_kv(train_path=str(train_csv), train_rows=len(train), prod_path=str(prod_csv), prod_rows=len(prod))

    # --- datetime column detection & cadence ---
    print_header("2) Datetime columns & cadence")
    for name, df in [("TRAIN", train), ("PROD", prod)]:
        if df.empty:
            print(f"{name}: empty")
            continue
        dt_col = "datetime" if "datetime" in df.columns else find_datetime_col(df)
        if dt_col is None:
            print(f"{name}: no datetime-like column found. Columns={list(df.columns)}")
            continue
        dts = pd.to_datetime(df[dt_col], errors="coerce").dropna()
        cadence = describe_cadence(dts)
        print(f"{name}: dt_col={dt_col}, rows={len(df)}, min={dts.min()}, max={dts.max()}, cadence={cadence}")

    # --- missingness ---
    print_header("3) Missingness (%) — top 10")
    if not train.empty:
        print("TRAIN:")
        print(pct_missing(train).head(10))
    if not prod.empty:
        print("\nPROD:")
        print(pct_missing(prod).head(10))

    # --- scale comparison for target ---
    print_header("4) Scale of target column in TRAIN vs PROD")
    if source_col in train.columns:
        train_summ = numeric_summary(train[source_col])
        print("TRAIN", source_col, train_summ)
    else:
        print(f"TRAIN: target source_col '{source_col}' not found. Columns={list(train.columns)}")

    if actual_col in prod.columns:
        prod_summ = numeric_summary(prod[actual_col])
        print("PROD ", actual_col, prod_summ)
    else:
        print(f"PROD: actual_col '{actual_col}' not found. Columns={list(prod.columns)}")

    # --- config sanity checks ---
    print_header("5) Config sanity")
    ok_first = False
    if not prod.empty and "datetime" in prod.columns:
        prod_dt = pd.to_datetime(prod["datetime"], errors="coerce")
        ok_first = (prod_dt == first_ts).any()
        print_kv(first_timestamp=str(first_ts), exists_in_prod=ok_first)
    else:
        print("Cannot check first_timestamp; prod is empty or has no 'datetime' column.")
    if not prod.empty:
        print_kv(num_inference_steps=num_steps, prod_rows=len(prod), steps_leq_rows=(num_steps <= len(prod)))
    print_kv(batch_size=batch_size, time_increment=time_inc)

    # --- base columns for FE ---
    print_header("6) Feature prerequisites in PROD (base columns present?)")
    base_ok = True
    need_cols = [source_col] + weather_deltas
    for c in need_cols:
        present = c in prod.columns if not prod.empty else False
        print(f"needs '{c}': {'OK' if present else 'MISSING'}")
        base_ok &= present

    # --- model feature names ---
    print_header("7) Model & feature names")
    if not model_path.exists():
        print(f"WARNING: model file not found: {model_path}")
    else:
        try:
            model = joblib.load(model_path)
            exp = getattr(model, "feature_names_in_", None)
            if exp is None:
                print("Model loaded. feature_names_in_ not available (older sklearn?).")
            else:
                print(f"Model loaded. feature_names_in_ count={len(exp)}")
                print("First 15 features:", list(exp[:15]))
        except Exception as e:
            print(f"ERROR loading model: {e}")

    # --- predictions metrics (optional) ---
    print_header("8) Predictions file (if exists) → MAE/RMSE & tail")
    if pred_csv.exists() and not prod.empty and ("datetime" in prod.columns):
        preds = load_csv(pred_csv)
        if not preds.empty and ("datetime" in preds.columns) and ("prediction" in preds.columns):
            # Should we display integers?
            cfg_output_int = bool(cfg.get("inference", {}).get("output_integer", False))
            make_int = args.int_preds or cfg_output_int

            preds_display = preds.copy()
            if make_int:
                preds_display["prediction"] = preds_display["prediction"].clip(lower=0).round().astype(int)

            if actual_col in prod.columns:
                merged = pd.merge(
                    preds_display[["datetime", "prediction"]],
                    prod[["datetime", actual_col]],
                    on="datetime",
                    how="inner",
                    validate="one_to_one",
                ).sort_values("datetime")
                if not merged.empty:
                    m = metrics_from_merge(merged, actual_col)
                    print_kv(rows_pred=len(preds), rows_matched=len(merged), **m)
                    tail_n = min(args.tail, len(merged))
                    print(f"\nLast {tail_n} rows (prediction vs {actual_col}):")
                    print(merged.tail(tail_n).to_string(index=False))
                else:
                    print("No matching timestamps between predictions and production actuals.")
            else:
                print(f"actual_col '{actual_col}' not found in production DB.")
        else:
            print(f"{pred_csv} exists but missing 'datetime'/'prediction' or file is empty.")
    else:
        print(f"predictions file not found at {pred_csv} (this is fine before first inference run).")

    print_header("Done")
    if not base_ok:
        print("NOTE: Some base columns are missing in PROD. Rolling features/deltas may be incomplete.")
    if not ok_first:
        print("NOTE: first_timestamp does not exist in PROD; your inference loop will skip until it appears.")


if __name__ == "__main__":
    main()

