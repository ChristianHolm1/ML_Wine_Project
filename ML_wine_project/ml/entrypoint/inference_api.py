#!/usr/bin/env python3
"""
Flask inference API for the taxi pipeline (CSV-only), placed under ml/entrypoint/.

Endpoints
---------
GET  /health
GET  /db/status
GET  /predictions/tail?n=10
POST /predict/next         -> take the next hour from real_time_data_prod.csv, append, predict t+1
POST /predict/from-row     -> accept a full JSON row, append, predict t+1
"""
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_from_directory, current_app
import pandas as pd
import numpy as np
import yaml

# --------- Resolve project paths ---------
ROOT = Path(__file__).resolve().parents[2]  # project root
ML_SRC = ROOT / "ml" / "src"

for p in (ROOT, ML_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(ROOT)

from common.data_manager import DataManager


# ---------------- Helpers ----------------
def read_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["datetime"])
    except Exception:
        return pd.read_csv(path)


def _to_native(obj):
    """
    Safely convert common numpy/pandas types (and nested containers) into plain
    Python types that Flask's jsonify can handle.
    """
    # dict / list / tuple: recurse
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]

    # None
    if obj is None:
        return None

    # Try numpy / pandas specifics in a guarded way
    try:
        import numpy as _np
        import pandas as _pd

        # pandas Timestamp
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat(sep=" ")

        # numpy scalar numbers
        if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
            return obj.item()

        # pandas / numpy NA-like
        if _pd.isna(obj):
            return None

        # numpy arrays -> convert to list then recurse
        if isinstance(obj, _np.ndarray):
            return [_to_native(v) for v in obj.tolist()]

    except Exception:
        # If numpy/pandas not available or something went wrong, continue
        pass

    # If it's a python int/float/bool already - return directly
    if isinstance(obj, (int, float, bool)):
        return obj

    # If it's a python string - return directly (no .item())
    if isinstance(obj, str):
        return obj

    # For any other object: attempt a safe conversion, otherwise string-ify
    try:
        # some libraries provide .tolist() or similar
        if hasattr(obj, "tolist"):
            return _to_native(obj.tolist())
    except Exception:
        pass

    # Fallback: return as-is if JSON serializable, else convert to str
    try:
        import json
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _append_row_to_csv(path: Path, row: dict, header_order: list = None):
    """
    Append a single-row dict to CSV using append mode. If file doesn't exist, write header.
    header_order: optional list to fix column order. If None, row.keys() order is used.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    if not file_exists:
        # create file and write header + row
        order = header_order or list(row.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=order)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in order})
        return

    # append mode: use existing header if present
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    order = header
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=order)
        writer.writerow({k: row.get(k, "") for k in order})


# ---------------- App & bootstrap ----------------
app = Flask(__name__)

CFG = read_config(ROOT / "config" / "config.yaml")
DM = DataManager(CFG)
RUNNER = None


def get_runner():
    global RUNNER
    if RUNNER is None:
        # import inside function to avoid circular import issues
        from pipelines.pipeline_runner import PipelineRunner
        RUNNER = PipelineRunner(CFG, DM)
    return RUNNER


# ---------------- Routes ----------------

@app.get("/")
def ui_index():
    ui_dir = Path(__file__).resolve().parents[0] / "static"
    return send_from_directory(str(ui_dir), "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/db/status")
def db_status():
    db_path = ROOT / CFG["data_manager"]["prod_database_path"]
    df_db = _load_csv(db_path)
    if df_db.empty:
        return jsonify({"rows": 0, "columns": [], "path": str(db_path)}), 200

    payload = {
        "rows": int(len(df_db)),
        "columns": list(df_db.columns),
        "path": str(db_path),
    }
    return jsonify(payload), 200


@app.post("/predict/value")
def predict_value():
    """
    Accept a single JSON object representing one row (no 'datetime' needed).
    Example body:
      {"fixed_acidity": 7.4, "volatile_acidity": 0.70, ...}
    """
    try:
        values = request.get_json(silent=True) or {}
        if not isinstance(values, dict) or not values:
            return jsonify({"status": "error", "message": "Provide a JSON object with feature names and values"}), 400

        result = get_runner().predict_from_values(values)
        return jsonify(_to_native({"status": "success", **result})), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/ingest")
def ingest_sample():
    """
    Expects features + 'quality' in JSON. Appends the row only to augmented_csv.
    """
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict) or "quality" not in payload:
            return jsonify(
                {"status": "error", "message": "Provide a JSON object with features AND a 'quality' field."}), 400

        # sanitize all values (convert numpy scalars and numeric strings to native int/float)
        sanitized = {}
        for k, v in payload.items():
            # numeric or none
            if isinstance(v, (int, float)) or v is None:
                sanitized[k] = v
                continue
            # numpy/pandas scalar safe extraction
            try:
                if hasattr(v, "item"):
                    try:
                        sanitized[k] = v.item()
                        continue
                    except Exception:
                        pass
            except Exception:
                pass
            # strings that are numeric
            if isinstance(v, str):
                s = v.strip()
                try:
                    sanitized[k] = int(s)
                    continue
                except Exception:
                    try:
                        sanitized[k] = float(s)
                        continue
                    except Exception:
                        sanitized[k] = s
                        continue
            # fallback: keep original
            sanitized[k] = v

        # Append only to augmented CSV
        aug_path = Path(CFG["data_manager"].get("augmented_csv", "data/prod_data/database_prod.csv"))
        _append_row_to_csv(aug_path, sanitized)
        return jsonify({"status": "success", "message": "Sample appended to augmented dataset."}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/retrain")
def retrain():
    """
    Retrain the model using the augmented CSV (raw + ingested).
    Synchronous: blocks until retraining finishes. Returns JSON status.
    """
    try:
        aug_path = Path(CFG["data_manager"].get("augmented_csv", "data/prod_data/database_prod.csv"))
        if not aug_path.exists() or aug_path.stat().st_size == 0:
            return jsonify({"status": "error", "message": f"Augmented file missing or empty: {aug_path}"}), 400

        df = pd.read_csv(aug_path)

        # Optional check: ensure label present
        if "quality" not in df.columns:
            return jsonify({"status": "error", "message": "Augmented CSV must include 'quality' column."}), 400

        # Use lazy runner (avoid circular import)
        runner = get_runner()

        # Prefer a static training entry if available; fallback to run_training()
        if hasattr(runner, "run_training_static"):
            runner.run_training_static(df)
        else:
            # If pipeline expects a file, write temp and call run_training()
            # or call run_training() if it consumes in-memory df in your refactor.
            runner.run_training_static(df)  # try this first; adjust if your runner uses a different method

        return jsonify({"status": "success", "message": "Retraining complete and model saved."}), 200

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        current_app.logger.error("Retrain error: %s\n%s", str(e), tb)
        return jsonify({"status": "error", "message": str(e), "traceback": tb}), 500


# ---------------- Main ----------------
if __name__ == "__main__":
    # Do NOT re-seed DB here (to avoid wiping history).
    app.run(host="0.0.0.0", port=5001, debug=False)
