# ===============================================================
# DataManager — TRAINING + INFERENCE (CSV only)
# Path: ml/src/common/data_manager.py
# ===============================================================
import os
import pandas as pd
from typing import Dict, Any, Optional


class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        dm = self.cfg["data_manager"]
        # === TRAINING ===
        self.raw_csv = dm["csv_path"]
        # === INFERENCE ===
        self.prod_db_csv = dm["prod_database_path"]

    # ------------------- CSV I/O -------------------
    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        """
        Read a CSV with best-effort datetime parsing for 'datetime'.
        Returns empty DataFrame if file doesn't exist.
        """
        if not os.path.exists(path):
            return pd.DataFrame()

        return pd.read_csv(path)

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to CSV, creating parent dirs if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    # ================= TRAINING =================
    def load_raw_csv(self) -> pd.DataFrame:
        """Load the training CSV."""
        return self._read_csv(self.raw_csv)

    def load_prod_data(self) -> pd.DataFrame:
        """Load the rolling production DB CSV."""
        return self._read_csv(self.prod_db_csv)

    def save_prod_data(self, df: pd.DataFrame) -> None:
        """Persist the rolling production DB CSV."""
        self._write_csv(df, self.prod_db_csv)


