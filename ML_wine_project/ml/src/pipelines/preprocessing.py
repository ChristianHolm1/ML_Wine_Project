# src/pipelines/preprocessing.py
import pandas as pd
from typing import Dict, List, Any


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["preprocessing"]

    # ---------- helpers ----------
    @staticmethod
    def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=mapping)

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        keep = [c for c in df.columns if c not in columns]
        return df[keep].copy()

    # ---------- hygiene ----------
    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    @staticmethod
    def _clip_outliers_minmax(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        clip_ranges = {
            "fixed_acidity": (3.5, 12.0),
            "volatile_acidity": (0.05, 1.0),
            "citric_acid": (0.0, 1.0),
            "residual_sugar": (0.0, 40.0),
            "chlorides": (0.005, 0.2),
            "free_sulfur_dioxide": (0.0, 150.0),
            "total_sulfur_dioxide": (0.0, 300.0),
            "density": (0.987, 1.010),
            "pH": (2.8, 3.8),
            "sulphates": (0.2, 1.0),
            "alcohol": (8.0, 14.5),
        }

        for col, (low, high) in clip_ranges.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=low, upper=high)

        return df

    # ---------- orchestrated run ----------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._drop_duplicates(df)
        df = self._clip_outliers_minmax(df)

        df = df.reset_index(drop=True)
        if self.cfg.get("column_mapping"):
            df = self.rename_columns(df, self.cfg["column_mapping"])
        if self.cfg.get("drop_columns"):
            df = self.drop_columns(df, self.cfg["drop_columns"])
        return df
