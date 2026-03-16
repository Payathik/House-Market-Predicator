"""
features.py
-----------
Feature engineering for the Ames Housing dataset.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns."""
    df = df.copy()

    # Total square footage (basement + 1st + 2nd floor)
    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0)
        + df.get("1stFlrSF", 0)
        + df.get("2ndFlrSF", 0)
    )

    # House age at time of sale
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Years since last remodel
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # Total bathrooms (half baths count as 0.5)
    df["TotalBaths"] = (
        df.get("FullBath", 0)
        + df.get("HalfBath", 0) * 0.5
        + df.get("BsmtFullBath", 0)
        + df.get("BsmtHalfBath", 0) * 0.5
    )

    # Total porch area
    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0)
        + df.get("EnclosedPorch", 0)
        + df.get("3SsnPorch", 0)
        + df.get("ScreenPorch", 0)
    )

    # Binary flags
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "2ndFlrSF" in df.columns:
        df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    return df


def log_transform_target(series: pd.Series) -> pd.Series:
    """Log-transform SalePrice to reduce right skew."""
    return np.log1p(series)


def inverse_log_transform(series: pd.Series) -> pd.Series:
    """Reverse the log transform to get back to dollar values."""
    return np.expm1(series)
