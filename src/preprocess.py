"""
preprocess.py
-------------
Functions for cleaning the Ames Housing dataset.
"""

import pandas as pd
import numpy as np


# Categorical columns where NaN means "None" (e.g. no pool, no garage)
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Numerical columns where NaN means 0
ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with sensible defaults."""
    df = df.copy()

    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill remaining categoricals with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fill remaining numericals with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns."""
    return pd.get_dummies(df, drop_first=True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known outliers flagged in the Ames dataset documentation."""
    df = df.copy()
    # Two extreme outliers noted in official docs (very large GrLivArea, low price)
    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]
    return df
