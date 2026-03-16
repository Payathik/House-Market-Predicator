"""
evaluate.py
-----------
Evaluation metrics and visualisation helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_scores(model_name: str, y_true, y_pred) -> None:
    """Print RMSE and R² for a model."""
    r = rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name:20s}  RMSE: ${r:,.0f}   R²: {r2:.4f}")


def plot_predicted_vs_actual(
    y_true, y_pred, model_name: str = "Model", save_path: str = None
):
    """Scatter plot of predicted vs actual prices."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="#1D9E75")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual SalePrice ($)")
    ax.set_ylabel("Predicted SalePrice ($)")
    ax.set_title(f"{model_name} — Predicted vs Actual")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(
    model, feature_names, top_n: int = 20, save_path: str = None
):
    """Horizontal bar chart of top N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    top.plot(kind="barh", ax=ax, color="#534AB7")
    ax.set_title(f"Top {top_n} feature importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
