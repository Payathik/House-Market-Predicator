# House Price Predictor

A machine learning project to predict residential house prices using the Ames Housing dataset.

## Project Overview

This project walks through the full ML pipeline: exploratory data analysis, data cleaning,
feature engineering, model training, and evaluation. Models compared: Ridge Regression,
Random Forest, and XGBoost.

## Results

| Model            | CV RMSE (log) | R²   |
|------------------|---------------|------|
| Ridge Regression | -             | -    |
| Random Forest    | -             | -    |
| XGBoost          | -             | -    |

> Fill in after running `notebooks/03_model_training.ipynb`

## Project Structure

```
house_price_predictor/
├── data/
│   ├── raw/          # Original downloaded data (do not modify)
│   └── processed/    # Cleaned and transformed data (auto-generated)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_preprocessing.ipynb     # Cleaning & feature engineering
│   └── 03_model_training.ipynb    # Model training & evaluation
├── src/
│   ├── __init__.py
│   ├── preprocess.py   # Reusable preprocessing functions
│   ├── features.py     # Feature engineering functions
│   └── evaluate.py     # Evaluation metrics and plots
├── models/             # Saved trained models (.pkl files)
├── reports/
│   └── figures/        # Output charts and visualizations
├── requirements.txt
└── README.md
```

## Setup

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from Kaggle:
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
   Place `train.csv`, `test.csv`, and `data_description.txt` in `data/raw/`

5. Run the notebooks in order:
   ```
   01_eda.ipynb  →  02_preprocessing.ipynb  →  03_model_training.ipynb
   ```

## Dataset

Ames Housing Dataset — 79 features describing residential homes in Ames, Iowa.

- `train.csv` — 1460 rows with SalePrice labels
- `test.csv`  — 1459 rows for Kaggle submission
- `data_description.txt` — full column documentation

## Key Findings

*(Update this section after EDA)*

- Strongest correlations with SalePrice: OverallQual, GrLivArea, GarageCars
- SalePrice is right-skewed → log-transforming improves all model scores
- Top engineered feature: TotalSF (basement + 1st floor + 2nd floor area)

## Skills Demonstrated

- Handling missing values (80+ features with NaNs)
- Exploratory data analysis with seaborn/matplotlib
- Feature engineering (TotalSF, HouseAge, TotalBaths, etc.)
- Model comparison and cross-validation
- Feature importance visualization with SHAP

## Author

Your Name — [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
