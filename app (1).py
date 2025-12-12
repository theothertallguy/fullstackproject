# app.py
from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# -----------------------------------------------------------------------------
# 1. Paths & artifact loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.pkl"
MAPPING_PATH = ARTIFACT_DIR / "stockcode_mapping.pkl"
BASELINES_PATH = ARTIFACT_DIR / "stock_baselines.pkl"
HISTORY_PATH = ARTIFACT_DIR / "daily_history.csv"

# Load artifacts
model = joblib.load(MODEL_PATH)
stockcode_to_id = joblib.load(MAPPING_PATH)
stock_baselines = joblib.load(BASELINES_PATH)

# History for forecasting (aggregated daily data)
daily = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
daily["StockCode"] = daily["StockCode"].astype(str).str.strip()

# This should match the y variable used in training (e.g. daily sales amount)
TARGET_COL = "Sales"  # or "Quantity" depending on how you trained


# -----------------------------------------------------------------------------
# 2. Helper: build features for one row (must match training features)
# -----------------------------------------------------------------------------
def _build_features_for_dates(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a history_df with columns:
        Date, StockCode_id, Sales (or Quantity)
    build the same feature set used in training:
        - DayOfWeek, Month
        - lag features (1, 7, 28)
        - rolling means (7, 28)
    Returns a DataFrame with feature columns only.
    """
    df = history_df.copy().sort_values("Date")

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    # lag features
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_7"] = df[TARGET_COL].shift(7)
    df["lag_28"] = df[TARGET_COL].shift(28)

    # rolling means
    df["roll_7"] = df[TARGET_COL].rolling(7).mean()
    df["roll_28"] = df[TARGET_COL].rolling(28).mean()

    # Drop rows with NA in features
    feature_cols = ["StockCode_id", "DayOfWeek", "Month",
                    "lag_1", "lag_7", "lag_28", "roll_7", "roll_28"]
    return df[feature_cols], df


# -----------------------------------------------------------------------------
# 3. Forecasting for one StockCode
# -----------------------------------------------------------------------------
def forecast_next_days(stock_code: str,
                       n_days: int,
                       model,
                       stockcode_to_id: dict,
                       history_df: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    """
    Recursive multi-step forecast for one StockCode.

    Returns list of (future_date, predicted_value) for n_days ahead.
    """

    code = stock_code.strip()
    if code not in stockcode_to_id:
        raise ValueError(f"Unknown StockCode: {code}")

    stock_id = stockcode_to_id[code]

    # Filter history for this product
    hist = history_df[history_df["StockCode"] == code].copy()
    if hist.empty:
        raise ValueError(f"No history found for StockCode {code}")

    # Ensure required columns exist
    if TARGET_COL not in hist.columns:
        raise ValueError(
            f"{TARGET_COL} column not found in history. "
            "Make sure daily_history.csv was created the same way as in training."
        )

    hist = hist.sort_values("Date")
    hist["StockCode_id"] = stock_id

    # Build initial features up to last known date
    X_all, hist_full = _build_features_for_dates(hist)

    # Keep only rows with complete features (after lags/rollings)
    valid_idx = X_all.dropna().index
    X_all = X_all.loc[valid_idx]
    hist_full = hist_full.loc[valid_idx]

    if X_all.empty:
        raise ValueError(f"Not enough history to build features for {code}")

    # Start from the last available row
    last_row = hist_full.iloc[-1]
    last_date = last_row["Date"]
    current_hist = hist_full.copy()

    preds = []

    for step in range(1, n_days + 1):
        future_date = last_date + timedelta(days=step)

        # Construct a single-row DataFrame for the future date,
        # using the latest available TARGET_COL values in current_hist
        tmp = current_hist.copy()

        # To build future lags/rollings we append a fake row for the future date
        future_df = pd.DataFrame(
            {
                "Date": [future_date],
                TARGET_COL: [tmp[TARGET_COL].iloc[-1]],  # placeholder, will be replaced
                "StockCode_id": [stock_id],
            }
        )
        tmp = pd.concat([tmp[["Date", TARGET_COL, "StockCode_id"]], future_df], ignore_index=True)

        X_tmp, tmp_full = _build_features_for_dates(tmp)

        # Last row in X_tmp corresponds to the future date
        X_future = X_tmp.iloc[[-1]]

        # If any feature is NA (short history), skip this step
        if X_future.isna().any(axis=1).iloc[0]:
            # stop forecasting if we can't build consistent features
            break

        y_hat = float(model.predict(X_future)[0])

        preds.append((future_date, y_hat))

        # Append predicted value to current history so that next step uses it as lag
        new_row = {
            "Date": future_date,
            TARGET_COL: y_hat,
            "StockCode_id": stock_id,
        }
        current_hist = pd.concat([current_hist, pd.DataFrame([new_row])], ignore_index=True)

    return preds


def forecast_next_days_df(stock_code: str,
                          n_days: int,
                          model,
                          stockcode_to_id: dict,
                          history_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper returning a DataFrame."""
    preds = forecast_next_days(stock_code, n_days, model, stockcode_to_id, history_df)
    return pd.DataFrame(
        {
            "Date": [d for d, _ in preds],
            "PredictedDailySales": [round(float(y), 2) for _, y in preds],
        }
    )


# -----------------------------------------------------------------------------
# 4. Flask app & routes
# -----------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """
    Simple homepage with a form.

    Expected form fields in index.html:
      - stock_code (text)
      - horizon   (select: week / month / year)
    """
    # You can optionally pass a few sample codes for the UI to suggest
    sample_codes = list(stockcode_to_id.keys())[:10]
    return render_template("index.html", sample_codes=sample_codes)


@app.route("/predict", methods=["POST"])
def predict():
    stock_code = request.form.get("stock_code", "").strip()
    horizon = request.form.get("horizon", "week")

    if not stock_code:
        return render_template("result.html", error="Please provide a StockCode.")

    if stock_code not in stockcode_to_id:
        return render_template(
            "result.html",
            error=f"Unknown StockCode: {stock_code}. Please check the value and try again.",
        )

    # Decide forecast length based on horizon
    if horizon == "week":
        n_days = 7
        baseline_key = "avg_week_sales"
    elif horizon == "month":
        n_days = 30
        baseline_key = "avg_month_sales"
    elif horizon == "year":
        n_days = 365
        baseline_key = "avg_year_sales"
    else:
        # default fallback
        n_days = 7
        baseline_key = "avg_week_sales"

    # 1) Forecast future daily values
    try:
        forecast_df = forecast_next_days_df(
            stock_code, n_days, model, stockcode_to_id, daily
        )
    except ValueError as e:
        return render_template("result.html", error=str(e))

    if forecast_df.empty:
        return render_template(
            "result.html",
            error=f"Could not generate a forecast for {stock_code}.",
        )

    total_forecast = float(forecast_df["PredictedDailySales"].sum())

    # 2) Compare against baseline to generate recommendation
    baseline_info = stock_baselines.get(stock_code, {})
    baseline = baseline_info.get(baseline_key)

    recommendation = None
    if baseline is not None and baseline > 0:
        if total_forecast >= 1.2 * baseline:
            recommendation = "Forecast is significantly above typical. Consider stocking up."
        elif total_forecast <= 0.8 * baseline:
            recommendation = "Forecast is significantly below typical. Consider reducing or delaying orders."
        else:
            recommendation = "Forecast is close to typical levels. Maintain current ordering plan."

    # Data for chart/table in result.html
    forecast_records = forecast_df.to_dict(orient="records")

    return render_template(
        "result.html",
        stock_code=stock_code,
        horizon=horizon,
        total_forecast=round(total_forecast, 2),
        baseline=round(float(baseline), 2) if baseline is not None else None,
        recommendation=recommendation,
        forecast_points=forecast_records,
    )


if __name__ == "__main__":
    # Debug mode for local development
    app.run(host="0.0.0.0", port=5000, debug=True)
