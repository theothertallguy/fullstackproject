# app.py
from pathlib import Path
from datetime import timedelta
import random

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import requests

# -----------------------------------------------------------------------------
# 1. Paths & artifact loading
# -----------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

HF_REPO = "theothertallguy/propredict"  # replace with your Hugging Face repo
ARTIFACT_FILES = ["model.pkl", "stockcode_mapping.pkl", "stock_baselines.pkl", "daily_history.csv"]

def download_hf_file(repo_id, filename, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"Downloading {filename} from {url} ...")
        r = requests.get(url)
        r.raise_for_status()
        local_path.write_bytes(r.content)
        print(f"{filename} downloaded successfully.")
    return local_path

# download all artifacts
for f in ARTIFACT_FILES:
    download_hf_file(HF_REPO, f, ARTIFACT_DIR / f)

# Load artifacts
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
MAPPING_PATH = ARTIFACT_DIR / "stockcode_mapping.pkl"
BASELINES_PATH = ARTIFACT_DIR / "stock_baselines.pkl"
HISTORY_PATH = ARTIFACT_DIR / "daily_history.csv"

model = joblib.load(MODEL_PATH)
stockcode_to_id = joblib.load(MAPPING_PATH)
stock_baselines = joblib.load(BASELINES_PATH)
daily = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
daily["StockCode"] = daily["StockCode"].astype(str).str.strip()

# This should match the y variable used in training (e.g. daily sales amount)
TARGET_COL = "DailySales"  # or "Quantity" depending on how you trained


# -----------------------------------------------------------------------------
# 2. Helper: build features for one row (must match training features)
# -----------------------------------------------------------------------------
def _build_features_for_dates(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for forecasting, matching the training order exactly.
    Returns: (features_df, full_df)
    """
    df = history_df.copy().sort_values("Date")
    
    # Date features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    
    # Lag features (require at least lag_7)
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_7"] = df[TARGET_COL].shift(7)
    
    # Rolling mean features
    df["roll_mean_7"] = df[TARGET_COL].rolling(7, min_periods=1).mean()
    df["roll_mean_28"] = df[TARGET_COL].rolling(28, min_periods=1).mean()
    
    # Ensure StockCode_id exists
    if "StockCode_id" not in df.columns:
        df["StockCode_id"] = df["StockCode"].map(stockcode_to_id)
    
    # Feature order must match training
    feature_cols = [
        "year", "month", "day_of_week", "week_of_year",
        "lag_1", "lag_7", "roll_mean_7", "roll_mean_28",
        "StockCode_id"
    ]
    
    return df[feature_cols], df


# -----------------------------------------------------------------------------
# 3. Forecasting for one StockCode
# -----------------------------------------------------------------------------
def forecast_next_days(stock_code: str,
                       n_days: int,
                       model,
                       stockcode_to_id: dict,
                       history_df: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:

    code = stock_code.strip()
    if code not in stockcode_to_id:
        raise ValueError(f"Unknown StockCode: {code}")

    stock_id = stockcode_to_id[code]

    # Filter history and restrict to valid date range
    hist = history_df[history_df["StockCode"] == code].copy()
    start_date = pd.to_datetime("01/12/2010", dayfirst=True)
    end_date = pd.to_datetime("09/12/2011", dayfirst=True)
    hist = hist[(hist["Date"] >= start_date) & (hist["Date"] <= end_date)]

    if hist.empty:
        raise ValueError(f"No history found for StockCode {code} in the valid date range")

    # Fill missing dates with 0 sales
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    hist = hist.set_index("Date").reindex(all_dates, fill_value=0).rename_axis("Date").reset_index()
    hist["StockCode"] = code
    hist["StockCode_id"] = stock_id

    # Build features and fill NaNs
    X_all, hist_full = _build_features_for_dates(hist)
    X_all = X_all.fillna(0)
    hist_full = hist_full.loc[X_all.index]

    if X_all.empty:
        raise ValueError(f"Not enough history to build features for {code}")

    # Start forecasting
    last_date = pd.to_datetime("08/12/2012", dayfirst=True)
    current_hist = hist_full.copy()
    preds = []

    for step in range(1, n_days + 1):
        future_date = last_date + timedelta(days=step)
        tmp = current_hist.copy()

        # Ensure StockCode_id is included for future row
        future_df = pd.DataFrame({
            "Date": [future_date],
            TARGET_COL: [tmp[TARGET_COL].iloc[-1]],  # placeholder
            "StockCode_id": [stock_id],
        })

        # Concatenate and ensure StockCode_id exists in all rows
        tmp = pd.concat([tmp[["Date", TARGET_COL, "StockCode_id"]], future_df], ignore_index=True)
        tmp["StockCode_id"] = stock_id

        # Build features
        X_tmp, tmp_full = _build_features_for_dates(tmp)
        X_tmp = X_tmp.fillna(0)  # fill any NaNs in lags/rollings
        X_future = X_tmp.iloc[[-1]]

        y_hat = float(model.predict(X_future)[0])
        preds.append((future_date, y_hat))

        # Append predicted value to current history
        new_row = {"Date": future_date, TARGET_COL: y_hat, "StockCode_id": stock_id}
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


@app.route("/")
def home():
    """
    Simple homepage with a form.

    Expected form fields in index.html:
      - stock_code (text)
      - horizon   (select: week / month / year)
    """
    # You can optionally pass a few sample codes for the UI to suggest
    random_codes = random.sample(list(stockcode_to_id.keys()), 12)

    # Prepend guaranteed good codes
    plot_codes = ["85123A", "22423", "85099B"]
    sample_codes = plot_codes + random_codes
    return render_template("index.html", sample_codes=sample_codes)


@app.route("/predict", methods=["POST"])
def predict():
    
    stock_code = request.form['stock_code']
    horizon = request.form.get("horizon", "week")

    if not stock_code:
        return render_template("result.html", error="Please provide a StockCode.", err="A",)

    if stock_code not in stockcode_to_id:
        return render_template(
            "result.html",
            error=f"Unknown StockCode: {stock_code}. Please check the value and try again.",
            err="D",
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
  #  try:
    forecast_df = forecast_next_days_df(
        stock_code, n_days, model, stockcode_to_id, daily
    )
  #  except ValueError as e:
    #    return render_template("result.html", error=str(e), err="C")

    if forecast_df.empty:
        return render_template(
            "result.html",
            error=f"Could not generate a forecast for {stock_code}.",
            err="B",
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
        err=""
    )


if __name__ == "__main__":
    # Debug mode for local development
    app.run(host="0.0.0.0", port=5000, debug=True)
