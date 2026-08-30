
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Page styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📈 Stock Market Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Historical market analysis, machine-learning prediction and performance evaluation.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Stock list
# -----------------------------
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
}

PERIODS = {
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
}


@st.cache_data(ttl=1800, show_spinner=False)
def download_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError("No data was returned for this ticker.")

    # Newer yfinance versions can return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns from Yahoo Finance: {missing}")

    data = data[required].copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.reset_index()
    data.rename(columns={"Date": "Date"}, inplace=True)

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna().drop_duplicates(subset=["Date"]).sort_values("Date")
    return data.reset_index(drop=True)


def clean_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Read a user CSV and standardize common stock column names."""
    data = pd.read_csv(uploaded_file)

    rename_map = {}
    for col in data.columns:
        key = str(col).strip().lower().replace(" ", "").replace("_", "")
        if key in {"date", "datetime", "timestamp"}:
            rename_map[col] = "Date"
        elif key in {"open", "openprice"}:
            rename_map[col] = "Open"
        elif key in {"high", "highprice"}:
            rename_map[col] = "High"
        elif key in {"low", "lowprice"}:
            rename_map[col] = "Low"
        elif key in {"close", "closeprice", "adjclose", "adjustedclose"}:
            rename_map[col] = "Close"
        elif key in {"volume", "tradingvolume"}:
            rename_map[col] = "Volume"

    data = data.rename(columns=rename_map)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "CSV must contain these columns: Date, Open, High, Low, Close, Volume. "
            f"Missing: {missing}"
        )

    data = data[required].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = (
        data.dropna()
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return data


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create time-series features without using future information."""
    df = data.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear

    for lag in [1, 7, 14, 28]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

    df["RollingMean_7"] = df["Close"].shift(1).rolling(7).mean()
    df["RollingMean_28"] = df["Close"].shift(1).rolling(28).mean()
    df["RollingStd_7"] = df["Close"].shift(1).rolling(7).std()

    # Previous-day OHLCV values are also safe predictors.
    for col in ["Open", "High", "Low", "Volume"]:
        df[f"{col}_Lag_1"] = df[col].shift(1)

    return df.dropna().reset_index(drop=True)


def train_model(data: pd.DataFrame):
    features = make_features(data)

    feature_cols = [
        "Year",
        "Month",
        "DayOfWeek",
        "DayOfYear",
        "Close_Lag_1",
        "Close_Lag_7",
        "Close_Lag_14",
        "Close_Lag_28",
        "RollingMean_7",
        "RollingMean_28",
        "RollingStd_7",
        "Open_Lag_1",
        "High_Lag_1",
        "Low_Lag_1",
        "Volume_Lag_1",
    ]

    X = features[feature_cols]
    y = features["Close"]

    split_index = int(len(features) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=18,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, test_pred),
        "MSE": mean_squared_error(y_test, test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "R2": r2_score(y_test, test_pred),
    }

    test_results = pd.DataFrame(
        {
            "Date": features.loc[X_test.index, "Date"].values,
            "Actual": y_test.values,
            "Predicted": test_pred,
        }
    )

    return model, feature_cols, metrics, test_results


def make_future_features(history: pd.DataFrame, future_date: pd.Timestamp, feature_cols):
    """Create one future row using only values available up to the previous day."""
    close = history["Close"].astype(float).tolist()
    open_values = history["Open"].astype(float).tolist()
    high_values = history["High"].astype(float).tolist()
    low_values = history["Low"].astype(float).tolist()
    volume_values = history["Volume"].astype(float).tolist()

    row = {
        "Year": future_date.year,
        "Month": future_date.month,
        "DayOfWeek": future_date.dayofweek,
        "DayOfYear": future_date.dayofyear,
        "Close_Lag_1": close[-1],
        "Close_Lag_7": close[-7],
        "Close_Lag_14": close[-14],
        "Close_Lag_28": close[-28],
        "RollingMean_7": float(np.mean(close[-7:])),
        "RollingMean_28": float(np.mean(close[-28:])),
        "RollingStd_7": float(np.std(close[-7:], ddof=1)),
        "Open_Lag_1": open_values[-1],
        "High_Lag_1": high_values[-1],
        "Low_Lag_1": low_values[-1],
        "Volume_Lag_1": volume_values[-1],
    }
    return pd.DataFrame([row])[feature_cols]


def forecast_future(model, data: pd.DataFrame, feature_cols, days: int) -> pd.DataFrame:
    """Recursively forecast the next trading days."""
    history = data.copy()
    predictions = []

    last_date = history["Date"].max()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days,
    )

    for future_date in future_dates:
        X_future = make_future_features(history, future_date, feature_cols)
        predicted_close = float(model.predict(X_future)[0])

        # OHLCV values for future rows are unavailable. For recursive prediction,
        # carry the latest observed values forward and replace Close with prediction.
        last_row = history.iloc[-1].copy()
        new_row = {
            "Date": future_date,
            "Open": float(last_row["Close"]),
            "High": predicted_close,
            "Low": predicted_close,
            "Close": predicted_close,
            "Volume": float(last_row["Volume"]),
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        predictions.append(
            {"Date": future_date, "Predicted_Close": predicted_close}
        )

    return pd.DataFrame(predictions)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    data_source = st.radio(
        "Choose data source",
        ["Yahoo Finance", "Upload CSV"],
        index=0,
    )

    if data_source == "Yahoo Finance":
        company = st.selectbox("Select stock/company", list(STOCKS.keys()))
        ticker = STOCKS[company]
        period_label = st.selectbox("Historical period", list(PERIODS.keys()), index=1)
        period = PERIODS[period_label]
    else:
        uploaded_file = st.file_uploader(
            "Upload stock CSV",
            type=["csv"],
            help="Required columns: Date, Open, High, Low, Close, Volume",
        )
        ticker = "Uploaded CSV"
        company = "Uploaded Stock"
        period = None

    forecast_days = st.slider(
        "Forecast trading days",
        min_value=7,
        max_value=30,
        value=15,
    )

    st.caption("⚠️ This project is for educational purposes, not financial advice.")


# -----------------------------
# Load data
# -----------------------------
try:
    with st.spinner("Loading market data..."):
        if data_source == "Yahoo Finance":
            data = download_stock_data(ticker, period)
        else:
            if uploaded_file is None:
                st.info(
                    "Upload a CSV to continue. The file must contain Date, Open, High, Low, Close and Volume."
                )
                st.stop()
            data = clean_uploaded_data(uploaded_file)

    if len(data) < 80:
        st.error("Please use a dataset with at least 80 rows for reliable train/test evaluation.")
        st.stop()

except Exception as exc:
    st.error("Unable to load the selected data.")
    st.exception(exc)
    st.stop()


# -----------------------------
# Dataset summary
# -----------------------------
st.subheader("1. Data Collection & Dataset Preparation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Start Date", data["Date"].min().strftime("%d %b %Y"))
c3.metric("End Date", data["Date"].max().strftime("%d %b %Y"))
c4.metric("Latest Close", f"{data['Close'].iloc[-1]:,.2f}")

with st.expander("View cleaned dataset"):
    st.dataframe(data.tail(100), use_container_width=True)

st.caption(
    "Preprocessing performed: date formatting, numeric conversion, missing-value removal, "
    "duplicate removal, sorting and feature preparation."
)


# -----------------------------
# EDA
# -----------------------------
st.subheader("2. Exploratory Data Analysis (EDA)")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price Trend", "📊 Volume", "〰️ Moving Averages", "🔥 Correlation"]
)

with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["Date"], data["Close"], label="Close Price")
    ax.set_title(f"{company} - Historical Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

with tab2:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(data["Date"], data["Volume"], width=1.0)
    ax.set_title(f"{company} - Trading Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

with tab3:
    eda = data.copy()
    eda["MA_20"] = eda["Close"].rolling(20).mean()
    eda["MA_50"] = eda["Close"].rolling(50).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eda["Date"], eda["Close"], label="Close")
    ax.plot(eda["Date"], eda["MA_20"], label="20-Day MA")
    ax.plot(eda["Date"], eda["MA_50"], label="50-Day MA")
    ax.set_title(f"{company} - Moving Average Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

with tab4:
    corr = data[["Open", "High", "Low", "Close", "Volume"]].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("OHLCV Correlation Matrix")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Model
# -----------------------------
st.subheader("3. Stock Price Prediction Model")

with st.spinner("Training Random Forest model..."):
    model, feature_cols, metrics, test_results = train_model(data)

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"{metrics['MAE']:.2f}")
m2.metric("MSE", f"{metrics['MSE']:.2f}")
m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
m4.metric("R² Score", f"{metrics['R2']:.3f}")

st.write(
    "The dataset is split chronologically into 80% training data and 20% testing data. "
    "A Random Forest Regression model is trained using lag, rolling-average and calendar features."
)


# -----------------------------
# Actual vs predicted
# -----------------------------
st.subheader("4. Model Training & Testing")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test_results["Date"], test_results["Actual"], label="Actual")
ax.plot(test_results["Date"], test_results["Predicted"], label="Predicted")
ax.set_title("Actual vs Predicted Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.grid(alpha=0.25)
fig.autofmt_xdate()
st.pyplot(fig, clear_figure=True)

with st.expander("View test predictions"):
    st.dataframe(test_results, use_container_width=True)


# -----------------------------
# Future forecast
# -----------------------------
st.subheader("5. Future Stock Price Forecast")

with st.spinner("Generating future predictions..."):
    future = forecast_future(model, data, feature_cols, forecast_days)

latest_close = float(data["Close"].iloc[-1])
future_last = float(future["Predicted_Close"].iloc[-1])
change_pct = ((future_last - latest_close) / latest_close) * 100

f1, f2, f3 = st.columns(3)
f1.metric("Latest Actual Close", f"{latest_close:,.2f}")
f2.metric(
    f"{forecast_days}-Day Forecast",
    f"{future_last:,.2f}",
    f"{change_pct:+.2f}%",
)
f3.metric("Predicted Trading Days", str(len(future)))

fig, ax = plt.subplots(figsize=(12, 4))
recent = data.tail(120)

ax.plot(recent["Date"], recent["Close"], label="Historical Close")
ax.plot(future["Date"], future["Predicted_Close"], label="Forecast")
ax.axvline(data["Date"].max(), linestyle="--", linewidth=1)
ax.set_title(f"{company} - Historical and Future Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(alpha=0.25)
fig.autofmt_xdate()
st.pyplot(fig, clear_figure=True)

st.dataframe(future, use_container_width=True)

csv = future.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Forecast CSV",
    data=csv,
    file_name="stock_forecast.csv",
    mime="text/csv",
)


# -----------------------------
# Project conclusion
# -----------------------------
st.subheader("6. Project Insights")

direction = "higher" if change_pct >= 0 else "lower"
st.write(
    f"Based on the trained model, the final forecasted closing price over the selected "
    f"horizon is **{direction}** than the latest observed closing price by approximately "
    f"**{abs(change_pct):.2f}%**. Model performance should be interpreted together with "
    f"the MAE, RMSE and R² values above."
)

st.info(
    "Important: stock prices are influenced by news, economic conditions, market sentiment "
    "and many external factors. This model is a learning project and should not be used as "
    "a guaranteed buy/sell signal."
)
