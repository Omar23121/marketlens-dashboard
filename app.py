import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

st.set_page_config(page_title="MarketLens Dashboard", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Utility helpers
# -------------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in df.columns.to_flat_index()
        ]
    return df


def standardize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Handles cases where yfinance returns columns like:
    Close, Open, High, Low, Volume
    OR
    Close_AAPL, Open_AAPL, ...
    """
    df = flatten_columns(df.copy())

    rename_map = {}
    base_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for base in base_cols:
        if base in df.columns:
            rename_map[base] = base
        elif f"{base}_{ticker}" in df.columns:
            rename_map[f"{base}_{ticker}"] = base
        elif f"{base}_{ticker.upper()}" in df.columns:
            rename_map[f"{base}_{ticker.upper()}"] = base

    df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in keep_cols:
        raise ValueError("Could not find a usable 'Close' column in the downloaded data.")

    df = df[keep_cols].copy()
    df = df.reset_index()
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return raw
    raw = standardize_ohlcv_columns(raw, ticker)
    return raw


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Return"] = data["Close"].pct_change()
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA100"] = data["Close"].rolling(100).mean()

    data["STD20"] = data["Close"].rolling(20).std()
    data["Volatility20"] = data["Return"].rolling(20).std() * np.sqrt(252)

    data["BB_Upper"] = data["MA20"] + 2 * data["STD20"]
    data["BB_Lower"] = data["MA20"] - 2 * data["STD20"]
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["MA20"]

    data["RSI14"] = compute_rsi(data["Close"], 14)

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    data["Lag1"] = data["Close"].shift(1)
    data["Lag2"] = data["Close"].shift(2)
    data["Lag3"] = data["Close"].shift(3)
    data["Return_Lag1"] = data["Return"].shift(1)
    data["Return_Lag2"] = data["Return"].shift(2)

    if "Volume" in data.columns:
        data["Volume_Change"] = data["Volume"].pct_change()
        data["Volume_MA20"] = data["Volume"].rolling(20).mean()
    else:
        data["Volume_Change"] = np.nan
        data["Volume_MA20"] = np.nan

    return data


def prepare_model_data(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    data = add_indicators(df)
    data["Target"] = data["Close"].shift(-horizon)

    feature_cols = [
        "Lag1",
        "Lag2",
        "Lag3",
        "Return",
        "Return_Lag1",
        "Return_Lag2",
        "MA20",
        "MA50",
        "MA100",
        "Volatility20",
        "RSI14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Width",
        "Volume_Change",
    ]

    model_df = data.dropna(subset=feature_cols + ["Target"]).copy()
    return model_df, feature_cols, data


def get_model(model_name: str):
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    raise ValueError("Unsupported model selected.")


def evaluate_model(model_df: pd.DataFrame, feature_cols: list[str], model_name: str):
    split_idx = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["Target"]

    X_test = test_df[feature_cols]
    y_test = test_df["Target"]

    model = get_model(model_name)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    result = {
        "model": model,
        "test_df": test_df,
        "y_test": y_test,
        "preds": preds,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }
    return result


def forecast_latest(model, latest_features: pd.DataFrame) -> float:
    pred = float(model.predict(latest_features)[0])
    return pred


def next_business_days(last_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    dates = []
    current = last_date
    while len(dates) < n:
        current += pd.Timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates


def make_price_chart(data: pd.DataFrame, ticker: str):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"] if "Open" in data.columns else data["Close"],
            high=data["High"] if "High" in data.columns else data["Close"],
            low=data["Low"] if "Low" in data.columns else data["Close"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    for col, name in [("MA20", "MA20"), ("MA50", "MA50"), ("MA100", "MA100")]:
        if col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[col],
                    mode="lines",
                    name=name,
                ),
                row=1,
                col=1,
            )

    if "BB_Upper" in data.columns and "BB_Lower" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["BB_Upper"],
                mode="lines",
                name="BB Upper",
                line=dict(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["BB_Lower"],
                mode="lines",
                name="BB Lower",
                line=dict(dash="dot"),
            ),
            row=1,
            col=1,
        )

    if "Volume" in data.columns:
        fig.add_trace(
            go.Bar(
                x=data["Date"],
                y=data["Volume"],
                name="Volume",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"{ticker} Price Overview",
        xaxis_rangeslider_visible=False,
        height=750,
        template="plotly_white",
    )
    return fig


def make_rsi_chart(data: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI14"], mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(
        title="RSI (14)",
        height=350,
        template="plotly_white",
    )
    return fig


def make_macd_chart(data: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=data["Date"], y=data["MACD_Hist"], name="Histogram"))
    fig.update_layout(
        title="MACD",
        height=400,
        template="plotly_white",
    )
    return fig


def make_actual_vs_pred_chart(test_df: pd.DataFrame, y_test: pd.Series, preds: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["Date"], y=y_test, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=test_df["Date"], y=preds, mode="lines", name="Predicted"))
    fig.update_layout(
        title="Actual vs Predicted",
        height=420,
        template="plotly_white",
    )
    return fig


def make_feature_importance_chart(model, feature_cols: list[str]):
    if not hasattr(model, "feature_importances_"):
        return None

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        height=420,
    )
    fig.update_layout(template="plotly_white", yaxis=dict(categoryorder="total ascending"))
    return fig


@st.cache_data(show_spinner=False)
def load_comparison_data(tickers: list[str], period: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            df = yf.download(t, period=period, auto_adjust=True, progress=False)
            if df.empty:
                continue

            df = flatten_columns(df)
            close_col = None
            if "Close" in df.columns:
                close_col = "Close"
            elif f"Close_{t}" in df.columns:
                close_col = f"Close_{t}"
            elif f"Close_{t.upper()}" in df.columns:
                close_col = f"Close_{t.upper()}"

            if close_col is None:
                continue

            temp = df[[close_col]].copy().reset_index()
            temp.columns = ["Date", "Close"]
            temp["Ticker"] = t
            frames.append(temp)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# -------------------------------------------------
# App header
# -------------------------------------------------
st.title("MarketLens: Stock Analysis and Forecasting Dashboard")
st.caption(
    "Interactive financial analytics dashboard built with Python, Streamlit, yfinance, Plotly, and scikit-learn."
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Controls")

ticker = st.sidebar.text_input("Primary stock ticker", "AAPL").upper().strip()
period = st.sidebar.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
forecast_horizon = st.sidebar.selectbox("Forecast horizon (trading days)", [1, 3, 5, 10, 20], index=2)
model_name = st.sidebar.selectbox(
    "Forecast model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    index=1,
)

comparison_input = st.sidebar.text_input(
    "Comparison tickers (comma-separated)",
    "MSFT,GOOGL,NVDA"
)

st.sidebar.markdown("---")
st.sidebar.write("Tip: use valid Yahoo Finance ticker symbols such as `AAPL`, `TSLA`, `MSFT`, `AMZN`, `BTC-USD`.")

# -------------------------------------------------
# Load main data
# -------------------------------------------------
if not ticker:
    st.warning("Please enter a stock ticker.")
    st.stop()

try:
    raw_data = load_data(ticker, period)
except Exception as e:
    st.error(f"Failed to load data for {ticker}: {e}")
    st.stop()

if raw_data.empty:
    st.error("No data found. Check the ticker and try again.")
    st.stop()

model_df, feature_cols, enriched_data = prepare_model_data(raw_data, forecast_horizon)

if len(model_df) < 120:
    st.warning("Not enough processed data for a stable forecast. Try a longer historical period like 2y or 5y.")
    st.stop()

results = evaluate_model(model_df, feature_cols, model_name)
model = results["model"]
test_df = results["test_df"]
y_test = results["y_test"]
preds = results["preds"]

latest_feature_row = model_df[feature_cols].iloc[[-1]]
latest_prediction = forecast_latest(model, latest_feature_row)

latest_close = float(raw_data["Close"].iloc[-1])
forecast_change_pct = ((latest_prediction - latest_close) / latest_close) * 100

future_dates = next_business_days(pd.to_datetime(raw_data["Date"].iloc[-1]), forecast_horizon)
projected_path = np.linspace(latest_close, latest_prediction, num=len(future_dates))

latest_return = float(enriched_data["Return"].dropna().iloc[-1] * 100)
latest_rsi = float(enriched_data["RSI14"].dropna().iloc[-1])
latest_vol = float(enriched_data["Volatility20"].dropna().iloc[-1] * 100)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Technical Analysis", "Forecasting", "Comparison", "Methodology"]
)

# -------------------------------------------------
# Tab 1: Overview
# -------------------------------------------------
with tab1:
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.4])
    col1.metric("Latest Close", f"${latest_close:,.2f}")
    col2.metric("Daily Return", f"{latest_return:,.2f}%")
    col3.metric("RSI (14)", f"{latest_rsi:,.2f}")
    col4.metric("Volatility", f"{latest_vol:,.2f}%")
    col5.metric(f"{forecast_horizon}-Day Forecast", f"${latest_prediction:,.2f}", f"{forecast_change_pct:,.2f}%")

    st.plotly_chart(make_price_chart(enriched_data, ticker), use_container_width=True)

    st.subheader("Recent Market Data")
    display_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI14"] if c in enriched_data.columns]
    st.dataframe(enriched_data[display_cols].tail(15), use_container_width=True)

# -------------------------------------------------
# Tab 2: Technical Analysis
# -------------------------------------------------
with tab2:
    left, right = st.columns(2)

    with left:
        st.plotly_chart(make_rsi_chart(enriched_data), use_container_width=True)

    with right:
        st.plotly_chart(make_macd_chart(enriched_data), use_container_width=True)

    st.subheader("Indicator Snapshot")

    latest_ma20 = float(enriched_data["MA20"].dropna().iloc[-1])
    latest_ma50 = float(enriched_data["MA50"].dropna().iloc[-1])
    latest_bb_upper = float(enriched_data["BB_Upper"].dropna().iloc[-1])
    latest_bb_lower = float(enriched_data["BB_Lower"].dropna().iloc[-1])

    ind1, ind2, ind3, ind4 = st.columns(4)
    ind1.metric("MA20", f"${latest_ma20:,.2f}")
    ind2.metric("MA50", f"${latest_ma50:,.2f}")
    ind3.metric("BB Upper", f"${latest_bb_upper:,.2f}")
    ind4.metric("BB Lower", f"${latest_bb_lower:,.2f}")

    st.markdown(
        """
        **Reading the indicators**

        - **RSI above 70** may indicate overbought conditions.  
        - **RSI below 30** may indicate oversold conditions.  
        - **MACD crossing above signal** can suggest positive momentum.  
        - **Price near Bollinger upper band** can indicate strength or possible overextension.  
        """
    )

# -------------------------------------------------
# Tab 3: Forecasting
# -------------------------------------------------
with tab3:
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Model", model_name)
    metric2.metric("MAE", f"{results['mae']:,.3f}")
    metric3.metric("RMSE", f"{results['rmse']:,.3f}")
    metric4.metric("R²", f"{results['r2']:,.3f}")

    metric5, metric6 = st.columns(2)
    metric5.metric("MAPE", f"{results['mape']:,.2f}%")
    metric6.metric(
        f"Predicted Close in {forecast_horizon} Trading Days",
        f"${latest_prediction:,.2f}",
        f"{forecast_change_pct:,.2f}% vs current",
    )

    st.plotly_chart(make_actual_vs_pred_chart(test_df, y_test, preds), use_container_width=True)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Projected Close": projected_path,
    })

    st.subheader("Projected Path")
    st.dataframe(forecast_df, use_container_width=True)

    proj_fig = go.Figure()
    proj_fig.add_trace(
        go.Scatter(
            x=enriched_data["Date"].tail(60),
            y=enriched_data["Close"].tail(60),
            mode="lines",
            name="Historical Close",
        )
    )
    proj_fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Projected Close"],
            mode="lines+markers",
            name="Projected Path",
        )
    )
    proj_fig.update_layout(
        title="Historical Close with Forward Projection",
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(proj_fig, use_container_width=True)

    importance_fig = make_feature_importance_chart(model, feature_cols)
    if importance_fig is not None:
        st.plotly_chart(importance_fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for Linear Regression in this dashboard view.")

# -------------------------------------------------
# Tab 4: Comparison
# -------------------------------------------------
with tab4:
    comparison_tickers = [x.strip().upper() for x in comparison_input.split(",") if x.strip()]
    comparison_tickers = list(dict.fromkeys([ticker] + comparison_tickers))[:5]

    comp_df = load_comparison_data(comparison_tickers, period)

    if comp_df.empty:
        st.warning("Comparison data could not be loaded.")
    else:
        comp_df = comp_df.sort_values(["Ticker", "Date"]).copy()
        comp_df["Normalized"] = comp_df.groupby("Ticker")["Close"].transform(lambda s: (s / s.iloc[0]) * 100)

        fig_norm = px.line(
            comp_df,
            x="Date",
            y="Normalized",
            color="Ticker",
            title="Normalized Performance Comparison (Base = 100)",
        )
        fig_norm.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_norm, use_container_width=True)

        summary_rows = []
        for t in comparison_tickers:
            sub = comp_df[comp_df["Ticker"] == t].copy()
            if sub.empty:
                continue

            total_return = ((sub["Close"].iloc[-1] / sub["Close"].iloc[0]) - 1) * 100
            daily_vol = sub["Close"].pct_change().std() * np.sqrt(252) * 100

            summary_rows.append({
                "Ticker": t,
                "Total Return (%)": round(total_return, 2),
                "Annualized Volatility (%)": round(daily_vol, 2),
                "Latest Close": round(float(sub["Close"].iloc[-1]), 2),
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("Total Return (%)", ascending=False)
        st.subheader("Comparison Summary")
        st.dataframe(summary_df, use_container_width=True)

# -------------------------------------------------
# Tab 5: Methodology
# -------------------------------------------------
with tab5:
    st.markdown(
        f"""
        ### Project methodology

        This dashboard combines:

        - **Live market data** from Yahoo Finance via `yfinance`
        - **Technical indicators** such as RSI, MACD, moving averages, and Bollinger Bands
        - **Machine learning models** from `scikit-learn`
        - **Interactive web visualization** with Streamlit and Plotly

        ### Forecast target

        The selected model predicts the **closing price {forecast_horizon} trading day(s) ahead**.

        ### Features used by the model

        - lagged closing prices
        - daily returns
        - moving averages
        - rolling volatility
        - RSI
        - MACD and MACD signal
        - Bollinger Band width
        - volume change

        ### Model evaluation

        The dataset is split chronologically:

        - **80% training**
        - **20% testing**

        Metrics displayed:

        - **MAE**: average absolute prediction error
        - **RMSE**: larger errors are penalized more heavily
        - **R²**: goodness of fit
        - **MAPE**: percentage-based average error

        ### Important limitation

        This is a **portfolio and educational analytics project**, not a production trading system.  
        Financial markets are noisy and difficult to predict reliably, and past price behavior does not guarantee future performance.

        ### Tech stack

        - Python
        - Streamlit
        - yfinance
        - pandas / numpy
        - Plotly
        - scikit-learn
        """
    )

st.info(
    "Disclaimer: This app is for educational, analytical, and portfolio purposes only. "
    "It does not provide financial advice or guaranteed investment predictions."
)