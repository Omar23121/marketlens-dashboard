# MarketLens: Stock Analysis and Forecasting Dashboard
MarketLens is an interactive stock analysis and forecasting dashboard built with Python and Streamlit. It allows users to explore stock price trends, apply technical indicators, and generate short-term predictions using machine learning models.

## Features
- Interactive candlestick price charts
- Technical indicators: Moving Averages (MA20, MA50, MA100), RSI, MACD, Bollinger Bands
- Machine learning models: Linear Regression, Random Forest, Gradient Boosting
- Model evaluation: MAE, RMSE, R², MAPE
- Multi-stock comparison dashboard
- Interactive visualizations using Plotly

## Tech Stack
- Python
- Streamlit
- yfinance
- pandas & numpy
- Plotly
- scikit-learn

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/marketlens-dashboard.git
cd marketlens-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Usage
- Enter a stock ticker (e.g., AAPL, TSLA, MSFT)
- Select time period and forecast horizon
- Choose a model
- Explore analysis and forecasts across tabs
- Compare multiple stocks

## Project Structure
```bash
marketlens-dashboard/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Disclaimer
This project is for educational and portfolio purposes only. It does not provide financial advice or guarantee prediction accuracy.

## Future Improvements
- Deploy as live web app
- Add deep learning models (LSTM)
- Add news sentiment analysis
- Improve UI/UX

## Author
Developed as a data science and financial analytics project using Python and Streamlit.
