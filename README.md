# MarketLens: Stock Analysis and Forecasting Dashboard

MarketLens is an interactive stock analysis and forecasting dashboard built with Python and Streamlit. It allows users to explore stock price trends, apply technical indicators, and generate short-term predictions using machine learning models.

---

## Features

- Interactive candlestick price charts
- Technical indicators:
  - Moving Averages (MA20, MA50, MA100)
  - RSI (Relative Strength Index)
  - MACD (Momentum indicator)
  - Bollinger Bands
- Machine learning forecasting:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Model evaluation metrics:
  - MAE, RMSE, R², MAPE
- Multi-stock comparison dashboard
- Interactive visualizations using Plotly

---

## Tech Stack

- Python
- Streamlit
- yfinance
- pandas & numpy
- Plotly
- scikit-learn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/marketlens-dashboard.git
cd marketlens-dashboard

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py
Usage
Enter a stock ticker (e.g., AAPL, TSLA, MSFT)
Select a time period and forecast horizon
Choose a machine learning model
Explore technical indicators and forecasts across different tabs
Compare multiple stocks in the comparison section
Project Structure
marketlens-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
Disclaimer

This project is for educational and portfolio purposes only.
It does not provide financial advice or guarantee prediction accuracy.
Stock markets are highly volatile and unpredictable.

Future Improvements
Deploy as a live web application
Add deep learning models (LSTM)
Include news sentiment analysis
Enhance UI/UX design
Author

Developed as a data science and financial analytics project using Python and Streamlit.
