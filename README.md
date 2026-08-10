# Gold Stock Price Analysis & Prediction 📈

A complete mini data-science project using the provided `gold_stock.csv` dataset.

## Objective
Analyze historical gold market prices and build a machine-learning model to predict the daily closing price using historical price, volume, lag and rolling-window features.

## Dataset
- Source file: `gold_stock.csv`
- Raw rows: 2,970
- Clean market rows: 2,968
- Columns: 6
- Date range: 2014-03-17 to 2025-12-31
- Target: `Close`

The input CSV contains two Yahoo Finance metadata rows (`Ticker` and `Date`). The project automatically removes these rows and converts the market columns to numeric types.

## Data Cleaning
- Removed metadata rows.
- Converted `Date` to datetime.
- Converted OHLCV columns to numeric.
- Removed invalid/missing market rows.
- Sorted observations chronologically.
- Created lag and rolling features.

## EDA
The notebook includes:
- Closing Price Trend
- Daily Trading Range
- Trading Volume
- Moving Averages (7-day and 30-day)
- Daily Return Distribution
- Closing Price vs Volume
- Correlation Heatmap

## Feature Engineering
Created:
- Daily Return
- Daily Range
- 7-day Moving Average
- 30-day Moving Average
- 7-day Volatility
- 1-day Close Lag
- 7-day Close Lag
- 7-day Volume Moving Average
- Year, Month and Day of Week

## Machine Learning
**Model:** Random Forest Regression

**Target:** Daily `Close`

**Split:** Chronological 80% training / 20% testing

**Metrics:**
- MAE: 0.5570
- RMSE: 1.1127
- R²: 0.9606

> Note: This is an educational forecasting project, not financial advice or a trading system.

## Power BI
Use `data/gold_stock_cleaned.csv` in Power BI to create:
- Current/Latest Close KPI
- Average Close
- Highest Close
- Lowest Close
- Closing Price Trend
- 7-day vs 30-day Moving Average
- Volume Trend
- Monthly Average Close
- Daily Range Analysis
- Date slicer

## Project Structure
```text
gold-stock-analysis-project/
├── data/
│   ├── gold_stock.csv
│   └── gold_stock_cleaned.csv
├── notebooks/
│   └── gold_stock_analysis.ipynb
├── gold_stock_analysis.py
├── requirements.txt
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
jupyter notebook
```

Open:
```text
notebooks/gold_stock_analysis.ipynb
```

Or run:
```bash
python gold_stock_analysis.py
```

## Conclusion
The project demonstrates a complete workflow from raw market data cleaning to EDA, feature engineering, visualization and machine-learning prediction of gold closing prices.
