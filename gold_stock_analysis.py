from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / "data" / "gold_stock.csv")

# Yahoo Finance export contains two metadata rows
df = raw.copy()
df["Date"] = df["Price"]
df = df.iloc[2:].copy()

for c in ["Close", "High", "Low", "Open", "Volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna().sort_values("Date").reset_index(drop=True)

# Feature engineering
df["Return"] = df["Close"].pct_change()
df["Daily_Range"] = df["High"] - df["Low"]
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_30"] = df["Close"].rolling(30).mean()
df["Volatility_7"] = df["Return"].rolling(7).std()
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Lag_1"] = df["Close"].shift(1)
df["Lag_7"] = df["Close"].shift(7)
df["Volume_MA_7"] = df["Volume"].rolling(7).mean()

features = [
    "Lag_1","Lag_7","MA_7","MA_30","Volatility_7","Volume_MA_7",
    "High","Low","Open","Volume","Daily_Range","Year","Month","DayOfWeek"
]
model_df = df.dropna().copy()

split = int(len(model_df) * 0.80)
X_train, X_test = model_df[features].iloc[:split], model_df[features].iloc[split:]
y_train, y_test = model_df["Close"].iloc[:split], model_df["Close"].iloc[split:]

model = RandomForestRegressor(
    n_estimators=120, max_depth=10, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Clean dataset shape:", df.shape)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("MAE :", round(mean_absolute_error(y_test, pred), 4))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 4))
print("R2  :", round(r2_score(y_test, pred), 4))
