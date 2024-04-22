from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

df = pd.read_csv("df_smoothed.csv")
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=len(test), include_history=False)
forecast = model.predict(future)

test['yhat'] = forecast['yhat'].values
rmse = np.sqrt(mean_squared_error(test['y'], test['yhat']))
print(f"RMSE: {rmse}")
