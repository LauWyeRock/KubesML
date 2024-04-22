from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import sys

df = pd.read_csv("df_smoothed.csv")
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df['hour'] = df['ds'].dt.hour
df['day_of_week'] = df['ds'].dt.dayofweek
df['day_of_year'] = df['ds'].dt.dayofyear

X = df[['hour', 'day_of_week', 'day_of_year']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

