import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# 1. LOAD DATA (FIXED FOR SPACE-SEPARATED VALUES)
# The sep='\s+' tells Python to split the columns by spaces
df = pd.read_csv('4) house Prediction Data Set.csv', header=None, sep='\s+')

# 2. NAME THE COLUMNS
df.columns = ['Price'] + [f'Feature_{i}' for i in range(1, df.shape[1])]

# 3. CREATE TIMELINE (Objective: Preprocessing)
df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='ME')
df.set_index('Date', inplace=True)

# 4. DECOMPOSITION (Objective 1)
decomposition = seasonal_decompose(df['Price'], model='additive', period=12)
decomposition.plot()
plt.suptitle('Objective 1: Time Series Decomposition', fontsize=16)
plt.tight_layout()
plt.show()

# 5. SMOOTHING & FORECASTING (Objectives 2 & 3)
model_es = ExponentialSmoothing(df['Price'], trend='add', seasonal='add', seasonal_periods=12).fit()
df['ES_Smoothing'] = model_es.fittedvalues
forecast = model_es.forecast(12)

# 6. EVALUATION & VISUALIZATION (Objective 4)
plt.figure(figsize=(12, 6))
plt.plot(df['Price'], label='Actual Price (History)', color='blue', alpha=0.4)
plt.plot(df['ES_Smoothing'], label='Model Fit', color='red')
plt.plot(forecast, label='12-Month Future Forecast', color='green', linestyle='--')
plt.title('Level 3 Task 1: Future Forecasting')
plt.legend()
plt.show()

# Final Metric
rmse = np.sqrt(mean_squared_error(df['Price'], df['ES_Smoothing']))
print(f"\n--- SUCCESS ---")
print(f"Model RMSE: {rmse:.2f}")