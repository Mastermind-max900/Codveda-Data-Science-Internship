import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD THE DATA
# Using sep='\s+' to handle the space-separated house data
df = pd.read_csv('4) house Prediction Data Set.csv', sep='\s+', header=None)

# 2. SELECT FEATURES AND TARGET
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# 3. SPLIT THE DATA (Objective 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL 1: LINEAR REGRESSION (Objective 2)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_preds = lin_model.predict(X_test)
r_lin = r2_score(y_test, lin_preds) # Defining r_lin here

# 5. MODEL 2: RANDOM FOREST (Objective 4)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
r_rf = r2_score(y_test, rf_preds) # Defining r_rf here

# 6. EVALUATE AND COMPARE (Objective 3)
print("--- Level 2 Task 1: Performance Comparison ---")
print(f"Linear Regression R2 Score: {r_lin:.4f}")
print(f"Random Forest R2 Score: {r_rf:.4f}")
print(f"Linear Regression MSE: {mean_squared_error(y_test, lin_preds):.2f}")

# 7. VISUALIZE
plt.figure(figsize=(10, 5))
plt.scatter(y_test, rf_preds, color='green', alpha=0.5, label='Random Forest')
plt.scatter(y_test, lin_preds, color='blue', alpha=0.3, label='Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs. Predicted: Linear Regression vs. Random Forest")
plt.legend()
plt.show()