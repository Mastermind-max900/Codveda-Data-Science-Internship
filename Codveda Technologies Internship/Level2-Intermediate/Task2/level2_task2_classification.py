import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD THE DATA
# No header=None here because file HAS headers
df = pd.read_csv('1) iris.csv')

# 2. PREPROCESS (Objective 1)
# Using the exact column names from the dataset
X = df.drop('species', axis=1) 
y = df['species']

# Feature Scaling: Essential for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. TRAIN MODELS (Objective 2 & 4)
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 5. EVALUATE (Objective 3)
print("--- Accuracy Comparison ---")
print(f"Logistic Regression: {accuracy_score(y_test, lr_preds):.4f}")
print(f"Random Forest: {accuracy_score(y_test, rf_preds):.4f}")

print("\n--- Detailed Classification Report (Random Forest) ---")
print(classification_report(y_test, rf_preds))

# 6. VISUALIZE WITH CONFUSION MATRIX
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)
plt.title('Iris Species Classification: Confusion Matrix')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.show()