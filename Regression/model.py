import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load dataset
df = pd.read_csv("data/train.csv")

# 2. Keep numeric columns only
df = df.select_dtypes(include=["number"])

# 3. Handle missing values
df = df.fillna(df.median())

# 4. Features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Show sample predictions
print("Sample Predictions:", y_pred[:5])

# 9. Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# 10. Save trained model
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully!")