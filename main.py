# stock_price_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download stock data
print("Downloading stock data...")
data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
print("Data download complete.\n")
print(data.head())

# Step 2: Select the 'Close' column for prediction
data = data[['Close']]
data['Target'] = data['Close'].shift(-1)  # Next day's price as target
data = data[:-1]  # Drop the last row with NaN target

# Step 3: Create features (X) and labels (y)
X = data[['Close']]
y = data['Target']

# Step 4: Split the data (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 5: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Step 8: Visualize the prediction vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.title('Stock Price Prediction using Linear Regression')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('stock_prediction_plot.png')  # Save plot as image
plt.show()

# Step 9: Save results to CSV
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_csv('prediction_results.csv', index=False)

print("\nScript completed successfully. Prediction plot and CSV saved.")
