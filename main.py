import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('synthetic_traffic_data - synthetic_traffic_data.csv')

# Traffic data exploration
print("Traffic Data Info:")
print(data.info())
print(data.describe())



# Check for missing values
print("\nMissing Values in Traffic Data:")
print(data.isnull().sum())

data = data.sort_values(by='timestamp')
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
# Adding time-based features to traffic data
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month


# One-hot encoding day of the week
data = pd.get_dummies(data, columns=['day_of_week'], prefix='weekday')


# Fill missing values or drop rows with missing data
data = data.dropna()

# Add time-based features
data['is_weekend'] = data['timestamp'].dt.dayofweek >= 5  # Weekend: Saturday (5) and Sunday (6)
data['is_rush_hour'] = data['hour'].isin([7, 8, 9, 16, 17, 18])  # Morning and evening rush hours

# Convert boolean features to integers for modeling
data['is_weekend'] = data['is_weekend'].astype(int)
data['is_rush_hour'] = data['is_rush_hour'].astype(int)

# Normalize temperature
scaler = MinMaxScaler()
weather_features = ['temperature','humidity']
data[weather_features] = scaler.fit_transform(data[weather_features])


# Create lag features
data['traffic_lag_1'] = data['traffic_flow'].shift(10)  # Previous hour
data['traffic_lag_24'] = data['traffic_flow'].shift(240)  # Previous day

# Drop rows with NaN values introduced by lagging
data = data.dropna()

# Normalize traffic volume and lag features
data[['traffic_flow', 'traffic_lag_1', 'traffic_lag_24']] = scaler.fit_transform(
    data[['traffic_flow', 'traffic_lag_1', 'traffic_lag_24']]
)
bool_columns = [col for col in data.columns if data[col].dtype == 'bool']
data[bool_columns] = data[bool_columns].astype(int)





print(data.head())

from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = data.drop(['traffic_flow', 'timestamp'], axis=1)  # Features
y = data['traffic_flow']  # Target

# Perform time-aware train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create a validation split from the training data
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# Reshape input data for LSTM (samples, timesteps, features)
# Assume X_train_final, X_val, and X_test have already been prepared
X_train_final_reshaped = X_train_final.values.reshape((X_train_final.shape[0], 1, X_train_final.shape[1]))
X_val_reshaped = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.dtypes)  # All columns should have numeric types (e.g., int64, float64)

# Build the LSTM model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(1, X_train_final.shape[1])),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Single output for regression
])

# Model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])



# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_final_reshaped, y_train_final,
    validation_data=(X_val_reshaped, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# Plot training history


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test_reshaped, y_test)

print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")


test_rmse = np.sqrt(test_loss)
print(f"Test Root Mean Squared Error (RMSE): {test_rmse}")

# Example: Make predictions on the test set
predictions = model.predict(X_test_reshaped)

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Traffic flow', alpha=0.7)
plt.plot(predictions, label='Predicted Traffic flow', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Traffic Volume')
plt.xlabel('Time Step')
plt.ylabel('Traffic flow')
plt.show()


new_data = pd.DataFrame({
    'intersection_id': [2],
    'temperature': [22.5],
    'humidity': [60],
    'rain': [0],
    'hour': [14],
    'month': [1],
    'weekday_0': [0],
    'weekday_1': [1],
    'weekday_2': [0],
    'weekday_3': [0],
    'weekday_4': [0],
    'weekday_5': [0],
    'weekday_6': [0],
    'is_weekend': [0],
    'is_rush_hour': [1],
    'traffic_lag_1': [25],
    'traffic_lag_24': [30]
})

# Example: Predict traffic for future periods (new_data is preprocessed future input features)
new_data_reshaped = new_data.values.reshape((new_data.shape[0], 1, new_data.shape[1]))
future_predictions = model.predict(new_data_reshaped)

# Display predictions
print("Future Traffic Predictions:")
print(future_predictions)

#model.save('traffic_forecasting_model.h5')









