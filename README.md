Traffic Flow Forecasting with LSTM
This project predicts traffic flow using an LSTM (Long Short-Term Memory) neural network. It processes traffic data, including weather and time-based features, to forecast future traffic volumes.

Requirements
Install the required libraries:

pip install pandas numpy scikit-learn keras tensorflow matplotlib
Dataset
The dataset (synthetic_traffic_data.csv) includes traffic flow, temperature, humidity, and time-based features such as hour and day of the week.

Code Overview
Data Preprocessing: The data is cleaned, missing values are handled, and time-based features (like hour, day of the week) are extracted. Lag features are also created for historical traffic data.

Model Building: An LSTM model with two LSTM layers and Dropout layers is used to predict traffic flow. The model is trained with early stopping to avoid overfitting.

Model Evaluation: The model's performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

Prediction: The trained model is used to predict future traffic flow based on new input data.

