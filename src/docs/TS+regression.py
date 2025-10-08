#%% imports
import numpy as np
import seaborn as sns
import statsmodels.api as sm
# Corrected Import: Import ARIMA directly from statsmodels.tsa.arima.model
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score

# Configure settings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')


#%%

# Load the original data
df = pd.read_csv('raw_data/sales.csv')
print("--- Original DataFrame Info ---")
df.info()
print("\n--- Original DataFrame Head ---")
print(df.head())

# Convert 'date' to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Set 'date' as the index
df.set_index('date', inplace=True)

# Sort the index to ensure chronological order
df.sort_index(inplace=True)

# Check for missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

target_variable = 'total'

start_date_plot = df.index.min()
end_date_plot = df.index.max()

# print(f"\nPlotting data from {start_date_plot.date()} to {end_date_plot.date()}")
# plt.figure(figsize=(14, 5))
# sns.lineplot(data=df, x=df.index, y='total')
# plt.title("Time Series Plot of Total Food Sales")
# plt.xlabel("Date")
# plt.ylabel("Total Food Sales")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ACF/PACF
fig, axes = plt.subplots(2, 1, figsize=(16, 5))
plot_acf(df[target_variable], lags=20, ax=axes[0], title=f'ACF of {target_variable}')
plot_pacf(df[target_variable], lags=20, ax=axes[1], title=f'PACF of {target_variable}')
plt.tight_layout()
plt.show()


# Encoding for potential independent variable
non_categorical_variables= ['school_code', 'item','free_meals', 'reduced_price_meals','full_price_meals','adults','alac_adult','alac_student','earned_student','earned_adult','earned_alac_student','earned_alac_adult','adj_alac','adj_meal']
categorical_variable = ['time_of_day']
#note that 'time_of_day' needs to be encoded
# Encoding 'time_of_day'
time_mapping = {'breakfast': 0, 'lunch': 1}

# Apply the mapping
df['time_of_day_encoded'] = df['time_of_day'].map(time_mapping)
independent_variables_for_corr = non_categorical_variables + ['time_of_day_encoded']

numerical_cols = [target_variable] + independent_variables_for_corr
df_encoded = df[independent_variables_for_corr + [target_variable]]

correlation_matrix = df_encoded.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()




#%% Split data
# Define the split point (80% of the data)
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

# Split the data

train_data = df_encoded.iloc[:split_index]
test_data = df_encoded.iloc[split_index:]

print(f"\n--- Data Splitting ---")
print(f"Total observations: {len(df_encoded)}")
print(f"Training set size: {len(train_data)} ({split_ratio*100:.0f}%)")
print(f"Test set size: {len(test_data)} ({(1-split_ratio)*100:.0f}%)")
# print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
# print(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")
num_training_days = len(train_data)
print(f"Number of data points in training set: {num_training_days}")

# %%
# Define function for stationarity tests
def check_stationarity(timeseries, series_name=""):
    """Performs ADF and KPSS tests and prints the results."""
    print(f'\n--- Stationarity Tests for {series_name} ---')
    # ADF Test
    adf_result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')
    if adf_result[1] <= 0.05:
        print("Result: Strong evidence against the null hypothesis (H0), reject H0. Data is likely stationary.")
    else:
        print("Result: Weak evidence against null hypothesis, fail to reject H0. Data is likely non-stationary.")

    print('\n')
    # KPSS Test
    kpss_result = kpss(timeseries, regression='c', nlags="auto") # 'c' for constant trend
    print(f'KPSS Statistic: {kpss_result[0]:.4f}')
    print(f'p-value: {kpss_result[1]:.4f}')
    print('Critical Values:')
    for key, value in kpss_result[3].items():
        print(f'\t{key}: {value:.4f}')
    if kpss_result[1] <= 0.05:
        print("Result: Strong evidence against the null hypothesis (H0), reject H0. Data is likely non-stationary.")
    else:
        print("Result: Weak evidence against null hypothesis, fail to reject H0. Data is likely stationary.")
    print('--- End Stationarity Tests ---')


# %%
# Plot rolling mean and variance for the raw training data
window_size = 30
rolling_mean = train_data[target_variable].rolling(window=window_size).mean()
rolling_std = train_data[target_variable].rolling(window=window_size).std()
rolling_var = rolling_std**2

plt.figure(figsize=(15, 9)) # Adjusted figure size for two plots

# Rolling Mean
plt.subplot(2, 1, 1)
plt.plot(train_data[target_variable], label='Original Training Data', color='blue', alpha=0.7)
plt.plot(rolling_mean, label=f'Rolling Mean (w={window_size} hours / ~30 days)', color='red', linewidth=2)
plt.title(f'{target_variable} and Rolling Statistics (Training Data)')
plt.ylabel('Total_Food_Sales')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
# Hide x-axis labels for the top plot to avoid overlap
plt.setp(plt.gca().get_xticklabels(), visible=False)
# Rolling Var
plt.subplot(2, 1, 2)
plt.plot(rolling_var, label=f'Rolling Variance (w={window_size} hours / ~30 days)', color='orange', linewidth=2)
plt.ylabel('Variance')
plt.xlabel('Date')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout and display
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap slightly
plt.show()

check_stationarity(train_data[target_variable], f"Raw {target_variable}")

# Apply first-order differencing
train_data_diff = train_data[target_variable].diff().dropna()

# Plot rolling mean and variance for the differenced training data
window_size = 24 * 30
rolling_mean_diff = train_data_diff.rolling(window=window_size).mean()
rolling_std_diff = train_data_diff.rolling(window=window_size).std()
rolling_var_diff = rolling_std_diff**2

plt.figure(figsize=(15, 9)) # Adjusted figure size for two plots

plt.subplot(2, 1, 1)
plt.plot(train_data_diff, label='Differenced Training Data', color='blue', alpha=0.7)
plt.plot(rolling_mean_diff, label=f'Differenced Rolling Mean (w={window_size} hours / ~30 days)', color='red', linewidth=2)
plt.title(f'{target_variable} and Differenced Rolling Statistics (Training Data)')
plt.ylabel('Total_Food_Sales')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
# Hide x-axis labels for the top plot to avoid overlap
plt.setp(plt.gca().get_xticklabels(), visible=False)

# Rolling Var
plt.subplot(2, 1, 2)
plt.plot(rolling_var_diff, label=f'Differenced Rolling Variance (w={window_size} hours / ~30 days)', color='orange', linewidth=2)
plt.ylabel('Variance')
plt.xlabel('Date')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)

#display
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap slightly
plt.show()

# Perform stationarity tests on the differenced training data
check_stationarity(train_data_diff, f"Differenced {target_variable}")

fig, axes = plt.subplots(2, 1, figsize=(16, 5))
plot_acf(train_data_diff, lags=20, ax=axes[0], title=f'ACF of Differenced {target_variable}')
plot_pacf(train_data_diff, lags=20, ax=axes[1], title=f'PACF of Differenced {target_variable}')
plt.tight_layout()
plt.show()


##########################################################################################
##########################################################################################
##########################################################################################

# %% Basic Regression Model

print("\n--- Building a Basic Regression Model ---")

# Define dependent and independent variables
Y_train = train_data[target_variable]
X_train = train_data[independent_variables_for_corr] # Using the previously defined independent variables

Y_test = test_data[target_variable]
X_test = test_data[independent_variables_for_corr]

# Add a constant to the independent variables for the intercept term
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


#####
#####

# %% AR(1) Model for Differenced Time Series

print("\n--- Building an AR(1) Model for the Differenced Training Data ---")

# Define the order for AR(1) on the differenced series: (p=1, d=0, q=0)
# We use d=0 because train_data_diff is already differenced.
# If we were to model the original series, it would be ARIMA(1,1,0) for an AR(1) on first-differenced data.
order_ar1 = (1, 0, 0)  # (AR order, Differencing order, MA order)

# Fit the AR(1) model
# Using the statsmodels ARIMA model, which is versatile.
# We fit it on the 'train_data_diff' which is the differenced 'total' series.
try:
    ar1_model = sm.tsa.arima.model.ARIMA(train_data_diff, order=order_ar1)
    ar1_results = ar1_model.fit()

    # Print the model summary
    print(ar1_results.summary())

    # --- Forecasting with AR(1) ---
    print("\n--- Forecasting with AR(1) Model ---")

    # Get the last observation from the training data (original 'total' series)
    last_train_value = train_data[target_variable].iloc[-1]


    # A simpler approach for step-by-step forecasting and converting back to original scale:
    history = [x for x in train_data[target_variable]]  # Original training data
    predictions_ar1 = list()


    # We iterate through the test set to make step-ahead predictions
    # This simulates how the model would be used in production for true out-of-sample forecasting
    for t in range(len(test_data)):

        # Let's forecast for the entire test period in one go for the differenced series
        forecast_diff_steps = len(test_data)
        forecast_results = ar1_results.get_forecast(steps=forecast_diff_steps)
        forecast_diff_values = forecast_results.predicted_mean

        # Convert differenced forecasts back to the original scale
        # We start with the last observed value from the training data
        last_observed_value = train_data[target_variable].iloc[-1]

        # Initialize list for original scale predictions
        original_scale_predictions = []

        # Iterate through differenced forecasts to reconstruct original scale
        current_prediction_value = last_observed_value
        for diff_forecast in forecast_diff_values:
            # The next original value is the current original value + the forecasted difference
            current_prediction_value = current_prediction_value + diff_forecast
            original_scale_predictions.append(current_prediction_value)

        # Convert to pandas Series with the test_data index for easy plotting and evaluation
        ar1_test_predictions = pd.Series(original_scale_predictions, index=test_data.index)

        # Evaluate the model on the test set
        mse_ar1_test = ((ar1_test_predictions - Y_test) ** 2).mean()
        rmse_ar1_test = np.sqrt(mse_ar1_test)

        ss_total_ar1 = ((Y_test - Y_test.mean()) ** 2).sum()
        ss_residual_ar1 = ((Y_test - ar1_test_predictions) ** 2).sum()
        r_squared_ar1_test = 1 - (ss_residual_ar1 / ss_total_ar1) if ss_total_ar1 > 0 else 0

        print(f"\n--- AR(1) Model Test Set Evaluation (Differenced Series) ---")
        print(f"Mean Squared Error (MSE): {mse_ar1_test:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_ar1_test:.4f}")
        print(f"R-squared: {r_squared_ar1_test:.4f}")

        # Plotting Actual vs. Predicted values for the test set
        plt.figure(figsize=(14, 7))
        plt.plot(Y_test.index, Y_test, label='Actual Total Sales', color='blue', alpha=0.7)
        plt.plot(ar1_test_predictions.index, ar1_test_predictions, label='Predicted Total Sales (AR(1))', color='green',
                 linestyle='--', alpha=0.7)
        plt.title('AR(1) Model: Actual vs. Predicted Total Sales (Test Set)')
        plt.xlabel('Date')
        plt.ylabel('Total Food Sales')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Residual Plot for AR(1)
        residuals_ar1 = Y_test - ar1_test_predictions
        plt.figure(figsize=(14, 7))
        plt.scatter(ar1_test_predictions, residuals_ar1, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residual Plot (AR(1) Model - Test Set)')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


except Exception as e:
    print(f"An error occurred while fitting or forecasting with the AR(1) model: {e}")
    print("Please ensure 'train_data_diff' has sufficient length and variance for model fitting.")


################
################
################
# %% LSTM Analysis for Time Series Forecasting

print("\n--- Starting LSTM Analysis ---")

# --- 1. Data Preparation for LSTM ---

# Select the target variable for LSTM
lstm_data = df[[target_variable]]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(lstm_data)

# Split scaled data into training and testing sets
# We use the same split point as before to ensure consistency.
train_scaled = scaled_data[:split_index]
test_scaled = scaled_data[split_index:]

print(f"Scaled Training data shape: {train_scaled.shape}")
print(f"Scaled Test data shape: {test_scaled.shape}")

# Define a function to create sequences for LSTM
def create_sequences(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30 # Number of previous time steps to use as input features to predict the next time step

# Create sequences for training and testing
X_train_lstm, Y_train_lstm = create_sequences(train_scaled, look_back)
X_test_lstm, Y_test_lstm = create_sequences(test_scaled, look_back)

# Reshape input to be [samples, timesteps, features] for LSTM
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

print(f"LSTM Training X shape: {X_train_lstm.shape}, Y shape: {Y_train_lstm.shape}")
print(f"LSTM Test X shape: {X_test_lstm.shape}, Y shape: {Y_test_lstm.shape}")


# --- 2. LSTM Model Architecture ---

print("\n--- Building LSTM Model ---")
lstm_model = Sequential()
lstm_model.add(LSTM(20, return_sequences=True, input_shape=(look_back, 1))) # 50 units, return sequences for next LSTM layer
lstm_model.add(LSTM(20, return_sequences=False)) # 50 units, don't return sequences for final dense layer
lstm_model.add(Dense(1)) # Output layer with 1 neuron for single value prediction

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.summary()


# --- 3. Training the LSTM Model ---

print("\n--- Training LSTM Model (This may take a while) ---")
history = lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=20, batch_size=16, verbose=1, shuffle=False)

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('LSTM Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()


# --- 4. Forecasting and Evaluation ---

print("\n--- Forecasting with LSTM Model ---")

# Make predictions on the scaled test data
lstm_predictions_scaled = lstm_model.predict(X_test_lstm)

# Inverse transform predictions to original scale
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
Y_test_original = scaler.inverse_transform(Y_test_lstm.reshape(-1, 1)) # Reshape for inverse_transform

# Ensure the index aligns with the original test data
# The first 'look_back' values of the test set are used as input, so predictions start from 'look_back' index onward.
# This means our LSTM predictions will align with `test_data.index[look_back:]`
lstm_test_index = test_data.index[look_back:]

# Convert predictions and actuals to pandas Series for consistent evaluation and plotting
lstm_predictions_series = pd.Series(lstm_predictions.flatten(), index=lstm_test_index)
Y_test_original_series = pd.Series(Y_test_original.flatten(), index=lstm_test_index)


# Evaluate the LSTM model on the test set
mse_lstm_test = mean_squared_error(Y_test_original_series, lstm_predictions_series)
rmse_lstm_test = np.sqrt(mse_lstm_test)
r_squared_lstm_test = r2_score(Y_test_original_series, lstm_predictions_series)

print(f"\n--- LSTM Model Test Set Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_lstm_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lstm_test:.4f}")
print(f"R-squared: {r_squared_lstm_test:.4f}")


# Plotting Actual vs. Predicted values for the test set (LSTM)
plt.figure(figsize=(14, 7))
plt.plot(Y_test_original_series.index, Y_test_original_series, label='Actual Total Sales', color='blue', alpha=0.7)
plt.plot(lstm_predictions_series.index, lstm_predictions_series, label='Predicted Total Sales (LSTM)', color='purple', linestyle='--', alpha=0.7)
plt.title('LSTM Model: Actual vs. Predicted Total Sales (Test Set)')
plt.xlabel('Date')
plt.ylabel('Total Food Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual Plot for LSTM
residuals_lstm = Y_test_original_series - lstm_predictions_series
plt.figure(figsize=(14, 7))
plt.scatter(lstm_predictions_series, residuals_lstm, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (LSTM Model - Test Set)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- LSTM Analysis Complete ---")
print("The LSTM model provides a non-linear approach to time series forecasting.")
print("Its performance can often capture complex patterns that linear models like AR(1) might miss.")
print("However, LSTMs require more data and computational resources for training.")
print("Comparing the evaluation metrics (MSE, RMSE, R-squared) of OLS, AR(1), and LSTM")
print("can help determine which model best captures the underlying patterns in the sales data.")
################
################
################


# Create and fit the OLS model
ols_model = sm.OLS(Y_train, X_train_sm)
ols_results = ols_model.fit()

# Print the model summary
print(ols_results.summary())

# Make predictions on the training set
train_predictions = ols_results.predict(X_train_sm)

# Evaluate the model on the training set
mse_train = ((train_predictions - Y_train)**2).mean()
rmse_train = np.sqrt(mse_train)
r_squared_train = ols_results.rsquared

print(f"\n--- Training Set Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_train:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_train:.4f}")
print(f"R-squared: {r_squared_train:.4f}")


# Make predictions on the test set
test_predictions = ols_results.predict(X_test_sm)

# Evaluate the model on the test set
mse_test = ((test_predictions - Y_test)**2).mean()
rmse_test = np.sqrt(mse_test)

# Calculate R-squared for the test set manually (statsmodels summary provides R-squared for training data)
ss_total = ((Y_test - Y_test.mean())**2).sum()
ss_residual = ((Y_test - test_predictions)**2).sum()
r_squared_test = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

print(f"\n--- Test Set Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"R-squared: {r_squared_test:.4f}")


# Plotting Actual vs. Predicted values for the test set
plt.figure(figsize=(14, 7))
plt.plot(Y_test.index, Y_test, label='Actual Total Sales', color='blue', alpha=0.7)
plt.plot(X_test.index, test_predictions, label='Predicted Total Sales (OLS)', color='red', linestyle='--', alpha=0.7)
plt.title('OLS Regression: Actual vs. Predicted Total Sales (Test Set)')
plt.xlabel('Date')
plt.ylabel('Total Food Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual Plot
residuals = Y_test - test_predictions
plt.figure(figsize=(14, 7))
plt.scatter(test_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (OLS Regression - Test Set)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Regression Model Interpretation ---")
print("From the OLS Regression Results (Summary Table):")
print(" - **R-squared:** Indicates the proportion of variance in the dependent variable ('total') that can be explained by the independent variables. A higher value suggests a better fit.")
print(" - **P-values (P>|t|):** For each independent variable, a p-value less than 0.05 (or other chosen significance level) suggests that the variable is statistically significant in predicting the dependent variable.")
print(" - **Coefficients (coef):** Indicate the change in the dependent variable for a one-unit increase in the independent variable, holding other variables constant.")
print(" - **F-statistic:** Tests the overall significance of the model. A low p-value (Prob (F-statistic)) suggests that at least one independent variable is useful in predicting the dependent variable.")
print("\nBased on the results, you can analyze which 'school_code', 'item', 'free_meals', 'reduced_price_meals', 'full_price_meals', 'alac_student', and 'time_of_day_encoded' significantly impact 'total' sales.")
print("The Test Set Evaluation provides an out-of-sample measure of the model's performance.")