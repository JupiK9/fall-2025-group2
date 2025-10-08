
#%% imports
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

#%%
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

start_date_plot = df.index.min()
end_date_plot = df.index.max()


#%%
target_variable = 'total'

# Group by date and sum the 'total' sales for each day
df_daily_total = df.groupby(df.index.date)[target_variable].sum().to_frame()
df_daily_total.index = pd.to_datetime(df_daily_total.index) # Convert index back to datetime
df_daily_total.rename(columns={'total': target_variable}, inplace=True)

# Plot the original time series
plt.figure(figsize=(14, 5))
sns.lineplot(data=df_daily_total, x=df_daily_total.index, y=target_variable)
plt.title(f"Time Series Plot of {target_variable} Food Sales (Daily Totals)")
plt.xlabel("Date")
plt.ylabel(f"{target_variable} Food Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ACF/PACF of the original series (for initial seasonality check at large lags)
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
plot_acf(df_daily_total[target_variable], lags=20, ax=axes[0], title=f'ACF of Original {target_variable} (Daily Totals, Lags 30)')
plot_pacf(df_daily_total[target_variable], lags=20, ax=axes[1], title=f'PACF of Original {target_variable} (Daily Totals, Lags 30)')
plt.tight_layout()
plt.show()
# # Look here for strong spikes at lag 7, 14, 21 etc. to initially confirm weekly seasonality.
#
#%% Split data
split_ratio = 0.8
split_index = int(len(df_daily_total) * split_ratio)
train_data_ar = df_daily_total[target_variable].iloc[:split_index]
test_data_ar = df_daily_total[target_variable].iloc[split_index:]

print(f"\n--- Data Splitting for AR Model ---")
print(f"Total observations: {len(df_daily_total)} (Daily Totals)")
print(f"Training set size: {len(train_data_ar)} ({split_ratio*100:.0f}%)")
print(f"Test set size: {len(test_data_ar)} ({(1-split_ratio)*100:.0f}%)")

# %% Define function for stationarity tests (Your function is good)
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
        print("Result: Strong evidence against the null hypothesis (H0), reject H0. Data is likely non-stationary (Trend/Unit Root detected).")
    else:
        print("Result: Weak evidence against null hypothesis, fail to reject H0. Data is likely stationary.")
    print('--- End Stationarity Tests ---')


# %% Stationarity analysis on the original training data
check_stationarity(train_data_ar, f"Raw {target_variable} (Daily Totals, Training Set)")

# Plot rolling mean and variance for the raw training data
window_size = 30 # Approx. one month
rolling_mean = train_data_ar.rolling(window=window_size).mean()
rolling_std = train_data_ar.rolling(window=window_size).std()

plt.figure(figsize=(15, 9))

plt.subplot(2, 1, 1)
plt.plot(train_data_ar, label='Original Training Data (Daily Totals)', color='blue', alpha=0.7)
plt.plot(rolling_mean, label=f'Rolling Mean (w={window_size} days)', color='red', linewidth=2)
plt.title(f'{target_variable} (Daily Totals) and Rolling Mean (Training Data)')
plt.ylabel(f'{target_variable} Sales')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.setp(plt.gca().get_xticklabels(), visible=False)

plt.subplot(2, 1, 2)
plt.plot(rolling_std**2, label=f'Rolling Variance (w={window_size} days)', color='orange', linewidth=2)
plt.ylabel('Variance')
plt.xlabel('Date')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# Apply first-order differencing
train_data_diff_1 = train_data_ar.diff().dropna()
print(f"\n--- First-Differenced Training Data Info ---")
print(f"Number of observations after first differencing: {len(train_data_diff_1)}")
print(train_data_diff_1.head())

# Perform stationarity tests on the first-differenced training data
check_stationarity(train_data_diff_1, f"First-Differenced {target_variable} (Daily Totals, Training Set)")

# Plot rolling mean and variance for the first-differenced training data
rolling_mean_diff_1 = train_data_diff_1.rolling(window=window_size).mean()
rolling_std_diff_1 = train_data_diff_1.rolling(window=window_size).std()

plt.figure(figsize=(15, 9))

plt.subplot(2, 1, 1)
plt.plot(train_data_diff_1, label='First-Differenced Training Data', color='blue', alpha=0.7)
plt.plot(rolling_mean_diff_1, label=f'First-Differenced Rolling Mean (w={window_size} days)', color='red', linewidth=2)
plt.title(f'{target_variable} (Daily Totals, First-Differenced) and Rolling Mean (Training Data)')
plt.ylabel(f'{target_variable} Sales (Differenced)')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.setp(plt.gca().get_xticklabels(), visible=False)

plt.subplot(2, 1, 2)
plt.plot(rolling_std_diff_1**2, label=f'First-Differenced Rolling Variance (w={window_size} days)', color='orange', linewidth=2)
plt.ylabel('Variance')
plt.xlabel('Date')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# ACF/PACF of the first-differenced series
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
plot_acf(train_data_diff_1, lags=20, ax=axes[0], title=f'ACF of First-Differenced {target_variable} (Daily Totals)')
plot_pacf(train_data_diff_1, lags=20, ax=axes[1], title=f'PACF of First-Differenced {target_variable} (Daily Totals)')
plt.tight_layout()
plt.show()
# IMPORTANT: Based on these plots, decide if d=1 is enough.
# If KPSS still suggests non-stationarity, proceed to d=2.

# If KPSS still fails (and plots show trend), apply second-order differencing
train_data_diff_2 = train_data_diff_1.diff().dropna() # This is already in your code, good.
print(f"\n--- Second-Differenced Training Data Info ---")
print(f"Number of observations after second differencing: {len(train_data_diff_2)}")
print(train_data_diff_2.head())

# Perform stationarity tests on the second-differenced training data
check_stationarity(train_data_diff_2, f"Second-Differenced {target_variable} (Daily Totals, Training Set)")

# Plot rolling mean and variance for the second-differenced training data
rolling_mean_diff_2 = train_data_diff_2.rolling(window=window_size).mean()
rolling_std_diff_2 = train_data_diff_2.rolling(window=window_size).std()

plt.figure(figsize=(15, 9))

plt.subplot(2, 1, 1)
plt.plot(train_data_diff_2, label='Second-Differenced Training Data', color='blue', alpha=0.7)
plt.plot(rolling_mean_diff_2, label=f'Second-Differenced Rolling Mean (w={window_size} days)', color='red', linewidth=2)
plt.title(f'{target_variable} (Daily Totals, Second-Differenced) and Rolling Mean (Training Data)')
plt.ylabel(f'{target_variable} Sales (Second-Differenced)')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.setp(plt.gca().get_xticklabels(), visible=False)

plt.subplot(2, 1, 2)
plt.plot(rolling_std_diff_2**2, label=f'Second-Differenced Rolling Variance (w={window_size} days)', color='orange', linewidth=2)
plt.ylabel('Variance')
plt.xlabel('Date')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# ACF/PACF of the second-differenced series - THIS IS WHERE YOU DETERMINE NON-SEASONAL P, Q
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
plot_acf(train_data_diff_2, lags=20, ax=axes[0], title=f'ACF of Second-Differenced {target_variable} (Daily Totals)')
plot_pacf(train_data_diff_2, lags=20, ax=axes[1], title=f'PACF of Second-Differenced {target_variable} (Daily Totals)')
plt.tight_layout()
plt.show()

# NOW, FOR SEASONAL ORDERS, look at ACF/PACF of the *latest stationary series* (e.g., train_data_diff_2)
# but with *longer lags* to capture seasonal spikes at 7, 14, 21 etc.
print("\n--- ACF/PACF for Seasonal Order Determination (Lags up to 30) ---")
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
plot_acf(train_data_diff_2, lags=20, ax=axes[0], title=f'ACF of Second-Differenced (for Seasonal Orders)')
plot_pacf(train_data_diff_2, lags=20, ax=axes[1], title=f'PACF of Second-Differenced (for Seasonal Orders)')
plt.tight_layout()
plt.show()
# Look for spikes at lags 7, 14, 21 etc. If these are significant,
# a seasonal difference (D=1) and/or seasonal AR/MA components (P, Q) are needed.

# If seasonal differencing is also required (i.e., strong seasonal spikes in ACF of train_data_diff_2):
# train_data_seasonally_diff = train_data_diff_2.diff(periods=7).dropna()
# check_stationarity(train_data_seasonally_diff, f"Second-Differenced and Seasonally Differenced {target_variable}")
# fig, axes = plt.subplots(2, 1, figsize=(16, 8))
# plot_acf(train_data_seasonally_diff, lags=30, ax=axes[0], title=f'ACF of Seasonally Differenced Data (for P, Q)')
# plot_pacf(train_data_seasonally_diff, lags=30, ax=axes[1], title=f'PACF of Seasonally Differenced Data (for P, Q)')
# plt.tight_layout()
# plt.show()


##########################################################################################
##########################################################################################
##########################################################################################

# %% SARIMAX Model for Time Series
print("\n--- Building a SARIMAX Model for the Training Data (Daily Totals) ---")

# **IMPORTANT**: RE-EVALUATE THESE ORDERS CAREFULLY BASED ON YOUR ACF/PACF PLOTS
# If your data is too short, start with smaller orders, especially for d and D.

# Suggested Starting Point for (p,d,q)(P,D,Q,s)
# Based on typical daily sales data and the assumption of weekly seasonality (s=7):
# 1. Non-seasonal differencing (d): Often 1 is enough for trend. Sometimes 0 or 2.
# 2. Seasonal differencing (D): Often 1 if strong weekly seasonality is present.
# 3. p, q, P, Q: Usually small values (0, 1, 2) based on significant spikes in ACF/PACF.

# Let's try a more conservative starting point for d and D if data length is a concern.
# For example, (1,1,1)(1,1,1,7) is a common general model.

# Based on common practice, let's start with d=1, D=1, and modest p, q, P, Q.
# YOU MUST adjust these based on your *specific* ACF/PACF plots after differencing.
p_order = 1
d_order = 1  # Start with d=1. Only increase to 2 if check_stationarity (KPSS) explicitly fails for train_data_diff_1.
q_order = 1

P_order = 1  # Seasonal AR order
D_order = 1  # Seasonal differencing for weekly cycles. Only use if ACF shows strong spikes at 7, 14, etc.
Q_order = 1  # Seasonal MA order
s_period = 7  # Weekly seasonality (7 days) - confirmed from previous discussion.

# If your 'train_data_ar' length is very short (e.g., < 30-40 observations),
# you might even need to set D_order=0 or reduce d_order.

arima_order = (p_order, d_order, q_order)
seasonal_arima_order = (P_order, D_order, Q_order, s_period)

print(f"\n--- Debugging SARIMAX Model Setup ---")
print(f"Training data length: {len(train_data_ar)}")
print(f"Non-seasonal order (p,d,q): {arima_order}")
print(f"Seasonal order (P,D,Q,s): {seasonal_arima_order}")

# Calculate effective length after differencing
effective_length = len(train_data_ar) - d_order - (D_order * s_period)
print(f"Effective data points after differencing: {effective_length}")

# Maximum order for AR/MA components (p, q, P, Q)
max_order_sum = p_order + q_order + P_order + Q_order

if effective_length <= 0:
    print("CRITICAL WARNING: Not enough data points left after differencing to fit the model!")
    print("ACTION: Reduce d_order or D_order drastically, or use a larger training set.")
elif effective_length < max_order_sum + 1:
    print(f"WARNING: Very few data points left ({effective_length}) compared to total AR/MA orders ({max_order_sum}).")
    print("This might lead to instability or errors. Consider reducing p,q,P,Q orders, or d/D.")

try:
    # Using SARIMAX instead of ARIMA for better robustness
    sarimax_model = SARIMAX(train_data_ar,
                            order=arima_order,
                            seasonal_order=seasonal_arima_order,
                            enforce_stationarity=False,
                            # Let the model handle differencing, don't force pre-differenced input
                            enforce_invertibility=False)  # Improves numerical stability for some datasets

    sarimax_results = sarimax_model.fit(disp=False)  # disp=False suppresses convergence messages

    print(sarimax_results.summary())

    # --- Forecasting with SARIMAX ---
    print("\n--- Forecasting with SARIMAX Model (Daily Totals) ---")

    forecast_steps = len(test_data_ar)

    # Use get_forecast from SARIMAX results
    forecast_results = sarimax_results.get_forecast(steps=forecast_steps)
    sarimax_test_predictions = forecast_results.predicted_mean
    conf_int = forecast_results.conf_int()  # Get confidence intervals

    # Ensure predictions have the correct index for easy plotting and evaluation
    sarimax_test_predictions.index = test_data_ar.index
    conf_int.index = test_data_ar.index

    # Evaluate the model on the test set
    mse_sarimax_test = mean_squared_error(test_data_ar, sarimax_test_predictions)
    rmse_sarimax_test = np.sqrt(mse_sarimax_test)
    r_squared_sarimax_test = r2_score(test_data_ar, sarimax_test_predictions)

    print(f"\n--- SARIMAX Model Test Set Evaluation (Daily Totals) ---")
    print(f"Mean Squared Error (MSE): {mse_sarimax_test:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_sarimax_test:.4f}")
    print(f"R-squared: {r_squared_sarimax_test:.4f}")

    # Plotting Actual vs. Predicted values for the test set
    plt.figure(figsize=(14, 7))
    plt.plot(train_data_ar.index, train_data_ar, label='Training Data', color='gray', alpha=0.7)
    plt.plot(test_data_ar.index, test_data_ar, label='Actual Total Sales (Daily Totals)', color='blue', alpha=0.7)
    plt.plot(sarimax_test_predictions.index, sarimax_test_predictions,
             label=f'Predicted Total Sales (SARIMAX{arima_order}{seasonal_arima_order} - Daily Totals)', color='green',
             linestyle='--', alpha=0.7)
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3,
                     label='Confidence Interval')
    plt.title(
        f'SARIMAX{arima_order}{seasonal_arima_order} Model: Actual vs. Predicted Total Sales (Daily Totals, Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Total Food Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #
    # Residual Plot for SARIMAX
    residuals_sarimax = test_data_ar - sarimax_test_predictions
    plt.figure(figsize=(14, 7))
    plt.scatter(sarimax_test_predictions, residuals_sarimax, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residual Plot (SARIMAX{arima_order}{seasonal_arima_order} Model - Daily Totals, Test Set)')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ACF/PACF of Residuals - THIS IS THE FINAL CHECK
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    plot_acf(residuals_sarimax, lags=20, ax=axes[0],
             title=f'ACF of SARIMAX{arima_order}{seasonal_arima_order} Residuals')
    plot_pacf(residuals_sarimax, lags=20, ax=axes[1],
              title=f'PACF of SARIMAX{arima_order}{seasonal_arima_order} Residuals')
    plt.tight_layout()
    plt.show()
    # Ideally, these plots should show no significant spikes (all within blue confidence bands).
    # If there are, it suggests your model is still missing some patterns, and you need to re-evaluate p, d, q, P, D, Q.


except Exception as e:
    print(f"An error occurred while fitting or forecasting with the SARIMAX model: {e}")
    print("Please carefully review your chosen (p,d,q)(P,D,Q,s) orders based on ACF/PACF plots and stationarity tests.")
    print(f"Current SARIMAX orders: Non-seasonal {arima_order}, Seasonal {seasonal_arima_order}")
    print(f"Training data length: {len(train_data_ar)}. Effective length after differencing: {effective_length}")



###########################################################
###########################################################
###########################################################
###########################################################
#%% Regression ( on milk, in may)


#%%
# 1. Filter data for 'milk' and 'May'
df_milk_may = df[(df['description'].isin(['ALC MILK', 'LACTOSE FREE MILK']))].copy()

if df_milk_may.empty:
    print("No record found for 'ALC MILK' or 'LACTOSE FREE MILK' in May. Please check your data or category name.")
else:
    # 2. Define dependent and initial list of independent variables
    dependent_variable = 'total'
    # Start with a very minimal set of independent variables for small datasets
    # Prioritize 'adults' and 'free_meals' as potentially robust indicators
    initial_independent_variables = ['free_meals', 'reduced_price_meals','full_price_meals']

    # --- Robust Numeric Conversion ---
    # Ensure dependent variable and initial independent variables are numeric
    all_potential_numeric_cols = initial_independent_variables + [dependent_variable]
    for col in all_potential_numeric_cols:
        if col in df_milk_may.columns:
            df_milk_may[col] = pd.to_numeric(df_milk_may[col], errors='coerce')
        else:
            print(f"Warning: Numeric column '{col}' not found in df_milk_may. It will be excluded from analysis.")
            if col in initial_independent_variables:
                initial_independent_variables.remove(col)  # Remove if not present

    independent_variables_for_regression = initial_independent_variables.copy()

    # --- Handle Categorical Variable Encoding ('time_of_day') ---
    # Only add time_of_day if it exists and we have enough data points for dummies
    if 'time_of_day' in df_milk_may.columns and len(
            df_milk_may) > 2:  # Need at least 3 points to have 2 categories + intercept
        df_milk_may['time_of_day'] = df_milk_may['time_of_day'].astype('category')
        # Check unique categories to ensure drop_first is valid
        if df_milk_may['time_of_day'].nunique() > 1:
            df_milk_may = pd.get_dummies(df_milk_may, columns=['time_of_day'], prefix='time', drop_first=True)
            time_dummy_cols = [col for col in df_milk_may.columns if col.startswith('time_')]
            independent_variables_for_regression.extend(time_dummy_cols)
        else:
            print("Skipping 'time_of_day' dummy encoding: Only one unique category found.")
    else:
        print("Warning: 'time_of_day' column not found or insufficient data for dummy encoding. It will be excluded.")

    # Ensure all required columns exist before proceeding
    final_cols_for_check = [dependent_variable] + [col for col in independent_variables_for_regression if
                                                   col in df_milk_may.columns]

    print(f"\n--- Data Types after initial numeric conversion and dummy encoding ---")
    print(df_milk_may[final_cols_for_check].dtypes)
    print("-" * 50)

    print(f"\nShape before dropna: {df_milk_may.shape}")
    df_milk_may_clean = df_milk_may[final_cols_for_check].dropna()
    print(f"Shape after dropna: {df_milk_may_clean.shape}")

    if df_milk_may_clean.empty:
        print("\nNo complete data points for regression after filtering, type conversion, and dropping NaNs.")
        print(
            "This is very likely due to limited data points and NaNs. Try to ensure your 'total', 'adults', and 'free_meals' columns have no missing values.")
        print("Please check the raw data and column names for these critical variables.")
    else:
        # Prepare data for statsmodels OLS
        Y = df_milk_may_clean[dependent_variable]
        X_cols = [col for col in df_milk_may_clean.columns if col != dependent_variable]
        X = df_milk_may_clean[X_cols]

        print(f"\n--- Final Data Types for Y and X before OLS ---")
        print(f"Y dtype: {Y.dtype}")
        print(f"X dtypes:\n{X.dtypes}")
        print("-" * 50)

        # Final check for object dtypes in Y or X (should be caught by previous to_numeric)
        if Y.dtype == 'object' or (X.dtypes == 'object').any():
            print("Critical Error: Object dtypes still present after numeric conversion attempts.")
            print("Please inspect your raw data for non-numeric entries in 'total', 'adults', 'free_meals'.")
            exit()

        # Drop columns with zero variance if they exist, as they cause issues in OLS
        # This is CRITICAL for small datasets where a column might be all the same value
        cols_to_drop_zero_variance = X.columns[X.var() == 0]
        if not cols_to_drop_zero_variance.empty:
            print(f"Warning: Dropping independent variable(s) with zero variance: {list(cols_to_drop_zero_variance)}")
            X = X.drop(columns=cols_to_drop_zero_variance)

        # Ensure we still have independent variables left after dropping zero variance columns
        if X.empty:
            print(
                "\nError: No independent variables left after cleaning and dropping zero-variance columns. Cannot perform regression.")
        elif len(X) <= X.shape[1]:  # Check if we have enough observations for the number of predictors
            print(
                f"\nError: Insufficient observations ({len(X)}) for the number of independent variables ({X.shape[1]}).")
            print("Regression is not possible with so few data points relative to predictors.")
            print("Consider reducing the number of independent variables further or collecting more data.")
        else:
            # Add a constant (intercept) to the independent variables
            X = sm.add_constant(X, has_constant='add')

            try:
                model = sm.OLS(Y, X)
                results = model.fit()

                print("\n--- Regression Results: Milk Sales in May (Minimal Model for Limited Data) ---")
                print(results.summary())

                # --- Plotting (Simplified for limited data) ---
                if len(Y) > 1:  # Only plot if there's more than one data point
                    plt.figure(figsize=(8, 6))
                    plt.scatter(results.predict(X), Y, alpha=0.7, label='Actual vs. Predicted')
                    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2, label='Perfect Fit Line')
                    plt.title('Milk Sales in May: Actual vs. Predicted (Minimal Model)')
                    plt.xlabel('Predicted Total Sales (Milk)')
                    plt.ylabel('Actual Total Sales (Milk)')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
                else:
                    print("\nSkipping plots: Too few data points after cleaning to generate meaningful visualizations.")


            except Exception as e:
                print(f"\nAn error occurred during OLS regression: {e}")
                print(
                    "This is often due to perfect multicollinearity (e.g., duplicate columns, too few observations for predictors, or highly correlated variables).")
                print(
                    "Given limited data, review the unique values in your independent variables. If any are constant, they will cause issues.")
                print("Try reducing independent variables further if this error persists.")

print("\n--- Regression Analysis Attempt Complete ---")




#
# %%
#
# ###########################################################
# ###########################################################
# ###########################################################
# ###########################################################
#

#%%
# LSTM
# Select the target variable for LSTM
lstm_data = df_daily_total[[target_variable]] # Use df_daily_total for consistency with SARIMAX

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(lstm_data)

# Split scaled data into training and testing sets
# We use the same split point as before to ensure consistency.
train_scaled = scaled_data[:split_index]
test_scaled = scaled_data[split_index:]

print(f"Scaled Training data shape: {train_scaled.shape}")
print(f"Scaled Test data shape: {test_scaled.shape}")

# Define a function to create sequences for LSTM for multi-step prediction
def create_sequences_multi_output(data, look_back=7, forecast_horizon=2):
    X, Y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        a = data[i:(i + look_back), 0] # Input sequence
        X.append(a)
        # Target sequence: 'forecast_horizon' steps immediately after the 'look_back' window
        Y.append(data[i + look_back : i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(Y)

look_back = 7 # Number of previous time steps to use as input features (e.g., a week)
forecast_horizon = 2 # Predict the next 2 days

# Create sequences for training and testing
X_train_lstm, Y_train_lstm = create_sequences_multi_output(train_scaled, look_back, forecast_horizon)
X_test_lstm, Y_test_lstm = create_sequences_multi_output(test_scaled, look_back, forecast_horizon)

# Reshape input to be [samples, timesteps, features] for LSTM
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

print(f"LSTM Training X shape: {X_train_lstm.shape}, Y shape: {Y_train_lstm.shape}")
print(f"LSTM Test X shape: {X_test_lstm.shape}, Y shape: {Y_test_lstm.shape}")

# Ensure there's enough data for LSTM training/testing
if len(X_train_lstm) == 0:
    print("Error: Not enough training data to create LSTM sequences. Adjust 'look_back' or get more data.")
if len(X_test_lstm) == 0:
    print("Error: Not enough test data to create LSTM sequences. Adjust 'look_back', 'forecast_horizon', or split ratio.")
    print(f"Test data length: {len(test_scaled)}, required: {look_back + forecast_horizon}")
    exit() # Exit if no test sequences can be formed

# --- 2. LSTM Model Architecture ---

print("\n--- Building LSTM Model ---")
lstm_model = Sequential()
# Increased units to 50, added 'relu' activation for potentially better learning
lstm_model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
# Output layer with 'forecast_horizon' neurons for multi-step prediction
lstm_model.add(Dense(forecast_horizon))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.summary()


# --- 3. Training the LSTM Model ---

print("\n--- Training LSTM Model (This may take a while) ---")
# Increased epochs to 50 and batch_size to 32 for potentially better training
history = lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=50, batch_size=32, verbose=1, shuffle=False)

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('LSTM Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()
print("LSTM Model Training Loss Plot: ")


# --- 4. Forecasting and Evaluation ---

print("\n--- Forecasting with LSTM Model ---")

# Make predictions on the scaled test data
lstm_predictions_scaled = lstm_model.predict(X_test_lstm)

# Inverse transform predictions and actual values to original scale
# Reshape -1, 1 is crucial for scaler to work correctly on single feature
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
Y_test_original = scaler.inverse_transform(Y_test_lstm)

# Ensure the index aligns with the original test data for plotting and evaluation
# Predictions correspond to dates starting from the (look_back + 1)-th day of the test set
# For a forecast horizon of H, the predictions at index i correspond to dates
# test_data_ar.index[look_back + i] to test_data_ar.index[look_back + i + H - 1]

# Create a list of actual future dates for each forecast
actual_forecast_dates = []
for i in range(len(Y_test_original)):
    start_date = test_data_ar.index[look_back + i]
    dates_for_forecast = pd.date_range(start=start_date, periods=forecast_horizon)
    actual_forecast_dates.append(dates_for_forecast)

# Flatten predictions and actuals for overall evaluation metrics
Y_test_original_flat = Y_test_original.flatten()
lstm_predictions_flat = lstm_predictions.flatten()

# Evaluate the LSTM model on the entire forecast horizon
mse_lstm_test = mean_squared_error(Y_test_original_flat, lstm_predictions_flat)
rmse_lstm_test = np.sqrt(mse_lstm_test)
r_squared_lstm_test = r2_score(Y_test_original_flat, lstm_predictions_flat)

print(f"\n--- LSTM Model Test Set Evaluation (Overall Multi-Step Forecast) ---")
print(f"Mean Squared Error (MSE): {mse_lstm_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lstm_test:.4f}")
print(f"R-squared: {r_squared_lstm_test:.4f}")

# Plotting Actual vs. Predicted values for the test set (LSTM - full forecast horizon)
plt.figure(figsize=(14, 7))
plt.plot(df_daily_total.index[:split_index], df_daily_total[target_variable][:split_index], label='Training Data', color='gray', alpha=0.7)

# Plot actual test data
plt.plot(test_data_ar.index, test_data_ar, label='Actual Total Sales (Test Set)', color='blue', alpha=0.7)

# Plot the predicted values, aligning them correctly with the forecast dates
# We can't simply plot lstm_predictions_flat against a continuous date range because
# each prediction corresponds to a *future* window.
# A more informative plot is the multi-step example one below.
# For a single line plot, we can plot the 'first day' of each forecast.
if len(Y_test_original) > 0:
    first_forecast_day_dates = [dates[0] for dates in actual_forecast_dates]
    first_forecast_day_predictions = lstm_predictions[:, 0]
    plt.plot(first_forecast_day_dates, first_forecast_day_predictions,
             label=f'Predicted Total Sales (LSTM - First Day of Forecast)', color='purple', linestyle='--', alpha=0.7)

plt.title(f'LSTM Model: Actual vs. Predicted Total Sales (Test Set - First Day of Each Forecast)')
plt.xlabel('Date')
plt.ylabel('Total Food Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("LSTM Model: Actual vs. Predicted Total Sales (Test Set - First Day of Each Forecast) Plot: ")


# Residual Plot for LSTM (using all flattened predictions)
residuals_lstm = Y_test_original_flat - lstm_predictions_flat
plt.figure(figsize=(14, 7))
plt.scatter(lstm_predictions_flat, residuals_lstm, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (LSTM Model - All Forecast Steps)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Residual Plot (LSTM Model - All Forecast Steps) Plot: ")


# --- Visualizing the full 2-day forecast (Example) ---
print("\n--- Visualizing Multi-Step Forecast Examples ---")
num_examples_to_plot = min(5, len(Y_test_original)) # Plot up to 5 examples or fewer if not enough data
start_plot_index = 0 # Starting from the first available test prediction

plt.figure(figsize=(15, 8))
plt.title(f'LSTM Model: Multi-Step Forecast Examples (Actual vs. Predicted, Horizon={forecast_horizon} days)')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)

# Plot actual data for context
plt.plot(df_daily_total.index, df_daily_total[target_variable], label='Full Historical Data', color='gray', alpha=0.6, linewidth=1)

for i in range(num_examples_to_plot):
    # Get the actual sequence and the predicted sequence
    current_actual_start_date = actual_forecast_dates[start_plot_index + i][0] # First date of the forecast window
    actual_forecast_window_dates = actual_forecast_dates[start_plot_index + i]
    actual_values_for_window = Y_test_original[start_plot_index + i, :]
    predicted_values_for_window = lstm_predictions[start_plot_index + i, :]

    # Plot actual future points
    plt.plot(actual_forecast_window_dates, actual_values_for_window,
             marker='o', linestyle='-', color='blue',
             label=f'Actual Future ({current_actual_start_date.strftime("%Y-%m-%d")})' if i == 0 else "", alpha=0.8)

    # Plot predicted future points
    plt.plot(actual_forecast_window_dates, predicted_values_for_window,
             marker='x', linestyle='--', color='red',
             label=f'Predicted Future ({current_actual_start_date.strftime("%Y-%m-%d")})' if i == 0 else "", alpha=0.8)

    # Plot the look-back window that led to this prediction (optional)
    # Reconstruct the input sequence dates
    look_back_start_date = current_actual_start_date - pd.Timedelta(days=look_back)
    look_back_dates = pd.date_range(start=look_back_start_date, periods=look_back)
    # Ensure the look-back data is from the scaled_data before inverse transform
    look_back_data_original_scale = scaler.inverse_transform(X_test_lstm[start_plot_index + i, :, 0].reshape(-1, 1)).flatten()
    plt.plot(look_back_dates, look_back_data_original_scale,
             color='green', linestyle=':',
             label=f'Look-back Window Input' if i == 0 else "", alpha=0.6)

# Create a single legend for all series
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()
print("LSTM Model: Multi-Step Forecast Examples Plot: ")