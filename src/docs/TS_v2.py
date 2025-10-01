#%% imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime as dt
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

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
non_categorical_variables= ['school_code', 'item','free_meals', 'reduced_price_meals','full_price_meals','alac_student']
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






