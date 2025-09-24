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
df_original = pd.read_csv('raw_data/sales.csv')
print("--- Original DataFrame Info ---")
df_original.info()
print("\n--- Original DataFrame Head ---")
print(df_original.head())


# --- Step 1: Preprocessing for time_of_day and initial date conversion ---
# Convert 'Date' to datetime objects in the original DataFrame
df_original['date'] = pd.to_datetime(df_original['date'])

# Encode 'time_of_day' BEFORE aggregation, if you need it for later analysis
# Make sure 'time_of_day' column actually exists in your CSV with these values
if 'time_of_day' in df_original.columns:
    df_original['tod'] = df_original['time_of_day'].map({'Breakfast': 0, 'Lunch': 1}).fillna(-1) # Added fillna for potentially missing values
    print("\n'tod' column after mapping in original df:")
    print(df_original['tod'].value_counts())
    print("New dtype:", df_original['tod'].dtype)
else:
    print("\nWARNING: 'time_of_day' column not found in the original CSV. Cannot encode 'tod'.")
    # You might want to handle this case, e.g., create a dummy 'tod' or skip encoding

# --- Step 2: Create the aggregated DataFrame for time series plotting ---
# Group by 'date' and sum 'total' to get daily totals
# This will be your 'df' for time series analysis
df = df_original.groupby('date')['total'].sum().reset_index()

# Set 'Date' as the index
df.set_index('date', inplace=True)

# Sort the index to ensure chronological order
df.sort_index(inplace=True)

print("\n--- Aggregated DataFrame Info (for time series) ---")
df.info()
print("\n--- Aggregated DataFrame Head (for time series) ---")
print(df.head())


# Check for missing values in the aggregated DataFrame
print("\n--- Missing Values Check (Aggregated DF) ---")
print(df.isnull().sum())

###################################
weekend_dates_in_df = df[df.index.dayofweek >= 5]
if not weekend_dates_in_df.empty:
    print(f"\nWARNING: Found {len(weekend_dates_in_df)} weekend entries in the DataFrame. "
          "If this is unexpected, check your data loading process.")
else:
    print("\nConfirmed: No weekend dates found in the DataFrame index.")


start_date_plot = df.index.min()
end_date_plot = df.index.max()


print(f"\nPlotting data from {start_date_plot.date()} to {end_date_plot.date()}")
plt.figure(figsize=(14, 5))
sns.lineplot(data=df, x=df.index, y='total')
plt.title("Time Series Plot of Total Food Sales (Daily Aggregated Data)")
plt.xlabel("Date")
plt.ylabel("Total Food Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

##############################################################################################


#%%
# define target variable and independent var
target_var = 'total'

# This line was incorrect as df after aggregation only has 'total'
# numerical_features_names = df.iloc[:,6:].columns.tolist()
# If you need other numerical features, you'll need them from df_original
# or create them during aggregation. For now, we'll assume target_var is the only one.
numerical_features_names = [] # No other numerical features in the aggregated df for now

# # plot sales over time (this section is fine with the aggregated df)
# daily_sales = df[target_var].resample('D').sum() # Resampling 'D' on an already daily df doesn't change much
#                                                 # but confirms daily frequency
# plt.figure(figsize=(15, 6))
# daily_sales.plot(title=f'{target_var} Over Time')
# plt.ylabel('sales')
# plt.xlabel('Date')
# plt.tight_layout()
# plt.show()


#%%
# Identify categorical features and encoding
# You now need to use the df_original (or a copy of it) if you want 'tod' in its full glory
# If you plan to merge 'tod' with the aggregated daily 'total', you need a strategy.

# For now, let's assume you want 'tod' for *individual transactions* or another type of model
# that uses the original transaction data.

# If you want to use 'tod' in a model alongside the *daily aggregated total*,
# you would need to aggregate 'tod' in some way (e.g., average 'tod' per day, most frequent 'tod' per day)
# or keep the original df_original for a different model.

# Let's show how you would proceed if you were using df_original for modeling with 'tod'.
# For the purpose of *this specific aggregated df*, 'tod' doesn't exist anymore.
# If you intend to use 'tod' as a feature, you need to revisit your data preparation strategy
# to either keep it during aggregation (e.g., group by date AND time_of_day)
# or use the df_original for a different modeling task.

# Example if you wanted to keep 'tod' in the aggregated daily data (more complex, depends on what you want to model):
# df_with_tod = df_original.groupby(['date', 'tod'])['total'].sum().reset_index()
# print("\nExample: Aggregated DF with 'tod' as well:")
# print(df_with_tod.head())

# The following lines will still cause an error with the *current* `df`
# because `df` only has 'date' and 'total'.
# To demonstrate, I'll comment out the problematic part and explain.

# categorical_features_names = 'tod'
# ind_names = numerical_features_names + [categorical_features_names]
# independent_var = df[ind_names] # This would fail because 'tod' is not in df
# print(independent_var)

print("\n--- End of Script ---")
print("Note: The 'tod' encoding was applied to the original DataFrame. "
      "If you need 'tod' in your aggregated daily DataFrame, "
      "you need to decide how to aggregate categorical features (e.g., one-hot encode before sum, or take mode).")

#
# Plot ACF and PACF for the target variable
fig, axes = plt.subplots(2, 1, figsize=(16, 5))
plot_acf(df[target_var], lags=20, ax=axes[0], title=f'ACF of {target_var}')
plot_pacf(df[target_var], lags=20, ax=axes[1], title=f'PACF of {target_var}')
plt.tight_layout()
plt.show()

#
#
# #%% check for stationary
#
# # %%
# # Select only numerical columns for the correlation matrix (including the target)
# numerical_cols_for_corr = target_var + indepenent_var
# correlation_matrix = df[numerical_cols_for_corr].corr()
#
# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()
#
# # %%
# # Define the split point (80% of the data)
# split_ratio = 0.8
# split_index = int(len(df) * split_ratio)
#
# # Split the data
# train_data = df.iloc[:split_index]
# test_data = df.iloc[split_index:]
#
# print(f"\n--- Data Splitting ---")
# print(f"Total observations: {len(df)}")
# print(f"Training set size: {len(train_data)} ({split_ratio*100:.0f}%)")
# print(f"Test set size: {len(test_data)} ({(1-split_ratio)*100:.0f}%)")
# print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
# print(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")
#
# # %%
# # Define function for stationarity tests
# def check_stationarity(timeseries, series_name=""):
#     """Performs ADF and KPSS tests and prints the results."""
#     print(f'\n--- Stationarity Tests for {series_name} ---')
#     # ADF Test
#     adf_result = adfuller(timeseries, autolag='AIC')
#     print(f'ADF Statistic: {adf_result[0]:.4f}')
#     print(f'p-value: {adf_result[1]:.4f}')
#     print('Critical Values:')
#     for key, value in adf_result[4].items():
#         print(f'\t{key}: {value:.4f}')
#     if adf_result[1] <= 0.05:
#         print("Result: Strong evidence against the null hypothesis (H0), reject H0. Data is likely stationary.")
#     else:
#         print("Result: Weak evidence against null hypothesis, fail to reject H0. Data is likely non-stationary.")
#
#     print('\n')
#     # KPSS Test
#     kpss_result = kpss(timeseries, regression='c', nlags="auto") # 'c' for constant trend
#     print(f'KPSS Statistic: {kpss_result[0]:.4f}')
#     print(f'p-value: {kpss_result[1]:.4f}')
#     print('Critical Values:')
#     for key, value in kpss_result[3].items():
#         print(f'\t{key}: {value:.4f}')
#     if kpss_result[1] <= 0.05:
#         print("Result: Strong evidence against the null hypothesis (H0), reject H0. Data is likely non-stationary.")
#     else:
#         print("Result: Weak evidence against null hypothesis, fail to reject H0. Data is likely stationary.")
#     print('--- End Stationarity Tests ---')
#
#
# # %%
# # Plot rolling mean and variance for the raw training data
# window_size = 24 * 30
# rolling_mean = train_data[target_var].rolling(window=24*30).mean()
# rolling_std = train_data[target_var].rolling(window=24*30).std()
# rolling_var = rolling_std**2
#
# plt.figure(figsize=(15, 9)) # Adjusted figure size for two plots
#
# # Rolling Mean
# plt.subplot(2, 1, 1)
# plt.plot(train_data[target_var], label='Original Training Data', color='blue', alpha=0.7)
# plt.plot(rolling_mean, label=f'Rolling Mean (w={window_size} hours / ~30 days)', color='red', linewidth=2)
# plt.title(f'{target_var} and Rolling Statistics (Training Data)')
# plt.ylabel('Sales')
# plt.legend(loc='best')
# plt.grid(True, linestyle='--', alpha=0.6)
# # Hide x-axis labels for the top plot to avoid overlap
# plt.setp(plt.gca().get_xticklabels(), visible=False)
# # Rolling Var
# plt.subplot(2, 1, 2)
# plt.plot(rolling_var, label=f'Rolling Variance (w={window_size} hours / ~30 days)', color='orange', linewidth=2)
# plt.ylabel('Variance')
# plt.xlabel('Date')
# plt.legend(loc='best')
# plt.grid(True, linestyle='--', alpha=0.6)
#
# # Adjust layout and display
# plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap slightly
# plt.show()
#
# check_stationarity(train_data[target_var], f"Raw {target_var}")
