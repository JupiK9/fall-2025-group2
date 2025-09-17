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
#load the dataset
df = pd.read_csv('raw_data/sales.csv')
df.head()
df.info()

# Convert 'Date' to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Set 'Date' as the index
df.set_index('date', inplace=True)

# Sort the index to ensure chronological order
df.sort_index(inplace=True)

# Check for missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())
###################################

df_weekdays_only = df[df.index.dayofweek < 5].copy() # Use .copy() to avoid SettingWithCopyWarning
start_date_plot = df_weekdays_only.index.min()
end_date_plot = df_weekdays_only.index.max()



# # Simulate 'total' sales data with some trend and weekly pattern
# np.random.seed(42)
# sales = np.linspace(100, 150, len(weekday_dates)) + \
#           5 * np.sin(np.arange(len(weekday_dates)) / 5 * 2 * np.pi) + \
#           np.random.normal(0, 5, len(weekday_dates))

# Create the DataFrame
# dfs = pd.DataFrame({'date': weekday_dates, 'total': sales})
# dfs.set_index('date', inplace=True)

# Plotting the time series
plt.figure(figsize=(14, 5))
sns.lineplot(data=df_weekdays_only, x=df_weekdays_only.index, y='total')
plt.title("Time Series Plot of Total Food Sales (Weekdays Only)")
plt.xlabel("Date")
plt.ylabel("Total Food Sales")
plt.grid(True) # Added a grid for better readability
plt.tight_layout()
plt.show()


##############################################################################################


#%%
# define target variable and independent var
target_var = 'total'
numerical_features_names = df.iloc[:,6:].columns.tolist()


# Identify categorical features and encoding
df['tod'] = df['time_of_day'].map({'breakfast': 0, 'lunch': 1})

print("\n'tod' column after mapping:")
print(df['tod'])
print("New dtype:", df['tod'].dtype)

print("\nDataFrame with new encoded column:")
print(df)

categorical_features_names = 'tod'
ind_names = numerical_features_names + [categorical_features_names]
indepenent_var = df[ind_names]
print(indepenent_var)



# plot sales over time
daily_sales = df[target_var].resample('D').sum()

plt.figure(figsize=(15, 6))
daily_sales.plot(title=f'{target_var} Over Time')
plt.ylabel('sales')
plt.xlabel('Date')
plt.tight_layout()
plt.show()


# # Plot ACF and PACF for the target variable
# fig, axes = plt.subplots(2, 1, figsize=(16, 5))
# plot_acf(df[target_var], lags=50, ax=axes[0], title=f'ACF of {target_var}')
# plot_pacf(df[target_var], lags=50, ax=axes[1], title=f'PACF of {target_var}')
# plt.tight_layout()
# plt.show()



#%% check for stationary

# %%
# Select only numerical columns for the correlation matrix (including the target)
numerical_cols_for_corr = target_var + indepenent_var
correlation_matrix = df[numerical_cols_for_corr].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# %%
# Define the split point (80% of the data)
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

# Split the data
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

print(f"\n--- Data Splitting ---")
print(f"Total observations: {len(df)}")
print(f"Training set size: {len(train_data)} ({split_ratio*100:.0f}%)")
print(f"Test set size: {len(test_data)} ({(1-split_ratio)*100:.0f}%)")
print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")

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
window_size = 24 * 30
rolling_mean = train_data[target_var].rolling(window=24*30).mean()
rolling_std = train_data[target_var].rolling(window=24*30).std()
rolling_var = rolling_std**2

plt.figure(figsize=(15, 9)) # Adjusted figure size for two plots

# Rolling Mean
plt.subplot(2, 1, 1)
plt.plot(train_data[target_var], label='Original Training Data', color='blue', alpha=0.7)
plt.plot(rolling_mean, label=f'Rolling Mean (w={window_size} hours / ~30 days)', color='red', linewidth=2)
plt.title(f'{target_var} and Rolling Statistics (Training Data)')
plt.ylabel('Sales')
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

check_stationarity(train_data[target_var], f"Raw {target_var}")
