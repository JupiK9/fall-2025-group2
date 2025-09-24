import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
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

# --- Resample the data to daily sums ---
# Group by date and sum the relevant columns
# 'total' for sales, and meal counts for free, paid, discount
daily_df = df.resample('D').agg({
    'total': 'sum',
    'free_meals': 'sum',
    'full_price_meals': 'sum',
    'reduced_price_meals': 'sum'
    # Add any other relevant numerical columns you want to sum daily
})

# Drop any days where there were no sales (if they exist after resampling)
daily_df.dropna(inplace=True)

print("\n--- Daily Resampled DataFrame Info ---")
daily_df.info()
print("\n--- Daily Resampled DataFrame Head ---")
print(daily_df.head())

target_variable = 'total'

start_date_plot = daily_df.index.min()
end_date_plot = daily_df.index.max()

print(f"\nPlotting daily data from {start_date_plot.date()} to {end_date_plot.date()}")

# 1. Time Series Plot of Total Daily Sales
plt.figure(figsize=(14, 6))
sns.lineplot(data=daily_df, x=daily_df.index, y=target_variable)
plt.title("Daily Total Food Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Food Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
#

# 2. Distribution of Daily Total Sales
plt.figure(figsize=(10, 6))
sns.histplot(daily_df[target_variable], kde=True, bins=30)
plt.title("Distribution of Daily Total Food Sales")
plt.xlabel("Total Food Sales")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
#
