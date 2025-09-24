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

# --- New Step: Exclude Weekends ---
# Create a boolean mask to identify weekdays (Monday=0, Sunday=6)
# We want to keep days where dayofweek is 0, 1, 2, 3, 4
df = df[df['date'].dt.dayofweek < 5]
print(f"\n--- DataFrame after excluding weekends ---")
print(f"Number of rows: {len(df)}")
print(f"Min Date: {df['date'].min().date()}, Max Date: {df['date'].max().date()}")


# --- New Step: Aggregate daily sales ---
# Group by 'date' and sum the relevant columns
# 'total' will be the sum of daily sales across all schools/items/times
# 'free_meals', 'paid_meals', 'discount_meals' will also be summed daily
daily_sales_df = df.groupby('date').agg(
    total=('total', 'sum'),
    free_meals=('free_meals', 'sum'),
    full_price_meals=('full_price_meals', 'sum'),
    reduced_price_meals=('reduced_price_meals', 'sum')
    # Add other independent variables here if you want their daily sums
).reset_index() # Reset index to make 'date' a column again before setting as index


# Set 'date' as the index for the new daily_sales_df
daily_sales_df.set_index('date', inplace=True)

# Sort the index to ensure chronological order (important for time series)
daily_sales_df.sort_index(inplace=True)

# Check for missing values in the aggregated DataFrame
print("\n--- Missing Values Check in Daily Sales DataFrame ---")
print(daily_sales_df.isnull().sum())

print("\n--- Daily Sales DataFrame Info ---")
daily_sales_df.info()
print("\n--- Daily Sales DataFrame Head ---")
print(daily_sales_df.head())


target_variable = 'total'

start_date_plot = daily_sales_df.index.min()
end_date_plot = daily_sales_df.index.max()

print(f"\nPlotting aggregated daily data from {start_date_plot.date()} to {end_date_plot.date()}")
plt.figure(figsize=(14, 6)) # Slightly larger figure for better visibility
sns.lineplot(data=daily_sales_df, x=daily_sales_df.index, y=target_variable)
plt.title(f"Aggregated Daily Total Food Sales Over Time (excluding weekends)")
plt.xlabel("Date")
plt.ylabel("Total Food Sales")
plt.grid(True)
plt.tight_layout()
plt.show()