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
#
# # Plot the original time series
# plt.figure(figsize=(14, 5))
# sns.lineplot(data=df_daily_total, x=df_daily_total.index, y=target_variable)
# plt.title(f"Time Series Plot of {target_variable} Food Sales (Daily Totals)")
# plt.xlabel("Date")
# plt.ylabel(f"{target_variable} Food Sales")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#%% Split data
split_ratio = 0.8
split_index = int(len(df_daily_total) * split_ratio)
train_data_ar = df_daily_total[target_variable].iloc[:split_index]
test_data_ar = df_daily_total[target_variable].iloc[split_index:]

print(f"\n--- Data Splitting for AR Model ---")
print(f"Total observations: {len(df_daily_total)} (Daily Totals)")
print(f"Training set size: {len(train_data_ar)} ({split_ratio*100:.0f}%)")
print(f"Test set size: {len(test_data_ar)} ({(1-split_ratio)*100:.0f}%)")


#%% Regression ( on milk, in may)
# %% Regression on milk sales for top 5 largest schools
# Load student counts
student_counts_path = 'raw_data/Student_counts.csv' # Assuming this path
top_N_schools = 5  # Changed from 4 back to 5 as requested

try:
    student_counts_df = pd.read_csv(student_counts_path)
    df_sizes = student_counts_df[['School_Name', '2024-2025']].dropna().copy()
    df_sizes.columns = ['school_name', 'count']

    # Normalize school names from student_counts_df for matching
    df_sizes['school_name_normalized'] = df_sizes['school_name'].astype(str).str.lower().str.replace(r'[^a-z0-9]+', '',
                                                                                                     regex=True)
    df_sizes['count'] = df_sizes['count'].astype(int)

    # Sort by count to find largest schools
    largest_schools_df = df_sizes.sort_values(by='count', ascending=False).head(top_N_schools)

    # Get the normalized names from the largest schools
    top_N_largest_school_names_normalized = largest_schools_df['school_name_normalized'].tolist()

    print(f"\n--- Top {top_N_schools} Largest Schools (by student count) ---")
    print(largest_schools_df[['school_name', 'count']])

except FileNotFoundError:
    print(f"Error: Student count file not found at {student_counts_path}. Skipping size-based analysis.")
    top_N_largest_school_names_normalized = []
except Exception as e:
    print(f"Error loading or processing student count file: {e}. Skipping size-based analysis.")
    top_N_largest_school_names_normalized = []

if top_N_largest_school_names_normalized:
    # Prepare sales.csv school names for matching
    # Create a normalized version of 'school_name' in df for matching
    df['school_name_original'] = df['school_name']  # Keep original for display if needed
    df['school_name_normalized_sales'] = df['school_name'].astype(str).str.lower().str.replace(r'[^a-z0-9]+', '',
                                                                                               regex=True)

    # Filter sales data for 'milk' and for the top N largest schools
    # Using 'description' column for milk products, similar to optimization_milk.py's implied item handling
    df_milk_top_schools = df[
        (df['description'].isin(['ALC MILK', 'LACTOSE FREE MILK'])) & # You can expand this list if other milk products exist
        (df['school_name_normalized_sales'].isin(top_N_largest_school_names_normalized))
        ].copy()

    if df_milk_top_schools.empty:
        print(
            f"\nNo records found for 'ALC MILK' or 'LACTOSE FREE MILK' in the top {top_N_schools} largest schools after name matching. Please check your data or category name.")
    else:
        # 2. Define dependent and initial list of independent variables
        dependent_variable = 'total'
        # Ensure these columns exist and are relevant in your sales.csv
        initial_independent_variables = ['free_meals', 'reduced_price_meals', 'full_price_meals', 'adults'] # Added 'adults' as it's a common predictor

        # --- Robust Numeric Conversion ---
        all_potential_numeric_cols = initial_independent_variables + [dependent_variable]
        for col in all_potential_numeric_cols:
            if col in df_milk_top_schools.columns:
                # Replace non-numeric with NaN and then convert
                df_milk_top_schools[col] = pd.to_numeric(
                    df_milk_top_schools[col].astype(str).str.replace(r'[$,]', '', regex=True),
                    errors='coerce'
                )
            else:
                print(f"Warning: Numeric column '{col}' not found in df_milk_top_schools. It will be excluded from analysis.")
                if col in initial_independent_variables:
                    initial_independent_variables.remove(col) # Remove missing column from predictors

        independent_variables_for_regression = initial_independent_variables.copy()

        # --- Handle Categorical Variable Encoding ('time_of_day') ---
        if 'time_of_day' in df_milk_top_schools.columns and len(df_milk_top_schools) > 2:
            df_milk_top_schools['time_of_day'] = df_milk_top_schools['time_of_day'].astype('category')
            if df_milk_top_schools['time_of_day'].nunique() > 1:
                df_milk_top_schools = pd.get_dummies(df_milk_top_schools, columns=['time_of_day'], prefix='time',
                                                     drop_first=True)
                time_dummy_cols = [col for col in df_milk_top_schools.columns if col.startswith('time_')]
                independent_variables_for_regression.extend(time_dummy_cols)
            else:
                print("Skipping 'time_of_day' dummy encoding: Only one unique category found.")
        else:
            print(
                "Warning: 'time_of_day' column not found or insufficient data for dummy encoding. It will be excluded.")

        # Ensure all required columns exist before proceeding
        # Only include columns that are actually in the dataframe after potential removals/creations
        final_cols_for_check = [dependent_variable] + [col for col in independent_variables_for_regression if col in df_milk_top_schools.columns]

        print(f"\n--- Data Types after initial numeric conversion and dummy encoding ---")
        print(df_milk_top_schools[final_cols_for_check].dtypes)
        print("-" * 50)

        # Drop rows with any NaN values in the selected columns for regression
        print(f"\nShape before dropna for regression: {df_milk_top_schools.shape}")
        df_milk_top_schools_clean = df_milk_top_schools[final_cols_for_check].dropna()
        print(f"Shape after dropna for regression: {df_milk_top_schools_clean.shape}")

        if df_milk_top_schools_clean.empty:
            print("\nNo complete data points for regression after filtering, type conversion, and dropping NaNs.")
            print(
                "This is very likely due to limited data points and NaNs in 'total', 'free_meals', 'reduced_price_meals', 'full_price_meals', or 'adults'.")
            print("Please check the raw data and column names for these critical variables.")
        else:
            # Prepare data for statsmodels OLS
            Y = df_milk_top_schools_clean[dependent_variable]
            X_cols = [col for col in df_milk_top_schools_clean.columns if col != dependent_variable]
            X = df_milk_top_schools_clean[X_cols]

            print(f"\n--- Final Data Types for Y and X before OLS ---")
            print(f"Y dtype: {Y.dtype}")
            print(f"X dtypes:\n{X.dtypes}")
            print("-" * 50)

            # Re-check for object dtypes (shouldn't happen with robust conversion but good for safety)
            if Y.dtype == 'object' or (X.dtypes == 'object').any():
                print("Critical Error: Object dtypes still present after numeric conversion attempts.")
                print("Please inspect your raw data for non-numeric entries in 'total' or the independent variables.")
                exit() # Exit if critical data type error persists

            # Drop independent variables with zero variance (they cause OLS issues)
            cols_to_drop_zero_variance = X.columns[X.var() == 0]
            if not cols_to_drop_zero_variance.empty:
                print(
                    f"Warning: Dropping independent variable(s) with zero variance: {list(cols_to_drop_zero_variance)}")
                X = X.drop(columns=cols_to_drop_zero_variance)

            if X.empty:
                print(
                    f"\nError: No independent variables left after cleaning and dropping zero-variance columns. Cannot perform regression for the top {top_N_schools} schools.")
            elif len(X) <= X.shape[1]: # Check if observations are sufficient for the number of predictors
                print(
                    f"\nError: Insufficient observations ({len(X)}) for the number of independent variables ({X.shape[1]}).")
                print("Regression is not possible with so few data points relative to predictors.")
                print("Consider reducing the number of independent variables further or collecting more data.")
            else:
                X = sm.add_constant(X, has_constant='add') # Add an intercept to the model

                try:
                    model = sm.OLS(Y, X)
                    results = model.fit()

                    print(f"\n--- Regression Results: Milk Sales in Top {top_N_schools} Largest Schools ---")
                    print(results.summary())

                    # Plotting actual vs. predicted values
                    if len(Y) > 1: # Only plot if there's enough data
                        plt.figure(figsize=(8, 6))
                        plt.scatter(results.predict(X), Y, alpha=0.7, label='Actual vs. Predicted')
                        plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2, label='Perfect Fit Line')
                        plt.title(f'Milk Sales in Top {top_N_schools} Largest Schools: Actual vs. Predicted')
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
else:
    print(
        f"\nSkipping milk regression by school size due to issues with loading student count data or identifying top {top_N_schools} schools.")

print("\n--- Regression Analysis Attempt Complete ---")