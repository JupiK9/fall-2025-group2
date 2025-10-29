import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- File Paths ---
BREAKFAST_CSV = 'raw_data/data_breakfast.csv'
LUNCH_CSV = 'raw_data/data_lunch.csv'
STUDENT_COUNTS_CSV = 'raw_data/student_counts.csv'


# --- Helper: Clean Numeric Columns ---
def clean_numeric_cols(df, cols_to_clean):
    for col in cols_to_clean:
        if col in df.columns:
            # Handle potential non-string types before .str accessor
            df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^0-9.\-]', '', regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Columns that should be numeric
numeric_obj_cols = [
    'served_non-reimbursable', 'discarded_total', 'discarded_cost',
    'subtotal_cost', 'left_over_percent_of_offered', 'left_over_cost',
    'production_cost_total', 'served_reimbursable', 'planned_reimbursable',
    'offered_reimbursable', 'left_over_total', 'offered_total', 'served_total',
    'planned_total'
]

# --- 1. Load and Preprocess ---
print("--- Loading Data ---")

try:
    df_breakfast_raw = pd.read_csv(BREAKFAST_CSV)
    df_lunch_raw = pd.read_csv(LUNCH_CSV)
except FileNotFoundError as e:
    print(f"Missing file: {e}")
    exit()

df_breakfast_raw.columns = df_breakfast_raw.columns.str.lower().str.strip()
df_lunch_raw.columns = df_lunch_raw.columns.str.lower().str.strip()

df_breakfast = clean_numeric_cols(df_breakfast_raw.copy(), numeric_obj_cols)
df_lunch = clean_numeric_cols(df_lunch_raw.copy(), numeric_obj_cols)

df_breakfast['school_name'] = df_breakfast['school_name'].str.lower().str.strip()
df_lunch['school_name'] = df_lunch['school_name'].str.lower().str.strip()

# --- Load Student Counts ---
try:
    df_sizes = pd.read_csv(STUDENT_COUNTS_CSV)
    df_sizes.columns = df_sizes.columns.str.lower().str.strip()
    df_sizes = df_sizes[['school_name', '2024-2025']].rename(columns={'2024-2025': 'student_count'})
    df_sizes['school_name'] = df_sizes['school_name'].str.lower().str.strip()
    df_sizes['student_count'] = pd.to_numeric(df_sizes['student_count'], errors='coerce').fillna(0)
except Exception as e:
    print(f"Warning: could not load student count file: {e}")
    df_sizes = None


def prepare_data(df, df_sizes, meal_type):
    df_temp = df.copy()
    if df_sizes is not None:
        df_temp = df_temp.merge(df_sizes, on='school_name', how='left')
    else:
        df_temp['student_count'] = np.nan
    df_temp['student_count'] = pd.to_numeric(df_temp['student_count'], errors='coerce').fillna(0)

    for col in numeric_obj_cols:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan)
    df_temp['meal_type'] = meal_type  # Add meal type for filtering
    return df_temp


combined_b = prepare_data(df_breakfast, df_sizes, 'breakfast')
combined_l = prepare_data(df_lunch, df_sizes, 'lunch')


# --- Helper Function to Run Regressions Safely ---
def run_regression(dependent_var, independent_vars, data, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if dependent_var not in data.columns:
        print(f"Error: '{dependent_var}' not found in dataset.")
        return

    # Filter out independent vars not in data
    available_independent_vars = [col for col in independent_vars if col in data.columns]
    if not available_independent_vars:
        print(f"No available independent variables for regression for {dependent_var}.")
        return

    X = data[available_independent_vars].copy()
    y = pd.to_numeric(data[dependent_var], errors='coerce')

    # Ensure all X columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with NaNs
    df_reg = pd.concat([y.rename(dependent_var), X], axis=1).dropna()
    y_clean = df_reg[dependent_var]
    X_clean = df_reg.drop(dependent_var, axis=1)

    if len(df_reg) == 0:
        print("No valid data after cleaning; skipping regression.")
        return

    # Final sanity check
    non_numeric = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print("⚠️ Non-numeric columns found:", non_numeric)
        return

    # Add constant
    X_sm = sm.add_constant(X_clean, has_constant='add')

    # Run regression
    try:
        model = sm.OLS(y_clean, X_sm).fit()
        print(model.summary())
    except Exception as e:
        print(f"Error during OLS regression: {e}")


def plot_regression_results(dependent_var, independent_vars_for_viz, data, title_prefix):
    print("\n" + "=" * 100)
    print(f"{title_prefix} Visualization (sklearn)")
    print("=" * 100)

    Y = pd.to_numeric(data[dependent_var], errors='coerce')

    # Filter for available independent variables
    available_independent_vars = [col for col in independent_vars_for_viz if col in data.columns]
    if not available_independent_vars:
        print(f"No available independent variables for visualization for {dependent_var}.")
        return

    X = data[available_independent_vars].copy()

    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_viz = pd.concat([Y.rename(dependent_var), X], axis=1).dropna()

    if len(df_viz) > 0:
        y = df_viz[dependent_var]
        X = df_viz.drop(dependent_var, axis=1)

        if len(X) == 0 or len(y) == 0:
            print(f"Not enough data for visualization for {dependent_var} after dropping NaNs.")
            return

        if len(X) < 2 or len(y) < 2:  # Need at least 2 samples for train_test_split
            print(f"Not enough samples ({len(X)}) for train_test_split for {dependent_var}.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(np.unique(y_train)) < 2:  # Check for single class in y_train
            print(f"Skipping visualization for {dependent_var} due to single unique value in y_train.")
            return

        try:
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except ValueError as e:
            print(f"Error fitting Linear Regression model for {dependent_var}: {e}")
            return

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f'Actual vs Predicted {dependent_var.replace("_", " ").title()}')
        plt.xlabel('Actual');
        plt.ylabel('Predicted')

        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Residuals vs Predicted for {dependent_var.replace("_", " ").title()}')
        plt.xlabel('Predicted');
        plt.ylabel('Residuals')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Not enough data for visualization for {dependent_var}.")


# --- Define Items for Analysis ---

breakfast_items = [
    'Apple Juice (Each)',
    'Orange Tangerine Juice (Each)',
    '1% White Milk (Each)',
    'Mini Maple Pancakes (Package)',
    'Red Delicious Apple (Each - serving)',
    'Honey Cheerios Cereal (Each)',
    'Cinnamon Toast Crunch Cereal, 25% Less Sugar (Each)',
    'Cinnamon Chex Cereal (Each)',
    'Fat Free White Milk (Each)',
    'Blueberry Chex Cereal (Each)'
]

lunch_items = [
    'Fat Free Chocolate Milk (Each)',
    'Lettuce Blend Salad Mix (1/2 cup)',
    'Ketchup Packet (Packet)',
    'Ranch Dressing (1 ounce)',
    'PB&J Power Pack (Power Packs)',
    '1% White Milk (Each)',
    'Manager\'s Vegetable Choice (1/4 cup)',
    'Cheese Pizza (Slice)',
    'Mayonnaise Packet (Packet)',
    'Chicken Tenders (3 tenders)'
]

# --- Regression Independent Variables ---
regression_1_ivs = ['served_reimbursable', 'planned_reimbursable', 'discarded_total', 'left_over_total',
                    'student_count', 'offered_reimbursable', 'served_total', 'planned_total', 'offered_total',
                    'subtotal_cost']
regression_2_ivs = ['served_reimbursable', 'offered_reimbursable', 'planned_reimbursable', 'left_over_total',
                    'student_count', 'served_total', 'planned_total', 'offered_total', 'subtotal_cost']
regression_3_ivs = ['planned_reimbursable', 'offered_reimbursable', 'student_count', 'served_total', 'planned_total',
                    'offered_total', 'subtotal_cost']


# --- Function to run all regressions and visualizations for a given item ---
def analyze_item(name, meal_data, meal_type_str):
    print(f"\n\n{'#' * 120}")
    print(f"### Analyzing {meal_type_str.upper()} ITEM: {name} ###")
    print(f"{'#' * 120}\n")

    # Filter data for the specific item
    item_data = meal_data[meal_data['name'].str.lower().str.strip() == name.lower().strip()].copy()

    if item_data.empty:
        print(f"No data found for item: {name} in {meal_type_str} data.")
        return

    # Regression 1: Predicting Production Cost
    run_regression(
        dependent_var='production_cost_total',
        independent_vars=regression_1_ivs,
        data=item_data,
        title=f"Regression 1 for {name}: Predicting Production Cost"
    )
    plot_regression_results(
        dependent_var='production_cost_total',
        independent_vars_for_viz=regression_1_ivs,
        data=item_data,
        title_prefix=f"Regression 1 for {name}: Production Cost"
    )

    # Regression 2: Predicting Discarded Cost
    run_regression(
        dependent_var='discarded_cost',
        independent_vars=regression_2_ivs,
        data=item_data,
        title=f"Regression 2 for {name}: Predicting Discarded Cost"
    )
    plot_regression_results(
        dependent_var='discarded_cost',
        independent_vars_for_viz=regression_2_ivs,
        data=item_data,
        title_prefix=f"Regression 2 for {name}: Discarded Cost"
    )

    # Regression 3: Predicting Served Reimbursable
    run_regression(
        dependent_var='served_reimbursable',
        independent_vars=regression_3_ivs,
        data=item_data,
        title=f"Regression 3 for {name}: Predicting Served Reimbursable"
    )
    plot_regression_results(
        dependent_var='served_reimbursable',
        independent_vars_for_viz=regression_3_ivs,
        data=item_data,
        title_prefix=f"Regression 3 for {name}: Served Reimbursable"
    )


# --- Run Analysis for Breakfast Items ---
print("\n" * 3)
print("=" * 150)
print("                           STARTING BREAKFAST ITEM ANALYSIS                           ")
print("=" * 150)
for item in breakfast_items:
    analyze_item(item, combined_b, 'breakfast')

# --- Run Analysis for Lunch Items ---
print("\n" * 3)
print("=" * 150)
print("                             STARTING LUNCH ITEM ANALYSIS                             ")
print("=" * 150)
for item in lunch_items:
    analyze_item(item, combined_l, 'lunch')