import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- File Paths ---
# Use Pathlib to find the data folder relative to this file
base_path = Path(__file__).resolve().parent.parent.parent / 'src' / 'data' / 'preprocessed-data'
BREAKFAST_CSV = base_path / 'milk_b.csv'
LUNCH_CSV = base_path / 'milk_l.csv'
STUDENT_COUNTS_CSV = base_path / 'student_counts.csv'

# --- Helper: Clean Numeric Columns ---
def clean_numeric_cols(df, cols_to_clean):
    for col in cols_to_clean:
        if col in df.columns:
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
    'offered_reimbursable', 'left_over_total'
]

# --- Helper Function to Run Regressions Safely ---
def run_regression_sm(dependent_var, independent_vars, data, title):
    """
    Runs a statsmodels OLS regression and returns the summary as text.
    """
    if dependent_var not in data.columns:
        return f"Error: '{dependent_var}' not found in dataset."

    X = data[independent_vars].copy()
    y = pd.to_numeric(data[dependent_var], errors='coerce')

    # Ensure all X columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with NaNs
    df_reg = pd.concat([y.rename(dependent_var), X], axis=1).dropna()
    y_clean = df_reg[dependent_var]
    X_clean = df_reg.drop(dependent_var, axis=1)

    if len(df_reg) == 0:
        return f"No valid data for {title}; skipping regression."

    # Final sanity check
    non_numeric = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        return f"⚠️ Non-numeric columns found: {non_numeric}"

    # Add constant
    X_sm = sm.add_constant(X_clean, has_constant='add')

    # Run regression
    model = sm.OLS(y_clean, X_sm).fit()
    
    # *** KEY CHANGE: Return the summary as text ***
    return model.summary().as_text()

# --- Helper Function to Generate Plot ---
def generate_regression_plot(combined_data):
    """
    Generates the regression visualization and returns the matplotlib figure.
    """
    Y = pd.to_numeric(combined_data['production_cost_total'], errors='coerce')
    X = combined_data[['served_reimbursable', 'planned_reimbursable', 'discarded_total',
                      'left_over_total', 'student_count']].copy()

    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_viz = pd.concat([Y.rename('production_cost_total'), X], axis=1).dropna()

    if len(df_viz) == 0:
        return None # Return None if no data

    y = df_viz['production_cost_total']
    X = df_viz.drop('production_cost_total', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # *** KEY CHANGE: Explicitly create a figure to return it ***
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_title('Actual vs Predicted Production Cost')
    ax1.set_xlabel('Actual'); ax1.set_ylabel('Predicted')

    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs Predicted')
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('Residuals')

    plt.tight_layout()
    
    return fig # Return the figure object

# --- Main Controller Function (to be called by Streamlit) ---
def perform_regression_analysis():
    """
    Loads all data, runs all regressions, and returns a dictionary of results.
    """
    # --- 1. Load and Preprocess ---
    try:
        df_breakfast_raw = pd.read_csv(BREAKFAST_CSV)
        df_lunch_raw = pd.read_csv(LUNCH_CSV)
    except FileNotFoundError as e:
        return {"error": f"Missing regression file: {e}"}

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
        df_sizes = None

    # --- Combine Datasets ---
    combined = pd.concat([df_breakfast, df_lunch], ignore_index=True)

    if df_sizes is not None:
        combined = combined.merge(df_sizes, on='school_name', how='left')
    else:
        combined['student_count'] = np.nan

    combined['student_count'] = pd.to_numeric(combined['student_count'], errors='coerce').fillna(0)

    # --- Universal Conversion ---
    for col in numeric_obj_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    combined = combined.replace([np.inf, -np.inf], np.nan)

    # --- 2. Run Regressions ---
    summary1 = run_regression_sm(
        dependent_var='production_cost_total',
        independent_vars=[
            'served_reimbursable', 'planned_reimbursable', 'discarded_total',
            'left_over_total', 'student_count'
        ],
        data=combined,
        title="Regression 1: Predicting Production Cost"
    )

    summary2 = run_regression_sm(
        dependent_var='discarded_cost',
        independent_vars=[
            'served_reimbursable', 'offered_reimbursable', 'planned_reimbursable',
            'left_over_total', 'student_count'
        ],
        data=combined,
        title="Regression 2: Predicting Discarded Cost"
    )

    summary3 = run_regression_sm(
        dependent_var='served_reimbursable',
        independent_vars=[
            'planned_reimbursable', 'offered_reimbursable', 'student_count'
        ],
        data=combined,
        title="Regression 3: Predicting Served Reimbursable"
    )

    # --- 3. Generate Plot ---
    plot_fig = generate_regression_plot(combined)

    # --- 4. Return all results ---
    return {
        "summary1": summary1,
        "summary2": summary2,
        "summary3": summary3,
        "plot": plot_fig
    }