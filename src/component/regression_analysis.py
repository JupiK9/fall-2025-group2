import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
def generate_regression_plot(data):
    """
    Trains a simple model and plots its predictions vs actuals.
    Uses UPDATED column names.
    """
    try:
        # Use 'planned' instead of 'planned_reimbursable'
        X = data[['planned', 'offered', 'student_count']]
        y = data['production_cost_total']

        # Fill any remaining NaNs just in case
        X = X.fillna(0)
        y = y.fillna(0)
        
        if X.empty or y.empty:
            print("No data for regression plot.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of Actual vs. Predicted
        sns.scatterplot(x='Actual', y='Predicted', data=plot_df, ax=ax, alpha=0.6, label='Predicted vs. Actual')
        
        # Add a 45-degree line (perfect prediction)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_title('Model 1 Performance: Actual vs. Predicted Production Cost')
        ax.set_xlabel('Actual Production Cost')
        ax.set_ylabel('Predicted Production Cost')
        ax.legend()
        ax.grid(True)
        
        return fig

    except Exception as e:
        print(f"Error generating regression plot: {e}")
        return None

# --- Main Controller Function (to be called by Streamlit) ---
def perform_regression_analysis(df_breakfast, df_lunch, student_counts_df):
    """
    Runs regression analyses on the *already cleaned* data from Streamlit.
    """
    results = {
        "summary1": "Model 1 failed to run.",
        "summary2": "Model 2 failed to run.",
        "summary3": "Model 3 failed to run.",
        "plot": None
    }

    try:
        # 1. Use DataFrames passed from Streamlit
        dfb = df_breakfast.copy()
        dfl = df_lunch.copy()
        student_counts_df = student_counts_df.copy() # Already clean

        # 2. Merge Student Counts
        # The 'student_counts_df' is already cleaned and binned in 5_Final.py
        # We just need to merge the 'count' column
        if 'school_name' in student_counts_df.columns and 'count' in student_counts_df.columns:
            school_counts = student_counts_df[['school_name', 'count']].drop_duplicates()
            dfb = pd.merge(dfb, school_counts, on='school_name', how='left')
            dfl = pd.merge(dfl, school_counts, on='school_name', how='left')
            # Rename 'count' to 'student_count' to match original regression script
            dfb.rename(columns={'count': 'student_count'}, inplace=True)
            dfl.rename(columns={'count': 'student_count'}, inplace=True)
        else:
            print("Warning: Could not merge student counts in regression_analysis.py")
            dfb['student_count'] = np.nan
            dfl['student_count'] = np.nan

        # 3. Combine DataFrames
        combined = pd.concat([dfb, dfl], ignore_index=True)
        
        # 4. Fill NaNs for regression
        # Data is already clean, just fill NaNs from merge
        combined['student_count'] = combined['student_count'].fillna(0)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
        
        # --- 5. Run Regressions with CORRECTED column names ---
        results["summary1"] = run_regression_sm(
            dependent_var='production_cost_total',
            # Use 'planned', 'discarded', 'leftover'
            independent_vars=[
                'served_reimbursable', 'planned', 'discarded',
                'leftover', 'student_count'
            ],
            data=combined,
            title="Regression 1: Predicting Production Cost"
        )

        results["summary2"] = run_regression_sm(
            dependent_var='discarded_cost',
            # Use 'planned', 'leftover'
            independent_vars=[
                'served_reimbursable', 'offered', 'planned',
                'leftover', 'student_count'
            ],
            data=combined,
            title="Regression 2: Predicting Discarded Cost"
        )

        results["summary3"] = run_regression_sm(
            dependent_var='served_reimbursable',
            # Use 'planned'
            independent_vars=[
                'planned', 'offered', 'student_count'
            ],
            data=combined,
            title="Regression 3: Predicting Served Reimbursable"
        )

        # --- 6. Generate Plot ---
        results["plot"] = generate_regression_plot(combined)

    except Exception as e:
        print(f"Error in perform_regression_analysis: {e}")
        import traceback
        traceback.print_exc()

    return results