import pandas as pd
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import geopandas as gpd
import folium
import json
import traceback
from pathlib import Path
from shapely.geometry import Point
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
import subprocess

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
from pathlib import Path
#
# # Current script location:
# script_path = Path(__file__).resolve()
#
# # Base directory (project root) is 2 levels above script_path (src/component → project root)
# project_root = script_path.parents[2]
#
# # PNG directory (Baseline Budget inside src/data/results)
# base_png_dir = project_root / "src" / "data" / "results" / "Baseline Budget"
#
# # PDF/EPS directory
# latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
#
#
# def save_all_formats(fig, base_png_path: Path, latex_fig_dir: Path):
#     """
#     Saves:
#       - PNG in original location (base_png_path)
#       - PDF + EPS in the research_paper/Latex/fig directory
#     """
#     base_png_path.parent.mkdir(parents=True, exist_ok=True)
#     latex_fig_dir.mkdir(parents=True, exist_ok=True)
#
#     stem = base_png_path.stem  # filename without extension
#
#     # 1. PNG — high-res (stored in original location)
#     fig.savefig(base_png_path.with_suffix(".png"), dpi=300, bbox_inches="tight")  # <-- added
#
#     # 2. PDF — vector (saved to latex_fig_dir)
#     fig.savefig(latex_fig_dir / f"{stem}.pdf", bbox_inches="tight")  # <-- added
#
#     # 3. EPS — vector (saved to latex_fig_dir)
#     fig.savefig(latex_fig_dir / f"{stem}.eps", format="eps", bbox_inches="tight")  # <-- added
#
#     print(f"Saved PNG → {base_png_path.with_suffix('.png')}")
#     print(f"Saved PDF/EPS → {latex_fig_dir}/{stem}.*")

def prepare_optimization_data(df_breakfast, df_lunch, df_sizes):
    """
    Prepares all necessary data for the optimization models.
    """
    print("\nPreparing Optimization Data...")
    try:
        dfb = df_breakfast.copy()
        dfl = df_lunch.copy()

        # Normalize column names
        dfb.columns = dfb.columns.str.lower()
        dfl.columns = dfl.columns.str.lower()

        # Parse dates
        if 'date' in dfb.columns:
            dfb['date'] = pd.to_datetime(dfb['date'], errors='coerce')
        if 'date' in dfl.columns:
            dfl['date'] = pd.to_datetime(dfl['date'], errors='coerce')

        # Lowercase school names for consistent joins/groupbys
        if 'school_name' in dfb.columns:
            dfb['school_name'] = dfb['school_name'].astype(str).str.strip().str.lower()
        if 'school_name' in dfl.columns:
            dfl['school_name'] = dfl['school_name'].astype(str).str.strip().str.lower()

        # Coerce numerics
        for _df in (dfb, dfl):
            for col in ('served_reimbursable', 'production_cost_total'):
                if col in _df.columns:
                    _df[col] = (
                        _df[col].astype(str)
                                .str.replace(r'[$,]', '', regex=True)
                                .str.strip()
                    )
                    _df[col] = pd.to_numeric(_df[col], errors='coerce').fillna(0)

        # Schools & meal types
        schools = sorted(dfb['school_name'].dropna().unique().tolist())
        meal_types = ['Breakfast', 'Lunch']

        # Average costs (Used as a fallback and for the ILP objective)
        bf_sum = dfb['served_reimbursable'].sum()
        ln_sum = dfl['served_reimbursable'].sum()
        avg_bf_cost = (dfb['production_cost_total'].sum() / bf_sum) if bf_sum > 0 else 0.0
        avg_ln_cost = (dfl['production_cost_total'].sum() / ln_sum) if ln_sum > 0 else 0.0
        meal_costs = [float(avg_bf_cost), float(avg_ln_cost)]

        # Demand per school (avg per day)
        demand = {}
        for school in schools:
            bf_school = dfb[dfb['school_name'] == school]
            ln_school = dfl[dfl['school_name'] == school]
            bf_days = bf_school['date'].nunique() if 'date' in bf_school else 0
            ln_days = ln_school['date'].nunique() if 'date' in ln_school else 0
            avg_bf = (bf_school['served_reimbursable'].sum() / bf_days) if bf_days > 0 else 0.0
            avg_ln = (ln_school['served_reimbursable'].sum() / ln_days) if ln_days > 0 else 0.0
            demand[school] = [float(avg_bf), float(avg_ln)]

        # School size lists
        all_school_lists = None
        if df_sizes is not None:
            df_sizes = df_sizes.copy()
            df_sizes.columns = df_sizes.columns.str.lower()
            if 'school_name' in df_sizes.columns:
                df_sizes['school_name'] = df_sizes['school_name'].astype(str).str.strip().str.lower()
            if 'size_category' in df_sizes.columns:
                all_school_lists = {
                    size: group['school_name'].tolist()
                    for size, group in df_sizes.groupby('size_category', observed=False)
                }
            else:
                print("Warning: 'size_category' missing in sizes dataframe.")
        else:
            print("Warning: student size dataframe is None.")

        print("Data preparation complete!")
        return {
            "dfb": dfb, "dfl": dfl,
            "schools": schools, "meal_types": meal_types,
            "meal_costs": meal_costs, "demand": demand,
            "all_school_lists": all_school_lists, "df_sizes": df_sizes,
        }
    except Exception as e:
        print(f"Error during data preparation: {e}")
        traceback.print_exc()
        return None

# ==============================================================================
# OPTIMIZATION MODELS
# ==============================================================================

def run_meal_optimization(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds):
    """
    Runs the linear programming model and prints an output with production bounds
    """
    num_vars = len(schools_to_optimize) * len(meal_types)

    # Objective Function
    c = []
    variable_names = []
    for school in schools_to_optimize:
        for j, meal_type_str in enumerate(meal_types):
            meal_cost = meal_costs[j]
            total_cost = meal_cost + (0.1 * waste_penalty[j])
            c.append(total_cost)
            variable_names.append({'school': school, 'meal_type': meal_type_str})

    print("\nPerforming Linear Programming Optimization...")

    # Constraints
    A_ineq = []
    b_ineq = []

    # School Budgets
    for i, school in enumerate(schools_to_optimize):
        constraint = [0] * num_vars
        for j in range(len(meal_types)):
            var_index = i * len(meal_types) + j
            constraint[var_index] = meal_costs[j]
        A_ineq.append(constraint)
        b_ineq.append(school_budgets[school])

    # Total Budget
    total_budget_constraint = []
    for i in range(len(schools_to_optimize)):
        total_budget_constraint.extend(meal_costs)
    A_ineq.append(total_budget_constraint)
    b_ineq.append(total_budget)

    # Running the Optimization
    try:
        result = linprog(
            c=c,
            A_ub=A_ineq,
            b_ub=b_ineq,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            results_df = pd.DataFrame(variable_names)
            results_df['optimal_quantity'] = result.x.astype(int)
            return results_df
        else:
            print(f"\nOptimization failed: {result.message}")
            return None

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None
    
    print("Linear Programming Optimization Complete!")

def run_meal_optimization_ilp(schools_to_optimize, meal_types, meal_costs, 
                              min_budgets, max_budgets,
                              waste_penalty, bounds):
    """
    Runs the ILP model with per-school MIN and MAX budget constraints.
    """
    num_vars = len(schools_to_optimize) * len(meal_types)

    # Objective Function
    c = []
    variable_names = []
    for school in schools_to_optimize:
        for j, meal_type_str in enumerate(meal_types):
            meal_cost = meal_costs[j]
            total_cost = meal_cost + (0.1 * waste_penalty[j])
            c.append(total_cost)
            variable_names.append({'school': school, 'meal_type': meal_type_str})

    print("\nPerforming Integer Linear Programming Optimization with Budget Ranges...")
    
    # Create the constraint matrix
    A_matrix = []
    lower_budget_bounds = []
    upper_budget_bounds = []

    for i, school in enumerate(schools_to_optimize):
        constraint_row = [0] * num_vars
        for j in range(len(meal_types)):
            var_index = i * len(meal_types) + j
            constraint_row[var_index] = meal_costs[j] # Cost of each meal
        
        A_matrix.append(constraint_row)
        
        # Get the min and max budget for this school
        lower_budget_bounds.append(min_budgets.get(school, 0)) 
        upper_budget_bounds.append(max_budgets.get(school, float('inf')))

    # Create the LinearConstraint object, this single object defines: min_budget <= (cost expression) <= max_budget
    budget_constraints = LinearConstraint(A_matrix, lb=lower_budget_bounds, ub=upper_budget_bounds)

    # Bounds for production quantities
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    bounds_obj = Bounds(lb=lower_bounds, ub=upper_bounds)
    integrality = [1] * num_vars

    # Running the Optimization
    try:
        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds_obj,
            constraints=budget_constraints
        )

        # Minimal success/failure handling
        if result.success:
            results_df = pd.DataFrame(variable_names)
            results_df['optimal_quantity'] = result.x.astype(int)
            return results_df
        else:
            print(f"\nOptimization failed: {result.message}")
            return None

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None
    
    print("Integer Linear Programming Optimization Complete!")

# ==============================================================================
# PIPELINE HELPER FUNCTIONS
# ==============================================================================

def _get_optimized_annual_cost_df(data, results_df, unit_costs_path):
    """
    Calculates the annual food cost per school based on the
    new item-based unit cost logic. This is the single source of truth.
    """
    MONTHS_IN_SCHOOL_YEAR = 10
    
    try:
        dfb = data['dfb']
        dfl = data['dfl']

        # 1. Get popularity proportions
        bf_popularity = dfb.groupby('name')['served_reimbursable'].sum()
        bf_total_served = bf_popularity.sum()
        bf_popularity_prop = (bf_popularity / bf_total_served).reset_index(name='proportion')
        bf_popularity_prop['meal_type'] = 'Breakfast'

        ln_popularity = dfl.groupby('name')['served_reimbursable'].sum()
        ln_total_served = ln_popularity.sum()
        ln_popularity_prop = (ln_popularity / ln_total_served).reset_index(name='proportion')
        ln_popularity_prop['meal_type'] = 'Lunch'

        item_popularity_df = pd.concat([bf_popularity_prop, ln_popularity_prop])
        
        # 2. Get recommended item quantities
        item_df = pd.merge(results_df, item_popularity_df, on='meal_type')
        item_df['recommended_quantity'] = (item_df['optimal_quantity'] * item_df['proportion']).round().astype(int)
        optimized_df = item_df[item_df['recommended_quantity'] > 0]
        optimized_df = optimized_df[['school', 'meal_type', 'name', 'recommended_quantity']].rename(columns={'name': 'food_item'})

        # 3. Load and merge unit costs
        df_costs = pd.read_csv(unit_costs_path)
        df_costs['join_key'] = df_costs['name'].str.strip().str.lower()
        optimized_df['join_key'] = optimized_df['food_item'].str.strip().str.lower()
        
        df_items_with_cost = pd.merge(
            optimized_df, 
            df_costs[['join_key', 'unit_cost']], 
            on='join_key', 
            how='left'
        )
        df_items_with_cost['unit_cost'] = df_items_with_cost['unit_cost'].fillna(0)

        # 4. Calculate monthly and annual costs
        df_items_with_cost['monthly_food_cost'] = df_items_with_cost['recommended_quantity'] * df_items_with_cost['unit_cost']
        school_monthly_costs = df_items_with_cost.groupby('school')['monthly_food_cost'].sum().reset_index()
        school_monthly_costs['optimized_annual_cost'] = school_monthly_costs['monthly_food_cost'] * MONTHS_IN_SCHOOL_YEAR * 10
        
        return school_monthly_costs[['school', 'optimized_annual_cost']]

    except Exception as e:
        print(f"Error in _get_optimized_annual_cost_df: {e}")
        traceback.print_exc()
        return None

def prepare_savings_analysis_df(data, results_df, unit_costs_path):
    """
    Prepares a DataFrame comparing actual and optimized annual costs for each school.
    """

    if results_df is None:
        return None

    dfb = data['dfb']
    dfl = data['dfl']
    df_sizes = data['df_sizes']
    MONTHS_IN_SCHOOL_YEAR = 10

    # Calculate Optimized Annual Cost per School
    optimized_costs = _get_optimized_annual_cost_df(data, results_df, unit_costs_path)
    if optimized_costs is None:
        print("Error: Could not calculate optimized costs for savings analysis.")
        return None

    # Calculate Actual Annual Cost per School
    actual_costs_b = dfb.groupby('school_name')['production_cost_total'].sum()
    actual_costs_l = dfl.groupby('school_name')['production_cost_total'].sum()
    actual_costs = (actual_costs_b.add(actual_costs_l, fill_value=0) * MONTHS_IN_SCHOOL_YEAR).reset_index(name='actual_annual_cost')
    actual_costs.rename(columns={'school_name': 'school'}, inplace=True)

    # Combine into a single DataFrame
    savings_df = pd.merge(actual_costs, optimized_costs, on='school', how='left')
    savings_df['optimized_annual_cost'] = savings_df['optimized_annual_cost'].fillna(0)
    savings_df['savings'] = savings_df['actual_annual_cost'] - savings_df['optimized_annual_cost']
    
    # Create outcome columns
    savings_df['outcome'] = np.where(savings_df['savings'] >= 0, 'Savings', 'Loss')
    savings_df['savings_magnitude'] = savings_df['savings'].abs()
    
    # Add size category
    try:
        size_map = _compute_size_category(df_sizes, preferred_year="2024-2025")
        savings_df = pd.merge(
            savings_df,
            size_map,
            left_on='school',
            right_on='school_name',
            how='left'
        )
        savings_df.drop(columns=['school_name'], inplace=True, errors='ignore')
        savings_df['size_category'] = savings_df['size_category'].astype(str).fillna('unknown')
    except Exception:
        savings_df['size_category'] = 'unknown'

    return savings_df

def analyze_savings_by_school_size(opt_data, results_df, unit_costs_path, df_sizes):
    """
    Aggregates the budget, optimized cost, and savings by school size category.
    """

    if results_df is None or df_sizes is None:
        print("Skipping analysis: Missing results or size information.")
        return None

    print("\nCalculating Savings by School Size Category (vs. Historical)...")

    dfb = opt_data['dfb']
    dfl = opt_data['dfl']
    MONTHS_IN_SCHOOL_YEAR = 10

    # 1. Calculate Optimized Annual Cost per School (NEW LOGIC)
    school_costs = _get_optimized_annual_cost_df(opt_data, results_df, unit_costs_path)
    if school_costs is None:
        print("Error: Could not calculate optimized costs for size analysis.")
        return None
    # Rename column for merging
    school_costs.rename(columns={'optimized_annual_cost': 'annual_food_cost'}, inplace=True)


    # 2. Calculate Actual Historical Annual Cost per School (Same as before)
    actual_costs_b = dfb.groupby('school_name')['production_cost_total'].sum()
    actual_costs_l = dfl.groupby('school_name')['production_cost_total'].sum()
    actual_costs = (actual_costs_b.add(actual_costs_l, fill_value=0) * MONTHS_IN_SCHOOL_YEAR).reset_index(name='actual_annual_cost')
    actual_costs.rename(columns={'school_name': 'school'}, inplace=True)

    # 3. Combine and merge with size data
    analysis_df = pd.merge(actual_costs, school_costs, on='school', how='left')
    analysis_df = pd.merge(analysis_df, df_sizes[['school_name', 'size_category']], left_on='school', right_on='school_name', how='left')

    # 4. Group by size category and sum the totals
    agg_df = analysis_df.groupby('size_category', observed=True)[['actual_annual_cost', 'annual_food_cost']].sum(numeric_only=True).reset_index()
    
    # 5. Calculate savings
    agg_df['total_savings'] = agg_df['actual_annual_cost'] - agg_df['annual_food_cost']
    agg_df['percent_savings'] = (agg_df['total_savings'] / agg_df['actual_annual_cost']) * 100

    return agg_df

def generate_item_breakdown(optimization_results_df, dfb, dfl, output_filename):
    """
    Generates and saves the detailed food item breakdown from optimization results.
    """

    if optimization_results_df is None:
        print("Skipping item breakdown: No optimization results.")
        return
    
    # Generate Food Item Breakdown
    if optimization_results_df is not None:
        # Calculate popularity proportion for each itm within its meal type
        bf_popularity = dfb.groupby('name')['served_reimbursable'].sum()
        bf_total_served = bf_popularity.sum()
        bf_popularity_prop = (bf_popularity / bf_total_served).reset_index(name='proportion')
        bf_popularity_prop['meal_type'] = 'Breakfast'

        ln_popularity = dfl.groupby('name')['served_reimbursable'].sum()
        ln_total_served = ln_popularity.sum()
        ln_popularity_prop = (ln_popularity / ln_total_served).reset_index(name='proportion')
        ln_popularity_prop['meal_type'] = 'Lunch'

        # Combining popularity data
        item_popularity_df = pd.concat([bf_popularity_prop, ln_popularity_prop])

        # Merge optimization results with item popularity
        item_df = pd.merge(optimization_results_df, item_popularity_df, on='meal_type')

        # Calculating the recommended quantity for each specific item
        item_df['recommended_quantity'] = (item_df['optimal_quantity'] * item_df['proportion']).round().astype(int)

        # Clean up and display results
        optimized_df = item_df[item_df['recommended_quantity'] > 0]
        optimized_df = optimized_df[['school', 'meal_type', 'name', 'recommended_quantity']].rename(columns={'name': 'food_item'})

        optimized_df = optimized_df.sort_values(['school', 'meal_type', 'recommended_quantity'], ascending=[True, True, False])

        # Save as csv file
        if output_filename:
            output_file_path = Path(output_filename)
        else:
            script_dir = Path(__file__).resolve().parent
            src_dir = script_dir.parent
            project_root = src_dir.parent
            output_file_path = project_root / 'src' / 'data' / 'optimization-data' / 'school_food_item_optimization.csv'

        optimized_df.to_csv(output_file_path, index=False)
        print(f"Saved item breakdown to {output_file_path}")

    else:
        print("Optimization did not produce a result")

def calculate_actual_annual_cost(data):
    """
    Calculates the actual total food cost from the source data and extrapolates it to a full 10-month school year.
    """

    print("\nCalculating Baseline (Actual) Annual Food Cost...")
    dfb = data['dfb']
    dfl = data['dfl']
    
    # Sum the costs from the one-month data period
    actual_monthly_cost = dfb['production_cost_total'].sum() + dfl['production_cost_total'].sum()
    
    # Scale to a 10-month school year
    actual_annual_cost = actual_monthly_cost * 10
    
    print(f"Result: Actual cost for the data period (1 month): ${actual_monthly_cost:,.2f}")
    print(f"Result: Estimated Actual Annual Food Cost: ${actual_annual_cost:,.2f}")
    
    return actual_annual_cost

def analyze_annual_budget(results_df, school_budgets, meal_costs, data, unit_costs_path, actual_annual_cost=None):
    """
    Calculates the remaining annual budget for each school after scaling the optimized monthly food costs to a full 10-month school year.
    """

    if results_df is None:
        print("Skipping budget analysis: No optimization results available.")
        return None

    print("\nCalculating Annual Budget Analysis (using Item Unit Costs)...")

    MONTHS_IN_SCHOOL_YEAR = 10
    
    optimized_cost_df = _get_optimized_annual_cost_df(data, results_df, unit_costs_path)
    if optimized_cost_df is None:
        print("Failed to calculate optimized costs. Aborting budget analysis.")
        return None
        
    # Merge the optimized annual cost with the budget info
    optimized_cost_df['monthly_food_cost'] = optimized_cost_df['optimized_annual_cost'] / MONTHS_IN_SCHOOL_YEAR

    annual_budgets_list = [
        (school, monthly_budget * MONTHS_IN_SCHOOL_YEAR)
        for school, monthly_budget in school_budgets.items()
    ]

    # Create a DataFrame for the analysis
    budget_analysis_df = pd.DataFrame(annual_budgets_list, columns=['school', 'proportional_annual_budget'])
    budget_analysis_df = pd.merge(budget_analysis_df, optimized_cost_df, on='school', how='left').fillna(0)

    # Rename the 'optimized_annual_cost' column to 'annual_food_cost' to match the rest of the function
    budget_analysis_df.rename(columns={'optimized_annual_cost': 'annual_food_cost'}, inplace=True)

    # Calculate remaining balance
    budget_analysis_df['remaining_annual_balance'] = budget_analysis_df['proportional_annual_budget'] - budget_analysis_df['annual_food_cost']
    
    # Calculate and print the grand totals for the full year
    total_budget = budget_analysis_df['proportional_annual_budget'].sum()
    grand_total_food_cost = budget_analysis_df['annual_food_cost'].sum()
    grand_total_remaining = budget_analysis_df['remaining_annual_balance'].sum()

    print("\nResult: Overall Financial Summary (Annual)")
    print(f"- Total Allocated Annual Budget: ${total_budget:,.2f}")
    print(f"- Grand Total Annual Food Expenses: ${grand_total_food_cost:,.2f}")
    print(f"- Grand Total Remaining for Other Expenses: ${grand_total_remaining:,.2f}")

    if actual_annual_cost is not None:
        savings = actual_annual_cost - grand_total_food_cost
        savings_percent = (savings / actual_annual_cost) * 100 if actual_annual_cost > 0 else 0
        print("\nResult: Savings Analysis")
        print(f"- Baseline Actual Annual Food Cost: ${actual_annual_cost:,.2f}")
        print(f"- Optimized Annual Food Cost: ${grand_total_food_cost:,.2f}")
        print(f"- Total Annual Savings: ${savings:,.2f} ({savings_percent:.2f}%)")

    # Format for display
    print("\nCalculating Detailed Breakdown by School (Annual)...")
    print('Saving results as chart...')
    display_df = budget_analysis_df[['school', 'proportional_annual_budget', 'annual_food_cost', 'remaining_annual_balance']].copy()
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].map('${:,.2f}'.format)

    return display_df

# ==============================================================================
# GEOSPATIAL AND PLOTTING FUNCTIONS
# ==============================================================================

def generate_overall_savings_bar_chart(opt_data, monthly_results_df, unit_costs_path, out_dir=None,
                                       file_suffix: str = ""):
    """
    Generates Total Savings Bar Chart.
    - Colors: Blue (Actual), Green (Optimized)
    - Saves PDF/EPS ONLY if it's the baseline run.
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to plot.")
        return None

    # --- PATH SETUP ---
    current_path = Path(__file__).resolve()
    project_root = current_path
    while project_root.name != 'src' and project_root.parent != project_root:
        project_root = project_root.parent
    project_root = project_root.parent

    latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
    latex_fig_dir.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        png_out_dir = project_root / "src" / "data" / "results"
    else:
        png_out_dir = Path(out_dir)
    png_out_dir.mkdir(parents=True, exist_ok=True)

    # --- FILENAME LOGIC ---
    if "baseline" in file_suffix:
        latex_filename = "total_savings_bar_color"
    else:
        latex_filename = f"total_savings_bar_color{file_suffix}"

    # --- DATA & PLOTTING ---
    actual_annual_cost = calculate_actual_annual_cost(opt_data)
    optimized_cost_df = _get_optimized_annual_cost_df(opt_data, monthly_results_df, unit_costs_path)

    if optimized_cost_df is None:
        return None

    optimized_annual_cost = optimized_cost_df['optimized_annual_cost'].sum()

    labels = ['Actual Cost (Baseline)', 'Optimized Cost']
    values = [actual_annual_cost, optimized_annual_cost]
    
    # COLORS: Red for Actual, Green for Optimized
    colors = ['#d62728', '#2ca02c']

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_title(f'Total Savings Analysis', fontsize=14)
    ax.set_ylabel('Cost (Millions $)', fontsize=11)
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x / 1e6:.0f}M'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'${h / 1e6:.2f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.tight_layout()

    # --- SAVE LOGIC (PDF/EPS only for Baseline) ---
    if "baseline" in file_suffix:
        fig.savefig(latex_fig_dir / f"{latex_filename}.pdf", format='pdf', bbox_inches='tight')
        fig.savefig(latex_fig_dir / f"{latex_filename}.eps", format='eps', bbox_inches='tight')
        print(f"DEBUG: Saved PDF/EPS to: {latex_fig_dir.resolve()}")

    # Always save PNG
    png_path = png_out_dir / f"{latex_filename}.png"
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight')

    plt.close(fig)
    return png_path


def generate_savings_analysis_chart(opt_data, monthly_results_df, unit_costs_path, out_dir=None, file_suffix: str = ""):
    """
    Generates Scatter Plot of Savings vs Loss.
    - Colors: Green (Savings), Red (Loss)
    - Saves PDF/EPS ONLY if it's the baseline run.
    """
    if monthly_results_df is None:
        return None

    savings_df = prepare_savings_analysis_df(opt_data, monthly_results_df, unit_costs_path)
    if savings_df is None or savings_df.empty:
        return None

    # --- PATH SETUP ---
    current_path = Path(__file__).resolve()
    project_root = current_path
    while project_root.name != 'src' and project_root.parent != project_root:
        project_root = project_root.parent
    project_root = project_root.parent

    latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
    latex_fig_dir.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        png_out_dir = project_root / "src" / "data" / "results"
    else:
        png_out_dir = Path(out_dir)
    png_out_dir.mkdir(parents=True, exist_ok=True)

    # --- FILENAME LOGIC ---
    if "baseline" in file_suffix:
        latex_filename = "savings_analysis_color"
    else:
        latex_filename = f"savings_analysis_color{file_suffix}"

    # --- CHART ---
    fig_static, ax = plt.subplots(figsize=(12, 6))

    savings_group = savings_df[savings_df['outcome'] == 'Savings']
    loss_group = savings_df[savings_df['outcome'] == 'Loss']

    max_mag = savings_df['savings_magnitude'].max()
    s_savings = (savings_group['savings_magnitude'] / max_mag) * 300 + 20
    s_loss = (loss_group['savings_magnitude'] / max_mag) * 300 + 20

    # COLORS: Green for Savings, Red for Loss
    ax.scatter(savings_group['actual_annual_cost'], savings_group['optimized_annual_cost'],
               s=s_savings, c='#2ca02c', alpha=0.6, edgecolors='none', label='Savings')

    ax.scatter(loss_group['actual_annual_cost'], loss_group['optimized_annual_cost'],
               s=s_loss, c='#d62728', alpha=0.7, edgecolors='none', label='Projected Cost Increase')

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='Break-even')

    ax.set_title(f"Actual vs. Optimized Cost", fontsize=14)
    ax.set_xlabel("Actual Annual Cost ($)", fontsize=11)
    ax.set_ylabel("Optimized Annual Cost ($)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x / 1e6:.1f}M'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x / 1e6:.1f}M'))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=9)

    fig_static.tight_layout()

    # --- SAVE LOGIC (PDF/EPS only for Baseline) ---
    if "baseline" in file_suffix:
        fig_static.savefig(latex_fig_dir / f"{latex_filename}.pdf", format='pdf', bbox_inches='tight')
        fig_static.savefig(latex_fig_dir / f"{latex_filename}.eps", format='eps', bbox_inches='tight')
        print(f"DEBUG: Saved PDF/EPS to: {latex_fig_dir.resolve()}")

    # Always save PNG
    fig_static.savefig(png_out_dir / f"{latex_filename}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_static)

    return png_out_dir / f"{latex_filename}.png"


def generate_savings_by_size_charts(opt_data, monthly_results_df, unit_costs_path, out_dir=None, file_suffix: str = ""):
    """
    Generates two bar charts: Total Savings and Percent Savings by School Size.
    - Colors: Blue for Total Savings, Green for Percent Savings.
    - Saves PDF/EPS ONLY if it's the baseline run.
    """
    if monthly_results_df is None:
        return None

    # --- PATH SETUP ---
    current_path = Path(__file__).resolve()
    project_root = current_path
    while project_root.name != 'src' and project_root.parent != project_root:
        project_root = project_root.parent
    project_root = project_root.parent

    latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
    latex_fig_dir.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        png_out_dir = project_root / "src" / "data" / "results"
    else:
        png_out_dir = Path(out_dir)
    png_out_dir.mkdir(parents=True, exist_ok=True)

    # --- DATA PREP ---
    try:
        size_map = _compute_size_category(opt_data['df_sizes'], preferred_year="2024-2025")
        df_sizes = opt_data['df_sizes'].copy()
        df_sizes.columns = df_sizes.columns.str.lower()
        df_sizes = df_sizes.merge(size_map, on='school_name', how='left', suffixes=('', '_mapped'))
        if 'size_category_mapped' in df_sizes.columns:
            df_sizes['size_category'] = df_sizes['size_category_mapped']
        df_sizes.drop(columns=[c for c in ['size_category_mapped'] if c in df_sizes.columns], inplace=True,
                      errors='ignore')
    except Exception:
        df_sizes = opt_data['df_sizes']

    savings_by_size_df = analyze_savings_by_school_size(opt_data, monthly_results_df, unit_costs_path, df_sizes)
    if savings_by_size_df is None or savings_by_size_df.empty:
        return None

    range_map = {'xxs': '0-499', 'xs': '500-999', 's': '1000-1499', 'm': '1500-1999', 'l': '2000-2499',
                 'xl': '2500-2999', 'xxl': '3000-3499', 'xxxl': '3500+'}
    order = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
    savings_by_size_df['count_range'] = savings_by_size_df['size_category'].map(range_map)
    savings_by_size_df['size_category'] = pd.Categorical(savings_by_size_df['size_category'], categories=order,
                                                         ordered=True)
    savings_by_size_df = savings_by_size_df.sort_values('size_category')

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass

    # --- CHART 1: TOTAL SAVINGS (Color: Blue) ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(savings_by_size_df['count_range'], savings_by_size_df['total_savings'], color='#1f77b4')
    ax1.set_title(f'Total Annual Savings by School Size', fontsize=14)
    ax1.set_xlabel("Student Population", fontsize=11)
    ax1.set_ylabel("Total Savings ($)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x / 1000:,.0f}k'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    fig1.tight_layout()

    # Filename Logic
    if "baseline" in file_suffix:
        base_name1 = "savings_by_size_total_color"
    else:
        base_name1 = f"savings_by_size_total_color{file_suffix}"

    # Save logic (PDF/EPS only for Baseline)
    if "baseline" in file_suffix:
        fig1.savefig(latex_fig_dir / f"{base_name1}.pdf", format='pdf', bbox_inches='tight')
        fig1.savefig(latex_fig_dir / f"{base_name1}.eps", format='eps', bbox_inches='tight')
    
    # Always save PNG
    fig1.savefig(png_out_dir / f"{base_name1}.png", format='png', dpi=600, bbox_inches='tight')
    plt.close(fig1)

    # --- CHART 2: PERCENT SAVINGS (Color: Green) ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    # CHANGED: Using Green instead of Purple
    ax2.bar(savings_by_size_df['count_range'], savings_by_size_df['percent_savings'], color='#2ca02c')
    ax2.set_title(f'Percentage of Budget Saved by School Size', fontsize=14)
    ax2.set_xlabel("Student Population", fontsize=11)
    ax2.set_ylabel("Savings (%)", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig2.tight_layout()

    if "baseline" in file_suffix:
        base_name2 = "savings_by_size_percent_color"
    else:
        base_name2 = f"savings_by_size_percent_color{file_suffix}"

    # Save logic (PDF/EPS only for Baseline)
    if "baseline" in file_suffix:
        fig2.savefig(latex_fig_dir / f"{base_name2}.pdf", format='pdf', bbox_inches='tight')
        fig2.savefig(latex_fig_dir / f"{base_name2}.eps", format='eps', bbox_inches='tight')

    # Always save PNG
    fig2.savefig(png_out_dir / f"{base_name2}.png", format='png', dpi=600, bbox_inches='tight')
    plt.close(fig2)

    return {"total_savings_png": png_out_dir / f"{base_name1}.png"}

def prepare_map_data_from_coordinates(savings_df, coordinates_csv_path):
    """
    Merges savings data with a CSV file containing school coordinates.
    """
    if savings_df is None:
        print("Skipping map data prep: No savings data available.")
        return None
    
    print("\nPreparing Map Data from Coordinates...")
    
    try:
        # Read the coordinates file
        coords_df = pd.read_csv(coordinates_csv_path, low_memory=False)
        coords_df.columns = coords_df.columns.str.lower()
        
        # Ensure school name column is consistent and select necessary columns
        if 'school_name' in coords_df.columns and 'school' not in coords_df.columns:
            coords_df.rename(columns={'school_name': 'school'}, inplace=True)
        
        if 'school' in coords_df.columns:
             coords_df['school'] = coords_df['school'].astype(str).str.strip().str.lower()
        else:
            print("Error: Could not find 'school_name' or 'school' column in coordinates file.")
            return None

        # Keep only the essential columns and remove duplicates
        # We keep 'latitude' and 'longitude' specifically
        coords_df = coords_df[['school', 'latitude', 'longitude']].drop_duplicates(subset='school')
        
        # Merge with savings data
        map_df = pd.merge(savings_df, coords_df, on='school', how='inner')
        
        # Drop rows with missing coordinates
        map_df.dropna(subset=['latitude', 'longitude'], inplace=True)

        print(f"Successfully merged coordinate data for {len(map_df)} schools.")
        return map_df
        
    except FileNotFoundError:
        print(f"Error: Coordinates file not found at {coordinates_csv_path}")
        return None
    except Exception as e:
        print(f"An error occurred during map data preparation: {e}")
        return None

def _load_coords_and_normalize(coords_csv_path):
    """
    Load coordinates and normalize the school column.
    """
    coords_df = pd.read_csv(coords_csv_path, low_memory=False)
    coords_df.columns = coords_df.columns.str.lower()
    if "school_name" in coords_df.columns and "school" not in coords_df.columns:
        coords_df = coords_df.rename(columns={"school_name": "school"})
    if "school" not in coords_df.columns:
        raise KeyError("Coordinates CSV needs a 'school' or 'school_name' column.")
    coords_df["school"] = coords_df["school"].astype(str).str.strip().str.lower()
    return coords_df

# ==============================================================================
# GEOSPATIAL AND PLOTTING FUNCTIONS (UPDATED WITH STATIC MAPS)
# ==============================================================================

def generate_savings_map(opt_data, monthly_results_df, unit_costs_path, coordinates_file, 
                         geojson_path=None, out_dir=None, file_suffix: str = ""):
    """
    1. Builds an interactive Folium bubble map (HTML).
    2. If Baseline: Generates a static Matplotlib map (PDF/EPS).
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to map.")
        return None

    # --- DATA PREP ---
    savings_df = prepare_savings_analysis_df(opt_data, monthly_results_df, unit_costs_path)
    if savings_df is None or savings_df.empty:
        return None

    map_df = prepare_map_data_from_coordinates(savings_df, coordinates_file)
    if map_df is None or map_df.empty:
        return None

    # --- PATH SETUP ---
    current_path = Path(__file__).resolve()
    project_root = current_path
    while project_root.name != 'src' and project_root.parent != project_root:
        project_root = project_root.parent
    project_root = project_root.parent

    latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
    latex_fig_dir.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        out_dir = project_root / "src" / "data" / "results"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. INTERACTIVE MAP (FOLIUM) - HTML
    # ---------------------------------------------------------
    try:
        lat0 = float(map_df['latitude'].mean())
        lon0 = float(map_df['longitude'].mean())
        center = [lat0, lon0]
    except Exception:
        center = [38.83, -77.27]

    m = folium.Map(location=center, zoom_start=10)
    max_abs = map_df['savings_magnitude'].max() if 'savings_magnitude' in map_df else None
    
    def scale_radius(v):
        if max_abs and v and v > 0:
            return (v / max_abs) ** (1 / 3) * 20 + 2
        return 2

    for _, row in map_df.iterrows():
        try:
            color = '#2ca02c' if row.get('outcome') == 'Savings' else '#d62728' # Green / Red
            popup_text = (
                f"<strong>School:</strong> {str(row.get('school','')).title()}<br>"
                f"<strong>Annual Savings:</strong> ${float(row.get('savings',0)):,.2f}"
            )
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=scale_radius(row.get('savings_magnitude', 0)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        except Exception:
            continue

    html_path = out_dir / f"overall_savings_map{file_suffix}.html"
    m.save(str(html_path))
    print(f"Saved interactive savings map to: {html_path}")

    # ---------------------------------------------------------
    # 2. STATIC MAP (MATPLOTLIB) - PDF/EPS (Baseline Only)
    # ---------------------------------------------------------
    if "baseline" in file_suffix:
        print("Generating static savings map (PDF/EPS)...")
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot Background (GeoJSON) if available
            if geojson_path and Path(geojson_path).exists():
                gdf_bg = gpd.read_file(geojson_path)
                gdf_bg.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc')
            
            # Prepare Data for Scatter
            # Colors: Green (#2ca02c) for Savings, Red (#d62728) for Loss
            colors = map_df['outcome'].apply(lambda x: '#2ca02c' if x == 'Savings' else '#d62728')
            
            # Size: Scale size for visibility
            sizes = map_df['savings_magnitude'] / map_df['savings_magnitude'].max() * 200 + 10

            scatter = ax.scatter(
                map_df['longitude'], 
                map_df['latitude'], 
                c=colors, 
                s=sizes, 
                alpha=0.7, 
                edgecolors='white', 
                linewidth=0.5
            )

            # Custom Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', label='Savings', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='Loss', markersize=10)
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            ax.set_title("Overall Savings by School", fontsize=14)
            ax.axis('off') # Hide axes for map look
            
            # Save
            latex_filename = "overall_savings_map_static"
            fig.savefig(latex_fig_dir / f"{latex_filename}.pdf", format='pdf', bbox_inches='tight')
            fig.savefig(latex_fig_dir / f"{latex_filename}.eps", format='eps', bbox_inches='tight')
            plt.close(fig)
            print(f"Saved static savings map to {latex_fig_dir}")
            
        except Exception as e:
            print(f"Failed to create static savings map: {e}")
            traceback.print_exc()

    return html_path


def generate_savings_maps_by_level(opt_data, monthly_results_df, unit_costs_path, coordinates_file, 
                                   geojson_path=None, out_dir=None, file_suffix: str = ""):
    """
    1. Creates interactive maps per level (HTML).
    2. If Baseline: Creates static maps per level (PDF/EPS).
    """
    if monthly_results_df is None:
        return None

    savings_df = prepare_savings_analysis_df(opt_data, monthly_results_df, unit_costs_path)
    if savings_df is None or savings_df.empty:
        return None

    coords_df = pd.read_csv(coordinates_file, low_memory=False)
    coords_df.columns = coords_df.columns.str.lower()

    if 'school_name' in coords_df.columns and 'school' not in coords_df.columns:
        coords_df = coords_df.rename(columns={'school_name': 'school'})
    if 'school' not in coords_df.columns:
        return None
    coords_df['school'] = coords_df['school'].astype(str).str.strip().str.lower()

    if 'level' not in coords_df.columns:
        return None

    map_df = pd.merge(
        savings_df,
        coords_df[['school', 'latitude', 'longitude', 'level']],
        on='school',
        how='inner'
    ).dropna(subset=['latitude', 'longitude'])

    if 'savings_magnitude' not in map_df.columns:
        map_df['savings_magnitude'] = map_df['savings'].abs()
    if 'outcome' not in map_df.columns:
        map_df['outcome'] = np.where(map_df['savings'] >= 0, 'Savings', 'Loss')

    # Setup Output Dirs
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
    latex_fig_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(out_dir) if out_dir else (project_root / "src" / "data" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- INTERACTIVE (HTML) ---
    out_paths = {}
    level_map = [('ES', 'elementary'), ('MS', 'middle'), ('HS', 'high')]
    
    # Pre-calculate max for consistent scaling across HTML maps
    vmax_global = map_df['savings_magnitude'].max()

    for code, label in level_map:
        df_level = map_df[map_df['level'] == code]
        if df_level.empty:
            continue
            
        # HTML Map
        try:
            center = [df_level['latitude'].mean(), df_level['longitude'].mean()]
        except:
            center = [38.83, -77.27]
            
        m = folium.Map(location=center, zoom_start=10)
        
        for _, row in df_level.iterrows():
            color = '#2ca02c' if row['outcome'] == 'Savings' else '#d62728'
            radius = (row['savings_magnitude'] / vmax_global) ** (1/3) * 20 + 2 if vmax_global > 0 else 2
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.5,
                popup=f"{row['school']}: ${row['savings']:,.0f}"
            ).add_to(m)
            
        html_file = results_dir / f"savings_map_{label}{file_suffix}.html"
        m.save(str(html_file))
        out_paths[label] = html_file
        
        # --- STATIC MAP (PDF/EPS) - BASELINE ONLY ---
        if "baseline" in file_suffix:
            try:
                fig, ax = plt.subplots(figsize=(8, 8))
                if geojson_path and Path(geojson_path).exists():
                    gdf_bg = gpd.read_file(geojson_path)
                    gdf_bg.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc')
                
                colors = df_level['outcome'].apply(lambda x: '#2ca02c' if x == 'Savings' else '#d62728')
                sizes = df_level['savings_magnitude'] / vmax_global * 200 + 10

                ax.scatter(df_level['longitude'], df_level['latitude'], c=colors, s=sizes, alpha=0.7, edgecolors='white')
                ax.set_title(f"Savings: {label.title()} Schools", fontsize=14)
                ax.axis('off')

                # Legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', label='Savings'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='Loss')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                latex_name = f"savings_map_{label}_static"
                fig.savefig(latex_fig_dir / f"{latex_name}.pdf", bbox_inches='tight')
                fig.savefig(latex_fig_dir / f"{latex_name}.eps", format='eps', bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Error static map {label}: {e}")

    return out_paths

def generate_fcps_region_choropleth(
    savings_df: pd.DataFrame,
    coords_csv_path: str | Path,
    geojson_path: str | Path,
    bins: int | None = 6,
    out_path: str | Path | None = None,
    map_title: str | None = None,
    file_suffix: str = ""
):
    """
    Generates Choropleth (HTML) AND Static Map (PDF/EPS) if baseline.
    """
    if savings_df is None or savings_df.empty:
        return None, None, None

    # --- Data Prep ---
    coords = pd.read_csv(coords_csv_path, low_memory=False)
    coords.columns = coords.columns.str.lower()
    if "school_name" in coords.columns and "school" not in coords.columns:
        coords = coords.rename(columns={"school_name": "school"})
    
    # Normalize
    coords["school"] = coords["school"].astype(str).str.strip().str.lower()
    
    # Merge
    lookup = coords[["school", "fcps region"]].drop_duplicates(subset="school")
    sdf = savings_df.copy()
    sdf["school"] = sdf["school"].astype(str).str.strip().str.lower()
    sdf["savings"] = pd.to_numeric(sdf["savings"], errors="coerce").fillna(0.0)
    
    merged = sdf.merge(lookup, on="school", how="inner")
    if merged.empty:
        return None, None, None

    # Aggregation
    regional = merged.groupby("fcps region", as_index=False).agg(total_optimization_savings=("savings", "sum"))
    regional["REGION"] = regional["fcps region"].astype(str).str.extract(r"(\d+)").astype(int)
    
    # Load GeoJSON
    gdf = gpd.read_file(geojson_path)
    # Ensure REGION is int for merge
    if 'REGION' in gdf.columns:
        gdf['REGION'] = gdf['REGION'].astype(int)
    
    # Merge Data into GeoDataFrame
    gdf_merged = gdf.merge(regional, on='REGION', how='left')
    gdf_merged['total_optimization_savings'] = gdf_merged['total_optimization_savings'].fillna(0)

    # --- FIX: Convert Timestamp columns to strings for JSON serialization ---
    for col in gdf_merged.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf_merged[col]):
            gdf_merged[col] = gdf_merged[col].astype(str)

    # --- 1. HTML INTERACTIVE MAP ---
    try:
        m = folium.Map(location=[38.85, -77.30], zoom_start=10, tiles="cartodbpositron")
        
        # We use gdf_merged.to_json() which should now work since datetimes are strings
        folium.Choropleth(
            geo_data=json.loads(gdf_merged.to_json()),
            data=regional,
            columns=["REGION", "total_optimization_savings"],
            key_on="feature.properties.REGION",
            fill_color="RdYlGn",
            fill_opacity=0.7,
            legend_name="Total Savings ($)"
        ).add_to(m)
        
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            m.save(str(out_path))

    except Exception as e:
        print(f"Error generating interactive choropleth: {e}")
        traceback.print_exc()

    # --- 2. STATIC MAP (PDF/EPS) - BASELINE ONLY ---
    if "baseline" in file_suffix:
        # Import patheffects for white outline around text
        import matplotlib.patheffects as patheffects

        # Path setup
        current_path = Path(__file__).resolve()
        project_root = current_path.parents[2]
        latex_fig_dir = project_root / "research_paper" / "Latex" / "fig"
        latex_fig_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create a categorical column for the Legend
            gdf_merged['Region_Label'] = "Region " + gdf_merged['REGION'].astype(str)
            
            # Plot: Color by Region (Categorical)
            gdf_merged.plot(
                column='Region_Label',
                cmap='Pastel1',       # Use distinct colors (Pastel1, Set3, or tab10)
                linewidth=0.8,
                edgecolor='black',
                legend=True,
                legend_kwds={'title': "FCPS Regions", 'loc': 'lower right'},
                ax=ax
            )
            
            # Add Savings Labels to the Center of Each Region
            for idx, row in gdf_merged.iterrows():
                # Calculate the center point of the polygon
                centroid = row.geometry.representative_point()
                
                # Format the savings value
                savings_val = row['total_optimization_savings']
                if abs(savings_val) >= 1_000_000:
                    label_text = f"${savings_val/1e6:.2f}M"
                else:
                    label_text = f"${savings_val/1e3:.0f}k"
                
                # Annotate map
                ax.annotate(
                    text=label_text,
                    xy=(centroid.x, centroid.y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    fontsize=11,
                    fontweight='bold',
                    color='black',
                    # Add white outline to text for readability against colors
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
                )
            
            clean_title = map_title.replace("FCPS Regions — ", "").replace(" ", "_").lower() if map_title else "choropleth"
            ax.set_title(f"{map_title} (Savings/Year)", fontsize=14)
            ax.axis('off')
            
            # Filename
            base_name = f"choropleth_{clean_title}_baseline"
            fig.savefig(latex_fig_dir / f"{base_name}.pdf", bbox_inches='tight')
            fig.savefig(latex_fig_dir / f"{base_name}.eps", format='eps', bbox_inches='tight')
            plt.close(fig)
            print(f"Saved static choropleth (Regions + Labels) to {latex_fig_dir / base_name}")

        except Exception as e:
            print(f"Error creating static choropleth: {e}")
            traceback.print_exc()

    return m, regional, out_path

def generate_all_region_choropleths(
    opt_data: dict,
    monthly_results_df: pd.DataFrame,
    unit_costs_path: str | Path,
    coords_csv_path: str | Path,
    geojson_path: str | Path,
    file_suffix: str = "",
    out_dir: str | Path | None = None
):
    """
    Wrapper to generate Overall + Per-Level choropleths (HTML + PDF/EPS).
    """
    if monthly_results_df is None:
        return None

    savings_df = prepare_savings_analysis_df(opt_data, monthly_results_df, unit_costs_path)
    if savings_df is None or savings_df.empty:
        return None
        
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "src" / "data" / "results"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Overall
    _, regional_df, path_overall = generate_fcps_region_choropleth(
        savings_df,
        coords_csv_path=coords_csv_path,
        geojson_path=geojson_path,
        bins=None,
        out_path=out_dir / f"fcps_region_choropleth_overall{file_suffix}.html",
        map_title="FCPS Regions — Overall Savings",
        file_suffix=file_suffix  # Pass suffix to trigger PDF gen
    )
    results["overall_html"] = path_overall

    # 2. Per Level
    coords_df = _load_coords_and_normalize(coords_csv_path)
    if "level" in coords_df.columns:
        savings_with_level = pd.merge(
            savings_df,
            coords_df[["school", "level"]],
            on="school",
            how="left"
        )

        for level_code, label in [("ES", "elementary"), ("MS", "middle"), ("HS", "high")]:
            filtered = savings_with_level[savings_with_level["level"] == level_code]
            if filtered.empty:
                continue
            _, reg_df, path_level = generate_fcps_region_choropleth(
                filtered,
                coords_csv_path=coords_csv_path,
                geojson_path=geojson_path,
                bins=None,
                out_path=out_dir / f"fcps_region_choropleth_{label}{file_suffix}.html",
                map_title=f"FCPS Regions — {label.title()} Schools",
                file_suffix=file_suffix # Pass suffix
            )
            results[f"{label}_html"] = path_level

    return results

# ==============================================================================
# PIPELINE RUNNERS
# ==============================================================================

def run_proportional_monthly_optimization_ilp(data, budget_cap_factor: float):
    """
    Runs a monthly optimization where the budget for each school
    is capped at a percentage (budget_cap_factor) of its historical demand cost.
    
    The production bounds are also adjusted to be relative to the budget cap
    to prevent infeasible solutions.
    """
    schools = data['schools']
    meal_types = data['meal_types']
    meal_costs = data['meal_costs']
    daily_demand = data['demand']
    dfb = data['dfb']
    dfl = data['dfl']

    # Using the Historical Demand Cost as budget baseline
    print(f"\n--- Running Optimization for Budget Cap: {budget_cap_factor*100:.0f}% ---")
    print("Calculating budget baseline from the cost of historical demand...")
    SCHOOL_DAYS_PER_MONTH = 20
    baseline_budgets = {}
    for school in schools:
        d_bf = daily_demand[school][0]
        d_ln = daily_demand[school][1]
        cost_bf = (d_bf * SCHOOL_DAYS_PER_MONTH) * meal_costs[0]
        cost_ln = (d_ln * SCHOOL_DAYS_PER_MONTH) * meal_costs[1]
        baseline_budgets[school] = cost_bf + cost_ln

    # Create Min/Max Budget Range from Baseline
    min_school_budgets = {s: 0 for s in baseline_budgets.keys()}
    max_school_budgets = {s: b * budget_cap_factor for s, b in baseline_budgets.items()}
    
    # Set a floor so we don't go below 70% production to prevent infeasible solutions
    min_prod_factor = max(0.70, budget_cap_factor - 0.15) 
    max_prod_factor = budget_cap_factor + 0.15
    
    print(f"Using dynamic production bounds: {min_prod_factor*100:.0f}% - {max_prod_factor*100:.0f}% of demand")

    waste_penalty = [0.50, 1.00]
    monthly_demand = {s: [d[0] * SCHOOL_DAYS_PER_MONTH, d[1] * SCHOOL_DAYS_PER_MONTH] for s, d in daily_demand.items()}
    monthly_bounds = []
    for school in schools:
        for i in range(len(meal_types)):
            demand_val = monthly_demand[school][i]
            monthly_bounds.append((demand_val * min_prod_factor, 
                                   demand_val * max_prod_factor))

    # Run the Optimization
    results_df = run_meal_optimization_ilp(
        schools, meal_types, meal_costs,
        min_school_budgets, max_school_budgets, 
        waste_penalty, monthly_bounds
    )
    
    return results_df, max_school_budgets, meal_costs

def _compute_size_category(df_sizes, preferred_year="2024-2025"):
    """
    Return a slim dataframe with ['school_name','size_category'] based on student counts.
    """

    sizes = df_sizes.copy()
    sizes.columns = sizes.columns.str.lower()

    # Normalize school name
    if 'school_name' not in sizes.columns:
        raise KeyError("Expected 'school_name' in df_sizes")
    sizes['school_name'] = sizes['school_name'].astype(str).str.strip().str.lower()

    # The year column
    year_col = preferred_year.lower() if preferred_year.lower() in sizes.columns else None
    if year_col is None:
        year_cols = []
        for c in sizes.columns:
            m = re.search(r'(\d{4})\D+(\d{4})', str(c))
            if m:
                start, end = int(m.group(1)), int(m.group(2))
                year_cols.append((c, max(start, end)))
        if not year_cols:
            raise KeyError("No year columns like '2024-2025' found in df_sizes")
        year_col = sorted(year_cols, key=lambda x: x[1])[-1][0]

    # Coerce counts
    sizes[year_col] = pd.to_numeric(sizes[year_col], errors='coerce').fillna(0)

    # Bin edges and labels per your spec
    # 0–499 → xxs, 500–999 → xs, ..., 3500+ → xxxl
    bins = [-1, 499, 999, 1499, 1999, 2499, 2999, 3499, float('inf')]
    labels = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']

    sizes['size_category'] = pd.cut(sizes[year_col], bins=bins, labels=labels, right=True)

    return sizes[['school_name', 'size_category']].drop_duplicates('school_name')

def run_daily_ilp_pipeline(
    opt_data: dict,
    total_budget: float = 139144760,
    waste_penalty = (0.50, 1.00),
    bounds_pct = (0.90, 1.10),
    save_item_breakdown: bool = True,
    item_breakdown_path: str | None = None
):
    """
    Runs the daily ILP pipeline using a 100% budget cap derived from the cost of daily demand.
    """

    if not opt_data:
        print("No optimization data provided.")
        return None

    # Use cost of daily demand as budget baseline
    print("Calculating daily budget baseline from the cost of daily demand...")
    baseline_budgets = {}
    for school in opt_data['schools']:
        d_bf = opt_data['demand'][school][0]
        d_ln = opt_data['demand'][school][1]
        cost_bf = d_bf * opt_data['meal_costs'][0]
        cost_ln = d_ln * opt_data['meal_costs'][1]
        baseline_budgets[school] = cost_bf + cost_ln

    # Set budget cap at 100% of the baseline cost
    min_school_budgets = {s: 0 for s in baseline_budgets.keys()}
    max_school_budgets = {s: b * 1.00 for s, b in baseline_budgets.items()}
    
    # Get Production Bounds
    daily_bounds = _daily_bounds_from_demand(opt_data, *bounds_pct)

    # Run the Optimization
    ilp_results = run_meal_optimization_ilp(
        opt_data['schools'],
        opt_data['meal_types'],
        opt_data['meal_costs'],
        min_school_budgets,
        max_school_budgets,
        list(waste_penalty),
        daily_bounds
    )

    if save_item_breakdown and ilp_results is not None:
        if item_breakdown_path is None:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent.parent
            item_breakdown_path = project_root / 'src' / 'data' / 'optimization-data' / 'school_food_item_optimization_ilp.csv'
        
        generate_item_breakdown(
            ilp_results, 
            opt_data['dfb'], 
            opt_data['dfl'], 
            str(item_breakdown_path)
        )

    return ilp_results

def run_monthly_proportional_pipeline(
    opt_data: dict,
    unit_costs_file: str | Path,
    total_budget: float = 139144760,
    coordinates_file: str | Path | None = None,
    geojson_file: str | Path | None = None
):
    """
    Runs the monthly ILP optimization for three scenarios and saves all CSVs and maps into their own subfolders.
    *** MODIFIED to pass unit_costs_file to all chart functions ***
    """
    
    run_maps = coordinates_file and geojson_file
    if not run_maps:
        print("\n[Warning] `coordinates_file` or `geojson_file` not provided.")
        print("CSV files will be generated, but maps will be skipped.")
        
    # Define the scenarios with their new folder names
    scenarios = [
        ("lower_bound", 0.80, "Lower Budget Bounds"),
        ("baseline", 1.00, "Baseline Budget"),
        ("upper_bound", 1.20, "Upper Budget bounds")
    ]
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    baseline_results = {}
    actual_cost_2025 = calculate_actual_annual_cost(opt_data)

    for name, factor, folder_name in scenarios:
        print(f"\n{'='*60}")
        print(f"RUNNING MONTHLY PIPELINE FOR: {folder_name}")
        print(f"{'='*60}")
        
        file_suffix = f"_{name}" # e.g., "_lower_bound"
        
        # Define the output directory for CSVs
        csv_output_dir = project_root / 'src' / 'data' / 'optimization-data'
        analysis_breakdown_path = csv_output_dir / f'annual_school_breakdown{file_suffix}.csv'
        
        # Define the output directory for all graphs for this scenario
        graph_output_dir = project_root / "src" / "data" / "results" / folder_name
        graph_output_dir.mkdir(parents=True, exist_ok=True) # Create the folder
        
        # Run the optimization
        monthly_results_df, school_budgets_for_this_run, meal_costs = \
            run_proportional_monthly_optimization_ilp(
                opt_data, 
                budget_cap_factor=factor 
            )

        if monthly_results_df is not None:
            # Run the annual analysis
            analysis_table = analyze_annual_budget(
                monthly_results_df,
                school_budgets_for_this_run,
                meal_costs,
                opt_data,
                unit_costs_file,
                actual_annual_cost=actual_cost_2025
            )
            
            if analysis_table is not None:
                # Save the unique annual breakdown CSV
                cols = ['school', 'proportional_annual_budget', 'annual_food_cost', 'remaining_annual_balance']
                analysis_table = analysis_table[cols]
                analysis_table.to_csv(analysis_breakdown_path, index=False)
                print(f"Successfully saved analysis to: {analysis_breakdown_path}")
            
            if run_maps:
                print(f"\nGenerating maps for {name} scenario...")
                
                generate_savings_map(
                    opt_data, 
                    monthly_results_df, 
                    unit_costs_file,
                    coordinates_file,
                    geojson_path=geojson_file, 
                    out_dir=graph_output_dir,
                    file_suffix=file_suffix
                )
                
                generate_savings_maps_by_level(
                    opt_data, 
                    monthly_results_df, 
                    unit_costs_file,  
                    coordinates_file,
                    geojson_path=geojson_file,
                    out_dir=graph_output_dir,
                    file_suffix=file_suffix
                )
                
                generate_all_region_choropleths(
                    opt_data, 
                    monthly_results_df, 
                    unit_costs_file,
                    coordinates_file, 
                    geojson_file, 
                    out_dir=graph_output_dir,
                    file_suffix=file_suffix
                )
            
            print(f"\nGenerating size-based bar charts for {name} scenario...")
            generate_savings_by_size_charts(
                opt_data,
                monthly_results_df,
                unit_costs_file,
                out_dir=graph_output_dir,
                file_suffix=file_suffix
            )

            print(f"\nGenerating overall summary charts for {name} scenario...")
            generate_savings_analysis_chart(
                opt_data,
                monthly_results_df,
                unit_costs_file,
                out_dir=graph_output_dir,
                file_suffix=file_suffix
            )
            generate_overall_savings_bar_chart(
                opt_data,
                monthly_results_df,
                unit_costs_file,
                out_dir=graph_output_dir,
                file_suffix=file_suffix
            )
            
            # Generate the item breakdown for *this* specific scenario
            print(f"\nGenerating item breakdown for {name} scenario...")
            item_breakdown_path = csv_output_dir / f'monthly_items_breakdown{file_suffix}.csv'
            
            generate_item_breakdown(
                monthly_results_df,
                opt_data['dfb'],
                opt_data['dfl'],
                str(item_breakdown_path)
            )
            
            if name == "baseline":
                baseline_results = {
                    'monthly_df': monthly_results_df,
                    'analysis_table': analysis_table,
                    'school_budgets': school_budgets_for_this_run,
                    'meal_costs': meal_costs,
                    'opt_data': opt_data
                }
        else:
            print(f"Optimization failed for scenario: {name}. Skipping analysis.")

    return (
        baseline_results.get('monthly_df'),
        baseline_results.get('analysis_table'),
        baseline_results.get('school_budgets'),
        baseline_results.get('meal_costs')
    )

def run_all_optimizations(
    breakfast_file: str | Path,
    lunch_file: str | Path,
    student_counts_file: str | Path,
    unit_costs_file: str | Path,
    coordinates_file: str | Path,
    geojson_file: str | Path,
    total_budget: float = 139144760
):
    """
    One-call end-to-end:
    - Prepare data
    - Daily ILP (equal budgets) + ILP item breakdown CSV
    - Monthly Proportional ILP + analysis table + (internal) item breakdown CSV
    - Generates all charts and maps for all 3 scenarios.
    Returns dict of outputs for optional programmatic use
    """

    # Expect cleaned CSVs
    bf_df = pd.read_csv(breakfast_file, low_memory=False)
    ln_df = pd.read_csv(lunch_file, low_memory=False)
    sc_df = pd.read_csv(student_counts_file, low_memory=False)

    opt_data = prepare_optimization_data(bf_df, ln_df, sc_df)

    ilp_daily_df = run_daily_ilp_pipeline(opt_data, total_budget=total_budget)

    monthly_df, analysis_table, monthly_school_budgets, monthly_meal_costs = run_monthly_proportional_pipeline(
        opt_data,
        unit_costs_file=unit_costs_file, 
        total_budget=total_budget,
        coordinates_file=coordinates_file,
        geojson_file=geojson_file
    )
    
    return {
        'opt_data': opt_data,
        'daily_ilp': ilp_daily_df,
        'monthly_ilp': monthly_df,
        'monthly_analysis': analysis_table,
        'monthly_school_budgets': monthly_school_budgets,
        'monthly_meal_costs': monthly_meal_costs
    }


def run_size_based_optimization(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, all_school_lists, dfb, dfl):
    """
    (Legacy) Runs the linear programming model using predefined school size lists
    and saves the output to a separate CSV file.
    (Accepts dfb and dfl as arguments)
    """

    # Create the school_sizes dictionary
    school_sizes = {}
    if all_school_lists:
        for size, school_list in all_school_lists.items():
            for school in school_list:
                school_sizes[school] = size
        print("--- Starting Size-Based Optimization ---")
        print(f"Categorized {len(school_sizes)} schools based on your lists.")
    else:
        print("Warning: all_school_lists is empty. Cannot run size-based optimization.")
        return

    # Define and set production bounds based on school size
    size_bounds_config = {
        'xxs': (0.80, 1.20), 'xs': (0.85, 1.15), 's': (0.90, 1.10),
        'm': (0.95, 1.05), 'l': (0.96, 1.04), 'xl': (0.97, 1.03),
        'xxl': (0.98, 1.02), 'xxxl':(0.99, 1.01)
    }
    bounds = []
    for school in schools_to_optimize:
        school_size = school_sizes.get(school, 'm')
        min_factor, max_factor = size_bounds_config[school_size]
        for i, meal_type in enumerate(meal_types):
            avg_demand = demand[school][i]
            bounds.append((avg_demand * min_factor, avg_demand * max_factor))
    print("Production bounds have been set based on custom school sizes.")

    print("Warning: run_size_based_optimization is a legacy function.")
    results_df = None # Bypassing

    # Generate the Food Item Breakdown and Save to a CSV file
    if results_df is not None:
        bf_popularity = dfb.groupby('name')['served_reimbursable'].sum()
        bf_total_served = bf_popularity.sum()
        bf_popularity_prop = (bf_popularity / bf_total_served).reset_index(name='proportion')
        bf_popularity_prop['meal_type'] = 'Breakfast'

        ln_popularity = dfl.groupby('name')['served_reimbursable'].sum()
        ln_total_served = ln_popularity.sum()
        ln_popularity_prop = (ln_popularity / ln_total_served).reset_index(name='proportion')
        ln_popularity_prop['meal_type'] = 'Lunch'

        item_popularity_df = pd.concat([bf_popularity_prop, ln_popularity_prop])
        item_df = pd.merge(results_df, item_popularity_df, on='meal_type')
        item_df['recommended_quantity'] = (item_df['optimal_quantity'] * item_df['proportion']).round().astype(int)

        optimized_df = item_df[item_df['recommended_quantity'] > 0][['school', 'meal_type', 'name', 'recommended_quantity']].rename(columns={'name': 'food_item'})
        optimized_df = optimized_df.sort_values(['school', 'meal_type', 'recommended_quantity'], ascending=[True, True, False])

        # Save to a CSV file
        script_dir = Path(__file__).resolve().parent
        src_dir = script_dir.parent
        project_root = src_dir.parent
        output_filename = project_root / 'src' / 'data' / 'school_food_item_optimization_by_size.csv'
        
        optimized_df.to_csv(output_filename, index=False)
        print(f"\nSaved the size-based optimization results to '{output_filename}'")
    else:
        print("Size-based optimization did not produce a result.")

def run_monthly_meal_optimization(schools_to_optimize, meal_types, meal_costs, demand, waste_penalty, bounds):
    """
    Runs the monthly meal optimization with no budget constraint.
    """

    num_vars = len(schools_to_optimize) * len(meal_types)
    c = []
    variable_names = []
    for school in schools_to_optimize:
        for j, meal_type_str in enumerate(meal_types):
            meal_cost = meal_costs[j]
            total_cost = meal_cost + (0.1 * waste_penalty[j])
            c.append(total_cost)
            variable_names.append({'school': school, 'meal_type': meal_type_str})
    print("\n--- Starting Monthly Optimization (No Budget Constraint) ---")
    print("Objective: Minimize Cost + Waste Penalty")

    # Define bounds for each variable
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    bounds_obj = Bounds(lb=lower_bounds, ub=upper_bounds)

    # Integrality constraint for all variables
    integrality = np.ones(num_vars)

    try:
        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds_obj,
            constraints=None
        )
        if result.success:
            print(f"\nOptimization Successful! Minimum Monthly Cost: ${result.fun:,.2f}")
            results_df = pd.DataFrame(variable_names)
            results_df['optimal_quantity'] = result.x.astype(int)
            print("\nOptimal Monthly Meal Production Plan (Sample):")

            for i, school in enumerate(schools_to_optimize[:5]):
                print(f"\n  {school}:")
                for j, meal_type in enumerate(meal_types):
                    var_index = i * len(meal_types) + j
                    quantity = int(result.x[var_index])
                    min_b, max_b = bounds[var_index]
                    print(f"    {meal_type}: {quantity:,} units "
                          f"(Monthly Demand: {demand[school][j]:,.0f}, Prod. Bounds: {int(min_b):,}-{int(max_b):,})")

            # Calculate total production cost based on meal costs only
            full_total_production_cost = 0
            cost_vector = meal_costs * len(schools_to_optimize)
            full_total_production_cost = sum(result.x * cost_vector)
            
            print("\n" + "="*50)
            print(f"Total Estimated Monthly Production Cost (All Schools): ${full_total_production_cost:,.2f}")
            print("="*50)
        else:
            print(f"\nOptimization failed: {result.message}")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")

def _daily_bounds_from_demand(opt_data, lo=0.90, hi=1.10):
    """
    Helper to create production bounds based on daily demand.
    """

    bounds = []
    for school in opt_data['schools']:
        for i in range(len(opt_data['meal_types'])):
            d = opt_data['demand'][school][i]
            bounds.append((d * lo, d * hi))
    return bounds