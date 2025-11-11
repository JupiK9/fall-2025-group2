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

        # Average costs
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
# Initial Linear Programming and Integer Linear Programming Optimization
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

    # Create the LinearConstraint object
    # This single object defines: min_budget <= (cost expression) <= max_budget
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
# Food Item popularity optimization
# ==============================================================================

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

# ==============================================================================
# Size based and monthly optimization
# ==============================================================================

def run_size_based_optimization(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, all_school_lists, dfb, dfl):
    """
    Runs the linear programming model using predefined school size lists
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

    # Run the base optimization logic
    results_df = run_meal_optimization_ilp(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds)

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

# ==============================================================================
# Monthly Budget Optimization based on School Size
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

# ==============================================================================
# Calculate the actual annual cost of producing the foods
# ==============================================================================

def calculate_actual_annual_cost(data):
    """
    Calculates the actual total food cost from the source data and
    extrapolates it to a full 10-month school year.
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

# ==============================================================================
# Creating a graph for savings analysis
# ==============================================================================

def analyze_annual_budget(results_df, school_budgets, meal_costs, actual_annual_cost=None):
    """
    Calculates the remaining annual budget for each school after scaling the
    optimized monthly food costs to a full 10-month school year.
    """

    if results_df is None:
        print("Skipping budget analysis: No optimization results available.")
        return None

    print("\nCalculating Annual Budget Analysis...")

    MONTHS_IN_SCHOOL_YEAR = 10

    # Map meal types to their costs
    meal_cost_map = {'Breakfast': meal_costs[0], 'Lunch': meal_costs[1]}
    results_df['monthly_food_cost'] = results_df.apply(
        lambda row: row['optimal_quantity'] * meal_cost_map[row['meal_type']],
        axis=1
    )

    # Calculate total food cost per school for one month
    school_monthly_costs = results_df.groupby('school')['monthly_food_cost'].sum().reset_index()

    annual_budgets_list = [
        (school, monthly_budget * MONTHS_IN_SCHOOL_YEAR)
        for school, monthly_budget in school_budgets.items()
    ]

    # Create a DataFrame for the analysis
    budget_analysis_df = pd.DataFrame(annual_budgets_list, columns=['school', 'proportional_annual_budget'])
    budget_analysis_df = pd.merge(budget_analysis_df, school_monthly_costs, on='school', how='left').fillna(0)

    # Scale monthly food cost to a full 10-month school year
    budget_analysis_df['annual_food_cost'] = budget_analysis_df['monthly_food_cost'] * MONTHS_IN_SCHOOL_YEAR
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
        savings_percent = (savings / actual_annual_cost) * 100
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

def prepare_savings_analysis_df(data, results_df, meal_costs):
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
    meal_cost_map = {'Breakfast': meal_costs[0], 'Lunch': meal_costs[1]}
    results_df['monthly_food_cost'] = results_df.apply(
        lambda row: row['optimal_quantity'] * meal_cost_map[row['meal_type']], axis=1
    )
    optimized_costs = results_df.groupby('school')['monthly_food_cost'].sum().reset_index()
    optimized_costs['optimized_annual_cost'] = optimized_costs['monthly_food_cost'] * MONTHS_IN_SCHOOL_YEAR

    # Calculate Actual Annual Cost per School
    actual_costs_b = dfb.groupby('school_name')['production_cost_total'].sum()
    actual_costs_l = dfl.groupby('school_name')['production_cost_total'].sum()
    actual_costs = (actual_costs_b.add(actual_costs_l, fill_value=0) * MONTHS_IN_SCHOOL_YEAR).reset_index(name='actual_annual_cost')
    actual_costs.rename(columns={'school_name': 'school'}, inplace=True)

    # Combine into a single DataFrame
    savings_df = pd.merge(actual_costs, optimized_costs[['school', 'optimized_annual_cost']], on='school', how='left')
    
    # This handles schools that were not in the optimization results.
    savings_df['optimized_annual_cost'] = savings_df['optimized_annual_cost'].fillna(0)

    savings_df['savings'] = savings_df['actual_annual_cost'] - savings_df['optimized_annual_cost']
    
    # Create an 'outcome' column and an absolute value column for size
    savings_df['outcome'] = np.where(savings_df['savings'] >= 0, 'Savings', 'Loss')
    savings_df['savings_magnitude'] = savings_df['savings'].abs()
    
    # Add size category from 2024-2025 counts using bins
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
        # If sizes missing or no year column, fall back to a default label
        savings_df['size_category'] = 'unknown'

    return savings_df

# ==============================================================================
# Data preparation for graphing savings by school size 
# ==============================================================================

def analyze_savings_by_school_size(opt_data, results_df, school_budgets, meal_costs, df_sizes):
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

    # Calculate Optimized Annual Cost per School (from results_df)
    meal_cost_map = {'Breakfast': meal_costs[0], 'Lunch': meal_costs[1]}
    results_df['monthly_food_cost'] = results_df.apply(
        lambda row: row['optimal_quantity'] * meal_cost_map[row['meal_type']], axis=1
    )
    school_costs = results_df.groupby('school')['monthly_food_cost'].sum().reset_index()
    school_costs['annual_food_cost'] = school_costs['monthly_food_cost'] * MONTHS_IN_SCHOOL_YEAR

    # Calculate Actual Historical Annual Cost per School (from source data)
    actual_costs_b = dfb.groupby('school_name')['production_cost_total'].sum()
    actual_costs_l = dfl.groupby('school_name')['production_cost_total'].sum()
    actual_costs = (actual_costs_b.add(actual_costs_l, fill_value=0) * MONTHS_IN_SCHOOL_YEAR).reset_index(name='actual_annual_cost')
    actual_costs.rename(columns={'school_name': 'school'}, inplace=True)

    # Combine and merge with size data
    analysis_df = pd.merge(actual_costs, school_costs[['school', 'annual_food_cost']], on='school', how='left')
    analysis_df = pd.merge(analysis_df, df_sizes[['school_name', 'size_category']], left_on='school', right_on='school_name', how='left')

    # Group by size category and sum the totals
    agg_df = analysis_df.groupby('size_category', observed=True)[['actual_annual_cost', 'annual_food_cost']].sum(numeric_only=True).reset_index()
    
    # Calculate savings in dollars and as a percentage
    agg_df['total_savings'] = agg_df['actual_annual_cost'] - agg_df['annual_food_cost']
    agg_df['percent_savings'] = (agg_df['total_savings'] / agg_df['actual_annual_cost']) * 100

    return agg_df

# ==============================================================================
# Geospatial analysis
# ==============================================================================

# Per school analysis
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
        if 'school_name' in coords_df.columns:
            coords_df.rename(columns={'school_name': 'school'}, inplace=True)
            coords_df['school'] = coords_df['school'].str.lower()
        else:
            print("Error: Could not find 'school_name' column in coordinates file.")
            return None

        # Keep only the essential columns and remove duplicates
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

# Regional analysis
def prepare_regional_map_data(savings_df, coordinates_csv_path):
    """
    Aggregates school savings data by FCPS Region.
    """
    if savings_df is None:
        print("Skipping regional map prep: No savings data available.")
        return None

    print("\n--- Preparing Regional Map Data ---")
    try:
        # Read coordinates file to get region data
        coords_df = pd.read_csv(coordinates_csv_path, low_memory=False)
        coords_df.columns = coords_df.columns.str.lower()

        # Clean and select necessary columns
        if 'school_name' in coords_df.columns:
            coords_df.rename(columns={'school_name': 'school'}, inplace=True)
            coords_df['school'] = coords_df['school'].str.lower()
        else:
            print("Error: 'school_name' column not found in coordinates file.")
            return None

        # Keep only the columns needed for regional aggregation
        region_df = coords_df[['school', 'fcps region', 'latitude', 'longitude']].drop_duplicates(subset='school')

        # Merge savings data with region data
        merged_df = pd.merge(savings_df, region_df, on='school', how='inner')

        # Group by region and aggregate the results
        regional_summary = merged_df.groupby('fcps region').agg(
            total_savings=('savings', 'sum'),
            latitude=('latitude', 'mean'),  # Get the average location for the bubble
            longitude=('longitude', 'mean')
        ).reset_index()

        # Determine the outcome (Savings or Loss) for the entire region
        regional_summary['outcome'] = np.where(regional_summary['total_savings'] >= 0, 'Savings', 'Loss')
        regional_summary['savings_magnitude'] = regional_summary['total_savings'].abs()

        print(f"Successfully aggregated savings for {len(regional_summary)} regions.")
        return regional_summary

    except Exception as e:
        print(f"An error occurred during regional data preparation: {e}")
        return None

def generate_fcps_region_choropleth(
    savings_df: "pd.DataFrame",
    coords_csv_path: str,
    geojson_path: str,
    legend_name: str = "Optimization Savings per Region (USD)",
    save_path: str | None = None,
    bins: int | None = 6,
    normalize_school_names: bool = True,
):
    """
    Build an FCPS region choropleth (green = savings, red = loss).

    Parameters
    ----------
    savings_df : pd.DataFrame
        Must have ['school','savings'] columns.
    coords_csv_path : str
        CSV with School_Name (or school_name), FCPS Region, latitude, longitude.
    geojson_path : str
        GeoJSON with properties.REGION (1..N).
    legend_name : str
        Title for the legend.
    save_path : str | None
        Optional output HTML file path.
    bins : int | None
        Number of bins for color breaks. Set None to let folium auto-scale.
    normalize_school_names : bool
        Lowercases and trims school names for safer joins.
    """
    
    # Load and normalize coordinate data
    coords = pd.read_csv(coords_csv_path, low_memory=False)
    coords.columns = coords.columns.str.lower()
    school_col = "school_name" if "school_name" in coords.columns else "school"

    lookup = (
        coords[[school_col, "fcps region", "latitude", "longitude"]]
        .drop_duplicates(subset=school_col)
        .rename(columns={school_col: "school"})
    )

    # Normalize and merge savings data
    sdf = savings_df.copy()
    sdf["savings"] = pd.to_numeric(sdf["savings"], errors="coerce")
    if normalize_school_names:
        sdf["school"] = sdf["school"].astype(str).str.strip().str.lower()
        lookup["school"] = lookup["school"].astype(str).str.strip().str.lower()

    merged = sdf.merge(lookup, on="school", how="inner")
    regional = (
        merged.groupby("fcps region", as_index=False)
        .agg(total_savings=("savings", "sum"))
    )
    regional["REGION"] = (
        regional["fcps region"].astype(str).str.extract(r"(\d+)").astype(int)
    )

    regional_map_df = regional[["REGION", "total_savings"]].copy()
    regional_map_df["REGION_KEY"] = regional_map_df["REGION"].astype(str)
    regional_map_df["total_savings"] = pd.to_numeric(
        regional_map_df["total_savings"], errors="coerce"
    ).fillna(0.0)

    # Load GeoJSON and align join key
    with open(geojson_path, "r") as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if "REGION" in props and props["REGION"] is not None:
            props["REGION_KEY"] = str(int(props["REGION"]))
        feat["properties"] = props

    # Compute map center
    def _geojson_center(geojson):
        centers = []
        for feat in geojson.get("features", []):
            geom = feat.get("geometry", {})
            if not geom or "coordinates" not in geom:
                continue

            def _flat(coords):
                if isinstance(coords[0][0], (float, int)):
                    return coords
                out = []
                for c in coords:
                    out.extend(_flat(c))
                return out

            pts = _flat(geom["coordinates"])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            centers.append((float(np.mean(ys)), float(np.mean(xs))))
        if centers:
            lat = float(np.mean([c[0] for c in centers]))
            lon = float(np.mean([c[1] for c in centers]))
            return lat, lon
        return (38.8462, -77.3064)  # Fairfax fallback

    center_lat, center_lon = _geojson_center(gj)

    # Build folium choropleth
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

    choropleth_kwargs = dict(
        geo_data=gj,
        data=regional_map_df,
        columns=["REGION_KEY", "total_savings"],
        key_on="feature.properties.REGION_KEY",
        fill_color="RdYlGn",
        fill_opacity=0.85,
        line_opacity=0.9,
        nan_fill_color="#f0f0f0",
        nan_fill_opacity=0.6,
        legend_name=legend_name,
        highlight=True,
    )

    vals = regional_map_df["total_savings"].astype(float).to_numpy()
    if bins and vals.size >= 2 and not np.allclose(np.nanmin(vals), np.nanmax(vals)):
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = (vmax - vmin) * 1e-6 or 1e-6
        vmin -= pad; vmax += pad
        span = max(1.0, abs(vmax - vmin))
        base = 10.0 ** max(0, int(np.floor(np.log10(span))) - 2)
        edges = np.linspace(vmin, vmax, int(bins) + 1)
        edges = np.round(edges / base) * base

        edges = np.array(sorted(set(edges)))
        if edges.size >= 3:
            # Ensure all data fits inside bins
            edges[0] = min(edges[0], np.nanmin(vals)) - 1e-9
            edges[-1] = max(edges[-1], np.nanmax(vals)) + 1e-9
            choropleth_kwargs["bins"] = edges.tolist()


    folium.Choropleth(**choropleth_kwargs).add_to(m)

    folium.GeoJson(
        gj,
        control=False,
        show=True,
        style_function=lambda f: {"fillOpacity": 0, "color": "#222222", "weight": 1.5},
    ).add_to(m)

    lut = dict(zip(regional_map_df["REGION_KEY"], regional_map_df["total_savings"]))
    def _fmt_money(v):
        try:
            return f"${v:,.0f}"
        except Exception:
            return "N/A"

    folium.GeoJson(
        gj,
        control=False,
        show=True,
        style_function=lambda f: {"fillOpacity": 0, "color": "#00000000", "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=["REGION"], aliases=["Region"]),
        popup=lambda f: folium.Popup(
            f"Region {f['properties'].get('REGION')} : {_fmt_money(lut.get(str(f['properties'].get('REGION')), None))}",
            max_width=260,
        ),
        highlight_function=lambda f: {"weight": 3, "color": "#000000"},
    ).add_to(m)

    if save_path:
        m.save(save_path)

    return m, regional_map_df[["REGION", "total_savings"]]

# ==============================================================================
# Pipeline Helpers
# ==============================================================================

def _equal_budgets(schools, total_budget):
    """
    Helper to create equal budgets for each school.
    """

    per = total_budget / max(1, len(schools))
    return {s: per for s in schools}

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

    # Pick the year column: prefer '2024-2025', else latest year-like col
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
    total_budget: float = 139144760,
    coordinates_file: str | Path | None = None,
    geojson_file: str | Path | None = None
):
    """
    Runs the monthly ILP optimization for three scenarios and saves all CSVs and maps into their own subfolders.
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
                    opt_data, monthly_results_df, meal_costs, 
                    coordinates_file, 
                    out_dir=graph_output_dir,
                    file_suffix=file_suffix
                )
                
                generate_savings_maps_by_level(
                    opt_data, monthly_results_df, meal_costs,
                    coordinates_file, 
                    out_dir=graph_output_dir,
                    file_suffix=file_suffix
                )
                
                generate_all_region_choropleths(
                    opt_data, monthly_results_df, meal_costs,
                    coordinates_file, geojson_file, 
                    out_dir=graph_output_dir,
                    file_suffix=file_suffix
                )
            
            print(f"\nGenerating size-based bar charts for {name} scenario...")
            generate_savings_by_size_charts(
                opt_data,
                monthly_results_df,
                school_budgets_for_this_run,
                meal_costs,
                out_dir=graph_output_dir,
                file_suffix=file_suffix
            )

            print(f"\nGenerating overall summary charts for {name} scenario...")
            generate_savings_analysis_chart(
                opt_data,
                monthly_results_df,
                meal_costs,
                out_dir=graph_output_dir,
                file_suffix=file_suffix
            )
            generate_overall_savings_bar_chart(
                opt_data,
                monthly_results_df,
                meal_costs,
                out_dir=graph_output_dir,
                file_suffix=file_suffix
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

    # Generate item breakdown once
    print(f"\n{'='*60}")
    print("Generating single item breakdown from 'baseline' (100%) run...")
    if 'monthly_df' in baseline_results and baseline_results['monthly_df'] is not None:
        item_breakdown_path = project_root / 'src' / 'data' / 'optimization-data' / 'monthly_items_breakdown.csv'
        
        generate_item_breakdown(
            baseline_results['monthly_df'],
            baseline_results['opt_data']['dfb'],
            baseline_results['opt_data']['dfl'],
            str(item_breakdown_path)
        )
        print(f"Successfully saved item breakdown to: {item_breakdown_path}")
    else:
        print("Skipping item breakdown: 'baseline' run did not produce results.")

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

def generate_savings_analysis_chart(opt_data, monthly_results_df, monthly_meal_costs, out_dir=None, file_suffix: str = ""):
    """
    Generates and saves the interactive savings analysis bubble chart
    to a specified output directory with a file suffix.
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to plot.")
        return None

    # Prepare data for plotting
    savings_df = prepare_savings_analysis_df(
        opt_data,
        monthly_results_df,
        monthly_meal_costs
    )

    if savings_df is None or savings_df.empty:
        print("No savings data available for plotting.")
        return None

    # Create Interactive Savings Analysis Bubble Chart
    title_suffix = file_suffix.strip("_").replace("_", " ").title()
    fig = px.scatter(
        savings_df,
        x='actual_annual_cost',
        y='optimized_annual_cost',
        size='savings_magnitude',
        color='outcome',
        color_discrete_map={'Savings': 'green', 'Loss': 'red'},
        hover_name='school',
        hover_data={
            'size_category': True,
            'savings': ':.2s',
            'savings_magnitude': False
        },
        title=f'Savings Analysis: Actual vs. Optimized Cost ({title_suffix})',
        labels={
            'actual_annual_cost': 'Actual Annual Food Cost',
            'optimized_annual_cost': 'Optimized Annual Food Cost',
            'outcome': 'Outcome',
            'savings_magnitude': 'Impact ($)'
        }
    )

    # Add the 45° "no savings" reference line
    max_val = max(
        savings_df['actual_annual_cost'].max(),
        savings_df['optimized_annual_cost'].max()
    ) * 1.05
    fig.add_shape(
        type='line',
        x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color='Gray', width=2, dash='dash')
    )
    fig.update_layout(legend=dict(font=dict(size=14)))

    # Save the interactive chart to the specified directory
    if out_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "src" / "data" / "results"
    else:
        results_dir = Path(out_dir)
        
    results_dir.mkdir(parents=True, exist_ok=True)
    chart_path = results_dir / f"savings_analysis_bubble_chart{file_suffix}.html"
    fig.write_html(str(chart_path))
    print(f"Saved interactive savings analysis chart to: {chart_path}")

    return chart_path

def generate_overall_savings_bar_chart(opt_data, monthly_results_df, monthly_meal_costs, out_dir=None, file_suffix: str = ""):
    """
    Builds and saves a bar chart comparing Actual (baseline) vs Optimized annual costs
    to a specified output directory with a file suffix.
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to plot.")
        return None

    # Calculate actual annual cost from source data
    actual_annual_cost = calculate_actual_annual_cost(opt_data)

    # Calculate optimized annual cost from monthly results
    meal_cost_map = {'Breakfast': monthly_meal_costs[0], 'Lunch': monthly_meal_costs[1]}
    results_copy = monthly_results_df.copy()
    results_copy['monthly_food_cost'] = results_copy.apply(
        lambda row: row['optimal_quantity'] * meal_cost_map[row['meal_type']], axis=1
    )
    optimized_monthly_cost = results_copy['monthly_food_cost'].sum()
    optimized_annual_cost = optimized_monthly_cost * 10  # months in school year

    labels = ['Actual Cost (2025 Baseline)', 'Optimized Cost']
    values = [actual_annual_cost, optimized_annual_cost]
    colors = ['#d9534f', '#5cb85c']  # red, green

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors)
    
    title_suffix = file_suffix.strip("_").replace("_", " ").title()
    ax.set_title(f'Overall Savings: Actual vs. Optimized ({title_suffix})', fontsize=16)
    ax.set_ylabel('Cost (in Millions of $)', fontsize=12)
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'${h/1e6:.2f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    fig.tight_layout()

    # Save to the specified directory
    if out_dir is None:
        out_path = Path(__file__).resolve().parents[2] / "src" / "data" / "results" / f"overall_savings_bar_chart{file_suffix}.png"
    else:
        out_path = Path(out_dir) / f"overall_savings_bar_chart{file_suffix}.png"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    savings = actual_annual_cost - optimized_annual_cost
    savings_percent = (savings / actual_annual_cost) * 100 if actual_annual_cost else 0.0
    print(f"Total Annual Savings: ${savings:,.2f} ({savings_percent:.2f}%)")
    print(f"Saved overall savings bar chart to: {out_path}")

    return out_path

def generate_savings_by_size_charts(opt_data, monthly_results_df, monthly_school_budgets, monthly_meal_costs, out_dir=None, file_suffix: str = ""):
    """
    Saves two bar charts with a unique file suffix:
      - Total Annual Savings by School Size Category  -> savings_by_size_total_[suffix].png
      - Percentage of Budget Saved by School Size     -> savings_by_size_percent_[suffix].png
    Returns dict with output paths.
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to plot by size.")
        return None

    # Ensure df_sizes has size_category
    try:
        size_map = _compute_size_category(opt_data['df_sizes'], preferred_year="2024-2025")
        df_sizes = opt_data['df_sizes'].copy()
        df_sizes.columns = df_sizes.columns.str.lower()
        df_sizes = df_sizes.merge(size_map, on='school_name', how='left', suffixes=('', '_mapped'))
        if 'size_category_mapped' in df_sizes.columns:
            df_sizes['size_category'] = df_sizes['size_category_mapped']
        df_sizes.drop(columns=[c for c in ['size_category_mapped'] if c in df_sizes.columns], inplace=True, errors='ignore')
    except Exception:
        df_sizes = opt_data['df_sizes']

    # Aggregate
    savings_by_size_df = analyze_savings_by_school_size(
        opt_data,
        monthly_results_df,
        monthly_school_budgets,
        monthly_meal_costs,
        df_sizes
    )
    if savings_by_size_df is None or savings_by_size_df.empty:
        print("No size-category aggregation available.")
        return None

    # Map ranges and order
    range_map = {
        'xxs': '0-499', 'xs': '500-999', 's': '1000-1499', 'm': '1500-1999',
        'l': '2000-2499', 'xl': '2500-2999', 'xxl': '3000-3499', 'xxxl': '3500+'
    }
    order = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
    savings_by_size_df['count_range'] = savings_by_size_df['size_category'].map(range_map)
    savings_by_size_df['size_category'] = pd.Categorical(savings_by_size_df['size_category'], categories=order, ordered=True)
    savings_by_size_df = savings_by_size_df.sort_values('size_category')

    # Output dir
    results_dir = Path(out_dir) if out_dir else (Path(__file__).resolve().parents[2] / "src" / "data" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Total Savings ($)
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.bar(savings_by_size_df['count_range'], savings_by_size_df['total_savings'], color='#5cb85c')
    ax1.set_title(f'Total Annual Savings by School Size Category ({file_suffix.strip("_").title()})', fontsize=16)
    ax1.set_xlabel("School Student Population Size")
    ax1.set_ylabel("Total Savings ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    fig1.tight_layout()

    out1 = results_dir / f"savings_by_size_total{file_suffix}.png"
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Chart 2: Savings (%)
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.bar(savings_by_size_df['count_range'], savings_by_size_df['percent_savings'], color='#428bca')
    ax2.set_title(f'Percentage of Budget Saved by School Size ({file_suffix.strip("_").title()})', fontsize=16)
    ax2.set_xlabel("School Population Size")
    ax2.set_ylabel("Savings (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    fig2.tight_layout()
    
    out2 = results_dir / f"savings_by_size_percent{file_suffix}.png"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    return {"total_savings_png": out1, "percent_savings_png": out2}

def generate_savings_map(opt_data, monthly_results_df, monthly_meal_costs, coordinates_file, out_dir=None, file_suffix: str = ""):
    """
    Builds an interactive Folium bubble map of savings by school and writes it to HTML.
    Saves to a specified out_dir.
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to map.")
        return None

    # Build savings table
    savings_df = prepare_savings_analysis_df(
        opt_data,
        monthly_results_df,
        monthly_meal_costs
    )
    if savings_df is None or savings_df.empty:
        print("No savings data available for mapping.")
        return None

    # Merge with coordinates
    map_df = prepare_map_data_from_coordinates(savings_df, coordinates_file)
    if map_df is None or map_df.empty:
        print("No coordinate-enhanced data available for mapping.")
        return None

    # Center (fallback to mean if present)
    try:
        lat0 = float(map_df['latitude'].mean())
        lon0 = float(map_df['longitude'].mean())
        center = [lat0, lon0]
    except Exception:
        center = [38.83, -77.27]  # Fairfax default

    m = folium.Map(location=center, zoom_start=10)

    max_abs = map_df['savings_magnitude'].max() if 'savings_magnitude' in map_df else None
    def scale_radius(v):
        if max_abs and v and v > 0:
            return (v / max_abs) ** (1 / 3) * 20 + 2
        return 2

    for _, row in map_df.iterrows():
        try:
            color = 'green' if row.get('outcome') == 'Savings' else 'red'
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
            # Skip rows with bad/missing coords
            continue

    # Save to a specified output directory
    if out_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "src" / "data" / "results"
    else:
        results_dir = Path(out_dir)
        
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the file_suffix to create a unique name inside the out_dir
    out_path = results_dir / f"overall_savings_map{file_suffix}.html"
    
    m.save(str(out_path))
    print(f"Saved interactive savings map to: {out_path}")
    return out_path

def generate_savings_maps_by_level(opt_data, monthly_results_df, monthly_meal_costs, coordinates_file, out_dir=None, file_suffix: str = ""):
    """
    Creates three interactive Folium bubble maps (Elementary / Middle / High)
    and saves them as HTML files under src/data/results by default.

    Returns a dict with file paths for each level.
    """

    if monthly_results_df is None:
        print("No monthly optimization results available to map by level.")
        return None

    # Build savings table
    savings_df = prepare_savings_analysis_df(
        opt_data,
        monthly_results_df,
        monthly_meal_costs
    )
    if savings_df is None or savings_df.empty:
        print("No savings data available for mapping by level.")
        return None

    # Load coordinates + level
    coords_df = pd.read_csv(coordinates_file, low_memory=False)
    coords_df.columns = coords_df.columns.str.lower()

    # Normalize school column
    if 'school_name' in coords_df.columns and 'school' not in coords_df.columns:
        coords_df = coords_df.rename(columns={'school_name': 'school'})
    if 'school' not in coords_df.columns:
        print("Coordinates file missing 'school' or 'school_name' column.")
        return None

    coords_df['school'] = coords_df['school'].astype(str).str.strip().str.lower()

    if 'level' not in coords_df.columns:
        print("Coordinates file missing 'level' column; cannot split by ES/MS/HS.")
        return None

    # Merge savings with coordinates + level
    map_df = pd.merge(
        savings_df,
        coords_df[['school', 'latitude', 'longitude', 'level']],
        on='school',
        how='inner'
    ).dropna(subset=['latitude', 'longitude'])

    # Ensure required fields
    if 'savings_magnitude' not in map_df.columns:
        map_df['savings_magnitude'] = map_df['savings'].abs()
    if 'outcome' not in map_df.columns:
        map_df['outcome'] = np.where(map_df['savings'] >= 0, 'Savings', 'Loss')

    # Map maker
    def _scale_radius(v, vmax):
        if vmax and v and v > 0:
            return (v / vmax) ** (1 / 3) * 20 + 2
        return 2

    def _build_map(df_level, title_center):
        m = folium.Map(location=title_center, zoom_start=10)
        vmax = df_level['savings_magnitude'].max()

        for _, row in df_level.iterrows():
            color = 'green' if row['outcome'] == 'Savings' else 'red'
            popup_text = (
                f"<strong>School:</strong> {str(row['school']).title()}<br>"
                f"<strong>Annual Savings:</strong> ${float(row['savings']):,.2f}"
            )

            # Single circle with translucent fill and outline
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=_scale_radius(row['savings_magnitude'], vmax),
                color=color,
                weight=2,
                opacity=0.4,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,        
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

        return m

    # Center on mean coords (fallback to Fairfax center)
    try:
        center = [float(map_df['latitude'].mean()), float(map_df['longitude'].mean())]
    except Exception:
        center = [38.83, -77.27]

    # Build maps per level and save
    results_dir = Path(out_dir) if out_dir else (Path(__file__).resolve().parents[2] / "src" / "data" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    level_map = [('ES', 'elementary'), ('MS', 'middle'), ('HS', 'high')]
    print("\nCreating interactive savings map...")
    for code, label in level_map:
        df_level = map_df[map_df['level'] == code]
        if df_level.empty:
            continue
        m = _build_map(df_level, center)
        out_file = results_dir / f"savings_map_{label}{file_suffix}.html"
        m.save(str(out_file))
        out_paths[label] = out_file
        print(f"Saved interactive savings map ({label}) to: {out_file}")

    if not out_paths:
        print("No data found for ES/MS/HS levels.")
        return None

    return out_paths

def _detect_region_prop(geojson_obj):
    """
    Find a likely region-name property on GeoJSON features.
    Tries common keys; falls back to the first string-like property.
    """

    candidates = ["region", "Region", "REGION", "school_region", "fcps_region", "Pyramid", "pyramid", "NAME"]
    props = geojson_obj["features"][0]["properties"]
    for k in candidates:
        if k in props:
            return k
    # fallback: first prop that looks stringy
    for k, v in props.items():
        if isinstance(v, str):
            return k
    # last resort: just use the first key
    return list(props.keys())[0]


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


def generate_fcps_region_choropleth(
    savings_df: pd.DataFrame,
    coords_csv_path: str | Path,
    geojson_path: str | Path,
    bins: int | None = 6,
    out_path: str | Path | None = None,
    map_title: str | None = None
):
    """
    Choropleth of TOTAL OPTIMIZATION SAVINGS per FCPS region.
    Assumes savings_df has ['school','savings'] where savings = actual_annual - optimized_annual.
    Coordinates CSV must have 'school_name' (or 'school') and 'fcps region' (with a space).
    GeoJSON must carry a numeric REGION property.
    """

    if savings_df is None or savings_df.empty:
        print("No savings data provided for region choropleth.")
        return None, None, None

    # Load/normalize coordinate lookup
    coords = pd.read_csv(coords_csv_path, low_memory=False)
    coords.columns = coords.columns.str.lower()

    if "school_name" in coords.columns and "school" not in coords.columns:
        coords = coords.rename(columns={"school_name": "school"})
    if "school" not in coords.columns:
        raise KeyError("Coordinates CSV needs 'school' or 'school_name'.")

    if "fcps region" not in coords.columns:
        raise KeyError("Coordinates CSV must contain 'fcps region' (note the space).")

    # slim lookup
    lookup = (
        coords[["school", "fcps region"]]
        .drop_duplicates(subset="school")
        .copy()
    )
    # normalize names
    lookup["school"] = lookup["school"].astype(str).str.strip().str.lower()

    # Normalize savings_df school names and merge region
    sdf = savings_df.copy()
    sdf["school"] = sdf["school"].astype(str).str.strip().str.lower()
    sdf["savings"] = pd.to_numeric(sdf["savings"], errors="coerce").fillna(0.0)

    merged = sdf.merge(lookup, on="school", how="inner")
    if merged.empty:
        print("Savings and coordinates did not overlap on 'school'.")
        return None, None, None

    # Aggregate: total optimization savings per region
    regional = (
        merged.groupby("fcps region", as_index=False)
              .agg(total_optimization_savings=("savings", "sum"))
    )

    # Extract numeric region id
    regional["REGION"] = (
        regional["fcps region"].astype(str).str.extract(r"(\d+)").astype(int)
    )
    regional_map = regional[["REGION", "total_optimization_savings"]].copy()
    regional_map["REGION_KEY"] = regional_map["REGION"].astype(str)

    # Load GeoJSON and align join key
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if "REGION" in props and props["REGION"] is not None:
            props["REGION_KEY"] = str(int(props["REGION"]))
        feat["properties"] = props

    # crude center from polygons
    def _geojson_center(geojson):
        centers = []
        for feat in geojson.get("features", []):
            geom = feat.get("geometry", {})
            if not geom or "coordinates" not in geom:
                continue
            def _flat(coords):
                if isinstance(coords[0][0], (float, int)):
                    return coords
                out = []
                for c in coords:
                    out.extend(_flat(c))
                return out
            pts = _flat(geom["coordinates"])
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            centers.append((float(np.mean(ys)), float(np.mean(xs))))
        if centers:
            lat = float(np.mean([c[0] for c in centers]))
            lon = float(np.mean([c[1] for c in centers]))
            return lat, lon
        return (38.8462, -77.3064)

    center_lat, center_lon = _geojson_center(gj)

    # Build choropleth using total optimization savings
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

    # Bin edges
    vals = regional_map["total_optimization_savings"].astype(float).to_numpy()
    chor_kwargs = dict(
        geo_data=gj,
        data=regional_map,
        columns=["REGION_KEY", "total_optimization_savings"],
        key_on="feature.properties.REGION_KEY",
        fill_color="RdYlGn",
        fill_opacity=0.85,
        line_opacity=0.9,
        nan_fill_color="#f0f0f0",
        nan_fill_opacity=0.6,
        legend_name="Total Optimization Savings ($)",
        highlight=True,
    )
    if bins and vals.size >= 2 and not np.allclose(np.nanmin(vals), np.nanmax(vals)):
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = (vmax - vmin) * 1e-6 or 1e-6
        edges = np.linspace(vmin - pad, vmax + pad, int(bins) + 1)
        chor_kwargs["bins"] = edges.tolist()

    folium.Choropleth(**chor_kwargs).add_to(m)

    # region outlines
    folium.GeoJson(
        gj, control=False, show=True,
        style_function=lambda f: {"fillOpacity": 0, "color": "#222222", "weight": 1.5},
    ).add_to(m)

    # simple tooltip/popup
    lut = dict(zip(regional_map["REGION_KEY"], regional_map["total_optimization_savings"]))
    def _fmt_money(v): 
        try: return f"${v:,.0f}"
        except: return "N/A"

    folium.GeoJson(
        gj, control=False, show=True,
        style_function=lambda f: {"fillOpacity": 0, "color": "#0000", "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=["REGION"], aliases=["Region"]),
        popup=lambda f: folium.Popup(
            f"Region {f['properties'].get('REGION')} : {_fmt_money(lut.get(str(f['properties'].get('REGION')), None))}",
            max_width=260,
        ),
        highlight_function=lambda f: {"weight": 3, "color": "#000000"},
    ).add_to(m)

    # save if requested
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "src" / "data" / "results" / "fcps_region_choropleth_overall.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Saved region choropleth to: {out_path}")

    return m, regional_map[["REGION", "total_optimization_savings"]], out_path

def generate_all_region_choropleths(
    opt_data: dict,
    monthly_results_df: pd.DataFrame,
    monthly_meal_costs: list[float],
    coords_csv_path: str | Path,
    geojson_path: str | Path,
    file_suffix: str = "",
    out_dir: str | Path | None = None
):
    """
    Creates:
      - Overall region choropleth
      - Per-level (ES/MS/HS) region choropleths
    Saves HTML to the specified out_dir and returns a dict of paths.
    """
    
    if monthly_results_df is None:
        print("No monthly results for region choropleths.")
        return None

    # Build savings_df once
    savings_df = prepare_savings_analysis_df(opt_data, monthly_results_df, monthly_meal_costs)
    if savings_df is None or savings_df.empty:
        print("No savings data for region choropleths.")
        return None
        
    # Define the output directory
    if out_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "src" / "data" / "results"
    else:
        results_dir = Path(out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Overall
    _, regional_df, path_overall = generate_fcps_region_choropleth(
        savings_df,
        coords_csv_path=coords_csv_path,
        geojson_path=geojson_path,
        bins=None,
        out_path=results_dir / f"fcps_region_choropleth_overall{file_suffix}.html",
        map_title="FCPS Regions — Overall Savings"
    )
    results["overall_html"] = path_overall
    results["overall_table"] = regional_df

    # Per level
    coords_df = _load_coords_and_normalize(coords_csv_path)
    if "level" not in coords_df.columns:
        print("Coordinates CSV missing 'level'; skipping per-level choropleths.")
        return results

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
            out_path=results_dir / f"fcps_region_choropleth_{label}{file_suffix}.html",
            map_title=f"FCPS Regions — {label.title()} Schools"
        )
        results[f"{label}_html"] = path_level
        results[f"{label}_table"] = reg_df

    return results
