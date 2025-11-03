import pandas as pd
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import geopandas as gpd
import folium
import json

# ==============================================================================
# Data Preparation for Optimization
# ==============================================================================
def prepare_optimization_data(breakfast_path, lunch_path, student_counts_path):
    """
    Loads and prepares all necessary data for the optimization models.
    """
    print("--- Preparing Optimization Data ---")
    try:
        dfb = pd.read_csv(breakfast_path, low_memory=False)
        dfl = pd.read_csv(lunch_path, low_memory=False)
        dfb.columns = dfb.columns.str.lower()
        dfl.columns = dfl.columns.str.lower()

        dfb['school_name'] = dfb['school_name'].str.lower()
        dfl['school_name'] = dfl['school_name'].str.lower()

        # Clean relevant numeric columns
        num_cols = ["served_reimbursable", "production_cost_total"]
        for col in num_cols:
            dfb[col] = pd.to_numeric(dfb[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            dfl[col] = pd.to_numeric(dfl[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
        
        schools = sorted(dfb['school_name'].unique().tolist())
        meal_types = ['Breakfast', 'Lunch']
        
        avg_bf_cost = dfb['production_cost_total'].sum() / dfb['served_reimbursable'].sum()
        avg_ln_cost = dfl['production_cost_total'].sum() / dfl['served_reimbursable'].sum()
        meal_costs = [avg_bf_cost, avg_ln_cost]

        demand = {}
        for school in schools:
            bf_school = dfb[dfb['school_name'] == school]
            bf_dates = bf_school['date'].nunique()
            avg_bf_demand = bf_school['served_reimbursable'].sum() / bf_dates if bf_dates > 0 else 0
            
            ln_school = dfl[dfl['school_name'] == school]
            ln_dates = ln_school['date'].nunique()
            avg_ln_demand = ln_school['served_reimbursable'].sum() / ln_dates if ln_dates > 0 else 0
            demand[school] = [avg_bf_demand, avg_ln_demand]
        
        all_school_lists = None
        try:
            student_counts_df = pd.read_csv(student_counts_path)
            df_sizes = student_counts_df[['School_Name', '2024-2025']].dropna().copy()
            df_sizes.columns = ['school_name', 'count']
            df_sizes['school_name'] = df_sizes['school_name'].str.lower().str.strip()
            df_sizes['count'] = df_sizes['count'].astype(int)
            bins = [-float('inf'), 499, 999, 1499, 1999, 2499, 2999, 3499, float('inf')]
            labels = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
            df_sizes['size_category'] = pd.cut(df_sizes['count'], bins=bins, labels=labels, right=True)
            all_school_lists = {size: group['school_name'].tolist() for size, group in df_sizes.groupby('size_category', observed=False)}
        except FileNotFoundError:
            print("Warning: Student count file not found.")
        
        print("Data preparation complete.")
        return {
            "dfb": dfb, "dfl": dfl, "schools": schools, "meal_types": meal_types, 
            "meal_costs": meal_costs, "demand": demand, "all_school_lists": all_school_lists,
            "df_sizes": df_sizes
        }
    except Exception as e:
        print(f"Error during data preparation: {e}")
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

    print("\n--- Linear Programming Optimization (Daily) ---")
    print("Objective: Minimize Cost + Waste Penalty")

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

         # Display Results
        if result.success:
            print(f"\nOptimization Successful!")
            print(f"Minimum Cost: ${result.fun:.2f}")

            results_df = pd.DataFrame(variable_names)
            results_df['optimal_quantity'] = result.x.astype(int)
            
            print("\nOptimal Meal Production Plan:")
            total_cost_check = 0
            for i, school in enumerate(schools_to_optimize):
                print(f"\n  {school}:")
                school_cost = 0
                for j, meal_type in enumerate(meal_types):
                    var_index = i * len(meal_types) + j
                    quantity = int(result.x[var_index])
                    cost = quantity * meal_costs[j]
                    school_cost += cost
                    
                    min_b, max_b = bounds[var_index]
                    print(f"    {meal_type}: {quantity} units "
                          f"(Demand: {demand[school][j]:.0f}, Prod. Bounds: {int(min_b)}-{int(max_b)})")

                print(f"    School Total Cost: ${school_cost:.2f} (Budget: ${school_budgets[school]:.2f})")
                total_cost_check += school_cost
            
            print("\n---")
            print(f"Total production cost across all schools: ${total_cost_check:.2f} (Total Budget: ${total_budget:.2f})")

            # Waste Analysis
            print(f"\nWaste Analysis")
            total_waste = 0
            for i, school in enumerate(schools_to_optimize):
                for j, meal_type in enumerate(meal_types):
                    var_index = i * len(meal_types) + j
                    produced = result.x[var_index]
                    expected_demand = demand[school][j]
                    waste = max(0, produced - expected_demand)
                    total_waste += waste
                    if waste > 0:
                        print(f"    {school} - {meal_type}: {waste:.1f} units waste")
            print(f"\nTotal Potential Waste: {total_waste:.1f} units")

            return results_df

        else:
            print(f"\nOptimization failed: {result.message}")
            return None

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None

def run_meal_optimization_ilp(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds):
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

    print("\n--- Integer Linear Programming Optimization (Daily) ---")
    print("Objective: Minimize Cost + Waste Penalty")

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

    num_constraints = len(b_ineq)
    b_l = np.full(num_constraints, -np.inf)
    constraints = LinearConstraint(A_ineq, lb=b_l, ub=b_ineq)

    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    bounds_obj = Bounds(lb=lower_bounds, ub=upper_bounds)

    # Running the Optimization
    try:
        integrality = [1] * num_vars

        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds_obj,
            constraints=constraints
        )

         # Display Results
        if result.success:
            print(f"\nOptimization Successful!")
            print(f"Minimum Cost: ${result.fun:.2f}")

            results_df = pd.DataFrame(variable_names)
            results_df['optimal_quantity'] = result.x.astype(int)
            
            print("\nOptimal Meal Production Plan:")
            total_cost_check = 0
            for i, school in enumerate(schools_to_optimize):
                print(f"\n  {school}:")
                school_cost = 0
                for j, meal_type in enumerate(meal_types):
                    var_index = i * len(meal_types) + j
                    quantity = int(result.x[var_index])
                    cost = quantity * meal_costs[j]
                    school_cost += cost
                    
                    min_b, max_b = bounds[var_index]
                    print(f"    {meal_type}: {quantity} units "
                          f"(Demand: {demand[school][j]:.0f}, Prod. Bounds: {int(min_b)}-{int(max_b)})")

                print(f"    School Total Cost: ${school_cost:.2f} (Budget: ${school_budgets[school]:.2f})")
                total_cost_check += school_cost
            
            print("\n---")
            print(f"Total production cost across all schools: ${total_cost_check:.2f} (Total Budget: ${total_budget:.2f})")

            # Waste Analysis
            print(f"\nWaste Analysis")
            total_waste = 0
            for i, school in enumerate(schools_to_optimize):
                for j, meal_type in enumerate(meal_types):
                    var_index = i * len(meal_types) + j
                    produced = result.x[var_index]
                    expected_demand = demand[school][j]
                    waste = max(0, produced - expected_demand)
                    total_waste += waste
                    if waste > 0:
                        print(f"    {school} - {meal_type}: {waste:.1f} units waste")
            print(f"\nTotal Potential Waste: {total_waste:.1f} units")

            return results_df

        else:
            print(f"\nOptimization failed: {result.message}")
            return None

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None

# ==============================================================================
# Food Item popularity optimization
# ==============================================================================
def generate_item_breakdown(optimization_results_df, dfb, dfl, output_filename):
    """Generates and saves the detailed food item breakdown from optimization results."""
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
        output_filename = '../data/school_food_item_optimization.csv'
        optimized_df.to_csv(output_filename, index=False)

    else:
        print("Optimization did not produce a result")

# ==============================================================================
# Size based and monthly optimization
# ==============================================================================
def run_size_based_optimization(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, all_school_lists):
    """
    Runs the linear programming model using predefined school size lists
    and saves the output to a separate CSV file.
    """

    dfb = pd.read_csv('../data/preprocessed-data/breakfast_combined.csv', low_memory=False)
    dfl = pd.read_csv('../data/preprocessed-data/lunch_combined.csv', low_memory=False)

    # --- Create the school_sizes dictionary ---
    school_sizes = {}
    for size, school_list in all_school_lists.items():
        for school in school_list:
            school_sizes[school] = size
    print("--- Starting Size-Based Optimization ---")
    print(f"Categorized {len(school_sizes)} schools based on your lists.")

    # --- Define and set production bounds based on school size ---
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

    # --- Run the base optimization logic ---
    results_df = run_meal_optimization_ilp(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds)

    # --- Generate the Food Item Breakdown and Save to a CSV file ---
    if results_df is not None:
        bf_popularity = dfb.groupby('name')['served_reimbursable'].sum()
        bf_total_served = bf_popularity.sum()
        bf_popularity_prop = (bf_popularity / bf_total_served).reset_index(name='proportion')
        bf_popularity_prop['meal_type'] = 'Breakfast'

        ln_popularity = dfl.groupby('name')['served_reimbursable'].sum()
        ln_total_served = ln_popularity.sum()
        ln_popularity_prop = (ln_popularity / ln_total_served).reset_index(name='proportion')
        ln_popularity_prop['meal_Type'] = 'Lunch'

        item_popularity_df = pd.concat([bf_popularity_prop, ln_popularity_prop])
        item_df = pd.merge(results_df, item_popularity_df, on='meal_type')
        item_df['recommended_quantity'] = (item_df['optimal_quantity'] * item_df['proportion']).round().astype(int)

        optimized_df = item_df[item_df['recommended_quantity'] > 0][['school', 'meal_type', 'name', 'recommended_quantity']].rename(columns={'name': 'food_item'})
        optimized_df = optimized_df.sort_values(['school', 'meal_type', 'recommended_quantity'], ascending=[True, True, False])

        # Save to a CSV file
        output_filename = '../data/school_food_item_optimization_by_size.csv'
        optimized_df.to_csv(output_filename, index=False)
        print(f"\nSaved the size-based optimization results to '{output_filename}'")
    else:
        print("Size-based optimization did not produce a result.")

def run_monthly_meal_optimization(schools_to_optimize, meal_types, meal_costs, demand, waste_penalty, bounds):
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

    # Integrality constraint for all variables (1 means integer)
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

            for i, school in enumerate(schools_to_optimize[:5]): # Limiting to first 5 schools for output
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
def run_proportional_monthly_optimization_ilp(data, total_budget=139144760):
    """
    Runs a monthly optimization where the budget is allocated to each school
    proportionally based on its student population.
    """
    print("\n" + "="*60)
    print("Proportional Budget Monthly Optimization (ILP)")
    print("="*60)

    schools = data['schools']
    meal_types = data['meal_types']
    meal_costs = data['meal_costs']
    daily_demand = data['demand']
    df_sizes = data['df_sizes']
    dfb = data['dfb']
    dfl = data['dfl']

    if df_sizes is None:
        print("Cannot run: Student size data is required for proportional budgeting.")
        return

    # --- Calculate Proportional Budget for Each School ---
    print("\n--- Allocating Budget Based on Student Population ---")
    relevant_schools_df = df_sizes[df_sizes['school_name'].isin(schools)].copy()
    total_population = relevant_schools_df['count'].sum()
    school_budgets = {}
    for index, row in relevant_schools_df.iterrows():
        school_name = row['school_name']
        proportion = row['count'] / total_population
        school_budgets[school_name] = total_budget * proportion
    print(f"Total budget of ${total_budget:,.2f} allocated proportionally.")

    # --- Aggregate Demand and Bounds to Monthly ---
    SCHOOL_DAYS_PER_MONTH = 20
    waste_penalty = [0.50, 1.00]
    monthly_demand = {s: [d[0] * SCHOOL_DAYS_PER_MONTH, d[1] * SCHOOL_DAYS_PER_MONTH] for s, d in daily_demand.items()}
    monthly_bounds = []
    for school in schools:
        for i in range(len(meal_types)):
            monthly_bounds.append((monthly_demand[school][i] * 0.85, monthly_demand[school][i] * 1.10))

    # --- Run the Optimization ---
    results_df = run_meal_optimization_ilp(
        schools, meal_types, meal_costs, monthly_demand,
        school_budgets, total_budget, waste_penalty, monthly_bounds
    )
    
    # --- Generate the item breakdown CSV ---
    if results_df is not None:
        generate_item_breakdown(
            results_df,
            dfb,
            dfl,
            '../data/preprocessed-data/monthly_proportional_to_size.csv'
        )
    return results_df, school_budgets, meal_costs

# ==============================================================================
# Calculate the actual annual cost of producing the foods
# ==============================================================================
def calculate_actual_annual_cost(data):
    """
    Calculates the actual total food cost from the source data and
    extrapolates it to a full 10-month school year.
    """
    print("\n--- Calculating Baseline (Actual) Annual Food Cost ---")
    dfb = data['dfb']
    dfl = data['dfl']
    
    # Sum the costs from the one-month data period
    actual_monthly_cost = dfb['production_cost_total'].sum() + dfl['production_cost_total'].sum()
    
    # Scale to a 10-month school year
    actual_annual_cost = actual_monthly_cost * 10
    
    print(f"Actual cost for the data period (1 month): ${actual_monthly_cost:,.2f}")
    print(f"Estimated Actual Annual Food Cost: ${actual_annual_cost:,.2f}")
    
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

    print("\n" + "="*60)
    print("ANALYSIS: Remaining Annual Budget for Non-Food Expenses")
    print("="*60)

    MONTHS_IN_SCHOOL_YEAR = 10

    # Map meal types to their costs
    meal_cost_map = {'Breakfast': meal_costs[0], 'Lunch': meal_costs[1]}
    results_df['monthly_food_cost'] = results_df.apply(
        lambda row: row['optimal_quantity'] * meal_cost_map[row['meal_type']],
        axis=1
    )

    # Calculate total food cost per school for one month
    school_monthly_costs = results_df.groupby('school')['monthly_food_cost'].sum().reset_index()

    # Create a DataFrame for the analysis
    budget_analysis_df = pd.DataFrame(list(school_budgets.items()), columns=['school', 'proportional_annual_budget'])
    budget_analysis_df = pd.merge(budget_analysis_df, school_monthly_costs, on='school', how='left').fillna(0)

    # Scale monthly food cost to a full 10-month school year
    budget_analysis_df['annual_food_cost'] = budget_analysis_df['monthly_food_cost'] * MONTHS_IN_SCHOOL_YEAR
    budget_analysis_df['remaining_annual_balance'] = budget_analysis_df['proportional_annual_budget'] - budget_analysis_df['annual_food_cost']
    
    # Calculate and print the grand totals for the full year
    total_budget = budget_analysis_df['proportional_annual_budget'].sum()
    grand_total_food_cost = budget_analysis_df['annual_food_cost'].sum()
    grand_total_remaining = budget_analysis_df['remaining_annual_balance'].sum()

    print("\n--- Overall Financial Summary (Annual) ---")
    print(f"Total Allocated Annual Budget: ${total_budget:,.2f}")
    print(f"Grand Total Annual Food Expenses: ${grand_total_food_cost:,.2f}")
    print(f"Grand Total Remaining for Other Expenses: ${grand_total_remaining:,.2f}")

    if actual_annual_cost is not None:
        savings = actual_annual_cost - grand_total_food_cost
        savings_percent = (savings / actual_annual_cost) * 100
        print("\n-- Savings Analysis ---")
        print(f"Baseline Actual Annual Food Cost: ${actual_annual_cost:,.2f}")
        print(f"Optimized Annual Food Cost: ${grand_total_food_cost:,.2f}")
        print(f"Total Annual Savings: ${savings:,.2f} ({savings_percent:.2f}%)")

    # Format for display
    print("\n--- Detailed Breakdown by School (Annual) ---")
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
    savings_df['optimized_annual_cost'].fillna(0, inplace=True)

    savings_df['savings'] = savings_df['actual_annual_cost'] - savings_df['optimized_annual_cost']
    
    # Create an 'outcome' column and an absolute value column for size
    savings_df['outcome'] = np.where(savings_df['savings'] >= 0, 'Savings', 'Loss')
    savings_df['savings_magnitude'] = savings_df['savings'].abs()
    
    # Add size category for additional hover data
    savings_df = pd.merge(savings_df, df_sizes[['school_name', 'size_category']], left_on='school', right_on='school_name', how='left')

    return savings_df

# ==============================================================================
# Data preparation for graphing savings by school size 
# ==============================================================================
def analyze_savings_by_school_size(results_df, school_budgets, meal_costs, df_sizes):
    """
    Aggregates the budget, optimized cost, and savings by school size category.
    """
    if results_df is None or df_sizes is None:
        print("Skipping analysis: Missing results or size information.")
        return None

    print("\n" + "="*60)
    print("ANALYSIS: Savings by School Size Category")
    print("="*60)

    # Calculate annual food cost for each school
    MONTHS_IN_SCHOOL_YEAR = 10
    meal_cost_map = {'Breakfast': meal_costs[0], 'Lunch': meal_costs[1]}
    results_df['monthly_food_cost'] = results_df.apply(
        lambda row: row['optimal_quantity'] * meal_cost_map[row['meal_type']], axis=1
    )
    school_costs = results_df.groupby('school')['monthly_food_cost'].sum().reset_index()
    school_costs['annual_food_cost'] = school_costs['monthly_food_cost'] * MONTHS_IN_SCHOOL_YEAR

    # Combine budget and cost data with school size data
    budget_df = pd.DataFrame(list(school_budgets.items()), columns=['school', 'proportional_annual_budget'])
    analysis_df = pd.merge(budget_df, school_costs[['school', 'annual_food_cost']], on='school', how='left')
    analysis_df = pd.merge(analysis_df, df_sizes[['school_name', 'size_category']], left_on='school', right_on='school_name', how='left')

    # Group by size category and sum the totals
    agg_df = analysis_df.groupby('size_category')[['proportional_annual_budget', 'annual_food_cost']].sum().reset_index()
    
    # Calculate savings in dollars and as a percentage
    agg_df['total_savings'] = agg_df['proportional_annual_budget'] - agg_df['annual_food_cost']
    agg_df['percent_savings'] = (agg_df['total_savings'] / agg_df['proportional_annual_budget']) * 100

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
    
    print("\n--- Preparing Map Data from Coordinates ---")
    
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