import pandas as pd
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# ==============================================================================
# Data Preparation for Optimization
# ==============================================================================
def prepare_optimization_data_m(breakfast_path, lunch_path, student_counts_path):
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
def run_meal_optimization_m(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds):
    """
    Runs the linear programming model and prints an output with production bounds
    """
    num_vars = len(schools_to_optimize) * len(meal_types)

    # --- Objective Function ---
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

    # --- Constraints ---
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

    # --- Running the Optimization ---
    try:
        result = linprog(
            c=c,
            A_ub=A_ineq,
            b_ub=b_ineq,
            bounds=bounds,
            method='highs'
        )

         # --- Display Results ---
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

            # --- Waste Analysis ---
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

def run_meal_optimization_ilp_m(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds):
    """
    Runs the linear programming model and prints an output with production bounds
    """
    num_vars = len(schools_to_optimize) * len(meal_types)

    # --- Objective Function ---
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

    # --- Constraints ---
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

    # --- Running the Optimization ---
    try:
        integrality = [1] * num_vars

        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds_obj,
            constraints=constraints
        )

         # --- Display Results ---
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

            # --- Waste Analysis ---
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
def generate_item_breakdown_m(optimization_results_df, dfb, dfl, output_filename):
    """Generates and saves the detailed food item breakdown from optimization results."""
    if optimization_results_df is None:
        print("Skipping item breakdown: No optimization results.")
        return
    
    # --- Generate Food Item Breakdown ---
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
        output_filename = '../data/school_milk_optimization.csv'
        optimized_df.to_csv(output_filename, index=False)

        print(f"Saved the full list to '{output_filename}'")

    else:
        print("Optimization did not produce a result")


# ==============================================================================
# Size based and monthly optimization
# ==============================================================================
def run_size_based_optimization_m(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, all_school_lists):
    """
    Runs the linear programming model using predefined school size lists
    and saves the output to a separate CSV file.
    """

    dfb = pd.read_csv('../data/preprocessed-data/breakfast_combined.csv')
    dfl = pd.read_csv('../data/preprocessed-data/lunch_combined.csv')

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
    results_df = run_meal_optimization_m(schools_to_optimize, meal_types, meal_costs, demand, school_budgets, total_budget, waste_penalty, bounds)

    # --- Generate the Food Item Breakdown and Save to a CSV file ---
    if results_df is not None:
        bf_popularity = dfb.groupby('Name')['Served_Reimbursable'].sum()
        bf_total_served = bf_popularity.sum()
        bf_popularity_prop = (bf_popularity / bf_total_served).reset_index(name='proportion')
        bf_popularity_prop['meal_type'] = 'Breakfast'

        ln_popularity = dfl.groupby('Name')['Served_Reimbursable'].sum()
        ln_total_served = ln_popularity.sum()
        ln_popularity_prop = (ln_popularity / ln_total_served).reset_index(name='proportion')
        ln_popularity_prop['meal_Type'] = 'Lunch'

        item_popularity_df = pd.concat([bf_popularity_prop, ln_popularity_prop])
        item_df = pd.merge(results_df, item_popularity_df, on='meal_type')
        item_df['recommended_quantity'] = (item_df['optimal_quantity'] * item_df['proportion']).round().astype(int)

        optimized_df = item_df[item_df['recommended_quantity'] > 0][['school', 'meal_type', 'Name', 'recommended_quantity']].rename(columns={'Name': 'food_item'})
        optimized_df = optimized_df.sort_values(['school', 'meal_type', 'recommended_quantity'], ascending=[True, True, False])

        # Save to a CSV file
        output_filename = '../data/school_milk_optimization_by_size.csv'
        optimized_df.to_csv(output_filename, index=False)
        print(f"\nSaved the size-based optimization results to '{output_filename}'")
    else:
        print("Size-based optimization did not produce a result.")

def run_monthly_meal_optimization_m(schools_to_optimize, meal_types, meal_costs, demand, waste_penalty, bounds):
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

    # Due to unfeasibility errors, budgetary constraints have been removed
    A_ineq = None
    b_ineq = None

    try:
        result = linprog(
            c=c,
            A_ub=A_ineq,
            b_ub=b_ineq,
            bounds=bounds,
            method='highs'
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
def run_proportional_monthly_optimization_m(data, total_budget=139144760):
    """
    Runs a monthly optimization where the budget is allocated to each school
    proportionally based on its student population.
    """
    print("\n" + "="*60)
    print("Proportional Budget Monthly Optimization (LP)")
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
    results_df = run_meal_optimization_m(
        schools, meal_types, meal_costs, monthly_demand,
        school_budgets, total_budget, waste_penalty, monthly_bounds
    )
    
    # --- Generate the item breakdown CSV ---
    if results_df is not None:
        generate_item_breakdown_m(
            results_df,
            dfb,
            dfl,
            '../data/monthly_proportional_to_size_milk.csv'
        )