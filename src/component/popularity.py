import pandas as pd
import numpy as np  # Added numpy for handling potential inf values

def prepare_popularity_data(breakfast_path, lunch_path, sales_path):
    """
    Prepares the breakfast, lunch, and sales files for food popularity analysis.

    Args:
        breakfast_path (str): Path to the combined breakfast CSV file.
        lunch_path (str): Path to the combined lunch CSV file.
        sales_path (str): Path to the sales CSV file.
    """
    try:
        dfb = pd.read_csv(breakfast_path, low_memory=False)
        dfl = pd.read_csv(lunch_path, low_memory=False)
        dfs = pd.read_csv(sales_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        return None, None, None  # Return None to avoid errors

    # converting column names to lowercase
    dfb.columns = dfb.columns.str.lower()
    dfl.columns = dfl.columns.str.lower()
    dfs.columns = dfs.columns.str.lower()

    # converting date columns into datetime objects
    dfb['date'] = pd.to_datetime(dfb['date'], errors='coerce')
    dfl['date'] = pd.to_datetime(dfl['date'], errors='coerce')

    return dfb, dfl, dfs


def clean_numeric(df, cols):
    """
    Cleans the numeric columns

    Args:
        df (str): Dataframe being used to clean numerics with
        cols (str): Columns to apply cleaning with
    """

    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[$,%]", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)
    return df


def food_popularity(breakfast_df, lunch_df, sales_df):
    """
    Analyzes and prints food popularity based on served and discarded totals.

    Args:
        breakfast_df (pd.DataFrame): Breakfast data.
        lunch_df (pd.DataFrame): Lunch data.
        sales_df (pd.DataFrame): Sales data.
    """
    if breakfast_df is None or lunch_df is None:
        print("DataFrames not loaded. Skipping food popularity analysis.")
        return

    # Popularity by served_reimbursable
    print("Top 10 Breakfast Items (by served_total):")
    b_pop_served = breakfast_df.groupby("name")["served_reimbursable"].sum().nlargest(10)
    print(breakfast_df[breakfast_df['name'].isin(b_pop_served.index)][['name', 'served_reimbursable']].groupby('name').sum().sort_values('served_reimbursable', ascending=False))

    print("\nTop 10 Lunch Items (by served_total):")
    l_pop_served = lunch_df.groupby("name")["served_reimbursable"].sum().nlargest(10)
    print(lunch_df[lunch_df['name'].isin(l_pop_served.index)][['name', 'served_reimbursable']].groupby('name').sum().sort_values('served_reimbursable', ascending=False))

    # Popularity by discarded_total (least popular)
    print("\nTop 10 Breakfast Items (least discarded_total):")
    b_pop_discarded = breakfast_df.groupby("name")["discarded_total"].sum().nsmallest(10)
    print(breakfast_df[breakfast_df['name'].isin(b_pop_discarded.index)][['name', 'discarded_total']].groupby('name').sum().sort_values('discarded_total', ascending=True))


    print("\nTop 10 Lunch Items (least discarded_total):")
    l_pop_discarded = lunch_df.groupby("name")["discarded_total"].sum().nsmallest(10)
    print(lunch_df[lunch_df['name'].isin(l_pop_discarded.index)][['name', 'discarded_total']].groupby('name').sum().sort_values('discarded_total', ascending=True))


def net_consumption(breakfast_df, lunch_df):
    """
    Analyzes and prints net consumption for breakfast and lunch.

    Args:
        breakfast_df (pd.DataFrame): Breakfast data.
        lunch_df (pd.DataFrame): Lunch data.
    """
    if breakfast_df is None or lunch_df is None:
        print("DataFrames not loaded. Skipping net consumption analysis.")
        return

    print("=== NET CONSUMPTION POPULARITY RANKINGS ===\n")

    # Breakfast
    breakfast_df["net_consumption"] = (
        breakfast_df["served_reimbursable"] - breakfast_df["discarded_total"]
    )
    b_net_consumption = (
        breakfast_df.groupby("name")
        .agg(
            net_consumption=("net_consumption", "sum"),
            total_served=("served_reimbursable", "sum"),
            total_discarded=("discarded_total", "sum"),
        )
        .nlargest(15, "net_consumption")
    )
    
    print("Top 15 Breakfast Items (by Net Consumption):")
    print("=" * 90)
    b_display = b_net_consumption[['net_consumption', 'total_served', 'total_discarded']]
    b_display.columns = ['Net Consumption', 'Total Served', 'Total Discarded']
    print(b_display)


    # Lunch
    lunch_df["net_consumption"] = (
        lunch_df["served_reimbursable"] - lunch_df["discarded_total"]
    )
    l_net_consumption = (
        lunch_df.groupby("name")
        .agg(
            net_consumption=("net_consumption", "sum"),
            total_served=("served_reimbursable", "sum"),
            total_discarded=("discarded_total", "sum"),
        )
        .nlargest(15, "net_consumption")
    )
    
    print("\nTop 15 Lunch Items (by Net Consumption):")
    print("=" * 90)
    l_display = l_net_consumption[['net_consumption', 'total_served', 'total_discarded']]
    l_display.columns = ['Net Consumption', 'Total Served', 'Total Discarded']
    print(l_display)


def leftover_rate(breakfast_df, lunch_df):
    """
    Analyzes and prints leftover rates for breakfast and lunch.

    Args:
        breakfast_df (pd.DataFrame): Breakfast data.
        lunch_df (pd.DataFrame): Lunch data.
    """
    if breakfast_df is None or lunch_df is None:
        print("DataFrames not loaded. Skipping leftover rate analysis.")
        return

    print("=== LEFTOVER RATE RANKINGS ===\n")

    # Breakfast
    b_item_waste = breakfast_df.groupby("name").agg(
        left_over_total=("left_over_total", "sum"),
        offered_reimbursable=("offered_reimbursable", "sum"),
    )
    b_item_waste["leftover_rate"] = (
        b_item_waste["left_over_total"] / b_item_waste["offered_reimbursable"]
    ) * 100
    b_item_waste = b_item_waste[b_item_waste["offered_reimbursable"] > 0]  # Avoid division by zero
    b_item_waste = b_item_waste.sort_values("leftover_rate", ascending=False)
    
    print("Top 15 Breakfast Items (Highest Leftover Rate - Most Waste):")
    print("=" * 100)
    breakfast_display = b_item_waste[['leftover_rate', 'left_over_total', 'offered_reimbursable']].head(15)
    breakfast_display.columns = ['Leftover Rate (%)', 'Total Left Over', 'Total Offered']
    breakfast_display['Leftover Rate (%)'] = breakfast_display['Leftover Rate (%)'].round(2)
    print(breakfast_display)


    # Lunch
    l_item_waste = lunch_df.groupby("name").agg(
        left_over_total=("left_over_total", "sum"),
        offered_reimbursable=("offered_reimbursable", "sum"),
    )
    l_item_waste["leftover_rate"] = (
        l_item_waste["left_over_total"] / l_item_waste["offered_reimbursable"]
    ) * 100
    l_item_waste = l_item_waste[l_item_waste["offered_reimbursable"] > 0]
    l_item_waste = l_item_waste.sort_values("leftover_rate", ascending=False)
    
    print("\nTop 15 Lunch Items (Highest Leftover Rate - Most Waste):")
    print("=" * 100)
    lunch_display = l_item_waste[['leftover_rate', 'left_over_total', 'offered_reimbursable']].head(15)
    lunch_display.columns = ['Leftover Rate (%)', 'Total Left Over', 'Total Offered']
    lunch_display['Leftover Rate (%)'] = lunch_display['Leftover Rate (%)'].round(2)
    print(lunch_display)

    # Calculate overall statistics
    print("\n=== OVERALL LEFTOVER STATISTICS ===")
    breakfast_total_leftover = breakfast_df['left_over_total'].sum()
    breakfast_total_offered = breakfast_df['offered_reimbursable'].sum()
    if breakfast_total_offered > 0:
        breakfast_overall_rate = (breakfast_total_leftover / breakfast_total_offered) * 100
        print(f"Breakfast Overall Leftover Rate: {breakfast_overall_rate:.2f}%")
        print(f"  Total Left Over: {breakfast_total_leftover:,.0f}")
        print(f"  Total Offered: {breakfast_total_offered:,.0f}")
    else:
        print("Breakfast: No offered items to calculate overall rate.")

    lunch_total_leftover = lunch_df['left_over_total'].sum()
    lunch_total_offered = lunch_df['offered_reimbursable'].sum()
    if lunch_total_offered > 0:
        lunch_overall_rate = (lunch_total_leftover / lunch_total_offered) * 100
        print(f"\nLunch Overall Leftover Rate: {lunch_overall_rate:.2f}%")
        print(f"  Total Left Over: {lunch_total_leftover:,.0f}")
        print(f"  Total Offered: {lunch_total_offered:,.0f}")
    else:
        print("\nLunch: No offered items to calculate overall rate.")

    # Combined
    total_leftover = breakfast_total_leftover + lunch_total_leftover
    total_offered = breakfast_total_offered + lunch_total_offered
    if total_offered > 0:
        combined_overall_rate = (total_leftover / total_offered) * 100
        print(f"\nCombined Overall Leftover Rate: {combined_overall_rate:.2f}%")
    else:
        print("\nCombined: No offered items to calculate overall rate.")


# --- MODIFIED FUNCTION BELOW ---

def get_net_consumption_by_school(df, meal_type):
    """
    Calculates the net consumption rate for each food item at each school.
    Assumes 'offered_reimbursable' and 'left_over_total' are already numeric.
    Net Consumption Rate = (Offered - Left Over) / Offered
    """
    print(f"Calculating Net Consumption Rate by Item and School for {meal_type}...")
    
    # Group by school AND item name
    item_summary = df.groupby(['school_name', 'name']).agg(
        total_offered=('offered_reimbursable', 'sum'),
        total_left_over=('left_over_total', 'sum')
    ).reset_index()

    # Calculate net consumption rate for each item at each school
    item_summary['net_consumption_rate'] = (
        (item_summary['total_offered'] - item_summary['total_left_over']) / item_summary['total_offered']
    ) * 100
    
    # Handle potential division by zero (if total_offered is 0)
    item_summary['net_consumption_rate'] = item_summary['net_consumption_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Sort by school, then by consumption rate
    item_summary = item_summary.sort_values(by=['school_name', 'net_consumption_rate'], ascending=[True, False])
    
    return item_summary

def get_leftover_rate_by_school(df, meal_type):
    """
    Calculates the leftover rate for each food item at each school.
    Assumes 'left_over_total' and 'offered_reimbursable' are already numeric.
    Leftover Rate = (Left Over / Offered) * 100
    """
    print(f"Calculating Leftover Rate by Item and School for {meal_type}...")
    
    # Group by school and item name, then sum up totals
    item_summary = df.groupby(['school_name', 'name']).agg(
        left_over_total=('left_over_total', 'sum'),
        offered_reimbursable=('offered_reimbursable', 'sum')
    ).reset_index()

    # Calculate leftover rate for each item at each school
    item_summary['leftover_rate'] = (
        (item_summary['left_over_total'] / item_summary['offered_reimbursable']) * 100
    )
    
    # Handle division by zero (if offered is 0)
    item_summary['leftover_rate'] = item_summary['leftover_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Sort by school, then by leftover rate (descending)
    item_summary = item_summary.sort_values(by=['school_name', 'leftover_rate'], ascending=[True, False])
    
    return item_summary