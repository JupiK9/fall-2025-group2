import pandas as pd

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
        return

    # converting column names to lowercase
    dfb.columns = dfb.columns.str.lower()
    dfl.columns = dfl.columns.str.lower()
    dfs.columns = dfs.columns.str.lower()

    # converting date columns into datetime objects
    dfb['date'] = pd.to_datetime(dfb['date'])
    dfl['date'] = pd.to_datetime(dfl['date'])

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
                .str.replace(r"[\$,%, ]", "", regex=True)
                .replace('nan', '0')
                .astype(float)
            )
    return df

def food_popularity(breakfast_df, lunch_df, sales_df):
    """
    """

    # from sales
    popular_sales = (
        sales_df.groupby(["time_of_day", "description"])['total']
        .sum()
        .reset_index()
        .sort_values(["time_of_day", "total"], ascending=[True, False])
    )

    # from breakfast (by served)
    popular_breakfast = (
        breakfast_df.groupby("name")["served_reimbursable"]
        .sum()
        .reset_index()
        .sort_values("served_reimbursable", ascending=False)
    )

    # From lunch (by served)
    popular_lunch = (
        lunch_df.groupby("name")["served_reimbursable"]
        .sum()
        .reset_index()
        .sort_values("served_reimbursable", ascending=False)
    )

    # --- Least discarded (lower is better) ---
    if "discarded_total" in breakfast_df.columns:
        popular_breakfast_low_discarded = (
            breakfast_df.groupby("name")["discarded_total"]
            .sum()
            .reset_index()
            .sort_values("discarded_total", ascending=True)
        )
    else:
        popular_breakfast_low_discarded = None

    if "discarded_total" in lunch_df.columns:
        popular_lunch_low_discarded = (
            lunch_df.groupby("name")["discarded_total"]
            .sum()
            .reset_index()
            .sort_values("discarded_total", ascending=True)
        )
    else:
        popular_lunch_low_discarded = None

    print("Top 10 Breakfast Items (by served_total):")
    print(popular_breakfast.head(10))

    print("\nTop 10 Lunch Items (by served_total):")
    print(popular_lunch.head(10))

    if popular_breakfast_low_discarded is not None:
        print("\nTop 10 Breakfast Items (least discarded_total):")
        print(popular_breakfast_low_discarded.head(10))

    if popular_lunch_low_discarded is not None:
        print("\nTop 10 Lunch Items (least discarded_total):")
        print(popular_lunch_low_discarded.head(10))

def net_consumption(breakfast_df, lunch_df):
    """
    Calculates the net consumption for breakfast and lunch items
    prints the top 15, and saves the full lists to CSV files
    """

    # Breakfast
    breakfast_net_consumption = (
        breakfast_df.groupby("name")
        .agg({
            'served_reimbursable': 'sum',
            'discarded_total': 'sum'
        })
        .reset_index()
    )

    # Calculating net consumption
    breakfast_net_consumption['net_consumption'] = (
        breakfast_net_consumption['served_reimbursable'] - breakfast_net_consumption['discarded_total']
    )

    # Sorting by net consumption (higher is better)
    breakfast_net_consumption = breakfast_net_consumption.sort_values('net_consumption', ascending=False)

    # Prepare Dataframe for output
    breakfast_output_df = breakfast_net_consumption[['name', 'net_consumption', 'served_reimbursable', 'discarded_total']].copy()
    breakfast_output_df.columns = ['Item Name', 'Net Consumption', 'Total Served', 'Total Discarded']

    # Save as CSV
    breakfast_output_df.to_csv('../data/preprocessed-data/breakfast_net_consumption.csv', index=False)
    print("Saved full breakfast net consumption data to 'breakfast_net_consumption.csv'")

    # Lunch
    lunch_net_consumption = (
        lunch_df.groupby("name")
        .agg({
            'served_reimbursable': 'sum',
            'discarded_total': 'sum'
        })
        .reset_index()
    )

    # Calculate net consumption
    lunch_net_consumption['net_consumption'] = (
        lunch_net_consumption['served_reimbursable'] - lunch_net_consumption['discarded_total']
    )

    # Sorting by net consumption (higher is better)
    lunch_net_consumption = lunch_net_consumption.sort_values('net_consumption', ascending=False)

    # Prepare DataFrame for output
    lunch_output_df = lunch_net_consumption[['name', 'net_consumption', 'served_reimbursable', 'discarded_total']].copy()
    lunch_output_df.columns = ['Item Name', 'Net Consumption', 'Total Served', 'Total Discarded']
    
    # --- Save to CSV ---
    lunch_output_df.to_csv('../data/preprocessed-data/lunch_net_consumption.csv', index=False)
    print("Saved full lunch net consumption data to 'lunch_net_consumption.csv'")

    # Display Net Consumption Results
    print("=== NET CONSUMPTION POPULARITY RANKINGS ===")

    print("\nTop 15 Breakfast Items (by Net Consumption):")
    print("=" * 90)
    breakfast_display = breakfast_net_consumption[['name', 'net_consumption', 'served_reimbursable', 'discarded_total']].head(15)
    breakfast_display.columns = ['Item Name', 'Net Consumption', 'Total Served', 'Total Discarded']
    print(breakfast_display)

    print("\nTop 15 Lunch Items (by Net Consumption):")
    print("=" * 90)
    lunch_display = lunch_net_consumption[['name', 'net_consumption', 'served_reimbursable', 'discarded_total']].head(15)
    lunch_display.columns = ['Item Name', 'Net Consumption', 'Total Served', 'Total Discarded']
    print(lunch_display)

def leftover_rate(breakfast_df, lunch_df):
    """
    Calculates the leftover rate for breakfast and lunch items,
    prints the top 15 food items, and saves the full lists to CSV files.
    """
    # ---------- Finding the items with the highest leftover rate ----------

    # Breakfast
    breakfast_leftover_rate = (
        breakfast_df.groupby("name")
        .agg({
            'left_over_total': 'sum',
            'offered_reimbursable': 'sum'
        })
        .reset_index()
    )

    # Calculate leftover rate (percentage)
    breakfast_leftover_rate['leftover_rate'] = (
        breakfast_leftover_rate['left_over_total'] / 
        breakfast_leftover_rate['offered_reimbursable'].replace(0, 1)
    ) * 100

    # Sorting by leftover rate (higher is worse)
    breakfast_leftover_rate = breakfast_leftover_rate.sort_values('leftover_rate', ascending=False)

    # Prepare DataFrame for output
    breakfast_output_df = breakfast_leftover_rate[['name', 'leftover_rate', 'left_over_total', 'offered_reimbursable']].copy()
    breakfast_output_df.columns = ['Item Name', 'Leftover Rate (%)', 'Total Left Over', 'Total Offered']
    breakfast_output_df['Leftover Rate (%)'] = breakfast_output_df['Leftover Rate (%)'].round(2)
    
    # Save to CSV
    breakfast_output_df.to_csv('../data/preprocessed-data/breakfast_leftover_rate.csv', index=False)
    print("Saved full breakfast leftover rate data to 'breakfast_leftover_rate.csv'")

    # Lunch
    lunch_leftover_rate = (
        lunch_df.groupby("name")
        .agg({
            'left_over_total': 'sum',
            'offered_reimbursable': 'sum'
        })
        .reset_index()
    )

    # Calculate leftover rate (percentage)
    lunch_leftover_rate['leftover_rate'] = (
        lunch_leftover_rate['left_over_total'] / 
        lunch_leftover_rate['offered_reimbursable'].replace(0, 1)
    ) * 100

    # Sorting by leftover rate (higher is worse)
    lunch_leftover_rate = lunch_leftover_rate.sort_values('leftover_rate', ascending=False)

    # Prepare DataFrame for output
    lunch_output_df = lunch_leftover_rate[['name', 'leftover_rate', 'left_over_total', 'offered_reimbursable']].copy()
    lunch_output_df.columns = ['Item Name', 'Leftover Rate (%)', 'Total Left Over', 'Total Offered']
    lunch_output_df['Leftover Rate (%)'] = lunch_output_df['Leftover Rate (%)'].round(2)
    
    # Save to CSV
    lunch_output_df.to_csv('../data/preprocessed-data/lunch_leftover_rate.csv', index=False)
    print("Saved full lunch leftover rate data to 'lunch_leftover_rate.csv'")

    # --- Display Leftover Rate Results ---
    print("=== LEFTOVER RATE RANKINGS ===")

    print("\nTop 15 Breakfast Items (Highest Leftover Rate - Most Waste):")
    print("=" * 100)
    breakfast_display = breakfast_leftover_rate[['name', 'leftover_rate', 'left_over_total', 'offered_reimbursable']].head(15)
    breakfast_display.columns = ['Item Name', 'Leftover Rate (%)', 'Total Left Over', 'Total Offered']
    breakfast_display['Leftover Rate (%)'] = breakfast_display['Leftover Rate (%)'].round(2)
    print(breakfast_display)

    print("\nTop 15 Lunch Items (Highest Leftover Rate - Most Waste):")
    print("=" * 100)
    lunch_display = lunch_leftover_rate[['name', 'leftover_rate', 'left_over_total', 'offered_reimbursable']].head(15)
    lunch_display.columns = ['Item Name', 'Leftover Rate (%)', 'Total Left Over', 'Total Offered']
    lunch_display['Leftover Rate (%)'] = lunch_display['Leftover Rate (%)'].round(2)
    print(lunch_display)

    # Calculate overall statistics
    print("\n=== OVERALL LEFTOVER STATISTICS ===")
    breakfast_total_leftover = breakfast_df['left_over_total'].sum()
    breakfast_total_offered = breakfast_df['offered_reimbursable'].sum()
    breakfast_overall_rate = (breakfast_total_leftover / breakfast_total_offered) * 100

    lunch_total_leftover = lunch_df['left_over_total'].sum()
    lunch_total_offered = lunch_df['offered_reimbursable'].sum()
    lunch_overall_rate = (lunch_total_leftover / lunch_total_offered) * 100

    print(f"Breakfast Overall Leftover Rate: {breakfast_overall_rate:.2f}%")
    print(f"  Total Left Over: {breakfast_total_leftover:,.0f}")
    print(f"  Total Offered: {breakfast_total_offered:,.0f}")

    print(f"\nLunch Overall Leftover Rate: {lunch_overall_rate:.2f}%")
    print(f"  Total Left Over: {lunch_total_leftover:,.0f}")
    print(f"  Total Offered: {lunch_total_offered:,.0f}")

    print(f"\nCombined Overall Leftover Rate: {((breakfast_total_leftover + lunch_total_leftover) / (breakfast_total_offered + lunch_total_offered)) * 100:.2f}%")
