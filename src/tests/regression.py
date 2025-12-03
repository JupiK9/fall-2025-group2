import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120

# --- File Paths ---
BREAKFAST_CSV = 'raw_data/data_breakfast_with_coordinates.csv'
LUNCH_CSV = 'raw_data/data_lunch_with_coordinates.csv'
STUDENT_COUNTS_CSV = 'raw_data/student_counts.csv'
REGIONS_GEOJSON = 'raw_data/School_Regions.geojson'



print(f"Loading breakfast data from: {BREAKFAST_CSV}")
# ... rest of your code

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
    'offered_reimbursable', 'left_over_total', 'planned_total', 'served_total'
]  # Added planned_total and served_total as they might be used in regressions


# --- Helper Function to Run Regressions Safely ---
def run_regression(dependent_var, independent_vars, data, title, min_rows=2, plot=True):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if dependent_var not in data.columns:
        print(f"Error: Dependent variable '{dependent_var}' not found in dataset.")
        return None, None

    missing_ivs = [iv for iv in independent_vars if iv not in data.columns]
    if missing_ivs:
        print(f"Error: Independent variables '{', '.join(missing_ivs)}' not found in dataset.")
        return None, None

    # Filter out independent variables that are constant in the current subset
    available_ivs = []
    for iv in independent_vars:
        if data[iv].nunique() > 1:
            available_ivs.append(iv)
        else:
            print(f"Warning: Independent variable '{iv}' is constant in this subset and will be excluded.")

    if not available_ivs:
        print(f"No valid independent variables left after filtering constant columns for {title}.")
        return None, None

    X_raw = data[available_ivs].copy().apply(pd.to_numeric, errors='coerce')
    y_raw = pd.to_numeric(data[dependent_var], errors='coerce')

    df_reg = pd.concat([y_raw.rename(dependent_var), X_raw], axis=1).dropna()

    if len(df_reg) < min_rows:
        print(f"Not enough valid data after cleaning ({len(df_reg)} rows); skipping regression.")
        return None, None

    y_clean = df_reg[dependent_var]
    X_clean = df_reg[available_ivs]

    X_sm = sm.add_constant(X_clean, has_constant='add')

    try:
        model = sm.OLS(y_clean, X_sm).fit()
        print(model.summary())

        if plot and len(df_reg) >= 5 and y_clean.nunique() > 1:  # Require more rows for meaningful plot
            try:
                # Use sklearn for plotting, as it's easier to get predictions from test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42
                )
                lin_reg_model = LinearRegression().fit(X_train, y_train)
                y_pred = lin_reg_model.predict(X_test)
                residuals = y_test - y_pred
                plot_regression_results(y_test, y_pred, residuals, title)
            except Exception as e:
                print(f"Warning: Could not create visualization for '{title}': {e}")

        return model, df_reg
    except Exception as e:
        print(f"Error running OLS regression for {title}: {e}")
        return None, None


# --- Helper Function to Create Visualizations Safely ---
def plot_regression_results(y_test, y_pred, residuals, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    if not y_test.empty and y_pred is not None and len(y_test) == len(y_pred):
        try:
            # Ensure min/max calculation handles potential NaN or infinite values safely
            all_vals = np.concatenate([y_test.values, y_pred])
            min_val = np.nanmin(all_vals) if np.any(~np.isnan(all_vals)) else 0
            max_val = np.nanmax(all_vals) if np.any(~np.isnan(all_vals)) else 1

            if not pd.isna(min_val) and not pd.isna(max_val) and min_val != max_val:
                plt.plot([min_val, max_val], [min_val, max_val], '--', linewidth=1, color='red')
        except Exception:
            pass  # Fallback if plotting diagonal line fails
    plt.title(f'Actual vs Predicted ({title})')
    plt.xlabel('Actual');
    plt.ylabel('Predicted')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(y=0, linestyle='--', color='red')
    plt.title(f'Residuals vs Predicted ({title})')
    plt.xlabel('Predicted');
    plt.ylabel('Residuals')

    plt.tight_layout()
    plt.show()
    plt.close()


# --- Main Regression Function ---
def run_general_regression(
        dependent_var='production_cost_total',
        independent_vars=None,
        selected_regions=None,
        selected_school_levels=None,
        data=None
):
    if data is None:
        print("Data must be provided to run the regression.")
        return

    if independent_vars is None:
        independent_vars = [
            'planned_total',
            'served_total',
            'discarded_total',
            'left_over_total',
            'student_count'
        ]

    filtered_data = data.copy()

    # Filter by regions
    if selected_regions:
        print(f"\n--- Filtering for Regions: {selected_regions} ---")
        # Ensure region_name is string for comparison
        filtered_data['region_name'] = filtered_data['region_name'].astype(str)
        # Check if selected regions are numeric strings (e.g., '1', '2')
        if all(isinstance(r, (int, str)) and str(r).isdigit() for r in selected_regions):
            selected_regions_str = [str(r) for r in selected_regions]
            filtered_data = filtered_data[filtered_data['region_name'].isin(selected_regions_str)]
        else:
            # Assume exact string matching for region names
            filtered_data = filtered_data[filtered_data['region_name'].isin(selected_regions)]

        if filtered_data.empty:
            print(f"No data found for the selected regions: {selected_regions}. Exiting.")
            return

    # Filter and aggregate by school levels
    if selected_school_levels and selected_school_levels != ['All']:
        print(f"\n--- Filtering for School Levels: {selected_school_levels} ---")
        school_level_data = filtered_data[filtered_data['school_level'].isin(selected_school_levels)].copy()
        if school_level_data.empty:
            print(f"No data found for the selected school levels: {selected_school_levels}. Exiting.")
            return

        # Aggregate by school level (mean of numeric columns)
        # Ensure only numeric columns are selected for aggregation
        cols_for_agg = [col for col in independent_vars + [dependent_var] if
                        col in school_level_data.columns and pd.api.types.is_numeric_dtype(school_level_data[col])]

        if not cols_for_agg:
            print("No numeric columns available for aggregation after filtering for school levels.")
            return

        aggregated_data = (
            school_level_data.groupby('school_level')[cols_for_agg]
            .mean()
            .reset_index()
        )
        print("\nAggregated data by school level:")
        print(aggregated_data.head())

        # Now run regression on the aggregated data
        run_regression(
            dependent_var=dependent_var,
            independent_vars=[v for v in independent_vars if v in aggregated_data.columns],
            data=aggregated_data,
            title=f"Aggregated Regression: {dependent_var} by {', '.join(selected_school_levels)} School Levels"
        )
    else:
        # Run regression on the filtered (by region) data without school level aggregation
        print(f"\n--- Running Regression on Filtered Data (Regions: {selected_regions}, School Levels: All) ---")
        run_regression(
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            data=filtered_data,
            title=f"General Regression: {dependent_var} (Filtered)"
        )


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    print("--- Loading Data ---")
    try:
        df_breakfast_raw = pd.read_csv(BREAKFAST_CSV, low_memory=False)
        df_lunch_raw = pd.read_csv(LUNCH_CSV, low_memory=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing data file: {e}. Please ensure 'raw_data/' exists and contains the necessary CSVs.")

    df_breakfast_raw.columns = df_breakfast_raw.columns.str.lower().str.strip()
    df_lunch_raw.columns = df_lunch_raw.columns.str.lower().str.strip()

    df_breakfast = clean_numeric_cols(df_breakfast_raw.copy(), numeric_obj_cols)
    df_lunch = clean_numeric_cols(df_lunch_raw.copy(), numeric_obj_cols)

    for df in (df_breakfast, df_lunch):
        if 'school_name' in df.columns:
            df['school_name'] = df['school_name'].astype(str).str.lower().str.strip()
        else:
            raise KeyError("Expected 'school_name' column in breakfast/lunch CSVs.")

    # Load Student Counts (optional)
    df_sizes = None
    try:
        df_sizes = pd.read_csv(STUDENT_COUNTS_CSV)
        df_sizes.columns = df_sizes.columns.str.lower().str.strip()
        # Find a plausible student count column
        student_count_col = None
        for c in ['2024-2025', 'student_count', 'enrollment']:  # Prioritize
            if c in df_sizes.columns:
                student_count_col = c
                break
        if student_count_col is None:  # Fallback to any numeric column if specific ones not found
            candidate_cols = [c for c in df_sizes.columns if
                              c != 'school_name' and pd.api.types.is_numeric_dtype(df_sizes[c])]
            if candidate_cols:
                student_count_col = candidate_cols[0]

        if student_count_col:
            df_sizes = df_sizes[['school_name', student_count_col]].rename(columns={student_count_col: 'student_count'})
            df_sizes['school_name'] = df_sizes['school_name'].astype(str).str.lower().str.strip()
            df_sizes['student_count'] = pd.to_numeric(df_sizes['student_count'], errors='coerce').fillna(0)
        else:
            print("Warning: No suitable student count column found in student_counts.csv.")
            df_sizes = None
    except FileNotFoundError:
        print(f"Warning: {STUDENT_COUNTS_CSV} not found. Continuing without student counts.")
    except Exception as e:
        print(f"Warning: Could not load or process student counts: {e}")

    combined = pd.concat([df_breakfast, df_lunch], ignore_index=True)

    if df_sizes is not None:
        combined = combined.merge(df_sizes, on='school_name', how='left')
    else:
        combined['student_count'] = np.nan

    combined['student_count'] = pd.to_numeric(combined['student_count'], errors='coerce').fillna(0)

    for col in numeric_obj_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined = combined.replace([np.inf, -np.inf], np.nan)

    # --- GEOSPATIAL PROCESSING ---
    print("--- Geospatial Processing ---")
    try:
        gdf_regions = gpd.read_file(REGIONS_GEOJSON)
        print(f"Loaded {len(gdf_regions)} regions from {REGIONS_GEOJSON}")

        REGION_NAME_COLUMN = None
        for col_name in ['name', 'region_name', 'region', 'NAME', 'Name', 'Region_ID']:  # Add common region ID names
            if col_name in gdf_regions.columns:
                REGION_NAME_COLUMN = col_name
                break

        if REGION_NAME_COLUMN is None:
            print("Warning: No obvious region name column found. Using index as region id.")
            gdf_regions = gdf_regions.reset_index().rename(columns={'index': 'region_index'})
            gdf_regions['region_name'] = gdf_regions['region_index'].astype(str)
        else:
            gdf_regions['region_name'] = gdf_regions[REGION_NAME_COLUMN].astype(str)

        if gdf_regions.crs is None:
            print("Warning: regions GeoJSON has no CRS. Assuming EPSG:4326.")
            gdf_regions.set_crs(epsg=4326, inplace=True)
        else:
            try:
                gdf_regions = gdf_regions.to_crs(epsg=4326)
            except Exception as e:
                print(f"Warning reprojecting regions to EPSG:4326: {e}")

        lat_candidates = [c for c in combined.columns if c.lower() in ('latitude', 'lat', 'y')]
        lon_candidates = [c for c in combined.columns if c.lower() in ('longitude', 'lon', 'lng', 'x')]

        if not lat_candidates or not lon_candidates:
            print("Error: Could not find latitude/longitude columns in combined dataset.")
            combined['region_name'] = 'No Region (Missing Coords)'
            # Do not raise, just continue without spatial join
        else:
            lat_col = lat_candidates[0]
            lon_col = lon_candidates[0]

            combined_for_geo = combined.dropna(subset=[lat_col, lon_col]).copy()
            if combined_for_geo.empty:
                print("No valid latitude/longitude pairs found after dropping NaNs. Cannot perform spatial join.")
                combined['region_name'] = 'No Region (No Valid Coords)'
            else:
                combined_for_geo['geometry'] = [Point(xy) for xy in zip(combined_for_geo[lon_col].astype(float),
                                                                        combined_for_geo[lat_col].astype(float))]
                gdf_combined = gpd.GeoDataFrame(combined_for_geo, geometry='geometry', crs="EPSG:4326")

                if gdf_combined.crs != gdf_regions.crs:
                    gdf_combined = gdf_combined.to_crs(gdf_regions.crs)

                try:
                    combined_with_regions = gpd.sjoin(gdf_combined, gdf_regions[['region_name', 'geometry']],
                                                      how="left", predicate='within')
                except TypeError:  # Older geopandas compatibility
                    combined_with_regions = gpd.sjoin(gdf_combined, gdf_regions[['region_name', 'geometry']],
                                                      how="left", op='within')

                unique_school_regions = combined_with_regions[['school_name', 'region_name']].drop_duplicates(
                    subset=['school_name'])
                unique_school_regions = unique_school_regions.rename(columns={'region_name': 'region_name_geo'})

                combined = combined.merge(unique_school_regions, on='school_name', how='left')
                if 'region_name' in combined.columns and 'region_name_geo' in combined.columns:
                    combined['region_name'] = combined['region_name_geo'].combine_first(combined['region_name'])
                    combined.drop(columns=['region_name_geo'], inplace=True)
                elif 'region_name_geo' in combined.columns:
                    combined['region_name'] = combined['region_name_geo']
                    combined.drop(columns=['region_name_geo'], inplace=True)
                else:
                    combined['region_name'] = np.nan

                combined['region_name'] = combined['region_name'].fillna('No Region (No Match)')
                print(
                    f"Assigned regions to {combined[combined['region_name'] != 'No Region (No Match)'].shape[0]} rows.")
                print("Top region counts:\n", combined['region_name'].value_counts().head())

    except FileNotFoundError:
        print(f"Error: Region GeoJSON file not found at {REGIONS_GEOJSON}. Skipping regional assignment.")
        combined['region_name'] = 'No Region (GeoJSON Missing)'
    except Exception as e:
        print(f"An error occurred during geospatial processing: {e}. Continuing without region assignments.")
        if 'region_name' not in combined.columns:
            combined['region_name'] = 'No Region (Processing Error)'

    # Infer school level
    def infer_school_level(name):
        name_lower = str(name).lower()
        if any(k in name_lower for k in ['elementary', 'el.']):
            return 'Elementary'
        elif any(k in name_lower for k in ['middle', 'intermediate', 'junior']):
            return 'Middle'
        elif any(k in name_lower for k in ['high', 'secondary', 'senior']):
            return 'High'
        else:
            return 'Other'

    combined['school_level'] = combined['school_name'].apply(infer_school_level)
    print("\nSchool level distribution:\n", combined['school_level'].value_counts())

    # Filter out placeholder/no-region markers before passing to regression function
    invalid_region_markers = [
        'No Region (Missing Coords)',
        'No Region (GeoJSON Missing)',
        'No Region (Processing Error)',
        'No Region (No Match)',
        'No Region (No Valid Coords)'
    ]
    combined_cleaned_regions = combined[~combined['region_name'].isin(invalid_region_markers)].copy()

    return combined_cleaned_regions


if __name__ == "__main__":
    preprocessed_data = load_and_preprocess_data()

    if preprocessed_data is None or preprocessed_data.empty:
        print("Failed to load or preprocess data. Exiting.")
    else:
        print("\n" + "#" * 80)
        print("### GENERAL REGRESSION TOOL ###")
        print("#" * 80)

        # --- User Inputs ---

        # Region Input
        available_regions = preprocessed_data['region_name'].dropna().unique().tolist()
        print(f"\nAvailable Regions: {sorted(available_regions)}")
        region_input = input(
            "Enter region numbers/names (e.g., '1,2,3' or 'North,South' or 'All' for all regions): ").strip()

        selected_regions = []
        if region_input.lower() == 'all':
            selected_regions = available_regions
        elif region_input:
            # Handle both comma-separated numbers and names
            temp_regions = [r.strip() for r in region_input.split(',')]
            for r in temp_regions:
                # Try to convert to int if it looks like a number
                if r.isdigit() and r in available_regions:  # Check if digit-string exists in available regions
                    selected_regions.append(r)
                elif r in available_regions:  # Check for exact string match
                    selected_regions.append(r)
                else:
                    print(f"Warning: Region '{r}' not found in available regions. Skipping.")

        if not selected_regions:
            print("No valid regions selected. Running regression on all available regions.")
            selected_regions = available_regions

        # School Level Input
        available_levels = ['Elementary', 'Middle', 'High', 'Other']  # Other is usually filtered out later
        print(f"\nAvailable School Levels: {available_levels[:-1]} (Note: 'Other' will be excluded from aggregation)")
        level_input = input(
            "Enter school levels to aggregate (e.g., 'Elementary,Middle' or 'All' for no aggregation by level): ").strip()

        selected_school_levels = []
        if level_input.lower() == 'all':
            # 'All' means don't aggregate by school level, but still filter for valid levels
            selected_school_levels = ['All']
            print("Regression will be run on data for all school levels selected without further aggregation.")
        elif level_input:
            temp_levels = [l.strip().title() for l in level_input.split(',')]  # Capitalize first letter
            for l in temp_levels:
                if l in available_levels:
                    selected_school_levels.append(l)
                else:
                    print(f"Warning: School level '{l}' is not recognized. Skipping.")

        if not selected_school_levels:
            print("No valid school levels selected. Running regression on all valid school levels without aggregation.")
            selected_school_levels = ['All']  # Default to all if nothing selected

        # Independent Variables Input (optional, uses default if empty)
        default_ivs = [
            'planned_total',
            'served_total',
            'discarded_total',
            'left_over_total',
            'student_count'
        ]
        print(f"\nDefault Independent Variables: {default_ivs}")
        iv_input = input("Enter comma-separated independent variables (leave empty for default): ").strip()

        custom_independent_vars = []
        if iv_input:
            custom_independent_vars = [iv.strip() for iv in iv_input.split(',')]
            # Validate if custom IVs exist in the preprocessed_data
            valid_custom_ivs = [iv for iv in custom_independent_vars if iv in preprocessed_data.columns]
            if len(valid_custom_ivs) != len(custom_independent_vars):
                missing = set(custom_independent_vars) - set(valid_custom_ivs)
                print(
                    f"Warning: The following custom independent variables were not found and will be ignored: {', '.join(missing)}")
            if valid_custom_ivs:
                independent_vars_to_use = valid_custom_ivs
            else:
                print("No valid custom independent variables provided. Using default.")
                independent_vars_to_use = default_ivs
        else:
            independent_vars_to_use = default_ivs

        # Dependent Variable Input
        dependent_var_input = input("Enter the dependent variable (e.g., 'production_cost_total'): ").strip()
        if not dependent_var_input:
            print("Dependent variable cannot be empty. Exiting.")
        elif dependent_var_input not in preprocessed_data.columns:
            print(f"Dependent variable '{dependent_var_input}' not found in data. Exiting.")
        else:
            # --- Run the generalized regression ---
            run_general_regression(
                dependent_var=dependent_var_input,
                independent_vars=independent_vars_to_use,
                selected_regions=selected_regions,
                selected_school_levels=selected_school_levels,
                data=preprocessed_data
            )

        print("\n--- Script Finished ---")
