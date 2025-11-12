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

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120

# --- File Paths ---
BREAKFAST_CSV = 'raw_data/data_breakfast_with_coordinates.csv'
LUNCH_CSV = 'raw_data/data_lunch_with_coordinates.csv'
STUDENT_COUNTS_CSV = 'raw_data/student_counts.csv'
REGIONS_GEOJSON = 'raw_data/School_Regions.geojson'

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

print("--- Loading Data ---")
# load CSVs (don't exit the process on missing file - raise)
try:
    df_breakfast_raw = pd.read_csv(BREAKFAST_CSV, low_memory=False)
    df_lunch_raw = pd.read_csv(LUNCH_CSV, low_memory=False)
except FileNotFoundError as e:
    raise

# normalize column names (lower/strip)
df_breakfast_raw.columns = df_breakfast_raw.columns.str.lower().str.strip()
df_lunch_raw.columns = df_lunch_raw.columns.str.lower().str.strip()

# clean numeric columns
df_breakfast = clean_numeric_cols(df_breakfast_raw.copy(), numeric_obj_cols)
df_lunch = clean_numeric_cols(df_lunch_raw.copy(), numeric_obj_cols)

# normalize school name
for df in (df_breakfast, df_lunch):
    if 'school_name' in df.columns:
        df['school_name'] = df['school_name'].astype(str).str.lower().str.strip()
    else:
        raise KeyError("Expected 'school_name' column in breakfast/lunch CSVs.")

# Load Student Counts (optional)
try:
    df_sizes = pd.read_csv(STUDENT_COUNTS_CSV)
    df_sizes.columns = df_sizes.columns.str.lower().str.strip()
    # find a plausible student count column; fallback if exact '2024-2025' missing
    if '2024-2025' in df_sizes.columns:
        df_sizes = df_sizes[['school_name', '2024-2025']].rename(columns={'2024-2025': 'student_count'})
    else:
        # try common alternatives
        candidate = None
        for c in df_sizes.columns:
            if 'student' in c and any(ch.isdigit() for ch in c):
                candidate = c
                break
        if candidate:
            df_sizes = df_sizes[['school_name', candidate]].rename(columns={candidate: 'student_count'})
        else:
            # fallback: try any column that looks numeric besides school_name
            other_cols = [c for c in df_sizes.columns if c != 'school_name']
            if other_cols:
                df_sizes = df_sizes[['school_name', other_cols[0]]].rename(columns={other_cols[0]: 'student_count'})
            else:
                df_sizes = None

    if df_sizes is not None:
        df_sizes['school_name'] = df_sizes['school_name'].astype(str).str.lower().str.strip()
        df_sizes['student_count'] = pd.to_numeric(df_sizes['student_count'], errors='coerce').fillna(0)
except FileNotFoundError:
    print(f"Warning: {STUDENT_COUNTS_CSV} not found. Continuing without student counts.")
    df_sizes = None
except Exception as e:
    print(f"Warning: could not load student counts: {e}")
    df_sizes = None

# Combine breakfast & lunch
combined = pd.concat([df_breakfast, df_lunch], ignore_index=True)

# Merge student counts if available
if df_sizes is not None:
    combined = combined.merge(df_sizes, on='school_name', how='left')
else:
    combined['student_count'] = np.nan

combined['student_count'] = pd.to_numeric(combined['student_count'], errors='coerce').fillna(0)

# Ensure all numeric columns are numeric
for col in numeric_obj_cols:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
combined = combined.replace([np.inf, -np.inf], np.nan)

# --- GEOSPATIAL PROCESSING ---
print("--- Geospatial Processing ---")
try:
    gdf_regions = gpd.read_file(REGIONS_GEOJSON)
    print(f"Loaded {len(gdf_regions)} regions from {REGIONS_GEOJSON}")
    # normalize region name column

    REGION_NAME_COLUMN = 'name'  # default attempt
    if REGION_NAME_COLUMN not in gdf_regions.columns:
        # try common alternatives
        for alt in ['region_name', 'region', 'NAME', 'Name']:
            if alt in gdf_regions.columns:
                REGION_NAME_COLUMN = alt
                break

    if REGION_NAME_COLUMN not in gdf_regions.columns:
        print(f"Warning: No obvious region name column found. Will use index as region id.")
        gdf_regions = gdf_regions.reset_index().rename(columns={'index': 'region_index'})
        gdf_regions['region_name'] = gdf_regions['region_index'].astype(str)
    else:
        gdf_regions['region_name'] = gdf_regions[REGION_NAME_COLUMN].astype(str)

    # Normalize region gdf geometry & CRS: ensure regions are in EPSG:4326 (lon/lat)
    if gdf_regions.crs is None:
        print("Warning: regions GeoJSON has no CRS. Assuming EPSG:4326.")
        gdf_regions.set_crs(epsg=4326, inplace=True)
    else:
        # Reproject regions to EPSG:4326 to match point lon/lat (we create points in EPSG:4326)
        try:
            gdf_regions = gdf_regions.to_crs(epsg=4326)
        except Exception as e:
            print(f"Warning reprojecting regions to EPSG:4326: {e}")

    # identify latitude/longitude columns in combined (support many names)
    lat_candidates = [c for c in combined.columns if c.lower() in ('latitude', 'lat', 'y')]
    lon_candidates = [c for c in combined.columns if c.lower() in ('longitude', 'lon', 'lng', 'x')]

    if not lat_candidates or not lon_candidates:
        print("Error: Could not find latitude/longitude columns in combined dataset. Found candidates:")
        print("lat candidates:", lat_candidates, "lon candidates:", lon_candidates)
        combined['region_name'] = 'No Region (Missing Coords)'
        raise ValueError("Missing lat/lon columns")

    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]

    print(f"Using lat column '{lat_col}' and lon column '{lon_col}' for spatial join.")
    print(f"Lat/Lon nulls before drop: {combined[lat_col].isnull().sum()}, {combined[lon_col].isnull().sum()}")

    combined_for_geo = combined.dropna(subset=[lat_col, lon_col]).copy()
    if combined_for_geo.empty:
        print("No valid latitude/longitude pairs found after dropping NaNs. Cannot perform spatial join.")
        combined['region_name'] = 'No Region (No Valid Coords)'
    else:
        # create geometry in EPSG:4326
        combined_for_geo['geometry'] = [Point(xy) for xy in zip(combined_for_geo[lon_col].astype(float),
                                                                  combined_for_geo[lat_col].astype(float))]
        gdf_combined = gpd.GeoDataFrame(combined_for_geo, geometry='geometry', crs="EPSG:4326")

        # make sure both GeoDataFrames are same CRS
        if gdf_combined.crs != gdf_regions.crs:
            gdf_combined = gdf_combined.to_crs(gdf_regions.crs)

        # perform spatial join - use predicate='within' (modern geopandas)
        try:
            combined_with_regions = gpd.sjoin(gdf_combined, gdf_regions[['region_name', 'geometry']],
                                              how="left", predicate='within')
        except TypeError:
            # older geopandas might still use op= within
            combined_with_regions = gpd.sjoin(gdf_combined, gdf_regions[['region_name', 'geometry']],
                                              how="left", op='within')

        # collect unique mapping of school_name -> region_name from join
        unique_school_regions = combined_with_regions[['school_name', 'region_name']].drop_duplicates(subset=['school_name'])

        # merge back to original combined, but prefer any existing region_name if present
        # rename to 'region_name_geo' to avoid accidental overwrite then coalesce
        unique_school_regions = unique_school_regions.rename(columns={'region_name': 'region_name_geo'})

        combined = combined.merge(unique_school_regions, on='school_name', how='left')
        # create final region_name by preferring geo result, else any existing one, else default
        if 'region_name' in combined.columns and 'region_name_geo' in combined.columns:
            combined['region_name'] = combined['region_name_geo'].combine_first(combined['region_name'])
            combined.drop(columns=['region_name_geo'], inplace=True)
        elif 'region_name_geo' in combined.columns:
            combined['region_name'] = combined['region_name_geo']
            combined.drop(columns=['region_name_geo'], inplace=True)
        else:
            combined['region_name'] = np.nan

        combined['region_name'] = combined['region_name'].fillna('No Region (No Match)')

        print(f"Assigned regions to {combined[combined['region_name'] != 'No Region (No Match)'].shape[0]} rows.")
        print("Top region counts:\n", combined['region_name'].value_counts().head())

except FileNotFoundError:
    print(f"Error: Region GeoJSON file not found at {REGIONS_GEOJSON}. Skipping regional regression.")
    combined['region_name'] = 'No Region (GeoJSON Missing)'
except Exception as e:
    print(f"An error occurred during geospatial processing: {e}. Continuing without region assignments.")
    if 'region_name' not in combined.columns:
        combined['region_name'] = 'No Region (Processing Error)'

# --- Helper Function to Run Regressions Safely ---
def run_regression(dependent_var, independent_vars, data, title, min_rows=2):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if dependent_var not in data.columns:
        print(f"Error: Dependent variable '{dependent_var}' not found in dataset.")
        return None

    missing_ivs = [iv for iv in independent_vars if iv not in data.columns]
    if missing_ivs:
        print(f"Error: Independent variables '{', '.join(missing_ivs)}' not found in dataset.")
        return None

    X_raw = data[independent_vars].copy().apply(pd.to_numeric, errors='coerce')
    y_raw = pd.to_numeric(data[dependent_var], errors='coerce')

    df_reg = pd.concat([y_raw.rename(dependent_var), X_raw], axis=1).dropna()

    if len(df_reg) < min_rows:
        print(f"Not enough valid data after cleaning ({len(df_reg)} rows); skipping regression.")
        return None

    y_clean = df_reg[dependent_var]
    X_clean = df_reg[independent_vars]

    # drop constant columns
    constant_cols = [col for col in X_clean.columns if X_clean[col].nunique() <= 1]
    if constant_cols:
        print(f"Warning: Dropping constant independent variable(s) in {title}: {constant_cols}")
        X_clean = X_clean.drop(columns=constant_cols)
    if X_clean.empty:
        print(f"No valid independent variables left after cleaning; skipping regression for {title}.")
        return None

    X_sm = sm.add_constant(X_clean, has_constant='add')

    try:
        model = sm.OLS(y_clean, X_sm).fit()
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error running OLS regression for {title}: {e}")
        return None

# --- Helper Function to Create Visualizations Safely ---
def plot_regression_results(y_test, y_pred, residuals, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    if not y_test.empty and not y_pred is None:
        try:
            min_val = min(y_test.min(), np.min(y_pred))
            max_val = max(y_test.max(), np.max(y_pred))
            if not pd.isna(min_val) and not pd.isna(max_val) and min_val != max_val:
                plt.plot([min_val, max_val], [min_val, max_val], '--', linewidth=1)
        except Exception:
            pass
    plt.title(f'Actual vs Predicted ({title})')
    plt.xlabel('Actual'); plt.ylabel('Predicted')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(y=0, linestyle='--')
    plt.title(f'Residuals vs Predicted ({title})')
    plt.xlabel('Predicted'); plt.ylabel('Residuals')

    plt.tight_layout()
    plt.show()
    plt.close()

# --- Independent variables ---
production_cost_independent_vars = [
    'planned_total', #proxy for production workload
    'served_total', # meals that are actually served
    'discarded_total', # inefficient waste
    'left_over_total', # overproduced leftovers
    'student_count' # scale of school population
]

# --- Run Global Regression ---
print("\n--- Running Global Regression ---")
global_model = run_regression(
    dependent_var='production_cost_total',
    independent_vars=production_cost_independent_vars,
    data=combined,
    title="Global Regression: Predicting Production Cost",
    min_rows=5  # be slightly stricter for the global model
)

# --- Regional Regression ---
print("\n" + "#" * 60)
print("### REGIONAL REGRESSION: Predicting Production Cost by Region ###")
print("#" * 60)

# Filter out placeholder/no-region markers
invalid_region_markers = [
    'No Region (Missing Coords)',
    'No Region (GeoJSON Missing)',
    'No Region (Processing Error)',
    'No Region (No Match)',
    'No Region (No Valid Coords)'
]
regional_data = combined[~combined['region_name'].isin(invalid_region_markers)].copy()

if regional_data.empty:
    print("No valid regional data available for regression after filtering.")
else:
    print("Regions found and row counts:\n", regional_data['region_name'].value_counts().head(30))
    unique_regions = regional_data['region_name'].unique()
    for region in unique_regions:
        region_df = regional_data[regional_data['region_name'] == region].copy()
        n_rows = len(region_df)
        print(f"\nPerforming regression for Region: {region} (N={n_rows})")

        region_ols_model = run_regression(
            dependent_var='production_cost_total',
            independent_vars=production_cost_independent_vars,
            data=region_df,
            title=f"Regional Regression: Production Cost in {region}",
            min_rows=2  # allow small-region OLS, but will skip if not enough valid rows
        )

        # Visualization: only if model succeeded AND enough cleaned rows for a train/test split
        if region_ols_model is not None:
            Y_reg = pd.to_numeric(region_df['production_cost_total'], errors='coerce')
            X_reg = region_df[production_cost_independent_vars].apply(pd.to_numeric, errors='coerce')
            df_viz_reg = pd.concat([Y_reg.rename('production_cost_total'), X_reg], axis=1).dropna()

            min_viz_rows = 5
            if len(df_viz_reg) >= min_viz_rows and df_viz_reg['production_cost_total'].nunique() > 1:
                try:
                    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                        df_viz_reg.drop('production_cost_total', axis=1),
                        df_viz_reg['production_cost_total'],
                        test_size=0.2,
                        random_state=42
                    )
                    model_reg = LinearRegression().fit(X_train_reg, y_train_reg)
                    y_pred_reg = model_reg.predict(X_test_reg)
                    residuals_reg = y_test_reg - y_pred_reg
                    plot_regression_results(y_test_reg, y_pred_reg, residuals_reg,
                                            f"Production Cost (Region: {region})")
                except Exception as e:
                    print(f"Visualization error for region {region}: {e}")
            else:
                print(f"Not enough data for visualization in region: {region}.")
        else:
            print(f"Skipping visualization for region {region} due to failed OLS regression.")

# --- School Level Classification, Aggregation & Regression ---
print("\n" + "#" * 60)
print("### SCHOOL LEVEL REGRESSION (Aggregated by School Type) ###")
print("#" * 60)

# 1️⃣ Determine school level (or use existing)
if 'school_level' not in combined.columns:
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

print("School level distribution:\n", combined['school_level'].value_counts())

# 2️⃣ Filter to valid levels
school_level_data = combined[combined['school_level'].isin(['Elementary', 'Middle', 'High'])].copy()
if school_level_data.empty:
    print("No valid school-level data available for regression.")
else:
    # 3️⃣ Aggregate by school level (mean of numeric columns)
    numeric_cols_for_agg = production_cost_independent_vars + ['production_cost_total']
    aggregated = (
        school_level_data.groupby('school_level')[numeric_cols_for_agg]
        .mean()
        .reset_index()
    )

    print("\nAggregated data by school level:")
    print(aggregated.head())

    # 4️⃣ Run regression per school level group
    for level in ['Elementary', 'Middle', 'High']:
        level_df = aggregated[aggregated['school_level'] == level]
        n_rows = len(level_df)
        print(f"\nPerforming regression for {level} Schools (Aggregated) (N={n_rows})")

        # You can still run regression across all levels combined — but if you want per-level,
        # we’ll use the mean-aggregated values as input (one row per level).
        # Note: For regression, we need >2 rows; with 3 levels total, this is usually too small.
        # So instead, let's run ONE regression using all levels together:
        # (skip individual regressions since each level = one row)

    print("\nRunning combined regression across aggregated school levels (Elementary/Middle/High)...")
    level_model = run_regression(
        dependent_var='production_cost_total',
        independent_vars=[v for v in production_cost_independent_vars if v in aggregated.columns],
        data=aggregated,
        title="Aggregated School Level Regression (All Levels)",
        min_rows=3
    )

    # 5️⃣ Visualization (optional)
    if level_model is not None:
        try:
            y_test_lvl = aggregated['production_cost_total']
            X_test_lvl = aggregated[[v for v in production_cost_independent_vars if v in aggregated.columns]]
            model_lin = LinearRegression().fit(X_test_lvl, y_test_lvl)
            y_pred_lvl = model_lin.predict(X_test_lvl)
            residuals_lvl = y_test_lvl - y_pred_lvl
            plot_regression_results(y_test_lvl, y_pred_lvl, residuals_lvl,
                                    "Aggregated School-Level Production Cost")
        except Exception as e:
            print(f"Visualization error for aggregated school levels: {e}")

print("\n--- Script Finished ---")
