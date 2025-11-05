import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import folium
from streamlit_folium import st_folium
import numpy as np
import sys
import json
import traceback

# --- Page Configuration ---
st.set_page_config(
    page_title="FCPS Food & Waste Analysis",
    page_icon="⭐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #07677F;
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
    background-color: #00AEC8 !important; /* Force the background color */
    color: white !important; /* Force the text color */
    border-radius: 0.25rem; /* Optional */
    border: none; /* Remove potential default border */
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child div {
    color: white !important;
}

[data-testid="stSelectbox"] label {
    color: white !important;
}

[data-testid="stSelectbox"] svg {
    fill: white !important; /* Force arrow color */
}

</style>
""", unsafe_allow_html=True)

# Setting path directories
src_path = str(Path(__file__).resolve().parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# importing functions from other files
try:
    from component.optimization import (
        prepare_optimization_data,
        prepare_savings_analysis_df,
        analyze_savings_by_school_size
    )

    from component.regression_analysis import perform_regression_analysis

    from component.pdf_generator import generate_pdf

except ImportError as e:
    st.error(f"Could not import functions from optimization.py: {e}.")
    # Define dummy functions
    def prepare_optimization_data(*args, **kwargs): return None
    def prepare_savings_analysis_df(*args, **kwargs): return None
    def prepare_map_data_from_coordinates(*args, **kwargs): return None
    def analyze_savings_by_school_size(*args, **kwargs): return None
    def perform_regression_analysis(*args, **kwargs): return None
    def generate_pdf(*args, **kwargs): return None

# --- Define the school we are focusing on ---
school = "aldrin elementary" 

st.title(f"⭐️ FCPS Waste & Food Management Analysis: {school.title()}")
st.markdown(f"Data analysis of food production, consumption, and waste for {school.title()}")

# --- Helper function kept locally ---
def map_educational_level(df):
    """Adds 'educational level' based on lowercase 'level'."""
    if 'level' in df.columns:
        level_map = {'es': 'Elementary', 'ms': 'Middle', 'hs': 'High', 'ss': 'High'}
        df['educational level'] = df['level'].str.lower().map(level_map).fillna('Other')
    return df

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads all data, CLEANS it, passes DataFrames to optimization function, and loads other files."""
    opt_data_dict = None
    school_opt_items_df = None
    popularity_files = {}
    bf_data_map_coords = None
    ln_data_map_coords = None
    nutrition_df = None
    school_list_df = None

    try:
        # --- Corrected Path Logic ---
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent 
        data_dir = project_root / 'src' / 'data'
        preprocessed_data_dir = data_dir / 'preprocessed-data'
        
        # --- Define Paths ---
        breakfast_path = preprocessed_data_dir / "breakfast_combined.csv"
        lunch_path = preprocessed_data_dir / "lunch_combined.csv"
        student_counts_path = preprocessed_data_dir / "2022-2025 Fairfax County School Student Count.csv"

        # --- 1. Load Raw Data ---
        df_breakfast_raw = pd.read_csv(breakfast_path)
        df_lunch_raw = pd.read_csv(lunch_path)
        df_student_counts_raw = pd.read_csv(student_counts_path)

        # --- 2. Clean Breakfast & Lunch Data ---
        new_columns = [
            "School_Name", "Date", "Identifier", "Name", "Planned", "Planned_Nonreimbursable",
            "Planned_Total", "Offered", "Served", "Served_Nonreimbursable", "Served_Total",
            "Served_Cost", "Discarded", "Discarded_POO", "Discarded_Cost", "Subtotal",
            "Leftover", "Leftover_POO", "Leftover_Cost", "Production_Cost"
        ]
        
        # --- Clean Breakfast ---
        if df_breakfast_raw.shape[1] > 20:
            dfb_cleaned = df_breakfast_raw.iloc[:, :20].copy()
            dfb_cleaned.columns = new_columns
        elif df_breakfast_raw.shape[1] == 20:
            dfb_cleaned = df_breakfast_raw.copy()
            dfb_cleaned.columns = new_columns
        else:
            st.error(f"Breakfast file has {df_breakfast_raw.shape[1]} columns, expected 20. Stopping.")
            return None, None, {}, None, None, None
            
        # --- Clean Lunch ---
        if df_lunch_raw.shape[1] > 20:
            dfl_cleaned = df_lunch_raw.iloc[:, :20].copy()
            dfl_cleaned.columns = new_columns
        elif df_lunch_raw.shape[1] == 20:
            dfl_cleaned = df_lunch_raw.copy()
            dfl_cleaned.columns = new_columns
        else:
            st.error(f"Lunch file has {df_lunch_raw.shape[1]} columns, expected 20. Stopping.")
            return None, None, {}, None, None, None

        # --- Standardize and Clean Numeric Data ---
        dfb_cleaned.columns = dfb_cleaned.columns.str.lower()
        dfl_cleaned.columns = dfl_cleaned.columns.str.lower()

        dfb_cleaned['school_name'] = dfb_cleaned['school_name'].str.lower()
        dfl_cleaned['school_name'] = dfl_cleaned['school_name'].str.lower()

        # NOTE: Renamed columns to match what optimization.py expects
        
        # 1. Rename 'production_cost' (from new_columns) to 'production_cost_total'
        dfb_cleaned.rename(columns={"production_cost": "production_cost_total"}, inplace=True)
        dfl_cleaned.rename(columns={"production_cost": "production_cost_total"}, inplace=True)
        
        # 2. Rename 'served' (from new_columns) to 'served_reimbursable'
        #    This was the main bug.
        dfb_cleaned.rename(columns={"served": "served_reimbursable"}, inplace=True)
        dfl_cleaned.rename(columns={"served": "served_reimbursable"}, inplace=True)

        # 3. NOW, clean the numeric columns using their final, correct names
        num_cols = ["served_reimbursable", "production_cost_total", "leftover_cost", "discarded_cost"]
        for col in num_cols:
            if col in dfb_cleaned.columns:
                dfb_cleaned[col] = pd.to_numeric(dfb_cleaned[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            if col in dfl_cleaned.columns:
                dfl_cleaned[col] = pd.to_numeric(dfl_cleaned[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
        
        # --- 4. Clean Student Data ---
        df_sizes_cleaned = df_student_counts_raw[['School_Name', '2024-2025']].dropna().copy()
        df_sizes_cleaned.columns = ['school_name', 'count']
        df_sizes_cleaned['school_name'] = df_sizes_cleaned['school_name'].str.lower().str.strip()
        df_sizes_cleaned['count'] = df_sizes_cleaned['count'].astype(int)
        bins = [-float('inf'), 499, 999, 1499, 1999, 2499, 2999, 3499, float('inf')]
        labels = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
        df_sizes_cleaned['size_category'] = pd.cut(df_sizes_cleaned['count'], bins=bins, labels=labels, right=True)

        # --- 5. Pass the *CLEANED* DataFrames to the function ---
        opt_data_dict = prepare_optimization_data(
            dfb_cleaned, 
            dfl_cleaned, 
            df_sizes_cleaned # Pass the cleaned df_sizes
        )

        if opt_data_dict is None:
             st.error("Failed to prepare optimization data from optimization.py.")
             return None, None, {}, None, None, None

        # --- 6. Load other needed files (this part remains the same) ---
        school_opt_items_df = pd.read_csv(data_dir / "school_food_item_optimization.csv")
        nutrition_df = pd.read_csv(preprocessed_data_dir / "fcps_nutrition_values.csv")
        if nutrition_df is not None:
            nutrition_df.columns = nutrition_df.columns.str.lower()
            
        bf_coords_file = preprocessed_data_dir / "data_breakfast_with_coordinates.csv"
        ln_coords_file = preprocessed_data_dir / "data_lunch_with_coordinates.csv"

        school_list_df = school_opt_items_df.copy()

        # Popularity Files
        popularity_files['bf_leftover'] = pd.read_csv(preprocessed_data_dir / "breakfast_leftover_rate.csv")
        popularity_files['ln_leftover'] = pd.read_csv(preprocessed_data_dir / "lunch_leftover_rate.csv")
        popularity_files['bf_consumption'] = pd.read_csv(preprocessed_data_dir / "breakfast_net_consumption.csv")
        popularity_files['ln_consumption'] = pd.read_csv(preprocessed_data_dir / "lunch_net_consumption.csv")
        popularity_files['bf_lr_school'] = pd.read_csv(preprocessed_data_dir / "breakfast_leftover_rate_by_school.csv")
        popularity_files['ln_lr_school'] = pd.read_csv(preprocessed_data_dir / "lunch_leftover_rate_by_school.csv")
        popularity_files['bf_nc_school'] = pd.read_csv(preprocessed_data_dir / "breakfast_net_consumption_by_school.csv")
        popularity_files['ln_nc_school'] = pd.read_csv(preprocessed_data_dir / "lunch_net_consumption_by_school.csv")

        # Clean column names for popularity files
        for name, df in popularity_files.items():
            if df is not None:
                df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_', regex=False)
                df.rename(columns={'leftover_rate_(%)': 'leftover_rate', 'item_name': 'name'}, inplace=True, errors='ignore')

        # Coordinate data
        bf_data_map_coords = pd.read_csv(bf_coords_file)
        if bf_data_map_coords is not None: bf_data_map_coords.columns = bf_data_map_coords.columns.str.lower()
        ln_data_map_coords = pd.read_csv(ln_coords_file)
        if ln_data_map_coords is not None: ln_data_map_coords.columns = ln_data_map_coords.columns.str.lower()

        # --- Perform merges needed for school_opt_items_df ---
        if school_opt_items_df is not None and bf_data_map_coords is not None:
             # Create metadata from coordinate data, Select only necessary columns and drop duplicates based on 'school_name'
             metadata_cols = ['school_name', 'fcps region', 'distribution kitchen (dk)', 'level']
             if all(col in bf_data_map_coords.columns for col in metadata_cols):
                 school_metadata = bf_data_map_coords[metadata_cols].drop_duplicates(subset=['school_name']).copy()
                 # Add educational level
                 school_metadata = map_educational_level(school_metadata)
                 # Ensure the merge key ('school_name') is clean
                 school_metadata['school_name'] = school_metadata['school_name'].astype(str).str.lower().str.strip()

                 # Prepare school_opt_items_df for merge, ensure its 'school' column is lowercase and stripped
                 school_opt_items_df['school'] = school_opt_items_df['school'].astype(str).str.lower().str.strip()

                 # Perform the merge: 'school' from opt_items on 'school_name' from metadata
                 school_opt_items_df = pd.merge(
                     school_opt_items_df,
                     school_metadata,
                     left_on='school',
                     right_on='school_name',
                     how='left'
                 )

             else:
                 st.warning("Coordinate data ('bf_data_map_coords') is missing required columns for metadata merge.")
                 placeholder_cols = ['fcps region', 'distribution kitchen (dk)', 'level', 'educational level', 'school_name']
                 for col in placeholder_cols:
                      if col not in school_opt_items_df.columns: school_opt_items_df[col] = pd.NA


             # Merge Nutrition Data
             if nutrition_df is not None and 'food_name' in nutrition_df.columns and 'sub-category' in nutrition_df.columns:
                 school_opt_items_df = pd.merge(school_opt_items_df, nutrition_df[['food_name', 'sub-category']], left_on='food_item', right_on='food_name', how='left')
                 school_opt_items_df['sub-category'] = school_opt_items_df['sub-category'].fillna('Other')
             else:
                 st.warning("Nutrition data not available or missing columns for merging sub-categories.")
                 if 'sub-category' not in school_opt_items_df.columns: school_opt_items_df['sub-category'] = 'Unknown'

             school_opt_items_df['recommended_quantity'] = pd.to_numeric(school_opt_items_df['recommended_quantity'], errors='coerce').fillna(0)
        else:
              st.warning("Could not perform merges for optimization tab as required dataframes were not loaded correctly.")


        return opt_data_dict, school_opt_items_df, popularity_files, bf_data_map_coords, ln_data_map_coords, school_list_df

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}.")
        return None, None, {}, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, {}, None, None, None

@st.cache_data
def load_regression_results(df_breakfast, df_lunch, df_student_counts):
    """Loads and runs all regression models"""
    return perform_regression_analysis(df_breakfast, df_lunch, df_student_counts)

# --- Load all data ---
opt_data_loaded, school_opt_data, popularity_files, bf_coord_data, ln_coord_data, school_list_df = load_data()

# --- Only run regression if data is loaded ---
regression_results = {} # Initialize
if opt_data_loaded:
    regression_results = load_regression_results(
        opt_data_loaded.get('dfb'), 
        opt_data_loaded.get('dfl'),
        opt_data_loaded.get('df_sizes')
    )
else:
    regression_results = {"error": "Primary data failed to load, regression not run."}

# Local Mapping Functions
def generate_fcps_region_choropleth(regional_map_df, geojson_path, columns, initial_column):
    """Generates choropleth map."""
    regional_map_df.rename(columns={'fcps region': 'FCPS Region'}, inplace=True, errors='ignore')

    try:
        with open(geojson_path) as f: gj = json.load(f)
        for feature in gj['features']:
            if 'properties' in feature and 'REGION' in feature['properties']:
                feature['properties']['REGION_KEY'] = str(feature['properties']['REGION'])
    except FileNotFoundError:
        st.error(f"GeoJSON file not found at {geojson_path}"); return None
    except Exception as e:
        st.error(f"Error processing GeoJSON: {e}"); return None

    m = folium.Map(location=[38.8, -77.3], zoom_start=10, tiles='CartoDB positron')
    try:
        # Check if region_key exists before creating Choropleth
        if 'region_key' not in regional_map_df.columns:
            st.error("'region_key' column missing in regional_savings data for Choropleth.")
            return None
        if initial_column not in regional_map_df.columns:
            st.error(f"Initial column '{initial_column}' missing for Choropleth.")
            return None

        choropleth = folium.Choropleth(
            geo_data=gj, data=regional_map_df,
            columns=['region_key', initial_column], # Use region_key and the value column
            key_on='feature.properties.REGION_KEY', # Match key
            fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
            legend_name=initial_column.replace('_',' ').title(), name=initial_column
        ).add_to(m)
        folium.GeoJsonTooltip(['REGION']).add_to(choropleth.geojson)
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Error creating Choropleth: {e}")
        st.write("Data passed to Choropleth:")
        st.dataframe(regional_map_df.head()) # Show data passed to choropleth
        return None

def prepare_eda_map_data(breakfast_data, lunch_data):
    """Prepares combined potential savings data from leftover cost (uses lowercase)."""
    req_cols = ['school_name', 'latitude', 'longitude', 'leftover_cost']
    if breakfast_data is None or lunch_data is None or \
       not all(col in breakfast_data.columns for col in req_cols) or \
       not all(col in lunch_data.columns for col in req_cols):
        st.error("Missing data or columns for EDA potential savings map.")
        return None

    bf_savings = breakfast_data.groupby(['school_name', 'latitude', 'longitude'])['leftover_cost'].sum().reset_index()
    bf_savings.rename(columns={'leftover_cost': 'Breakfast Savings'}, inplace=True)
    ln_savings = lunch_data.groupby(['school_name', 'latitude', 'longitude'])['leftover_cost'].sum().reset_index()
    ln_savings.rename(columns={'leftover_cost': 'Lunch Savings'}, inplace=True)

    combined = pd.merge(bf_savings, ln_savings, on=['school_name', 'latitude', 'longitude'], how='outer').fillna(0)
    combined['Total Savings'] = combined['Breakfast Savings'] + combined['Lunch Savings']
    combined['School_Name_Display'] = combined['school_name'].str.title()
    return combined

# Check if essential data loaded before proceeding
if opt_data_loaded and school_opt_data is not None and popularity_files and bf_coord_data is not None and ln_coord_data is not None:

    # Extract dfb, dfl for filtering popularity data
    dfb = opt_data_loaded.get('dfb')
    dfl = opt_data_loaded.get('dfl')
    df_sizes = opt_data_loaded.get('df_sizes')

    # --- Create metadata and merge into raw dfb and dfl ---
    school_metadata_df = pd.DataFrame()
    if bf_coord_data is not None:
        metadata_cols = ['school_name', 'fcps region', 'distribution kitchen (dk)', 'level']
        if all(col in bf_coord_data.columns for col in metadata_cols):
            school_metadata_df = bf_coord_data[metadata_cols].drop_duplicates(subset=['school_name']).copy()
            school_metadata_df['school_name'] = school_metadata_df['school_name'].astype(str).str.lower().str.strip()
        else:
            st.warning("Coordinate data is missing required metadata columns (e.g., 'fcps region'). Filters may not work.")

    if dfb is not None and not school_metadata_df.empty:
        # Columns are already lowercase and clean from load_data
        dfb = pd.merge(dfb, school_metadata_df, on='school_name', how='left')
    
    if dfl is not None and not school_metadata_df.empty:
        # Columns are already lowercase and clean from load_data
        dfl = pd.merge(dfl, school_metadata_df, on='school_name', how='left')

    
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    selected_meal_period = st.sidebar.selectbox("Select a Meal Period", ["Overall", "Breakfast", "Lunch"])

    # --- ALL OTHER FILTERS REMOVED ---

    # --- Data Filtering for Popularity/EDA Raw Data ---
    # Filter the dataframes for *only* the hard-coded school
    school_name_clean = school.lower().strip()
    
    bf_filtered_raw = pd.DataFrame()
    if dfb is not None:
        bf_filtered_raw = dfb[dfb['school_name'] == school_name_clean].copy()
        bf_filtered_raw = map_educational_level(bf_filtered_raw)

    ln_filtered_raw = pd.DataFrame()
    if dfl is not None:
        ln_filtered_raw = dfl[dfl['school_name'] == school_name_clean].copy()
        ln_filtered_raw = map_educational_level(ln_filtered_raw)
    
    # --- FILTERS (region, dk, level, school) REMOVED ---

    # Clean remaining numeric cols
    # Note: leftover_cost and discarded_cost are already clean from load_data
    # 'production_cost_total' is also clean
    num_cols_raw = ['leftover_total', 'offered_total', 'served_total', 'discarded_total', 'planned_total'] 
    for col in num_cols_raw:
        if col in bf_filtered_raw.columns:
            bf_filtered_raw[col] = pd.to_numeric(bf_filtered_raw[col], errors='coerce').fillna(0)
        if col in ln_filtered_raw.columns:
            ln_filtered_raw[col] = pd.to_numeric(ln_filtered_raw[col], errors='coerce').fillna(0)
    # Ensure date column is datetime
    if 'date' in bf_filtered_raw.columns:
        bf_filtered_raw['date'] = pd.to_datetime(bf_filtered_raw['date'], errors='coerce')
    if 'date' in ln_filtered_raw.columns:
        ln_filtered_raw['date'] = pd.to_datetime(ln_filtered_raw['date'], errors='coerce')


    # Main Dashboard Tabs
    tab_eda, tab_pop, tab_opt, tab_reg, tab_sav, tab_pdf = st.tabs([
        "📈 Exploratory Data Analysis",
        "⭐ Popularity",
        "⚙️ Optimization",
        "📊 Regression",
        "💰 Savings/Loss",
        "🧾 Generate Recommendation"
    ])

    # EDA Tab
    with tab_eda:
        
        # --- Map and Student Count Logic ---
        # This logic is now simplified to always show the hard-coded school
        student_count_val = None # Initialize
        school_info = None
        school_count_info = None
        
        # Get School Coordinates from bf_coord_data
        if bf_coord_data is not None and 'school_name' in bf_coord_data.columns:
            school_info = bf_coord_data[
                bf_coord_data['school_name'].str.lower() == school.lower()
            ].iloc[0] if not bf_coord_data[bf_coord_data['school_name'].str.lower() == school.lower()].empty else None

        # Get Student Count from df_sizes
        if df_sizes is not None and 'school_name' in df_sizes.columns:
            school_count_info = df_sizes[
                df_sizes['school_name'].str.lower() == school.lower()
            ].iloc[0] if not df_sizes[df_sizes['school_name'].str.lower() == school.lower()].empty else None
        
        # Set student count for the metric
        student_count_val = school_count_info['count'] if school_count_info is not None and 'count' in school_count_info else "N/A"

        # Display the Map
        if school_info is not None and 'latitude' in school_info and 'longitude' in school_info:
            st.subheader(f"Location of: {school.title()}")
            school_lat = school_info['latitude']
            school_lon = school_info['longitude']
            
            if pd.notna(school_lat) and pd.notna(school_lon):
                m = folium.Map(location=[school_lat, school_lon], zoom_start=15, tiles="cartodbpositron")
                folium.Marker(
                    [school_lat, school_lon],
                    popup=f"{school.title()}",
                    tooltip=f"{school.title()}"
                ).add_to(m)
                st_folium(m, use_container_width=True, height=300)
            else:
                st.warning(f"Map cannot be displayed: Missing coordinate data for {school.title()}.")
        else:
            st.warning(f"Map cannot be displayed: Location data not found for {school.title()}.")

        st.header(f"High-Level Overview for {school.title()}")
        st.markdown("Summary of production costs and waste based on the selected meal period.")

        # --- DYNAMIC METRIC CALCULATION ---
        prod_cost, lo_cost, disc_cost = 0, 0, 0
        
        if selected_meal_period == "Breakfast":
            prod_cost = bf_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in bf_filtered_raw.columns else 0
            lo_cost = bf_filtered_raw['leftover_cost'].sum()
            disc_cost = bf_filtered_raw['discarded_cost'].sum()
        
        elif selected_meal_period == "Lunch":
            prod_cost = ln_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in ln_filtered_raw.columns else 0
            lo_cost = ln_filtered_raw['leftover_cost'].sum()
            disc_cost = ln_filtered_raw['discarded_cost'].sum()
        
        else: # "Overall"
            bf_prod_cost = bf_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in bf_filtered_raw.columns else 0
            ln_prod_cost = ln_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in ln_filtered_raw.columns else 0
            bf_lo_cost = bf_filtered_raw['leftover_cost'].sum()
            ln_lo_cost = ln_filtered_raw['leftover_cost'].sum()
            bf_disc_cost = bf_filtered_raw['discarded_cost'].sum()
            ln_disc_cost = ln_filtered_raw['discarded_cost'].sum()

            prod_cost = bf_prod_cost + ln_prod_cost
            lo_cost = bf_lo_cost + ln_lo_cost
            disc_cost = bf_disc_cost + ln_disc_cost
        
        waste_perc = ((lo_cost + disc_cost) / prod_cost * 100) if prod_cost > 0 else 0

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(f"{selected_meal_period} Production Cost", f"${prod_cost:,.2f}")
        col2.metric(f"{selected_meal_period} Leftover Cost", f"${lo_cost:,.2f}")
        col3.metric(f"{selected_meal_period} Discarded Cost", f"${disc_cost:,.2f}")
        col4.metric(f"{selected_meal_period} Waste Percentage", f"{waste_perc:.2f}%")
        col5.metric("Student Population", student_count_val)


        st.markdown("---")

        # --- DYNAMIC CHART ---
        st.subheader("Cost Over Time")
        cost_cols = ['date', 'production_cost_total', 'leftover_cost', 'discarded_cost']
        cost_agg = pd.DataFrame()

        if selected_meal_period == "Breakfast":
            if all(col in bf_filtered_raw.columns for col in cost_cols) and not bf_filtered_raw.empty:
                cost_agg = bf_filtered_raw.groupby('date')[cost_cols[1:]].sum().reset_index()
        
        elif selected_meal_period == "Lunch":
             if all(col in ln_filtered_raw.columns for col in cost_cols) and not ln_filtered_raw.empty:
                cost_agg = ln_filtered_raw.groupby('date')[cost_cols[1:]].sum().reset_index()
        
        else: # "Overall"
            bf_valid = all(col in bf_filtered_raw.columns for col in cost_cols) and not bf_filtered_raw.empty
            ln_valid = all(col in ln_filtered_raw.columns for col in cost_cols) and not ln_filtered_raw.empty

            if bf_valid or ln_valid:
                dfs_to_concat = []
                if bf_valid:
                    dfs_to_concat.append(bf_filtered_raw[cost_cols])
                if ln_valid:
                    dfs_to_concat.append(ln_filtered_raw[cost_cols])
                
                cost_over_time = pd.concat(dfs_to_concat)
                cost_over_time['date'] = pd.to_datetime(cost_over_time['date'])
                cost_agg = cost_over_time.groupby('date').sum().reset_index()

        if not cost_agg.empty:
            fig_cost_time = px.line(cost_agg, x='date',
                                    y=['production_cost_total', 'leftover_cost', 'discarded_cost'],
                                    title=f'Daily Production and Waste Costs ({selected_meal_period})',
                                    labels={'value': 'Cost (USD)', 'variable': 'Cost Type', 'date':'Date'})
            st.plotly_chart(fig_cost_time, use_container_width=True)
        else:
            st.warning(f"No cost data available for the selected filters to plot a time-series chart.")

        # --- RAW DATA DISPLAY ---
        st.markdown("---")
        st.subheader(f"Raw Filtered Data for {school.title()}")
        
        if selected_meal_period == "Breakfast":
            st.dataframe(bf_filtered_raw)
        elif selected_meal_period == "Lunch":
            st.dataframe(ln_filtered_raw)
        else: # Overall
            st.markdown("#### Breakfast")
            st.dataframe(bf_filtered_raw)
            st.markdown("#### Lunch")
            st.dataframe(ln_filtered_raw)

    # Popularity Tab
    with tab_pop:
        st.header(f"Food Item Popularity Analysis for {school.title()}")

        # Get cleaned dataframes from dictionary
        bf_lr_school_df = popularity_files.get('bf_lr_school')
        ln_lr_school_df = popularity_files.get('ln_lr_school')
        bf_nc_school_df = popularity_files.get('bf_nc_school')
        ln_nc_school_df = popularity_files.get('ln_nc_school')

        # --- BREAKFAST SECTION ---
        if selected_meal_period == "Breakfast" or selected_meal_period == "Overall":
            st.subheader(f"Breakfast Insights for: {school.title()}")
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.subheader("Items by Leftover Rate")
                if bf_lr_school_df is not None and 'school_name' in bf_lr_school_df.columns:
                    bf_lr_school_data_filtered = bf_lr_school_df[bf_lr_school_df['school_name'].str.lower() == school.lower()]
                    bf_lr_chart_data = bf_lr_school_data_filtered.sort_values(by='leftover_rate', ascending=True)
                    if not bf_lr_chart_data.empty:
                        chart_h = max(400, len(bf_lr_chart_data) * 20)
                        fig = px.bar(bf_lr_chart_data, x='leftover_rate', y='name', orientation='h', title='Leftover Rates', labels={'name':'Item', 'leftover_rate':'Rate (%)'}, height=chart_h, color='leftover_rate', color_continuous_scale=px.colors.sequential.Reds)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No pre-computed Breakfast Leftover Rate data found for {school.title()}.")
                    with st.expander("View Data"): st.dataframe(bf_lr_chart_data.sort_values(by='leftover_rate', ascending=False))
                else:
                    st.warning("Breakfast school leftover rate data not loaded or 'school_name' column missing.")

            with b_col2:
                st.subheader("Items by Net Consumption Rate")
                if bf_nc_school_df is not None and 'school_name' in bf_nc_school_df.columns:
                    bf_nc_school_data_filtered = bf_nc_school_df[bf_nc_school_df['school_name'].str.lower() == school.lower()]
                    bf_nc_chart_data = bf_nc_school_data_filtered.sort_values(by='net_consumption_rate', ascending=True)
                    if not bf_nc_chart_data.empty:
                        chart_h = max(400, len(bf_nc_chart_data) * 20)
                        fig = px.bar(bf_nc_chart_data, x='net_consumption_rate', y='name', orientation='h', title='Net Consumption Rate', labels={'name':'Item', 'net_consumption_rate':'Rate (%)'}, height=chart_h, color='net_consumption_rate', color_continuous_scale=px.colors.sequential.Greens)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No pre-computed Breakfast Net Consumption Rate data found for {school.title()}.")
                    with st.expander("View Data"): st.dataframe(bf_nc_chart_data.sort_values(by='net_consumption_rate', ascending=False))
                else:
                    st.warning("Breakfast school net consumption rate data not loaded or 'school_name' column missing.")
            st.markdown("---")


        # --- LUNCH SECTION ---
        if selected_meal_period == "Lunch" or selected_meal_period == "Overall":
            st.subheader(f"Lunch Insights for: {school.title()}")
            
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.subheader("Items by Leftover Rate")
                if ln_lr_school_df is not None and 'school_name' in ln_lr_school_df.columns:
                    ln_lr_school_data_filtered = ln_lr_school_df[ln_lr_school_df['school_name'].str.lower() == school.lower()]
                    ln_lr_chart_data = ln_lr_school_data_filtered.sort_values(by='leftover_rate', ascending=True)
                    if not ln_lr_chart_data.empty:
                        chart_h = max(400, len(ln_lr_chart_data) * 20)
                        fig = px.bar(ln_lr_chart_data, x='leftover_rate', y='name', orientation='h', title='Leftover Rates', labels={'name':'Item', 'leftover_rate':'Rate (%)'}, height=chart_h, color='leftover_rate', color_continuous_scale=px.colors.sequential.Reds)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No pre-computed Lunch Leftover Rate data found for {school.title()}.")
                    with st.expander("View Data"): st.dataframe(ln_lr_chart_data.sort_values(by='leftover_rate', ascending=False))
                else:
                        st.warning("Lunch school leftover rate data not loaded or 'school_name' column missing.")

            with l_col2:
                st.subheader("Items by Net Consumption Rate")
                if ln_nc_school_df is not None and 'school_name' in ln_nc_school_df.columns:
                    ln_nc_school_data_filtered = ln_nc_school_df[ln_nc_school_df['school_name'].str.lower() == school.lower()]
                    ln_nc_chart_data = ln_nc_school_data_filtered.sort_values(by='net_consumption_rate', ascending=True)
                    if not ln_nc_chart_data.empty:
                        chart_h = max(400, len(ln_nc_chart_data) * 20)
                        fig = px.bar(ln_nc_chart_data, x='net_consumption_rate', y='name', orientation='h', title='Net Consumption Rate', labels={'name':'Item', 'net_consumption_rate':'Rate (%)'}, height=chart_h, color='net_consumption_rate', color_continuous_scale=px.colors.sequential.Greens)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No pre-computed Lunch Net Consumption Rate data found for {school.title()}.")
                    with st.expander("View Data"): st.dataframe(ln_nc_chart_data.sort_values(by='net_consumption_rate', ascending=False))
                else:
                        st.warning("Lunch school net consumption rate data not loaded or 'school_name' column missing.")

        if selected_meal_period not in ["Breakfast", "Lunch", "Overall"]:
            st.info("Select a meal period (Breakfast, Lunch, or Overall) from the sidebar to see popularity data.")

    # Optimization Tab
    with tab_opt:
        st.header("Optimization Recommendations")
        opt_display_data = school_opt_data.copy() if school_opt_data is not None else pd.DataFrame()

        if not opt_display_data.empty:
            # Filter by Meal Period first
            if selected_meal_period == "Breakfast":
                if 'meal_type' in opt_display_data.columns:
                    opt_display_data = opt_display_data[opt_display_data['meal_type'] == 'Breakfast']
                else: st.warning("Cannot filter by meal type - 'meal_type' column missing.")
            elif selected_meal_period == "Lunch":
                if 'meal_type' in opt_display_data.columns:
                    opt_display_data = opt_display_data[opt_display_data['meal_type'] == 'Lunch']
                else: st.warning("Cannot filter by meal type - 'meal_type' column missing.")

            # --- Logic now *only* shows the specific hard-coded school ---
            st.subheader(f"Recommendations for: {school.title()}")
            if 'school_name' in opt_display_data.columns:
                selected_school_clean = school.lower().strip()
                school_opt_filtered = opt_display_data[
                    opt_display_data['school_name'].astype(str).str.lower().str.strip() == selected_school_clean
                ].copy()
            else:
                st.warning("Cannot filter optimization data by school - 'school_name' column missing (merge may have failed).")
                school_opt_filtered = pd.DataFrame()

            # Display data if filtering was successful
            if not school_opt_filtered.empty:
                total_items = int(school_opt_filtered['recommended_quantity'].sum())
                st.metric("Total Recommended Monthly Items", f"{total_items:,}")
                st.subheader("Breakdown by Sub-Category")
                if 'sub-category' in school_opt_filtered.columns:
                    subcat_totals = school_opt_filtered.groupby('sub-category')['recommended_quantity'].sum().reset_index().sort_values(by='recommended_quantity', ascending=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(subcat_totals, x='recommended_quantity', y='sub-category', orientation='h', title=f"Sub-Category Breakdown", labels={'recommended_quantity': 'Qty', 'sub-category': 'Category'})
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig_pie = px.pie(subcat_totals, values='recommended_quantity', names='sub-category', title=f"Sub-Category Proportions")
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                else: st.warning("Cannot show sub-category breakdown - 'sub-category' column missing.")
                st.markdown("---")
                st.subheader("Recommended Monthly Item List")
                table_cols = ['food_item', 'sub-category', 'recommended_quantity']
                if all(col in school_opt_filtered.columns for col in table_cols):
                        display_table = school_opt_filtered[table_cols].rename(columns={'food_item': 'Item', 'sub-category': 'Category', 'recommended_quantity': 'Qty'}).sort_values(by='Qty', ascending=False)
                        st.dataframe(display_table, use_container_width=True, hide_index=True)
                else: st.warning(f"Cannot display item list - missing one or more columns: {table_cols}")
            else:
                st.warning(f"No specific optimization data found for {school.title()} and {selected_meal_period} after filtering.")
            
        else:
             st.warning("Optimization data could not be loaded or processed correctly.")
    
    # Regression Tab
    with tab_reg:
        st.header("Regression Analysis")

        if "error" in regression_results:
            st.error(f"Could not run regression analysis: {regression_results['error']}")
        
        elif not regression_results:
             st.error("Regression analysis returned no results.")

        else:
            st.markdown("""
            This analysis attempts to model the relationships between key variables in the milk dataset. 
            Below are three Ordinary Least Squares (OLS) regression models.
            """)
            
            st.subheader("Regression 1: Predicting Production Cost")
            st.markdown("This model predicts the `production_cost_total` based on served, planned, discarded, leftover, and student counts.")
            st.code(regression_results.get('summary1', 'Error: Summary 1 not found.'))
            
            st.subheader("Regression 2: Predicting Discarded Cost")
            st.markdown("This model predicts the `discarded_cost` based on served, offered, planned, leftover, and student counts.")
            st.code(regression_results.get('summary2', 'Error: Summary 2 not found.'))
            
            st.subheader("Regression 3: Predicting Served Reimbursable")
            st.markdown("This model predicts the `served_reimbursable` (number of meals) based on planned, offered, and student counts.")
            st.code(regression_results.get('summary3', 'Error: Summary 3 not found.'))
            
            st.markdown("---")
            
            st.subheader("Visualization for Regression 1 (Production Cost)")
            st.markdown("""
            The plots below show the performance of a simple linear regression model 
            (trained on 80% of the data) in predicting production cost.
            """)
            
            plot_fig = regression_results.get('plot')
            if plot_fig:
                st.pyplot(plot_fig)
            else:
                st.warning("Could not generate the regression plot (not enough data).")

    # Savings/Loss Tab
    with tab_sav:
        st.header(f"Savings/Loss from Optimization for {school.title()}")

        try:
            # Load Base Data
            dfb_lower = opt_data_loaded.get('dfb')
            dfl_lower = opt_data_loaded.get('dfl')
            df_sizes_lower = opt_data_loaded.get('df_sizes')
            monthly_meal_costs = opt_data_loaded.get('meal_costs', [0, 0])
            all_schools_list = opt_data_loaded.get('schools', [])
            savings_input_data = {'dfb': dfb_lower, 'dfl': dfl_lower, 'df_sizes': df_sizes_lower}

            # Create Base DataFrames for Filtering
            aggregated_opt_results = None
            if school_opt_data is not None and 'school' in school_opt_data.columns and 'meal_type' in school_opt_data.columns and 'recommended_quantity' in school_opt_data.columns:
                 aggregated_opt_results = school_opt_data.groupby(['school', 'meal_type'], as_index=False)['recommended_quantity'].sum()
                 aggregated_opt_results = aggregated_opt_results.rename(columns={'recommended_quantity': 'optimal_quantity'})
            else:
                 st.error("Cannot aggregate optimization results - required columns missing.")

            # Create savings_df
            savings_df = None
            if aggregated_opt_results is not None:
                savings_df = prepare_savings_analysis_df(savings_input_data, aggregated_opt_results, monthly_meal_costs)

            # Create school_budgets
            total_budget = 139144760 # Default total budget
            school_budgets = {}
            if df_sizes_lower is not None:
                relevant_schools_df = df_sizes_lower[df_sizes_lower['school_name'].isin(all_schools_list)].copy()
                total_population = relevant_schools_df['count'].sum()
                if total_population > 0:
                    for index, row in relevant_schools_df.iterrows():
                        # --- FIX: Need to define proportion before using it ---
                        proportion = row['count'] / total_population
                        school_budgets[row['school_name']] = total_budget * proportion
                else:
                     st.warning("Total student population is zero, cannot calculate proportional budgets.")
            
            # --- Filter data for the hard-coded school ---
            filtered_savings_df = pd.DataFrame()
            if savings_df is not None:
                filtered_savings_df = savings_df[savings_df['school'].str.lower() == school.lower()]

            filtered_opt_results = pd.DataFrame()
            if aggregated_opt_results is not None:
                filtered_opt_results = aggregated_opt_results[aggregated_opt_results['school'].str.lower() == school.lower()]

            filtered_school_budgets = {k: v for k, v in school_budgets.items() if k == school.lower()}
            
            # --- Logic now *only* shows the specific hard-coded school ---
            st.subheader(f"Cost Comparison for: {school.title()}")
            
            # --- Chart 1 (Actual vs. Opt) ---
            if not filtered_savings_df.empty:
                school_data = filtered_savings_df.iloc[0]
                actual_vs_opt_data = {
                    'Cost Type': ['Actual Annual Cost', 'Optimized Annual Cost'],
                    'Amount (USD)': [school_data['actual_annual_cost'], school_data['optimized_annual_cost']]
                }
                actual_vs_opt_df = pd.DataFrame(actual_vs_opt_data)
                
                fig_actual_vs_opt = px.bar(
                    actual_vs_opt_df,
                    x='Cost Type',
                    y='Amount (USD)',
                    color='Cost Type',
                    title=f'Actual vs. Optimized Cost for {school.title()}',
                    color_discrete_map={'Actual Annual Cost': '#d62728', 'Optimized Annual Cost': '#2ca02c'}
                )
                st.plotly_chart(fig_actual_vs_opt, use_container_width=True)
            else:
                st.warning(f"No savings data found for {school.title()}.")

            st.markdown("---")

            # Budget vs. Opt
            st.subheader(f"Budget vs. Optimized Cost for: {school.title()}")
            if not filtered_opt_results.empty and school.lower() in filtered_school_budgets:
                school_opt_cost_df = filtered_opt_results.copy()
                meal_cost_map = {'Breakfast': monthly_meal_costs[0], 'Lunch': monthly_meal_costs[1]}
                school_opt_cost_df['monthly_food_cost'] = school_opt_cost_df.apply(
                    lambda row: row['optimal_quantity'] * meal_cost_map.get(row['meal_type'], 0),
                    axis=1
                )
                optimized_annual_cost = school_opt_cost_df['monthly_food_cost'].sum() * 10
                
                budget_data = {
                    'Cost Type': ['Proportional Budget', 'Optimized Food Cost'],
                    'Amount (USD)': [filtered_school_budgets[school.lower()], optimized_annual_cost]
                }
                budget_vs_opt_df = pd.DataFrame(budget_data)
                
                fig_budget_vs_opt = px.bar(
                    budget_vs_opt_df,
                    x='Cost Type',
                    y='Amount (USD)',
                    color='Cost Type',
                    title=f'Budget vs. Optimized Cost for {school.title()}'
                )
                st.plotly_chart(fig_budget_vs_opt, use_container_width=True)
            else:
                st.warning(f"No budget or optimization data found for {school.title()}.")

            # --- "All Schools" charts (bubble plot, bar charts by size) are removed ---

            # Savings Map
            st.markdown("---")
            map_df = None
            if filtered_savings_df is not None and not filtered_savings_df.empty:
                coords_for_map = bf_coord_data[['school_name', 'latitude', 'longitude']].drop_duplicates(subset='school_name').copy()
                coords_for_map.rename(columns={'school_name': 'school'}, inplace=True)
                
                map_data_base = filtered_savings_df.copy() # Use filtered data
                map_data_base['school'] = map_data_base['school'].astype(str).str.lower().str.strip()
                coords_for_map['school'] = coords_for_map['school'].astype(str).str.lower().str.strip()

                map_df = pd.merge(map_data_base, coords_for_map, on='school', how='left')
                map_df.dropna(subset=['latitude', 'longitude'], inplace=True)

            if map_df is not None and not map_df.empty:
                st.subheader(f"Map of Savings/Loss for {school.title()}")
                map_center = [map_df['latitude'].mean(), map_df['longitude'].mean()]
                m = folium.Map(location=map_center, zoom_start=15, tiles="cartodbpositron")
                
                # Simplified map for one school
                row = map_df.iloc[0]
                color = 'green' if row.get('outcome', 'Savings') == 'Savings' else 'red'
                popup_txt = f"<strong>School:</strong> {row['school'].title()}<br><strong>Annual Savings:</strong> ${row['savings']:,.2f}"
                size_cat = row.get('size_category', None)
                if size_cat and pd.notna(size_cat): popup_txt += f"<br><strong>Size:</strong> {size_cat.upper()}"
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_txt, max_width=300),
                    tooltip=row['school'].title(),
                    icon=folium.Icon(color=color, icon='dollar-sign', prefix='fa')
                ).add_to(m)

                st_folium(m, use_container_width=True)
            elif savings_df is not None: st.warning("Could not generate map data. Check coordinate matching.")
            else: st.warning("Could not calculate savings data needed for the map.")

        except NameError as ne: st.error(f"Required variable/function missing: {ne}. Import issue?")
        except Exception as e: st.error(f"Error generating savings map or chart: {e}"); import traceback; st.text(traceback.format_exc())
    
    with tab_pdf:
        st.header("Recommendation Report Generator")
        st.markdown(f"Generate a PDF food order recommendation for **{school.title()}**.")
        st.markdown("---")

        # --- Simplified PDF logic ---
        school_for_pdf = school # Use hard-coded school

        st.write(f"**Selected School for Report:** {school_for_pdf.title()}")

        meal_type_selection = st.radio(
            "Select Meal Type for Report:",
            ('Both', 'Breakfast', 'Lunch'),
            horizontal=True,
            key='pdf_meal_type'
        )

        if st.button(f"Generate PDF Report", use_container_width=True):
            if generate_pdf:
                with st.spinner(f"Creating {meal_type_selection} report for {school_for_pdf.title()}..."):
                    try:
                        pdf_bytes = generate_pdf(school_for_pdf.lower(), meal_type_selection)
                        if pdf_bytes:
                            st.success("PDF generated successfully!")
                            file_name_school = school_for_pdf.replace(' ', '_').lower()
                            file_name_meal = meal_type_selection.lower()
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"recommendation_{file_name_school}_{file_name_meal}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error(f"Could not generate a report. There may be no optimization data for '{school_for_pdf.title()}' with meal type '{meal_type_selection}'.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        st.text(traceback.format_exc())
            else:
                st.error("The PDF generation function is not available due to an import error.")

else:
    st.warning("⚠️ Could not load primary data required for the application. Please check file paths and availability.")