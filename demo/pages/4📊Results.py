import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import folium
from streamlit_folium import st_folium
import numpy as np
import sys
import json

src_path = str(Path(__file__).resolve().parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from component.optimization import (
        prepare_optimization_data,
        prepare_savings_analysis_df,
        prepare_map_data_from_coordinates,
        analyze_savings_by_school_size
    )

except ImportError as e:
    st.error(f"Could not import functions from optimization.py: {e}.")
    # Define dummy functions
    def prepare_optimization_data(*args, **kwargs): return None
    def prepare_savings_analysis_df(*args, **kwargs): return None
    def prepare_map_data_from_coordinates(*args, **kwargs): return None
    def analyze_savings_by_school_size(*args, **kwargs): return None

# --- Page Configuration ---
st.set_page_config(
    page_title="FCPS Food & Waste Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 FCPS Waste & Food Management Analysis")
st.markdown("An exploratory data analysis of food production, consumption, and waste within the Fairfax County Public Schools system.")

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
    """Loads data using prepare_optimization_data and separately loads other files."""
    opt_data_dict = None
    school_opt_items_df = None
    popularity_files = {}
    bf_data_map_coords = None
    ln_data_map_coords = None
    nutrition_df = None # Initialize nutrition_df

    try:
        base_path = Path(__file__).resolve().parent.parent.parent / 'src' / 'data' / 'preprocessed-data'
        breakfast_path = base_path / "breakfast_combined.csv"
        lunch_path = base_path / "lunch_combined.csv"
        student_counts_path = base_path / "2022-2025 Fairfax County School Student Count.csv" # Check filename

        # --- Call prepare_optimization_data ---
        opt_data_dict = prepare_optimization_data(str(breakfast_path), str(lunch_path), str(student_counts_path))
        if opt_data_dict is None:
             st.error("Failed to prepare optimization data from optimization.py.")
             return None, None, {}, None, None

        # --- Load other needed files ---
        school_opt_items_df = pd.read_csv(base_path / "school_food_item_optimization.csv")
        nutrition_df = pd.read_csv(base_path / "fcps_nutrition_values.csv")
        # Ensure nutrition_df columns are lowercase immediately
        if nutrition_df is not None:
            nutrition_df.columns = nutrition_df.columns.str.lower()

        # Popularity Files
        popularity_files['bf_leftover'] = pd.read_csv(base_path / "breakfast_leftover_rate.csv")
        popularity_files['ln_leftover'] = pd.read_csv(base_path / "lunch_leftover_rate.csv")
        popularity_files['bf_consumption'] = pd.read_csv(base_path / "breakfast_net_consumption.csv")
        popularity_files['ln_consumption'] = pd.read_csv(base_path / "lunch_net_consumption.csv")
        popularity_files['bf_lr_school'] = pd.read_csv(base_path / "breakfast_leftover_rate_by_school.csv")
        popularity_files['ln_lr_school'] = pd.read_csv(base_path / "lunch_leftover_rate_by_school.csv")
        popularity_files['bf_nc_school'] = pd.read_csv(base_path / "breakfast_net_consumption_by_school.csv")
        popularity_files['ln_nc_school'] = pd.read_csv(base_path / "lunch_net_consumption_by_school.csv")

        # Clean column names for popularity files
        for name, df in popularity_files.items():
            if df is not None:
                df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_', regex=False)
                df.rename(columns={'leftover_rate_(%)': 'leftover_rate', 'item_name': 'name'}, inplace=True, errors='ignore')

        # Coordinate data
        bf_data_map_coords = pd.read_csv(base_path / "data_breakfast_with_coordinates.csv")
        if bf_data_map_coords is not None: bf_data_map_coords.columns = bf_data_map_coords.columns.str.lower()
        ln_data_map_coords = pd.read_csv(base_path / "data_lunch_with_coordinates.csv")
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


        return opt_data_dict, school_opt_items_df, popularity_files, bf_data_map_coords, ln_data_map_coords

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}.")
        return None, None, {}, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, {}, None, None

# --- Load all data ---
opt_data_loaded, school_opt_data, popularity_files, bf_coord_data, ln_coord_data = load_data()

# Local Mapping Functions
def generate_fcps_region_choropleth(regional_map_df, geojson_path, columns, initial_column):
    """Generates choropleth map."""
    # Ensure input keys match the expected format
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
    req_cols = ['school_name', 'latitude', 'longitude', 'left_over_cost']
    if breakfast_data is None or lunch_data is None or \
       not all(col in breakfast_data.columns for col in req_cols) or \
       not all(col in lunch_data.columns for col in req_cols):
        st.error("Missing data or columns for EDA potential savings map.")
        return None

    bf_savings = breakfast_data.groupby(['school_name', 'latitude', 'longitude'])['left_over_cost'].sum().reset_index()
    bf_savings.rename(columns={'left_over_cost': 'Breakfast Savings'}, inplace=True)
    ln_savings = lunch_data.groupby(['school_name', 'latitude', 'longitude'])['left_over_cost'].sum().reset_index()
    ln_savings.rename(columns={'left_over_cost': 'Lunch Savings'}, inplace=True)

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
        dfb['school_name'] = dfb['school_name'].astype(str).str.lower().str.strip()
        dfb = pd.merge(dfb, school_metadata_df, on='school_name', how='left')
    
    if dfl is not None and not school_metadata_df.empty:
        dfl['school_name'] = dfl['school_name'].astype(str).str.lower().str.strip()
        dfl = pd.merge(dfl, school_metadata_df, on='school_name', how='left')

    # prepare_optimization_data only cleans production_cost_total and served_reimbursable
    cost_cols_to_clean = ['left_over_cost', 'discarded_cost']
    if dfb is not None:
        for col in cost_cols_to_clean:
            if col in dfb.columns:
                dfb[col] = pd.to_numeric(dfb[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            else:
                st.warning(f"Column '{col}' not found in breakfast data ('dfb'). Metrics might be inaccurate.")
                dfb[col] = 0 # Add column with zeros if missing
    if dfl is not None:
        for col in cost_cols_to_clean:
            if col in dfl.columns:
                dfl[col] = pd.to_numeric(dfl[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            else:
                 st.warning(f"Column '{col}' not found in lunch data ('dfl'). Metrics might be inaccurate.")
                 dfl[col] = 0 # Add column with zeros if missing

    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_meal_period = st.sidebar.selectbox("Select a Meal Period", ["Overall", "Breakfast", "Lunch"])

    all_regions = sorted(bf_coord_data['fcps region'].dropna().unique())
    selected_region = st.sidebar.selectbox("Select an FCPS Region", ["All Regions"] + all_regions)

    all_dks = sorted(bf_coord_data['distribution kitchen (dk)'].dropna().unique())
    selected_dk = st.sidebar.selectbox("Select a Distribution Kitchen", ["All Distribution Kitchens"] + all_dks)

    all_levels = ['Elementary', 'Middle', 'High']
    selected_level = st.sidebar.selectbox("Select an Educational Level", ["All Levels"] + all_levels)

    # --- Dynamic School Filtering ---
    temp_schools_df = bf_coord_data.copy()
    temp_schools_df = map_educational_level(temp_schools_df)

    if selected_region != "All Regions":
        temp_schools_df = temp_schools_df[temp_schools_df['fcps region'] == selected_region]
    if selected_dk != "All Distribution Kitchens":
         temp_schools_df = temp_schools_df[temp_schools_df['distribution kitchen (dk)'] == selected_dk]
    if selected_level != "All Levels":
        temp_schools_df = temp_schools_df[temp_schools_df['educational level'] == selected_level]

    all_schools = sorted(temp_schools_df['school_name'].unique())
    selected_school = st.sidebar.selectbox("Select a School", ["All Schools"] + all_schools)

    # Data Filtering for Popularity/EDA Raw Data
    bf_filtered_raw = dfb.copy() if dfb is not None else pd.DataFrame()
    ln_filtered_raw = dfl.copy() if dfl is not None else pd.DataFrame()

    bf_filtered_raw = map_educational_level(bf_filtered_raw)
    ln_filtered_raw = map_educational_level(ln_filtered_raw)

    # --- Apply ALL filters sequentially ---
    
    # Apply school filter
    if selected_school != "All Schools":
        if 'school_name' in bf_filtered_raw.columns:
             bf_filtered_raw = bf_filtered_raw[bf_filtered_raw['school_name'].str.lower().str.strip() == selected_school.lower().strip()]
        if 'school_name' in ln_filtered_raw.columns:
             ln_filtered_raw = ln_filtered_raw[ln_filtered_raw['school_name'].str.lower().str.strip() == selected_school.lower().strip()]
    
    # Apply other filters
    if selected_region != "All Regions":
        if 'fcps region' in bf_filtered_raw.columns:
            bf_filtered_raw = bf_filtered_raw[bf_filtered_raw['fcps region'] == selected_region]
        if 'fcps region' in ln_filtered_raw.columns:
            ln_filtered_raw = ln_filtered_raw[ln_filtered_raw['fcps region'] == selected_region]
    
    if selected_dk != "All Distribution Kitchens":
        if 'distribution kitchen (dk)' in bf_filtered_raw.columns:
            bf_filtered_raw = bf_filtered_raw[bf_filtered_raw['distribution kitchen (dk)'] == selected_dk]
        if 'distribution kitchen (dk)' in ln_filtered_raw.columns:
            ln_filtered_raw = ln_filtered_raw[ln_filtered_raw['distribution kitchen (dk)'] == selected_dk]
    
    if selected_level != "All Levels":
        if 'educational level' in bf_filtered_raw.columns:
            bf_filtered_raw = bf_filtered_raw[bf_filtered_raw['educational level'] == selected_level]
        if 'educational level' in ln_filtered_raw.columns:
            ln_filtered_raw = ln_filtered_raw[ln_filtered_raw['educational level'] == selected_level]

    # Clean remaining numeric cols
    num_cols_raw = ['left_over_total', 'offered_total', 'served_total', 'discarded_total']
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
    tab_eda, tab_pop, tab_opt, tab_reg, tab_sav = st.tabs([
        "📈 Exploratory Data Analysis",
        "⭐ Popularity",
        "⚙️ Optimization",
        "📊 Regression",
        "💰 Savings/Loss"
    ])

    # EDA Tab
    with tab_eda:
        
        # --- NEW: Map and Student Count Logic ---
        student_count_val = None # Initialize
        if selected_school != "All Schools":
            school_info = None
            school_count_info = None
            
            # Get School Coordinates from bf_coord_data
            if bf_coord_data is not None and 'school_name' in bf_coord_data.columns:
                school_info = bf_coord_data[
                    bf_coord_data['school_name'].str.lower() == selected_school.lower()
                ].iloc[0] if not bf_coord_data[bf_coord_data['school_name'].str.lower() == selected_school.lower()].empty else None

            # Get Student Count from df_sizes
            if df_sizes is not None and 'school_name' in df_sizes.columns:
                school_count_info = df_sizes[
                    df_sizes['school_name'].str.lower() == selected_school.lower()
                ].iloc[0] if not df_sizes[df_sizes['school_name'].str.lower() == selected_school.lower()].empty else None
            
            # Set student count for the metric
            student_count_val = school_count_info['count'] if school_count_info is not None and 'count' in school_count_info else "N/A"

            # Display the Map
            if school_info is not None and 'latitude' in school_info and 'longitude' in school_info:
                st.subheader(f"{selected_school}")
                school_lat = school_info['latitude']
                school_lon = school_info['longitude']
                
                if pd.notna(school_lat) and pd.notna(school_lon):
                    m = folium.Map(location=[school_lat, school_lon], zoom_start=15, tiles="cartodbpositron")
                    folium.Marker(
                        [school_lat, school_lon],
                        popup=f"{selected_school}",
                        tooltip=f"{selected_school}"
                    ).add_to(m)
                    st_folium(m, use_container_width=True, height=300)
                else:
                    st.warning(f"Map cannot be displayed: Missing coordinate data for {selected_school}.")
            else:
                st.warning(f"Map cannot be displayed: Location data not found for {selected_school}.")

        st.header(f"High-Level Overview")
        st.markdown("Summary of production costs and waste based on all active filters.")

        # --- DYNAMIC METRIC CALCULATION ---
        prod_cost, lo_cost, disc_cost = 0, 0, 0
        
        if selected_meal_period == "Breakfast":
            prod_cost = bf_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in bf_filtered_raw.columns else 0
            lo_cost = bf_filtered_raw['left_over_cost'].sum()
            disc_cost = bf_filtered_raw['discarded_cost'].sum()
        
        elif selected_meal_period == "Lunch":
            prod_cost = ln_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in ln_filtered_raw.columns else 0
            lo_cost = ln_filtered_raw['left_over_cost'].sum()
            disc_cost = ln_filtered_raw['discarded_cost'].sum()
        
        else: # "Overall"
            bf_prod_cost = bf_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in bf_filtered_raw.columns else 0
            ln_prod_cost = ln_filtered_raw['production_cost_total'].sum() if 'production_cost_total' in ln_filtered_raw.columns else 0
            bf_lo_cost = bf_filtered_raw['left_over_cost'].sum()
            ln_lo_cost = ln_filtered_raw['left_over_cost'].sum()
            bf_disc_cost = bf_filtered_raw['discarded_cost'].sum()
            ln_disc_cost = ln_filtered_raw['discarded_cost'].sum()

            prod_cost = bf_prod_cost + ln_prod_cost
            lo_cost = bf_lo_cost + ln_lo_cost
            disc_cost = bf_disc_cost + ln_disc_cost
        
        waste_perc = ((lo_cost + disc_cost) / prod_cost * 100) if prod_cost > 0 else 0

        if student_count_val is not None:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(f"{selected_meal_period} Production Cost", f"${prod_cost:,.2f}")
            col2.metric(f"{selected_meal_period} Leftover Cost", f"${lo_cost:,.2f}")
            col3.metric(f"{selected_meal_period} Discarded Cost", f"${disc_cost:,.2f}")
            col4.metric(f"{selected_meal_period} Waste Percentage", f"{waste_perc:.2f}%")
            col5.metric("Student Population", student_count_val)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{selected_meal_period} Production Cost", f"${prod_cost:,.2f}")
            col2.metric(f"{selected_meal_period} Leftover Cost", f"${lo_cost:,.2f}")
            col3.metric(f"{selected_meal_period} Discarded Cost", f"${disc_cost:,.2f}")
            col4.metric(f"{selected_meal_period} Waste Percentage", f"{waste_perc:.2f}%")

        st.markdown("---")

        # --- DYNAMIC CHART ---
        st.subheader("Cost Over Time")
        cost_cols = ['date', 'production_cost_total', 'left_over_cost', 'discarded_cost']
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
                                    y=['production_cost_total', 'left_over_cost', 'discarded_cost'],
                                    title=f'Daily Production and Waste Costs ({selected_meal_period})',
                                    labels={'value': 'Cost (USD)', 'variable': 'Cost Type', 'date':'Date'})
            st.plotly_chart(fig_cost_time, use_container_width=True)
        else:
            st.warning(f"No cost data available for the selected filters to plot a time-series chart.")

        # --- RAW DATA DISPLAY ---
        if selected_school != "All Schools":
            st.markdown("---")
            st.subheader(f"Raw Filtered Data for {selected_school}")
            
            if selected_meal_period == "Breakfast":
                st.dataframe(bf_filtered_raw)
            elif selected_meal_period == "Lunch":
                st.dataframe(ln_filtered_raw)
            else: # Overall
                st.markdown("#### Breakfast")
                st.dataframe(bf_filtered_raw)
                st.markdown("#### Lunch")
                st.dataframe(ln_filtered_raw)
        else:
            st.info(f"Aggregated food item visualizations are available in the '⭐ Popularity' tab.")

    # Popularity Tab
    with tab_pop:
        st.header("Food Item Popularity Analysis")
        filters_active = (selected_region != "All Regions" or
                          selected_dk != "All Distribution Kitchens" or
                          selected_level != "All Levels")

        # Get cleaned dataframes from dictionary
        bf_leftover_static = popularity_files.get('bf_leftover')
        ln_leftover_static = popularity_files.get('ln_leftover')
        bf_consumption_static = popularity_files.get('bf_consumption')
        ln_consumption_static = popularity_files.get('ln_consumption')
        bf_lr_school_df = popularity_files.get('bf_lr_school')
        ln_lr_school_df = popularity_files.get('ln_lr_school')
        bf_nc_school_df = popularity_files.get('bf_nc_school')
        ln_nc_school_df = popularity_files.get('ln_nc_school')

        # --- BREAKFAST SECTION ---
        if selected_meal_period == "Breakfast" or selected_meal_period == "Overall":
            st.subheader(f"Breakfast Insights for: {selected_school}")
            if selected_school == "All Schools":
                b_col1, b_col2 = st.columns(2)
                if filters_active:
                    st.markdown("Showing aggregate data for schools matching filters.")
                    with b_col1:
                        st.subheader("Items by Leftover Rate")
                        bf_lr_agg = bf_filtered_raw.groupby('name').agg(lo=('left_over_total', 'sum'), off=('offered_total', 'sum')).reset_index()
                        bf_lr_agg = bf_lr_agg[bf_lr_agg['off'] > 0]
                        bf_lr_agg['rate'] = (bf_lr_agg['lo'] / bf_lr_agg['off']) * 100
                        bf_lr_chart_data = bf_lr_agg.sort_values(by='rate', ascending=True)
                        chart_h = max(400, len(bf_lr_chart_data) * 20)
                        fig = px.bar(bf_lr_chart_data, x='rate', y='name', orientation='h', title='Leftover Rates (Filtered)', labels={'name':'Item', 'rate':'Rate (%)'}, height=chart_h, color='rate', color_continuous_scale=px.colors.sequential.Reds)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("View Data"): st.dataframe(bf_lr_chart_data.sort_values(by='rate', ascending=False))
                    with b_col2:
                        st.subheader("Items by Net Consumption Rate")
                        bf_nc_agg = bf_filtered_raw.groupby('name').agg(off=('offered_total', 'sum'), lo=('left_over_total', 'sum')).reset_index()
                        bf_nc_agg = bf_nc_agg[bf_nc_agg['off'] > 0]
                        bf_nc_agg['cons'] = bf_nc_agg['off'] - bf_nc_agg['lo']
                        bf_nc_agg['rate'] = (bf_nc_agg['cons'] / bf_nc_agg['off']) * 100
                        bf_nc_chart_data = bf_nc_agg.sort_values(by='rate', ascending=True)
                        chart_h = max(400, len(bf_nc_chart_data) * 20)
                        fig = px.bar(bf_nc_chart_data, x='rate', y='name', orientation='h', title='Net Consumption Rate (Filtered)', labels={'name':'Item', 'rate':'Rate (%)'}, height=chart_h, color='rate', color_continuous_scale=px.colors.sequential.Greens)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("View Data"): st.dataframe(bf_nc_chart_data.sort_values(by='rate', ascending=False))
                else: # County-wide static files
                    st.markdown("Showing aggregate data (county-wide).")
                    with b_col1:
                        st.subheader("Items by Leftover Rate")
                        if bf_leftover_static is not None and 'leftover_rate' in bf_leftover_static.columns and 'name' in bf_leftover_static.columns:
                            bf_lr_chart_data = bf_leftover_static.sort_values(by='leftover_rate', ascending=True)
                            chart_h = max(400, len(bf_lr_chart_data) * 20)
                            fig = px.bar(bf_lr_chart_data, x='leftover_rate', y='name', orientation='h', title='Leftover Rates (County-Wide)', labels={'name':'Item', 'leftover_rate':'Rate (%)'}, height=chart_h, color='leftover_rate', color_continuous_scale=px.colors.sequential.Reds)
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("View Data"): st.dataframe(bf_lr_chart_data.sort_values(by='leftover_rate', ascending=False))
                        else: st.warning("Breakfast leftover rate data not loaded or columns incorrect.")
                    with b_col2:
                         st.subheader("Items by Net Consumption")
                         if bf_consumption_static is not None and 'net_consumption' in bf_consumption_static.columns and 'name' in bf_consumption_static.columns:
                             bf_nc_chart_data = bf_consumption_static.sort_values(by='net_consumption', ascending=True)
                             chart_h = max(400, len(bf_nc_chart_data) * 20)
                             fig = px.bar(bf_nc_chart_data, x='net_consumption', y='name', orientation='h', title='Net Consumption (County-Wide)', labels={'name':'Item', 'net_consumption':'Count'}, height=chart_h, color='net_consumption', color_continuous_scale=px.colors.sequential.Greens)
                             st.plotly_chart(fig, use_container_width=True)
                             with st.expander("View Data"): st.dataframe(bf_nc_chart_data.sort_values(by='net_consumption', ascending=False))
                         else: st.warning("Breakfast net consumption data not loaded or columns incorrect.")

            # --- SPECIFIC SCHOOL ---
            else:
                st.markdown(f"Showing data for {selected_school} only.")
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.subheader("Items by Leftover Rate")
                    if bf_lr_school_df is not None and 'school_name' in bf_lr_school_df.columns:
                        bf_lr_school_data_filtered = bf_lr_school_df[bf_lr_school_df['school_name'].str.lower() == selected_school.lower()]
                        bf_lr_chart_data = bf_lr_school_data_filtered.sort_values(by='leftover_rate', ascending=True)
                        if not bf_lr_chart_data.empty:
                            chart_h = max(400, len(bf_lr_chart_data) * 20)
                            fig = px.bar(bf_lr_chart_data, x='leftover_rate', y='name', orientation='h', title='Leftover Rates', labels={'name':'Item', 'leftover_rate':'Rate (%)'}, height=chart_h, color='leftover_rate', color_continuous_scale=px.colors.sequential.Reds)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No pre-computed Breakfast Leftover Rate data found for {selected_school}.")
                        with st.expander("View Data"): st.dataframe(bf_lr_chart_data.sort_values(by='leftover_rate', ascending=False))
                    else:
                        st.warning("Breakfast school leftover rate data not loaded or 'school_name' column missing.")

                with b_col2:
                    st.subheader("Items by Net Consumption Rate")
                    if bf_nc_school_df is not None and 'school_name' in bf_nc_school_df.columns:
                        bf_nc_school_data_filtered = bf_nc_school_df[bf_nc_school_df['school_name'].str.lower() == selected_school.lower()]
                        bf_nc_chart_data = bf_nc_school_data_filtered.sort_values(by='net_consumption_rate', ascending=True)
                        if not bf_nc_chart_data.empty:
                            chart_h = max(400, len(bf_nc_chart_data) * 20)
                            fig = px.bar(bf_nc_chart_data, x='net_consumption_rate', y='name', orientation='h', title='Net Consumption Rate', labels={'name':'Item', 'net_consumption_rate':'Rate (%)'}, height=chart_h, color='net_consumption_rate', color_continuous_scale=px.colors.sequential.Greens)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No pre-computed Breakfast Net Consumption Rate data found for {selected_school}.")
                        with st.expander("View Data"): st.dataframe(bf_nc_chart_data.sort_values(by='net_consumption_rate', ascending=False))
                    else:
                        st.warning("Breakfast school net consumption rate data not loaded or 'school_name' column missing.")

                # Keep raw data expander
                with st.expander("View Raw Filtered Data (Matching Sidebar Filters)"):
                    st.dataframe(bf_filtered_raw)
            st.markdown("---")


        # --- LUNCH SECTION ---
        if selected_meal_period == "Lunch" or selected_meal_period == "Overall":
            st.subheader(f"Lunch Insights for: {selected_school}")
            if selected_school == "All Schools":
                 l_col1, l_col2 = st.columns(2)
                 if filters_active:
                     st.markdown("Showing aggregate data for schools matching filters.")
                     with l_col1:
                         st.subheader("Items by Leftover Rate")
                         ln_lr_agg = ln_filtered_raw.groupby('name').agg(lo=('left_over_total', 'sum'), off=('offered_total', 'sum')).reset_index()
                         ln_lr_agg = ln_lr_agg[ln_lr_agg['off'] > 0]
                         ln_lr_agg['rate'] = (ln_lr_agg['lo'] / ln_lr_agg['off']) * 100
                         ln_lr_chart_data = ln_lr_agg.sort_values(by='rate', ascending=True)
                         chart_h = max(400, len(ln_lr_chart_data) * 20)
                         fig = px.bar(ln_lr_chart_data, x='rate', y='name', orientation='h', title='Leftover Rates (Filtered)', labels={'name':'Item', 'rate':'Rate (%)'}, height=chart_h, color='rate', color_continuous_scale=px.colors.sequential.Reds)
                         st.plotly_chart(fig, use_container_width=True)
                         with st.expander("View Data"): st.dataframe(ln_lr_chart_data.sort_values(by='rate', ascending=False))
                     with l_col2:
                         st.subheader("Items by Net Consumption Rate")
                         ln_nc_agg = ln_filtered_raw.groupby('name').agg(off=('offered_total', 'sum'), lo=('left_over_total', 'sum')).reset_index()
                         ln_nc_agg = ln_nc_agg[ln_nc_agg['off'] > 0]
                         ln_nc_agg['cons'] = ln_nc_agg['off'] - ln_nc_agg['lo']
                         ln_nc_agg['rate'] = (ln_nc_agg['cons'] / ln_nc_agg['off']) * 100
                         ln_nc_chart_data = ln_nc_agg.sort_values(by='rate', ascending=True)
                         chart_h = max(400, len(ln_nc_chart_data) * 20)
                         fig = px.bar(ln_nc_chart_data, x='rate', y='name', orientation='h', title='Net Consumption Rate (Filtered)', labels={'name':'Item', 'rate':'Rate (%)'}, height=chart_h, color='rate', color_continuous_scale=px.colors.sequential.Greens)
                         st.plotly_chart(fig, use_container_width=True)
                         with st.expander("View Data"): st.dataframe(ln_nc_chart_data.sort_values(by='rate', ascending=False))

                 else: # County-wide static files
                     st.markdown("Showing aggregate data (county-wide).")
                     with l_col1:
                         st.subheader("Items by Leftover Rate")
                         if ln_leftover_static is not None and 'leftover_rate' in ln_leftover_static.columns and 'name' in ln_leftover_static.columns:
                             ln_lr_chart_data = ln_leftover_static.sort_values(by='leftover_rate', ascending=True)
                             chart_h = max(400, len(ln_lr_chart_data) * 20)
                             fig = px.bar(ln_lr_chart_data, x='leftover_rate', y='name', orientation='h', title='Leftover Rates (County-Wide)', labels={'name':'Item', 'leftover_rate':'Rate (%)'}, height=chart_h, color='leftover_rate', color_continuous_scale=px.colors.sequential.Reds)
                             st.plotly_chart(fig, use_container_width=True)
                             with st.expander("View Data"): st.dataframe(ln_lr_chart_data.sort_values(by='leftover_rate', ascending=False))
                         else: st.warning("Lunch leftover rate data not loaded or columns incorrect.")
                     with l_col2:
                          st.subheader("Items by Net Consumption")
                          if ln_consumption_static is not None and 'net_consumption' in ln_consumption_static.columns and 'name' in ln_consumption_static.columns:
                              ln_nc_chart_data = ln_consumption_static.sort_values(by='net_consumption', ascending=True)
                              chart_h = max(400, len(ln_nc_chart_data) * 20)
                              fig = px.bar(ln_nc_chart_data, x='net_consumption', y='name', orientation='h', title='Net Consumption (County-Wide)', labels={'name':'Item', 'net_consumption':'Count'}, height=chart_h, color='net_consumption', color_continuous_scale=px.colors.sequential.Greens)
                              st.plotly_chart(fig, use_container_width=True)
                              with st.expander("View Data"): st.dataframe(ln_nc_chart_data.sort_values(by='net_consumption', ascending=False))
                          else: st.warning("Lunch net consumption data not loaded or columns incorrect.")

            # --- SPECIFIC SCHOOL ---
            else:
                 st.markdown(f"Showing data for {selected_school} only.")
                 l_col1, l_col2 = st.columns(2)
                 with l_col1:
                     st.subheader("Items by Leftover Rate")
                     if ln_lr_school_df is not None and 'school_name' in ln_lr_school_df.columns:
                         ln_lr_school_data_filtered = ln_lr_school_df[ln_lr_school_df['school_name'].str.lower() == selected_school.lower()]
                         ln_lr_chart_data = ln_lr_school_data_filtered.sort_values(by='leftover_rate', ascending=True)
                         if not ln_lr_chart_data.empty:
                             chart_h = max(400, len(ln_lr_chart_data) * 20)
                             fig = px.bar(ln_lr_chart_data, x='leftover_rate', y='name', orientation='h', title='Leftover Rates', labels={'name':'Item', 'leftover_rate':'Rate (%)'}, height=chart_h, color='leftover_rate', color_continuous_scale=px.colors.sequential.Reds)
                             st.plotly_chart(fig, use_container_width=True)
                         else:
                             st.info(f"No pre-computed Lunch Leftover Rate data found for {selected_school}.")
                         with st.expander("View Data"): st.dataframe(ln_lr_chart_data.sort_values(by='leftover_rate', ascending=False))
                     else:
                          st.warning("Lunch school leftover rate data not loaded or 'school_name' column missing.")

                 with l_col2:
                     st.subheader("Items by Net Consumption Rate")
                     if ln_nc_school_df is not None and 'school_name' in ln_nc_school_df.columns:
                         ln_nc_school_data_filtered = ln_nc_school_df[ln_nc_school_df['school_name'].str.lower() == selected_school.lower()]
                         ln_nc_chart_data = ln_nc_school_data_filtered.sort_values(by='net_consumption_rate', ascending=True)
                         if not ln_nc_chart_data.empty:
                             chart_h = max(400, len(ln_nc_chart_data) * 20)
                             fig = px.bar(ln_nc_chart_data, x='net_consumption_rate', y='name', orientation='h', title='Net Consumption Rate', labels={'name':'Item', 'net_consumption_rate':'Rate (%)'}, height=chart_h, color='net_consumption_rate', color_continuous_scale=px.colors.sequential.Greens)
                             st.plotly_chart(fig, use_container_width=True)
                         else:
                             st.info(f"No pre-computed Lunch Net Consumption Rate data found for {selected_school}.")
                         with st.expander("View Data"): st.dataframe(ln_nc_chart_data.sort_values(by='net_consumption_rate', ascending=False))
                     else:
                          st.warning("Lunch school net consumption rate data not loaded or 'school_name' column missing.")

                 # Keep raw data expander
                 with st.expander("View Raw Filtered Data (Matching Sidebar Filters)"): st.dataframe(ln_filtered_raw)

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

            # Specific School Logic
            if selected_school != "All Schools":
                st.subheader(f"Recommendations for: {selected_school}")
                if 'school_name' in opt_display_data.columns:
                    selected_school_clean = selected_school.lower().strip()
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
                    st.subheader("Recommended Item List")
                    table_cols = ['food_item', 'sub-category', 'recommended_quantity']
                    if all(col in school_opt_filtered.columns for col in table_cols):
                         display_table = school_opt_filtered[table_cols].rename(columns={'food_item': 'Item', 'sub-category': 'Category', 'recommended_quantity': 'Qty'}).sort_values(by='Qty', ascending=False)
                         st.dataframe(display_table, use_container_width=True, hide_index=True)
                    else: st.warning(f"Cannot display item list - missing one or more columns: {table_cols}")
                else:
                    st.warning(f"No specific optimization data found for {selected_school} and {selected_meal_period} after filtering.")

            # --- All Schools Logic ---
            else:
                filtered_opt_agg = opt_display_data.copy()
                title = "Aggregated Recommendations"
                filters_applied_list = []
                if 'fcps region' in filtered_opt_agg.columns and selected_region != "All Regions":
                    filtered_opt_agg = filtered_opt_agg[filtered_opt_agg['fcps region'] == selected_region]
                    filters_applied_list.append(selected_region)
                if 'distribution kitchen (dk)' in filtered_opt_agg.columns and selected_dk != "All Distribution Kitchens":
                    filtered_opt_agg = filtered_opt_agg[filtered_opt_agg['distribution kitchen (dk)'] == selected_dk]
                    filters_applied_list.append(selected_dk)
                if 'educational level' in filtered_opt_agg.columns and selected_level != "All Levels":
                    filtered_opt_agg = filtered_opt_agg[filtered_opt_agg['educational level'] == selected_level]
                    filters_applied_list.append(selected_level)

                title = f"Agg. Recommendations for: {', '.join(filters_applied_list)}" if filters_applied_list else "County-Wide Recommendations"
                st.subheader(title)

                if filtered_opt_agg.empty: st.warning("No optimization data found for the selected filters.")
                else:
                    total_items = int(filtered_opt_agg['recommended_quantity'].sum())
                    st.metric("Total Recommended Monthly Items", f"{total_items:,}")
                    st.subheader("Breakdown by Sub-Category")
                    if 'sub-category' in filtered_opt_agg.columns:
                        subcat_totals = filtered_opt_agg.groupby('sub-category')['recommended_quantity'].sum().reset_index().sort_values(by='recommended_quantity', ascending=False)
                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.bar(subcat_totals, x='recommended_quantity', y='sub-category', orientation='h', title=f"Sub-Category Breakdown (Agg.)", labels={'recommended_quantity': 'Qty', 'sub-category': 'Category'})
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            fig_pie = px.pie(subcat_totals, values='recommended_quantity', names='sub-category', title="Sub-Category Proportions (Agg.)")
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else: st.warning("Cannot show sub-category breakdown - 'sub-category' column missing.")
        else:
             st.warning("Optimization data could not be loaded or processed correctly.")
    
    # Regression Tab
    with tab_reg:
        st.header("Regression Analysis")
        st.markdown("Content to be added here.")

    # Savings/Loss Tab
    with tab_sav:
        st.header("Savings/Loss from Optimization")

        try:
            # Prepare inputs based on opt_data_loaded
            dfb_lower = opt_data_loaded.get('dfb')
            dfl_lower = opt_data_loaded.get('dfl')
            df_sizes_lower = opt_data_loaded.get('df_sizes')
            monthly_meal_costs = opt_data_loaded.get('meal_costs', [0, 0])
            all_schools_list = opt_data_loaded.get('schools', [])

            savings_input_data = {'dfb': dfb_lower, 'dfl': dfl_lower, 'df_sizes': df_sizes_lower}

            # Mimic monthly_results_df
            aggregated_opt_results = None
            if school_opt_data is not None and 'school' in school_opt_data.columns and 'meal_type' in school_opt_data.columns and 'recommended_quantity' in school_opt_data.columns:
                 aggregated_opt_results = school_opt_data.groupby(['school', 'meal_type'], as_index=False)['recommended_quantity'].sum()
                 aggregated_opt_results = aggregated_opt_results.rename(columns={'recommended_quantity': 'optimal_quantity'})
            else:
                 st.error("Cannot aggregate optimization results - required columns missing.")

            # Calculate Savings Dataframe
            savings_df = None
            if aggregated_opt_results is not None:
                savings_df = prepare_savings_analysis_df(savings_input_data, aggregated_opt_results, monthly_meal_costs)

            st.subheader("Actual vs. Optimized Cost by School Size")
            
            if savings_df is not None:
                
                # Aggregate actual vs. optimized cost by size
                actual_vs_opt_size_df = savings_df.groupby('size_category')[
                    ['actual_annual_cost', 'optimized_annual_cost']
                ].sum().reset_index()

                # Melt this new dataframe for plotting
                actual_vs_opt_melted = pd.melt(
                    actual_vs_opt_size_df,
                    id_vars=['size_category'],
                    value_vars=['actual_annual_cost', 'optimized_annual_cost'],
                    var_name='Cost Type', 
                    value_name='Amount (USD)'
                )
                actual_vs_opt_melted['Cost Type'] = actual_vs_opt_melted['Cost Type'].replace({
                    'actual_annual_cost': 'Actual Annual Cost',
                    'optimized_annual_cost': 'Optimized Annual Cost'
                })
                
                # Define category order
                category_order = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
                actual_vs_opt_melted['size_category'] = pd.Categorical(
                    actual_vs_opt_melted['size_category'], 
                    categories=category_order, 
                    ordered=True
                )
                actual_vs_opt_melted = actual_vs_opt_melted.sort_values('size_category')

                # Create the Plotly bar chart
                fig_actual_vs_opt = px.bar(
                    actual_vs_opt_melted,
                    x='size_category',
                    y='Amount (USD)',
                    color='Cost Type',
                    barmode='group',
                    title='Actual Annual Cost vs. Optimized Annual Cost by School Size',
                    labels={'size_category': 'School Size Category'},
                    category_orders={'size_category': category_order},
                    color_discrete_map={
                         'Actual Annual Cost': '#d62728', # Red
                         'Optimized Annual Cost': '#2ca02c' # Green
                    }
                )
                fig_actual_vs_opt.update_layout(yaxis_title='Amount (USD)', xaxis_title='School Size Category')
                
                st.plotly_chart(fig_actual_vs_opt, use_container_width=True)
                st.info("Graph represents the actual costs incurred from May data provided, Optimized costs is what our model produced considering budgetary and production constraints")
            else:
                st.warning("Could not generate Actual vs. Optimized cost chart. Missing savings or size data.")


            # --- Budget vs. Optimized Cost Chart ---
            st.markdown("---") # Separator
            st.subheader("Budget vs. Optimized Cost by School Size")
            agg_savings_by_size = None
            if aggregated_opt_results is not None and df_sizes_lower is not None:
                # Replicate Proportional Budget Calculation
                total_budget = 139144760 # Default total budget
                relevant_schools_df = df_sizes_lower[df_sizes_lower['school_name'].isin(all_schools_list)].copy()
                total_population = relevant_schools_df['count'].sum()
                school_budgets = {}
                if total_population > 0:
                    for index, row in relevant_schools_df.iterrows():
                        proportion = row['count'] / total_population
                        school_budgets[row['school_name']] = total_budget * proportion
                else:
                     st.warning("Total student population is zero, cannot calculate proportional budgets.")

                # Call analyze_savings_by_school_size
                agg_savings_by_size = analyze_savings_by_school_size(
                    aggregated_opt_results,
                    school_budgets,
                    monthly_meal_costs,
                    df_sizes_lower
                )

                if agg_savings_by_size is not None and not agg_savings_by_size.empty:
                    agg_melted = pd.melt(agg_savings_by_size,
                                         id_vars=['size_category'],
                                         value_vars=['proportional_annual_budget', 'annual_food_cost'],
                                         var_name='Cost Type', value_name='Amount (USD)')
                    agg_melted['Cost Type'] = agg_melted['Cost Type'].replace({
                        'proportional_annual_budget': 'Proportional Budget',
                        'annual_food_cost': 'Optimized Food Cost'
                    })

                    # Define category order
                    category_order = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
                    agg_melted['size_category'] = pd.Categorical(agg_melted['size_category'], categories=category_order, ordered=True)
                    agg_melted = agg_melted.sort_values('size_category')


                    # Create the Plotly bar chart
                    fig_size_savings = px.bar(
                        agg_melted,
                        x='size_category',
                        y='Amount (USD)',
                        color='Cost Type',
                        barmode='group',
                        title='Annual Budget vs. Optimized Food Cost by School Size',
                        labels={'size_category': 'School Size Category'},
                        category_orders={'size_category': category_order}
                    )
                    fig_size_savings.update_layout(yaxis_title='Amount (USD)', xaxis_title='School Size Category')
                    st.plotly_chart(fig_size_savings, use_container_width=True)
                    st.info("This graph represents the optimized costs compared to the annual approved budget which is divided and allocated based on school size")

                    # Display the aggregated data table
                    with st.expander("View Aggregated Data by Size"):
                        display_agg = agg_savings_by_size[['size_category', 'proportional_annual_budget', 'annual_food_cost', 'total_savings', 'percent_savings']].copy()
                        st.dataframe(display_agg, hide_index=True)

                else:
                    st.warning("Could not generate aggregated savings data by school size.")
            else:
                 st.warning("Missing data required for savings by size analysis (Optimization results or School sizes).")

            st.markdown("---") # Separator


            # Prepare and Generate Savings Map
            map_df = None
            if savings_df is not None and not savings_df.empty:
                coords_for_map = bf_coord_data[['school_name', 'latitude', 'longitude']].drop_duplicates(subset='school_name').copy()
                coords_for_map.rename(columns={'school_name': 'school'}, inplace=True)

                savings_df['school'] = savings_df['school'].astype(str).str.lower().str.strip()
                coords_for_map['school'] = coords_for_map['school'].astype(str).str.lower().str.strip()

                map_df = pd.merge(savings_df, coords_for_map, on='school', how='inner')
                map_df.dropna(subset=['latitude', 'longitude'], inplace=True)

            if map_df is not None and not map_df.empty:
                st.subheader("Savings/Loss per School (Bubble Map)")
                map_center = [38.83, -77.27]
                m = folium.Map(location=map_center, zoom_start=10, tiles="cartodbpositron")
                if df_sizes_lower is not None:
                     map_df = pd.merge(map_df, df_sizes_lower[['school_name', 'size_category']], left_on='school', right_on='school_name', how='left')

                max_abs_savings = map_df['savings_magnitude'].max() if 'savings_magnitude' in map_df.columns and not map_df['savings_magnitude'].empty else 1
                def scale_radius(val):
                    if pd.isna(val) or max_abs_savings == 0: return 2
                    magnitude = abs(val)
                    return (magnitude / max_abs_savings)**(1/3) * 20 + 2 if magnitude > 0 else 2

                for _, row in map_df.iterrows():
                    if pd.isna(row['latitude']) or pd.isna(row['longitude']): continue
                    color = 'green' if row.get('outcome', 'Savings') == 'Savings' else 'red'
                    popup_txt = f"<strong>School:</strong> {row['school'].title()}<br><strong>Annual Savings:</strong> ${row['savings']:,.2f}"
                    size_cat = row.get('size_category', None)
                    if size_cat and pd.notna(size_cat): popup_txt += f"<br><strong>Size:</strong> {size_cat.upper()}"
                    folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=scale_radius(row.get('savings_magnitude', 0)), color=color, fill=True, fill_color=color, fill_opacity=0.6, popup=folium.Popup(popup_txt, max_width=300)).add_to(m)
                st_folium(m, use_container_width=True)
            elif savings_df is not None: st.warning("Could not generate map data. Check coordinate matching.")
            else: st.warning("Could not calculate savings data needed for the map.")

        except NameError as ne: st.error(f"Required variable/function missing: {ne}. Import issue?")
        except Exception as e: st.error(f"Error generating savings map or chart: {e}"); import traceback; st.text(traceback.format_exc())
else:
    st.warning("⚠️ Could not load primary data required for the application. Please check file paths and availability.")