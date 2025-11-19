import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
from datetime import datetime

# Page Configuration
SCHOOL_NAME = "Coates Elementary"
SCHOOL_NAME_LOWER = SCHOOL_NAME.lower() # "coates elementary"

# Set page config
st.set_page_config(
    layout="wide",
    page_title=f"Food Analysis Dashboard"
)

# Path Definitions
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd().parent.parent

# Data source directories
DATA_DIR = PROJECT_ROOT / "src" / "data"
CLEANED_DATA_DIR = DATA_DIR / "clean-data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-data"
OPTIMIZATION_DATA_DIR = DATA_DIR / "optimization-data"
RESULTS_DIR = DATA_DIR / "results"
EDA_DIR = RESULTS_DIR / "EDA"
POPULARITY_DATA_DIR = DATA_DIR / "popularity-data"
LEFTOVER_DATA_DIR = DATA_DIR / "leftover-data"
BASELINE_BUDGET_DIR = RESULTS_DIR / "Baseline Budget"
LOWER_BOUND_BUDGET_DIR = RESULTS_DIR / "Lower Budget Bounds"
UPPER_BOUND_BUDGET_DIR = RESULTS_DIR / "Upper Budget bounds"

# Define paths for 'Coates Elementary' View and 'EDA'
BF_PATH = CLEANED_DATA_DIR / "data_breakfast.csv" 
LN_PATH = CLEANED_DATA_DIR / "data_lunch.csv" 
BF_LEFTOVER_PATH = LEFTOVER_DATA_DIR / "breakfast_leftover_rate_by_school.csv"
LN_LEFTOVER_PATH = LEFTOVER_DATA_DIR / "lunch_leftover_rate_by_school.csv"

# Define paths for 'Fairfax County' Popularity Tab (with 'Level' column)
BF_COORD_PATH = PREPROCESSED_DATA_DIR / "data_breakfast_with_coordinates.csv"
LN_COORD_PATH = PREPROCESSED_DATA_DIR / "data_lunch_with_coordinates.csv"
POPULATION_PATH = PREPROCESSED_DATA_DIR / "2022-2025 Fairfax County School Student Count.csv"

# Define paths for financial data
ANNUAL_BASELINE_PATH = OPTIMIZATION_DATA_DIR / "annual_school_breakdown_baseline.csv"
ANNUAL_LOWER_PATH = OPTIMIZATION_DATA_DIR / "annual_school_breakdown_lower_bound.csv"
ANNUAL_UPPER_PATH = OPTIMIZATION_DATA_DIR / "annual_school_breakdown_upper_bound.csv"

# Define paths for monthly optimization data
MONTHLY_BASELINE_PATH = OPTIMIZATION_DATA_DIR / "monthly_items_breakdown_baseline.csv"
MONTHLY_LOWER_PATH = OPTIMIZATION_DATA_DIR / "monthly_items_breakdown_lower_bound.csv"
MONTHLY_UPPER_PATH = OPTIMIZATION_DATA_DIR / "monthly_items_breakdown_upper_bound.csv"

# Define Image Paths
BF_COST_LEVEL_PATH = EDA_DIR / "bf_production_cost_by_level.png"
BF_COST_REGION_PATH = EDA_DIR / "bf_production_cost_by_region.png"
LN_COST_LEVEL_PATH = EDA_DIR / "lunch_production_cost_by_level.png"
LN_COST_REGION_PATH = EDA_DIR / "lunch_production_cost_by_region.png"
COST_LEVEL_PATH = EDA_DIR / "production_cost_by_level.png"
COST_REGION_PATH = EDA_DIR / "production_cost_by_region.png"
TIMESERIES_PATH = EDA_DIR / "timeseries_total_food.png"

# Define Baseline HTML Map Paths
CHORO_ELEM_BASELINE_PATH = BASELINE_BUDGET_DIR / "fcps_region_choropleth_elementary_baseline.html"
CHORO_MID_BASELINE_PATH = BASELINE_BUDGET_DIR / "fcps_region_choropleth_middle_baseline.html"
CHORO_HIGH_BASELINE_PATH = BASELINE_BUDGET_DIR / "fcps_region_choropleth_high_baseline.html"
CHORO_ALL_BASELINE_PATH = BASELINE_BUDGET_DIR / "fcps_region_choropleth_overall_baseline.html"
SAVINGS_ELEM_BASELINE_PATH = BASELINE_BUDGET_DIR / "savings_map_elementary_baseline.html"
SAVINGS_MID_BASELINE_PATH = BASELINE_BUDGET_DIR / "savings_map_middle_baseline.html"
SAVINGS_HIGH_BASELINE_PATH = BASELINE_BUDGET_DIR / "savings_map_high_baseline.html"
OVERALL_SAVINGS_BAR_PATH = BASELINE_BUDGET_DIR / "overall_savings_bar_chart_baseline.png"
OVERALL_SAVINGS_MAP_PATH = BASELINE_BUDGET_DIR / "overall_savings_map_baseline.html"
BUBBLE_CHART_PATH = BASELINE_BUDGET_DIR / "savings_analysis_bubble_chart_baseline.html"
SIZE_PERCENT_PATH = BASELINE_BUDGET_DIR / "savings_by_size_percent_baseline.png"
SIZE_TOTAL_PATH = BASELINE_BUDGET_DIR / "savings_by_size_total_baseline.png"

# Define Lower Bound HTML Map Paths
CHORO_ELEM_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "fcps_region_choropleth_elementary_lower_bound.html"
CHORO_MID_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "fcps_region_choropleth_middle_lower_bound.html"
CHORO_HIGH_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "fcps_region_choropleth_high_lower_bound.html"
CHORO_ALL_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "fcps_region_choropleth_overall_lower_bound.html"
SAVINGS_ELEM_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "savings_map_elementary_lower_bound.html"
SAVINGS_MID_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "savings_map_middle_lower_bound.html"
SAVINGS_HIGH_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "savings_map_high_lower_bound.html"
OVERALL_SAVINGS_BAR_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "overall_savings_bar_chart_lower_bound.png"
OVERALL_SAVINGS_MAP_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "overall_savings_map_lower_bound.html"
BUBBLE_CHART_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "savings_analysis_bubble_chart_lower_bound.html"
SIZE_PERCENT_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "savings_by_size_percent_lower_bound.png"
SIZE_TOTAL_LOWER_BOUND_PATH = LOWER_BOUND_BUDGET_DIR / "savings_by_size_total_lower_bound.png"

# Define Upper Bound HTML Map Paths
CHORO_ELEM_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "fcps_region_choropleth_elementary_upper_bound.html"
CHORO_MID_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "fcps_region_choropleth_middle_upper_bound.html"
CHORO_HIGH_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "fcps_region_choropleth_high_upper_bound.html"
CHORO_ALL_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "fcps_region_choropleth_overall_upper_bound.html"
SAVINGS_ELEM_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "savings_map_elementary_upper_bound.html"
SAVINGS_MID_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "savings_map_middle_upper_bound.html"
SAVINGS_HIGH_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "savings_map_high_upper_bound.html"
OVERALL_SAVINGS_BAR_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "overall_savings_bar_chart_upper_bound.png"
OVERALL_SAVINGS_MAP_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "overall_savings_map_upper_bound.html"
BUBBLE_CHART_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "savings_analysis_bubble_chart_upper_bound.html"
SIZE_PERCENT_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "savings_by_size_percent_upper_bound.png"
SIZE_TOTAL_UPPER_BOUND_PATH = UPPER_BOUND_BUDGET_DIR / "savings_by_size_total_upper_bound.png"


# System Path & Module Imports
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from component.pdf_generator import generate_pdf 
except ImportError as e:
    st.error(
        f"Fatal Error: Could not import 'component.pdf_generator'."
        f"Please ensure 'pdf_generator.py' is inside a 'component' folder in 'src'."
        f"Error: {e}"
    )

# Data Loading
@st.cache_data
def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# HTML File Loader
@st.cache_data
def load_html(file_path):
    """Loads content from an HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

# Helper function for styling profile data
def display_styled_metric(label, value, is_financial=False):
    """Displays a styled metric using HTML markdown."""
    if pd.isna(value):
        value = "N/A"
    
    if is_financial:
        value = str(value).replace('"', '')
    
    label_style = "font-size: 14px; color: #808080; margin-bottom: 0;"
    data_style = "font-size: 18px;"
    
    st.markdown(f'<p style="{label_style}">{label}</p>\n<p style="{data_style}">{value}</p>', unsafe_allow_html=True)


def run_app():
    """Main function to run the Streamlit app."""
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    view_selection = st.sidebar.radio(
        "Select View",
        ['Fairfax County', 'Coates Elementary'],
        key="view_select"
    )

    # Sidebar Meal Filter
    st.sidebar.title("Filters")
    meal_selection = st.sidebar.radio(
        "Select Meal Type",
        ["Breakfast", "Lunch"],
        key="meal_select"
    )

    # Budget Filter
    budget_selection = st.sidebar.radio(
        "Select Budget Scenario",
        ['Lower Bound', 'Baseline', 'Upper Bound'],
        key="budget_select"
    )

    # School Level Filter (Conditional)
    school_level_selection = 'Overall' # Set a default
    if view_selection == 'Fairfax County':
        school_level_selection = st.sidebar.radio(
            "Select School Level",
            ['Overall', 'Elementary', 'Middle', 'High'],
            key="school_level_select",
        )


    # Load Data for Coates View & EDA
    df_bf_all = load_data(BF_PATH)
    df_ln_all = load_data(LN_PATH)
    df_bf_leftover_all = load_data(BF_LEFTOVER_PATH)
    df_ln_leftover_all = load_data(LN_LEFTOVER_PATH)
    
    # Load coordinate data for both views
    df_bf_coord = None
    df_ln_coord = load_data(LN_COORD_PATH) # Load lunch data to get school info
    
    # Load population data
    df_population = load_data(POPULATION_PATH)

    # Load Financial Data
    df_annual_baseline = load_data(ANNUAL_BASELINE_PATH)
    df_annual_lower = load_data(ANNUAL_LOWER_PATH)
    df_annual_upper = load_data(ANNUAL_UPPER_PATH)
    
    # Load Monthly Optimization Data
    df_monthly_baseline = load_data(MONTHLY_BASELINE_PATH)
    df_monthly_lower = load_data(MONTHLY_LOWER_PATH)
    df_monthly_upper = load_data(MONTHLY_UPPER_PATH)

    if view_selection == 'Fairfax County':
        # Load breakfast data only if needed for FCPS popularity tab
        df_bf_coord = load_data(BF_COORD_PATH)


    # Process Data for 'Coates Elementary' (Original)
    try:
        school_bf_data = df_bf_all[df_bf_all['school_name'].str.lower() == SCHOOL_NAME_LOWER].copy()
        school_ln_data = df_ln_all[df_ln_all['school_name'].str.lower() == SCHOOL_NAME_LOWER].copy()
        school_bf_leftover = df_bf_leftover_all[df_bf_leftover_all['school_name'].str.lower() == SCHOOL_NAME_LOWER].copy()
        school_ln_leftover = df_ln_leftover_all[df_ln_leftover_all['school_name'].str.lower() == SCHOOL_NAME_LOWER].copy()
    except Exception as e:
        st.error(f"Error filtering data for {SCHOOL_NAME} (from original files). Check 'school_name' column. Error: {e}")

    # Process Data for 'Coates Elementary' Profile Tab
    coates_info = None
    if df_ln_coord is not None:
        try:
            # Use 'Original_School_Name_csv' for matching
            coates_info_df = df_ln_coord[df_ln_coord['Original_School_Name_csv'].str.lower() == SCHOOL_NAME_LOWER]
            if not coates_info_df.empty:
                coates_info = coates_info_df.iloc[0]
        except Exception as e:
            st.warning(f"Could not extract Coates info from coordinate file. Error: {e}")
    
    # Process Population Data
    coates_pop_info = None
    if df_population is not None:
        try:
            pop_row_df = df_population[df_population['School_Name'].str.lower() == SCHOOL_NAME_LOWER]
            if not pop_row_df.empty:
                coates_pop_info = pop_row_df.iloc[0]
        except Exception as e:
            st.warning(f"Could not find population data for {SCHOOL_NAME}. Error: {e}")

    # Process Financial Data
    baseline_finance_info = None
    lower_finance_info = None
    upper_finance_info = None
    
    if df_annual_baseline is not None:
        try:
            baseline_df = df_annual_baseline[df_annual_baseline['school'] == SCHOOL_NAME_LOWER]
            if not baseline_df.empty:
                baseline_finance_info = baseline_df.iloc[0]
        except Exception as e:
            st.warning(f"Could not find baseline financial data: {e}")
            
    if df_annual_lower is not None:
        try:
            lower_df = df_annual_lower[df_annual_lower['school'] == SCHOOL_NAME_LOWER]
            if not lower_df.empty:
                lower_finance_info = lower_df.iloc[0]
        except Exception as e:
            st.warning(f"Could not find lower bound financial data: {e}")

    if df_annual_upper is not None:
        try:
            upper_df = df_annual_upper[df_annual_upper['school'] == SCHOOL_NAME_LOWER]
            if not upper_df.empty:
                upper_finance_info = upper_df.iloc[0]
        except Exception as e:
            st.warning(f"Could not find upper bound financial data: {e}")


    # ==================================================================
    # FAIRFAX COUNTY
    # ==================================================================
    if view_selection == 'Fairfax County':
        st.title(f"Fairfax County Public Schools: Overall Food and Waste Management Analysis")
        
        tab_eda, tab_pop, tab_opt, tab_reg = st.tabs([
            'EDA', 'Popularity', 'Optimization', 'Regression'
        ])

        with tab_eda:
            st.header(f"Exploratory Data Analysis (County-Wide) - {meal_selection}")
            
            st.subheader("Overall Production Cost Analysis")
            col_gen1, col_gen2 = st.columns(2)
            try:
                with col_gen1:
                    st.image(Image.open(COST_LEVEL_PATH), caption="Total Production Cost by Level", use_container_width=True)
                with col_gen2:
                    st.image(Image.open(COST_REGION_PATH), caption="Total Production Cost by Region", use_container_width=True)
            except FileNotFoundError as e:
                st.error(f"Could not find general cost images (production_cost_by_level.png or production_cost_by_region.png). Make sure they are in {EDA_DIR}")
                st.warning(e)
            
            st.subheader("Sales Time Series - Total Items")
            try:
                st.image(Image.open(TIMESERIES_PATH), caption="Sales Time Series - Total Items", use_container_width=True)
            except FileNotFoundError as e:
                st.error(f"Could not find time series image (timeseries_total_food.png). Make sure it is in {EDA_DIR}")
                st.warning(e)

            st.divider()

            if meal_selection == "Breakfast":
                st.subheader("Breakfast Production Cost Analysis")
                col1, col2 = st.columns(2)
                try:
                    with col1:
                        st.image(Image.open(BF_COST_LEVEL_PATH), caption="Breakfast Production Cost by Level", use_container_width=True)
                    with col2:
                        st.image(Image.open(BF_COST_REGION_PATH), caption="Breakfast Production Cost by Region", use_container_width=True)
                except FileNotFoundError as e:
                    st.error(f"Could not find Breakfast EDA images. Make sure they are in {EDA_DIR}")
                    st.warning(e)

            else: # if meal_selection == "Lunch"
                st.subheader("Lunch Production Cost Analysis")
                col1, col2 = st.columns(2)
                try:
                    with col1:
                        st.image(Image.open(LN_COST_LEVEL_PATH), caption="Lunch Production Cost by Level", use_container_width=True)
                    with col2:
                        st.image(Image.open(LN_COST_REGION_PATH), caption="Lunch Production Cost by Region", use_container_width=True)
                except FileNotFoundError as e:
                    st.error(f"Could not find Lunch EDA images. Make sure they are in {EDA_DIR}")
                    st.warning(e)


        with tab_pop:
            st.header(f"Popularity Analysis (County-Wide) - {meal_selection}")
            
            st.info("Note: The 'Select Budget Scenario' filter does not apply to this tab.")

            # Logic to filter dataframes based on school level
            level_map = {'Elementary': 'ES', 'Middle': 'MS', 'High': 'HS'}
            df_bf_filtered = df_bf_coord
            df_ln_filtered = df_ln_coord
            
            if school_level_selection in level_map:
                filter_level = level_map[school_level_selection]
                if df_bf_coord is not None:
                    df_bf_filtered = df_bf_coord[df_bf_coord['Level'] == filter_level].copy()
                if df_ln_coord is not None:
                    df_ln_filtered = df_ln_coord[df_ln_coord['Level'] == filter_level].copy()
            
            if meal_selection == "Breakfast":
                st.subheader(f"Top 5 Served Breakfast Items ({school_level_selection})")
                if (df_bf_filtered is not None) and (not df_bf_filtered.empty):
                    try:
                        # Use new column names from _with_coordinates file
                        df_bf_filtered['Served_Reimbursable'] = pd.to_numeric(df_bf_filtered['Served_Reimbursable'], errors='coerce').fillna(0)
                        top_bf_items_county = df_bf_filtered.groupby('Name')['Served_Reimbursable'].sum().nlargest(5).reset_index()
                        
                        fig_bf_top_county = px.bar(
                            top_bf_items_county, 
                            x='Served_Reimbursable', y='Name', orientation='h', 
                            title=f"Top 5 Served Breakfast Items ({school_level_selection})"
                        )
                        fig_bf_top_county.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_bf_top_county, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate county-wide breakfast chart: {e}")
                else:
                    st.warning(f"No breakfast data found for {school_level_selection} level.")
                
                st.subheader(f"Top 10 Breakfast Leftover Rates ({school_level_selection})")
                if (df_bf_filtered is not None) and (not df_bf_filtered.empty):
                    try:
                        # Recalculate leftovers using new data and column names
                        df_bf_leftover_grouped = df_bf_filtered.groupby('Name').agg(
                            Left_Over_Total=('Left_Over_Total', 'sum'),
                            Offered_Total=('Offered_Total', 'sum')
                        ).reset_index()
                        
                        df_bf_leftover_grouped = df_bf_leftover_grouped[df_bf_leftover_grouped['Offered_Total'] > 0] # Avoid division by zero
                        df_bf_leftover_grouped['leftover_rate'] = (df_bf_leftover_grouped['Left_Over_Total'] / df_bf_leftover_grouped['Offered_Total']) * 100
                        
                        top_bf_leftover_county = df_bf_leftover_grouped.sort_values(by='leftover_rate', ascending=False).head(10)

                        fig_bf_leftover_county = px.bar(
                            top_bf_leftover_county,
                            x='leftover_rate', y='Name', orientation='h',
                            title=f"Top 10 Breakfast Leftover Rates ({school_level_selection})"
                        )
                        fig_bf_leftover_county.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_bf_leftover_county, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate county-wide breakfast leftover chart: {e}")
                else:
                    st.warning(f"No breakfast data found for {school_level_selection} level.")
            
            else: # if meal_selection == "Lunch"
                st.subheader(f"Top 5 Served Lunch Items ({school_level_selection})")
                if (df_ln_filtered is not None) and (not df_ln_filtered.empty):
                    try:
                        # Use new column names from _with_coordinates file
                        df_ln_filtered['Served_Reimbursable'] = pd.to_numeric(df_ln_filtered['Served_Reimbursable'], errors='coerce').fillna(0)
                        top_ln_items_county = df_ln_filtered.groupby('Name')['Served_Reimbursable'].sum().nlargest(5).reset_index()
                        
                        fig_ln_top_county = px.bar(
                            top_ln_items_county, 
                            x='Served_Reimbursable', y='Name', orientation='h', 
                            title=f"Top 5 Served Lunch Items ({school_level_selection})"
                        )
                        fig_ln_top_county.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_ln_top_county, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate county-wide lunch chart: {e}")
                else:
                    st.warning(f"No lunch data found for {school_level_selection} level.")

                st.subheader(f"Top 10 Lunch Leftover Rates ({school_level_selection})")
                if (df_ln_filtered is not None) and (not df_ln_filtered.empty):
                    try:
                        # Recalculate leftovers using new data and column names
                        df_ln_leftover_grouped = df_ln_filtered.groupby('Name').agg(
                            Left_Over_Total=('Left_Over_Total', 'sum'),
                            Offered_Total=('Offered_Total', 'sum')
                        ).reset_index()
                        
                        df_ln_leftover_grouped = df_ln_leftover_grouped[df_ln_leftover_grouped['Offered_Total'] > 0]
                        df_ln_leftover_grouped['leftover_rate'] = (df_ln_leftover_grouped['Left_Over_Total'] / df_ln_leftover_grouped['Offered_Total']) * 100
                        
                        top_ln_leftover_county = df_ln_leftover_grouped.sort_values(by='leftover_rate', ascending=False).head(10)

                        fig_ln_leftover_county = px.bar(
                            top_ln_leftover_county,
                            x='leftover_rate', y='Name', orientation='h',
                            title=f"Top 10 Lunch Leftover Rates ({school_level_selection})"
                        )
                        fig_ln_leftover_county.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_ln_leftover_county, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate county-wide lunch leftover chart: {e}")
                else:
                    st.warning(f"No lunch data found for {school_level_selection} level.")

        
        with tab_opt:
            st.header(f"Optimization Results: {school_level_selection} - {budget_selection} Budget")

            # Initialize paths
            choropleth_path, savings_path, bubble_path, bar_path, size_total_path, size_percent_path = (None,)*6

            # Assign paths based on Budget Selection
            if budget_selection == 'Baseline':
                if school_level_selection == 'Overall':
                    choropleth_path = CHORO_ALL_BASELINE_PATH
                    savings_path = OVERALL_SAVINGS_MAP_PATH 
                    bubble_path = BUBBLE_CHART_PATH
                    bar_path = OVERALL_SAVINGS_BAR_PATH
                    size_total_path = SIZE_TOTAL_PATH
                    size_percent_path = SIZE_PERCENT_PATH
                elif school_level_selection == 'Elementary':
                    choropleth_path = CHORO_ELEM_BASELINE_PATH
                    savings_path = SAVINGS_ELEM_BASELINE_PATH
                elif school_level_selection == 'Middle':
                    choropleth_path = CHORO_MID_BASELINE_PATH
                    savings_path = SAVINGS_MID_BASELINE_PATH
                elif school_level_selection == 'High':
                    choropleth_path = CHORO_HIGH_BASELINE_PATH
                    savings_path = SAVINGS_HIGH_BASELINE_PATH
            
            elif budget_selection == 'Lower Bound':
                if school_level_selection == 'Overall':
                    choropleth_path = CHORO_ALL_LOWER_BOUND_PATH
                    savings_path = OVERALL_SAVINGS_MAP_LOWER_BOUND_PATH
                    bubble_path = BUBBLE_CHART_LOWER_BOUND_PATH
                    bar_path = OVERALL_SAVINGS_BAR_LOWER_BOUND_PATH
                    size_total_path = SIZE_TOTAL_LOWER_BOUND_PATH
                    size_percent_path = SIZE_PERCENT_LOWER_BOUND_PATH
                elif school_level_selection == 'Elementary':
                    choropleth_path = CHORO_ELEM_LOWER_BOUND_PATH
                    savings_path = SAVINGS_ELEM_LOWER_BOUND_PATH
                elif school_level_selection == 'Middle':
                    choropleth_path = CHORO_MID_LOWER_BOUND_PATH
                    savings_path = SAVINGS_MID_LOWER_BOUND_PATH
                elif school_level_selection == 'High':
                    choropleth_path = CHORO_HIGH_LOWER_BOUND_PATH
                    savings_path = SAVINGS_HIGH_LOWER_BOUND_PATH

            elif budget_selection == 'Upper Bound':
                if school_level_selection == 'Overall':
                    choropleth_path = CHORO_ALL_UPPER_BOUND_PATH
                    savings_path = OVERALL_SAVINGS_MAP_UPPER_BOUND_PATH
                    bubble_path = BUBBLE_CHART_UPPER_BOUND_PATH
                    bar_path = OVERALL_SAVINGS_BAR_UPPER_BOUND_PATH
                    size_total_path = SIZE_TOTAL_UPPER_BOUND_PATH
                    size_percent_path = SIZE_PERCENT_UPPER_BOUND_PATH
                elif school_level_selection == 'Elementary':
                    choropleth_path = CHORO_ELEM_UPPER_BOUND_PATH
                    savings_path = SAVINGS_ELEM_UPPER_BOUND_PATH
                elif school_level_selection == 'Middle':
                    choropleth_path = CHORO_MID_UPPER_BOUND_PATH
                    savings_path = SAVINGS_MID_UPPER_BOUND_PATH
                elif school_level_selection == 'High':
                    choropleth_path = CHORO_HIGH_UPPER_BOUND_PATH
                    savings_path = SAVINGS_HIGH_UPPER_BOUND_PATH
            
            # Row 1: Maps (Display if paths are set)
            if choropleth_path or savings_path:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Region Choropleth ({school_level_selection})")
                    if choropleth_path:
                        html_content = load_html(choropleth_path)
                        if html_content:
                            components.html(html_content, height=500, scrolling=True)
                        else:
                            st.warning(f"File not found or is empty: {choropleth_path.name}")
                    else:
                        st.info("No choropleth map available for this selection.")
                
                with col2:
                    st.subheader(f"Potential Savings Map ({school_level_selection})")
                    if savings_path:
                        html_content = load_html(savings_path)
                        if html_content:
                            components.html(html_content, height=500, scrolling=True)
                        else:
                            st.warning(f"File not found or is empty: {savings_path.name}")
                    else:
                        st.info(f"No savings map available for '{school_level_selection}'.")
            else:
                st.warning(f"No data available to display for {school_level_selection} - {budget_selection}. Please check file paths.")

            # Add extra graphs for 'Overall'
            if school_level_selection == 'Overall' and (bubble_path or bar_path or size_total_path or size_percent_path):
                st.divider()
                st.subheader(f"Overall Savings Analysis ({budget_selection})")
                
                # Row 2: Bubble & Bar Charts
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("##### Savings Analysis: Actual vs. Optimized Cost")
                    if bubble_path:
                        html_content_bubble = load_html(bubble_path)
                        if html_content_bubble:
                            components.html(html_content_bubble, height=500, scrolling=True)
                        else:
                            st.error(f"File not found or is empty: {bubble_path.name}")
                    else:
                        st.warning("Bubble chart not found.")
                
                with col4:
                    st.markdown("##### Overall Savings Bar Chart")
                    if bar_path:
                        try:
                            st.image(Image.open(bar_path), caption="Overall Savings", use_container_width=True)
                        except FileNotFoundError:
                            st.error(f"File not found: {bar_path.name}")
                    else:
                        st.warning("Bar chart not found.")

                # Row 3: Savings by Size Charts
                col5, col6 = st.columns(2)
                with col5:
                    st.markdown("##### Savings by Size (Total)")
                    if size_total_path:
                        try:
                            st.image(Image.open(size_total_path), caption="Savings by Size (Total)", use_container_width=True)
                        except FileNotFoundError:
                            st.error(f"File not found: {size_total_path.name}")
                    else:
                        st.warning("Size (Total) chart not found.")
                
                with col6:
                    st.markdown("##### Savings by Size (%)")
                    if size_percent_path:
                        try:
                            st.image(Image.open(size_percent_path), caption="Savings by Size (%)", use_container_width=True)
                        except FileNotFoundError:
                            st.error(f"File not found: {size_percent_path.name}")
                    else:
                        st.warning("Size (%) chart not found.")

        with tab_reg:
            st.header("Regression Analysis (County-Wide)")
            st.write(f"Content for county-wide regression models for {meal_selection} using the **{budget_selection}** scenario goes here.")

    # ==================================================================
    # COATES ELEMENTARY
    # ==================================================================
    else: # if view_selection == 'Coates Elementary'
        st.title(f"Food and Waste Management Analysis: {SCHOOL_NAME}")
        
        tab_profile, tab_pop, tab_opt, tab_rec = st.tabs([
            'School Profile', 'Popularity', 'Optimization', 'Recommendation'
        ])

        with tab_profile:
            st.header(f"School Profile: {SCHOOL_NAME}")
            if coates_info is not None:
                correct_lat = 38.95
                correct_lon = -77.42
                map_data = pd.DataFrame({'lat': [correct_lat], 'lon': [correct_lon]})
                st.map(map_data, zoom=12, use_container_width=True, height=450)

                st.divider() 
                
                st.subheader("School Details")
                
                # Prepare Address Line
                address = coates_info.get('address', 'N/A')
                zipcode = coates_info.get('zipcode')
                address_line = str(address)
                if pd.notna(zipcode):
                    try:
                        address_line += f", {int(zipcode)}"
                    except ValueError:
                        address_line += f", {zipcode}"
                
                # Create 3 columns for details
                col1, col2, col3 = st.columns(3)
                with col1:
                    display_styled_metric("Address", address_line)
                
                with col2:
                    display_styled_metric("Region", coates_info.get('FCPS Region', 'N/A'))
                
                with col3:
                    display_styled_metric("Distribution Kitchen", coates_info.get('Distribution Kitchen (DK)', 'N/A'))
                
                
                st.divider()
                st.subheader("Student Population")
                
                # Create 3 columns for population
                col_pop1, col_pop2, col_pop3 = st.columns(3)
                if coates_pop_info is not None:
                    with col_pop1:
                        display_styled_metric("2022-2023", coates_pop_info.get('2022-2023'))

                    with col_pop2:
                        display_styled_metric("2023-2024", coates_pop_info.get('2023-2024'))
                    
                    with col_pop3:
                        display_styled_metric("2024-2025", coates_pop_info.get('2024-2025'))
                else:
                    st.warning("Population data not found for this school.")
                
            else:
                st.warning("Could not load school profile information from 'data_lunch_with_coordinates.csv'.")
            
        with tab_pop:
            st.header(f"Popularity Analysis for {SCHOOL_NAME}")

            if meal_selection == "Breakfast":
                # --- Breakfast Tab Content ---
                st.subheader(f"Breakfast Analysis for {SCHOOL_NAME}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Served Items (Historical)")
                    st.write("Based on total 'Served Reimbursable' from historical data.")
                    if not school_bf_data.empty:
                        try:
                            school_bf_data['served_reimbursable'] = pd.to_numeric(school_bf_data['served_reimbursable'], errors='coerce').fillna(0)
                            top_bf_items = school_bf_data.groupby('name')['served_reimbursable'].sum().sort_values(ascending=False).reset_index()
                            
                            fig_bf_top = px.bar(
                                top_bf_items, x='served_reimbursable', y='name', 
                                orientation='h', 
                                title="Served Breakfast Items (All)",
                                height=800
                            )
                            fig_bf_top.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_bf_top, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate historical breakfast chart: {e}")
                    else:
                        st.warning(f"No historical breakfast data found for {SCHOOL_NAME}.")

                with col2:
                    st.subheader("Historical Leftover Rate")
                    st.write("Items with the highest leftover rate.")
                    if not school_bf_leftover.empty:
                        try:
                            school_bf_leftover['leftover_rate'] = pd.to_numeric(school_bf_leftover['leftover_rate'], errors='coerce').fillna(0)
                            top_bf_leftover = school_bf_leftover.sort_values(by='leftover_rate', ascending=False)
                            
                            fig_bf_leftover = px.bar(
                                top_bf_leftover, x='leftover_rate', y='name',
                                orientation='h', 
                                title="Breakfast Leftover Rates (All Items)",
                                height=800
                            )
                            fig_bf_leftover.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_bf_leftover, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load optimized breakfast data: {e}")
                    else:
                        st.warning(f"No breakfast leftover rate data found for {SCHOOL_NAME}.")

            else: # if meal_selection == "Lunch"
                # Lunch Tab Content
                st.subheader(f"Lunch Analysis for {SCHOOL_NAME}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Served Items (Historical)")
                    st.write("Based on total 'Served Reimbursable' from historical data.")
                    if not school_ln_data.empty:
                        try:
                            school_ln_data['served_reimbursable'] = pd.to_numeric(school_ln_data['served_reimbursable'], errors='coerce').fillna(0)
                            top_ln_items = school_ln_data.groupby('name')['served_reimbursable'].sum().sort_values(ascending=False).reset_index()
                            
                            fig_ln_top = px.bar(
                                top_ln_items, x='served_reimbursable', y='name', 
                                orientation='h', 
                                title="Served Lunch Items (All)",
                                height=800
                            )
                            fig_ln_top.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_ln_top, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate historical lunch chart: {e}")
                    else:
                        st.warning(f"No historical lunch data found for {SCHOOL_NAME}.")
                
                with col2:
                    st.subheader("Historical Leftover Rate")
                    st.write("Items with the highest leftover rate.")
                    if not school_ln_leftover.empty:
                        try:
                            school_ln_leftover['leftover_rate'] = pd.to_numeric(school_ln_leftover['leftover_rate'], errors='coerce').fillna(0)
                            top_ln_leftover = school_ln_leftover.sort_values(by='leftover_rate', ascending=False)
                            
                            fig_ln_leftover = px.bar(
                                top_ln_leftover, x='leftover_rate', y='name',
                                orientation='h', 
                                title="Lunch Leftover Rates (All Items)",
                                height=800
                            )
                            fig_ln_leftover.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_ln_leftover, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load optimized lunch data: {e}")
                    else:
                        st.warning(f"No lunch leftover rate data found for {SCHOOL_NAME}.")

        # Coates Optimization Tab
        with tab_opt:
            
            # Financial Summary Section
            st.subheader(f"Financial Summary: {budget_selection} Scenario")
            
            # Create 3 columns for horizontal layout
            col1, col2, col3 = st.columns(3)
            
            # Helper function to fill financial columns
            def fill_financial_columns(finance_info):
                if finance_info is not None:
                    with col1:
                        display_styled_metric("Proportional Annual Budget", finance_info.get('proportional_annual_budget'), is_financial=True)
                    with col2:
                        display_styled_metric("Annual Food Cost", finance_info.get('annual_food_cost'), is_financial=True)
                    with col3:
                        display_styled_metric("Remaining Annual Balance", finance_info.get('remaining_annual_balance'), is_financial=True)
                else:
                    st.warning(f"No {budget_selection} financial data found.")

            # Call the helper based on selection
            if budget_selection == 'Baseline':
                fill_financial_columns(baseline_finance_info)
            
            elif budget_selection == 'Lower Bound':
                fill_financial_columns(lower_finance_info)
            
            elif budget_selection == 'Upper Bound':
                fill_financial_columns(upper_finance_info)
            
            st.divider()

            st.header(f"Optimized Monthly Production for {SCHOOL_NAME}")
            st.write(f"This table shows the recommended monthly item production for **{SCHOOL_NAME}** for **{meal_selection}**, based on the **{budget_selection}** scenario selected in the sidebar.")
            
            # Map sidebar selection to data
            scenario_map_opt = {
                'Baseline': ('Baseline', df_monthly_baseline),
                'Lower Bound': ('Lower Bound', df_monthly_lower),
                'Upper Bound': ('Upper Bound', df_monthly_upper)
            }
            
            scenario_name_opt, df_opt_items = scenario_map_opt.get(budget_selection)
            
            # Display Table
            if df_opt_items is not None:
                try:
                    data_to_show = df_opt_items[
                        (df_opt_items['school'] == SCHOOL_NAME_LOWER) &
                        (df_opt_items['meal_type'] == meal_selection)
                    ][['food_item', 'recommended_quantity']].sort_values(by='recommended_quantity', ascending=False)
                    
                    st.dataframe(data_to_show, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not display {scenario_name_opt} optimization data. Error: {e}")
            else:
                st.warning(f"{scenario_name_opt} optimization file not loaded.")

        # Coates Recommendation Tab
        with tab_rec:
            st.header(f"Generate PDF Report for {SCHOOL_NAME}")
            st.write(f"Click the button below to generate a PDF summary for the **{budget_selection}** scenario.")
            st.write(f"This report will contain a financial summary and item recommendations for **both Breakfast and Lunch**.")
            
            # Map sidebar selection to data
            scenario_map_pdf = {
                'Baseline': ('_baseline', 'Baseline'),
                'Lower Bound': ('_lower_bound', 'Lower Bound'),
                'Upper Bound': ('_upper_bound', 'Upper Bound')
            }
            
            scenario_suffix, scenario_name = scenario_map_pdf.get(budget_selection)
            
            # Add PDF Download Button
            st.divider()
            
            pdf_data = None
            try:
                # Call the PDF generator function from pdf_generator.py
                pdf_data = generate_pdf(SCHOOL_NAME_LOWER, scenario_suffix, scenario_name)
            except Exception as e:
                st.error(f"Error initializing PDF generator: {e}")
                st.error("This is likely due to an incorrect file path *inside* 'pdf_generator.py'. Please ensure it can access 'unit_costs.csv' at `src/data/preprocessed-data/unit_costs.csv`.")

            if pdf_data:
                st.download_button(
                    label=f"Download {scenario_name} Report (PDF)",
                    data=bytes(pdf_data),
                    file_name=f"{SCHOOL_NAME_LOWER.replace(' ', '_')}{scenario_suffix}_recommendation.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("PDF generation failed. Please check that all required data files (including `unit_costs.csv`) are in the correct 'preprocessed-data' folder.")


# Run the App
if __name__ == "__main__":
    run_app()