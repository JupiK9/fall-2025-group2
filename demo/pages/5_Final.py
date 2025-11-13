import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from fpdf import FPDF
import plotly.express as px
from datetime import datetime
import sys 

# Page Configuration
st.set_page_config(layout="wide", page_title="School Food Dashboard")

# Path Definitions
try:
    # This path is '.../Group2_Fall_2025/demo/pages'
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd().parent.parent

# Data source directories
DATA_DIR = PROJECT_ROOT / "src" / "data"
CLEANED_DATA_DIR = DATA_DIR / "clean-data" 
OPTIMIZATION_DATA_DIR = DATA_DIR / "optimization-data"
RESULTS_DIR = DATA_DIR / "results"

# Specific file paths for data loading
BF_PATH = CLEANED_DATA_DIR / "data_breakfast.csv" 
LN_PATH = CLEANED_DATA_DIR / "data_lunch.csv" 

if not DATA_DIR.exists() or not CLEANED_DATA_DIR.exists():
    st.error(f"Error: Data directories not found at {DATA_DIR}.")
    st.stop()

# System Path & Module Imports
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from component.pdf_generator import generate_pdf 
except ImportError as e:
    st.error(
        f"Fatal Error: Could not import 'component.pdf_generator'."
        f"Please ensure 'src/component/__init__.py' exists."
        f"Error: {e}"
    )
    st.stop()

# Helper Functions
def render_html(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=500, scrolling=True)
    except FileNotFoundError:
        st.error(f"Map file not found. Please run the pipeline. Missing: {path.name}")
    except Exception as e:
        st.error(f"An error occurred while loading the map: {e}")

@st.cache_data
def get_top_items(df, school_name, meal_type, top_n=5):
    school_df = df[
        (df['school_name'].str.lower() == school_name.lower()) & 
        (df['name'].notna())
    ].copy()
    school_df['served_reimbursable'] = pd.to_numeric(
        school_df['served_reimbursable'], errors='coerce'
    ).fillna(0)
    top_items = school_df.groupby('name')['served_reimbursable'].sum().nlargest(top_n).reset_index()
    return top_items

@st.cache_data
def load_historical_data(bf_path, ln_path):
    df_b = pd.read_csv(bf_path, low_memory=False)
    df_l = pd.read_csv(ln_path, low_memory=False)
    
    df_b['school_name'] = df_b['school_name'].astype(str).str.lower().str.strip()
    df_l['school_name'] = df_l['school_name'].astype(str).str.lower().str.strip()
    
    for df in [df_b, df_l]:
        if 'production_cost_total' in df.columns:
            df['production_cost_total'] = (
                df['production_cost_total'].astype(str)
                                         .str.replace(r'[$,]', '', regex=True)
                                         .str.strip()
            )
            df['production_cost_total'] = pd.to_numeric(
                df['production_cost_total'], errors='coerce'
            ).fillna(0)
            
    bf_costs = df_b.groupby('school_name')['production_cost_total'].sum()
    ln_costs = df_l.groupby('school_name')['production_cost_total'].sum()
    
    total_costs_dict = (bf_costs.add(ln_costs, fill_value=0) * 10).to_dict()
    return df_b, df_l, total_costs_dict

# Main Navigation Sidebar
st.sidebar.title("📊 Project Dashboard")

page = st.sidebar.radio(
    "Select a View:",
    [
        "🔍 Exploratory Data Analysis (EDA)",
        "🥪 Popularity Analysis",
        "📈 Optimization",
        "📉 Regression Modeling",
        "📄 Recommendation"
    ]
)
st.sidebar.markdown("---")

# EDA View
if page == "🔍 Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    st.info("This page is under construction. 🏗️")
    st.write("This section will contain visualizations and insights from the initial data cleaning and exploration process.")

# Popularity Analysis View
elif page == "🥪 Popularity Analysis":
    st.title("Popularity Analysis")
    st.info("This view shows the most popular food items based on historical data. This data is used to create the item-by-item production plan.")
    st.header("Recommended Monthly Item Production")
    try:
        item_csv_path = OPTIMIZATION_DATA_DIR / "monthly_items_breakdown.csv"
        df_items = pd.read_csv(item_csv_path)
        
        st.subheader("Breakfast")
        st.dataframe(df_items[df_items['meal_type'] == 'Breakfast'].head(20))
        st.subheader("Lunch")
        st.dataframe(df_items[df_items['meal_type'] == 'Lunch'].head(20))
        
    except FileNotFoundError: st.error(f"Item breakdown file not found. Please run the pipeline. Missing: {item_csv_path.name}")

# Optimization View
elif page == "📈 Optimization":
    
    st.sidebar.header("Optimization Controls")
    SCENARIO_MAP = {
        "Baseline Budget (100%)": ("Baseline Budget", "_baseline"),
        "Lower Budget Bounds (80%)": ("Lower Budget Bounds", "_lower_bound"),
        "Upper Budget bounds (120%)": ("Upper Budget bounds", "_upper_bound")
    }
    selected_scenario_name = st.sidebar.selectbox(
        "Select a Budget Scenario:", list(SCENARIO_MAP.keys())
    )
    scenario_folder, scenario_suffix = SCENARIO_MAP[selected_scenario_name]
    SCENARIO_RESULTS_DIR = RESULTS_DIR / scenario_folder

    st.sidebar.markdown("---")
    st.sidebar.header("About This View")
    st.sidebar.write(
        "This view visualizes the overall optimization results "
        "for the selected budget scenario."
    )
    
    st.title(f"School Food Optimization: {selected_scenario_name}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Financial Summary", "🏫 Savings by School Size",
        "🗺️ Geospatial Savings", "🌍 Regional Savings"
    ])

    with tab1:
        st.header("Overall Financial Impact")
        try:
            bar_chart_path = SCENARIO_RESULTS_DIR / f"overall_savings_bar_chart{scenario_suffix}.png"
            st.image(Image.open(bar_chart_path), caption=f"Overall Savings: {selected_scenario_name}", use_column_width=True)
        except FileNotFoundError: st.error(f"Chart file not found. Missing: {bar_chart_path.name}")
        
        st.markdown("---")
        st.header("Detailed Annual School Breakdown")
        try:
            breakdown_csv_path = OPTIMIZATION_DATA_DIR / f"annual_school_breakdown{scenario_suffix}.csv"
            df_breakdown = pd.read_csv(breakdown_csv_path)
            
            for col in ['proportional_annual_budget', 'annual_food_cost', 'remaining_annual_balance']:
                if col in df_breakdown.columns:
                    df_breakdown[col] = (
                        df_breakdown[col].astype(str)
                                         .str.replace(r'[$,]', '', regex=True)
                                         .str.strip()
                    )
                    df_breakdown[col] = pd.to_numeric(df_breakdown[col], errors='coerce').fillna(0)

            st.dataframe(df_breakdown)
        except FileNotFoundError: st.error(f"Data file not found. Missing: {breakdown_csv_path.name}")

    with tab2:
        st.header(f"Savings by School Size: {selected_scenario_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total Savings ($)")
            try:
                total_png_path = SCENARIO_RESULTS_DIR / f"savings_by_size_total{scenario_suffix}.png"
                st.image(Image.open(total_png_path), use_column_width=True)
            except FileNotFoundError: st.error(f"Chart file not found. Missing: {total_png_path.name}")
        with col2:
            st.subheader("Savings (%)")
            try:
                percent_png_path = SCENARIO_RESULTS_DIR / f"savings_by_size_percent{scenario_suffix}.png"
                st.image(Image.open(percent_png_path), use_column_width=True)
            except FileNotFoundError: st.error(f"Chart file not found. Missing: {percent_png_path.name}")
    with tab3:
        st.header(f"Geospatial Savings Analysis: {selected_scenario_name}")
        st.subheader("Savings Analysis: Actual vs. Optimized Cost")
        bubble_path = SCENARIO_RESULTS_DIR / f"savings_analysis_bubble_chart{scenario_suffix}.html"
        render_html(bubble_path)
    with tab4:
        st.header(f"Regional Savings Analysis (Choropleth): {selected_scenario_name}")
        st.write("Maps showing aggregated savings per region.")
        
        region_tabs = st.tabs(["Overall Map", "Elementary Schools", "Middle Schools", "High Schools"])
    
        with region_tabs[0]:
            st.write("All Schools")
            map_path = SCENARIO_RESULTS_DIR / f"fcps_region_choropleth_overall{scenario_suffix}.html"
            render_html(map_path)
            
        with region_tabs[1]:
            st.write("Elementary Schools (ES)")
            map_path = SCENARIO_RESULTS_DIR / f"fcps_region_choropleth_elementary{scenario_suffix}.html"
            render_html(map_path)
            
        with region_tabs[2]:
            st.write("Middle Schools (MS)")
            map_path = SCENARIO_RESULTS_DIR / f"fcps_region_choropleth_middle{scenario_suffix}.html"
            render_html(map_path)
            
        with region_tabs[3]:
            st.write("High Schools (HS)")
            map_path = SCENARIO_RESULTS_DIR / f"fcps_region_choropleth_high{scenario_suffix}.html"
            render_html(map_path)

# Regression Modeling View
elif page == "📉 Regression Modeling":
    st.title("Regression Modeling")
    st.info("This page is under construction. 🏗️")
    st.write("This section will feature regression models for demand forecasting or waste prediction.")

# Recommendation View
elif page == "📄 Recommendation":
    st.title("School-Specific Recommendation")
    
    # Load all necessary data
    try:
        df_b, df_l, total_costs_dict = load_historical_data(BF_PATH, LN_PATH)
        
        item_csv_path = OPTIMIZATION_DATA_DIR / "monthly_items_breakdown.csv"
        df_opt_items = pd.read_csv(item_csv_path)
        
        baseline_csv_path = OPTIMIZATION_DATA_DIR / "annual_school_breakdown_baseline.csv"
        df_school_list = pd.read_csv(baseline_csv_path)
        school_list = sorted(df_school_list['school'].unique())

    except FileNotFoundError:
        st.error("Missing critical data files. Please run the main data pipeline first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        st.stop()
        
    # Sidebar Controls
    st.sidebar.header("Report Controls")
    
    SCENARIO_MAP = {
        "Baseline Budget (100%)": ("Baseline Budget", "_baseline"),
        "Lower Budget Bounds (80%)": ("Lower Budget Bounds", "_lower_bound"),
        "Upper Budget bounds (120%)": ("Upper Budget bounds", "_upper_bound")
    }
    selected_scenario_name = st.sidebar.selectbox(
        "Select a Budget Scenario:", list(SCENARIO_MAP.keys())
    )
    scenario_folder, scenario_suffix = SCENARIO_MAP[selected_scenario_name]
    SCENARIO_RESULTS_DIR = RESULTS_DIR / scenario_folder
    
    selected_school = st.sidebar.selectbox("Select a School:", school_list)
    
    # Load Data for Selected School & Scenario
    try:
        breakdown_csv_path = OPTIMIZATION_DATA_DIR / f"annual_school_breakdown{scenario_suffix}.csv"
        df_breakdown = pd.read_csv(breakdown_csv_path)

        for col in ['proportional_annual_budget', 'annual_food_cost', 'remaining_annual_balance']:
            if col in df_breakdown.columns:
                df_breakdown[col] = (
                    df_breakdown[col].astype(str)
                                     .str.replace(r'[$,]', '', regex=True)
                                     .str.strip()
                )
                df_breakdown[col] = pd.to_numeric(df_breakdown[col], errors='coerce').fillna(0)
        
        df_breakdown['school'] = df_breakdown['school'].astype(str).str.lower().str.strip()
        
        school_data = df_breakdown[df_breakdown['school'] == selected_school].iloc[0]

        proportional_budget = school_data['proportional_annual_budget']
        annual_food_cost = school_data['annual_food_cost']
        remaining_balance = school_data['remaining_annual_balance']
        
        school_actual_cost = total_costs_dict.get(selected_school.lower().strip(), 0)
        
        total_savings = school_actual_cost - annual_food_cost
        percent_savings = (total_savings / school_actual_cost) * 100 if school_actual_cost > 0 else 0

    except FileNotFoundError:
        st.error(f"Breakdown file for {selected_scenario_name} not found. Please run the pipeline.")
        st.stop()
    except IndexError: 
        st.error(f"Could not find data for {selected_school} in scenario {selected_scenario_name}.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading scenario data: {e}")
        st.exception(e)
        st.stop()

    # --- PDF Generation Button ---
    st.sidebar.markdown("---")
    st.sidebar.write("This PDF generator uses the **Baseline (100%)** scenario data.")
    if st.sidebar.button("Generate PDF Report"):
        with st.spinner(f"Generating report for {selected_school.title()}..."):
            
            # Call 'generate_pdf' with 'school_to_test'
            pdf_data = generate_pdf(
                school_to_test=selected_school
            )
            
            if pdf_data is None:
                st.sidebar.error("PDF generation failed. Check terminal for errors.")
            else:
                st.sidebar.download_button(
                    label="✅ Download Report as PDF",
                    data=bytes(pdf_data), # pdf_data is already bytes
                    file_name=f"{selected_school.replace(' ', '_')}_report.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("Report ready!")

    # Main Page Content
    st.header(f"School Report: {selected_school.title()}")
    st.subheader(f"Scenario: {selected_scenario_name}")
    st.markdown("---")

    # Create Tabs
    tab_financial, tab_breakfast, tab_lunch = st.tabs([
        "Financial Summary", "Breakfast Details", "Lunch Details"
    ])

    with tab_financial:
        st.header("Financial Summary (from Optimization Pipeline)")
        st.write(f"Key financial metrics for **{selected_school.title()}** under the **{selected_scenario_name}** scenario.")
        
        st.subheader("Budget vs. Cost")
        col1, col2, col3 = st.columns(3)
        col1.metric("Allocated Annual Budget", f"${proportional_budget:,.2f}")
        col2.metric("Optimized Annual Cost", f"${annual_food_cost:,.2f}")
        col3.metric("Remaining Budget", f"${remaining_balance:,.2f}")

        st.subheader("Savings vs. Historical Spending")
        col4, col5 = st.columns(2)
        col4.metric("Actual Historical Cost", f"${school_actual_cost:,.2f}")
        col5.metric(
            "Total Annual Savings", 
            f"${total_savings:,.2f}",
            delta=f"{percent_savings:.2f}%",
            delta_color="normal" if total_savings >= 0 else "inverse"
        )
        
    with tab_breakfast:
        st.header("Breakfast Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Historical Top 5 Items")
            st.write("Based on total 'Served Reimbursable' from historical data.")
            try:
                top_bf = get_top_items(df_b, selected_school, 'breakfast', 5)
                fig = px.bar(top_bf, x='served_reimbursable', y='name', orientation='h', title="Top 5 Served (Historical)")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                
                st.dataframe(top_bf) # Corrected from top_f
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate historical breakfast chart: {e}")
        
        with col2:
            st.subheader("Optimized Monthly Production")
            st.write("Recommended monthly item production from the *baseline* optimization.")
            try:
                opt_bf_items = df_opt_items[
                    (df_opt_items['school'].str.lower() == selected_school.lower()) &
                    (df_opt_items['meal_type'] == 'Breakfast')
                ].sort_values(by='recommended_quantity', ascending=False)
                
                st.dataframe(opt_bf_items)
            except Exception as e:
                st.warning(f"Could not load optimized breakfast data: {e}")

    with tab_lunch:
        st.header("Lunch Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Historical Top 5 Items")
            st.write("Based on total 'Served Reimbursable' from historical data.")
            try:
                top_ln = get_top_items(df_l, selected_school, 'lunch', 5)
                fig = px.bar(top_ln, x='served_reimbursable', y='name', orientation='h', title="Top 5 Served (Historical)")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.dataframe(top_ln)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
        
                st.warning(f"Could not generate historical lunch chart: {e}")
        
        with col2:
            st.subheader("Optimized Monthly Production")
            st.write("Recommended monthly item production from the *baseline* optimization.")
            try:
                opt_ln_items = df_opt_items[
                    (df_opt_items['school'].str.lower() == selected_school.lower()) &
                    (df_opt_items['meal_type'] == 'Lunch')
                ].sort_values(by='recommended_quantity', ascending=False)
                
                st.dataframe(opt_ln_items)
            except Exception as e:
                st.warning(f"Could not load optimized lunch data. Error: {e}")