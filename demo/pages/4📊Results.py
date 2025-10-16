import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="FCPS Food & Waste Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 FCPS Waste & Food Management Analysis")
st.markdown("An exploratory data analysis of food production, consumption, and waste within the Fairfax County Public Schools system.")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Builds file paths and loads all necessary CSV files."""
    try:
        # Construct the correct path relative to the script's location
        # Assumes the script is in a folder like 'pages' and data is in 'src/data/preprocessed-data'
        base_path = Path(__file__).resolve().parent.parent.parent / 'src' / 'data' / 'preprocessed-data'
        
        bf_combined = pd.read_csv(base_path / "breakfast_combined.csv")
        ln_combined = pd.read_csv(base_path / "lunch_combined.csv")
        bf_leftover = pd.read_csv(base_path / "breakfast_leftover_rate.csv")
        ln_leftover = pd.read_csv(base_path / "lunch_leftover_rate.csv")
        bf_consumption = pd.read_csv(base_path / "breakfast_net_consumption.csv")
        ln_consumption = pd.read_csv(base_path / "lunch_net_consumption.csv")
        
        return bf_combined, ln_combined, bf_leftover, ln_leftover, bf_consumption, ln_consumption
        
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure the directory structure is correct (e.g., your app is in a 'pages' folder and data is in 'src/data/preprocessed-data/').")
        return None, None, None, None, None, None

bf_combined, ln_combined, bf_leftover, ln_leftover, bf_consumption, ln_consumption = load_data()

if bf_combined is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    all_schools = sorted(pd.concat([bf_combined['School_Name'], ln_combined['School_Name']]).unique())
    selected_school = st.sidebar.selectbox(
        "Select a School",
        options=["All Schools"] + all_schools,
        help="Filter data for a specific school or view aggregate data for all schools."
    )

    # --- Data Preprocessing & Filtering ---
    def preprocess_and_filter(df, school):
        if school != "All Schools":
            df = df[df['School_Name'] == school].copy()
        for col in ['Discarded_Cost', 'Subtotal_Cost', 'Left_Over_Cost', 'Production_Cost_Total']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    bf_filtered = preprocess_and_filter(bf_combined, selected_school)
    ln_filtered = preprocess_and_filter(ln_combined, selected_school)

    # --- Main Dashboard Tabs ---
    tab_overview, tab_breakfast, tab_lunch = st.tabs(["📈 Overview", "🥐 Breakfast Analysis", "🍔 Lunch Analysis"])

    with tab_overview:
        st.header(f"High-Level Overview for {selected_school}")
        st.markdown("This section provides a summary of production costs and waste across both breakfast and lunch.")
        
        col1, col2, col3, col4 = st.columns(4)

        total_production_cost = bf_filtered['Production_Cost_Total'].sum() + ln_filtered['Production_Cost_Total'].sum()
        total_leftover_cost = bf_filtered['Left_Over_Cost'].sum() + ln_filtered['Left_Over_Cost'].sum()
        total_discarded_cost = bf_filtered['Discarded_Cost'].sum() + ln_filtered['Discarded_Cost'].sum()
        waste_percentage = ((total_leftover_cost + total_discarded_cost) / total_production_cost * 100) if total_production_cost > 0 else 0
        
        col1.metric("Total Production Cost", f"${total_production_cost:,.2f}")
        col2.metric("Total Leftover Cost", f"${total_leftover_cost:,.2f}")
        col3.metric("Total Discarded Cost", f"${total_discarded_cost:,.2f}")
        col4.metric("Waste Percentage", f"{waste_percentage:.2f}%")

        st.markdown("---")

        st.subheader("Cost Over Time")
        cost_over_time = pd.concat([
            bf_filtered[['Date', 'Production_Cost_Total', 'Left_Over_Cost', 'Discarded_Cost']],
            ln_filtered[['Date', 'Production_Cost_Total', 'Left_Over_Cost', 'Discarded_Cost']]
        ]).groupby('Date').sum().reset_index()
        
        fig_cost_time = px.line(
            cost_over_time, 
            x='Date', 
            y=['Production_Cost_Total', 'Left_Over_Cost', 'Discarded_Cost'],
            title='Daily Production and Waste Costs',
            labels={'value': 'Cost (USD)', 'variable': 'Cost Type'}
        )
        st.plotly_chart(fig_cost_time, use_container_width=True)


    with tab_breakfast:
        st.header(f"Breakfast Insights for {selected_school}")
        b_col1, b_col2 = st.columns(2)
        
        with b_col1:
            st.subheader("Top 10 Items by Leftover Rate")
            bf_leftover_chart = bf_leftover.nlargest(10, 'Leftover Rate (%)')
            fig_bf_leftover = px.bar(
                bf_leftover_chart, x='Leftover Rate (%)', y='Item Name', orientation='h',
                title='Breakfast Items with Highest Leftover Rates',
                labels={'Item Name': 'Food Item'}, color='Leftover Rate (%)',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig_bf_leftover.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bf_leftover, use_container_width=True)

        with b_col2:
            st.subheader("Top 10 Items by Net Consumption")
            bf_consumption_chart = bf_consumption.nlargest(10, 'Net Consumption')
            fig_bf_consumption = px.bar(
                bf_consumption_chart, x='Net Consumption', y='Item Name', orientation='h',
                title='Most Consumed Breakfast Items',
                labels={'Item Name': 'Food Item'}, color='Net Consumption',
                color_continuous_scale=px.colors.sequential.Greens
            )
            fig_bf_consumption.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bf_consumption, use_container_width=True)
            
        with st.expander("📋 View Raw Breakfast Data"):
            st.dataframe(bf_filtered)

    with tab_lunch:
        st.header(f"Lunch Insights for {selected_school}")
        l_col1, l_col2 = st.columns(2)

        with l_col1:
            st.subheader("Top 10 Items by Leftover Rate")
            ln_leftover_chart = ln_leftover.nlargest(10, 'Leftover Rate (%)')
            fig_ln_leftover = px.bar(
                ln_leftover_chart, x='Leftover Rate (%)', y='Item Name', orientation='h',
                title='Lunch Items with Highest Leftover Rates',
                labels={'Item Name': 'Food Item'}, color='Leftover Rate (%)',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig_ln_leftover.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_ln_leftover, use_container_width=True)

        with l_col2:
            st.subheader("Top 10 Items by Net Consumption")
            ln_consumption_chart = ln_consumption.nlargest(10, 'Net Consumption')
            fig_ln_consumption = px.bar(
                ln_consumption_chart, x='Net Consumption', y='Item Name', orientation='h',
                title='Most Consumed Lunch Items',
                labels={'Item Name': 'Food Item'}, color='Net Consumption',
                color_continuous_scale=px.colors.sequential.Greens
            )
            fig_ln_consumption.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_ln_consumption, use_container_width=True)
            
        with st.expander("📋 View Raw Lunch Data"):
            st.dataframe(ln_filtered)
else:
    st.warning("⚠️ Could not load data. Please check file paths and availability.")

