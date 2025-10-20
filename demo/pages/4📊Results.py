import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="FCPS Food & Waste Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 FCPS Waste & Food Management Analysis")
st.markdown("An exploratory data analysis of food production, consumption, and waste within the Fairfax County Public Schools system.")

def map_educational_level(df):
    """
    Creates a new 'Educational Level' column by mapping the 'Level' column.
    It groups 'SS' (Secondary) with 'HS' (High).
    """
    if 'Level' in df.columns:
        # Define the mapping
        level_map = {
            'ES': 'Elementary',
            'MS': 'Middle',
            'HS': 'High',
        }
        # Create the new column
        df['Educational Level'] = df['Level'].map(level_map).fillna('Other')
    return df

# --- Data Loading ---
@st.cache_data
def load_data():
    """Builds file paths and loads all necessary CSV files."""
    try:
        # Construct the correct path relative to the script's location
        base_path = Path(__file__).resolve().parent.parent.parent / 'src' / 'data' / 'preprocessed-data'
        
        # --- Load Original Files ---
        bf_combined = pd.read_csv(base_path / "breakfast_combined.csv")
        ln_combined = pd.read_csv(base_path / "lunch_combined.csv")
        bf_leftover = pd.read_csv(base_path / "breakfast_leftover_rate.csv")
        ln_leftover = pd.read_csv(base_path / "lunch_leftover_rate.csv")
        bf_consumption = pd.read_csv(base_path / "breakfast_net_consumption.csv")
        ln_consumption = pd.read_csv(base_path / "lunch_net_consumption.csv")

        bf_data_map = pd.read_csv(base_path / "data_breakfast_with_coordinates.csv")
        ln_data_map = pd.read_csv(base_path / "data_lunch_with_coordinates.csv")

        # --- Load Optimization Files ---
        opt_df = pd.read_csv(base_path / "school_food_item_optimization.csv")
        nutrition_df = pd.read_csv(base_path / "fcps_nutrition_values.csv")

        # --- Load New School-Specific Popularity Files ---
        bf_lr_school = pd.read_csv(base_path / "breakfast_leftover_rate_by_school.csv")
        ln_lr_school = pd.read_csv(base_path / "lunch_leftover_rate_by_school.csv")

        bf_nc_school = pd.read_csv(base_path / "breakfast_net_consumption_by_school.csv")
        ln_nc_school = pd.read_csv(base_path / "lunch_net_consumption_by_school.csv")

        # --- Add Educational Level to files ---
        bf_combined = map_educational_level(bf_combined)
        ln_combined = map_educational_level(ln_combined)
        bf_data_map = map_educational_level(bf_data_map)
        ln_data_map = map_educational_level(ln_data_map)
        
        # --- Process and Merge Optimization Data ---
        
        # Create a clean school metadata table
        school_metadata = bf_data_map[['School_Name', 'FCPS Region', 'Distribution Kitchen (DK)', 'Level', 'Educational Level']].drop_duplicates()
        school_metadata['school_join_key'] = school_metadata['School_Name'].str.lower()

        # Merge optimization data with school metadata
        opt_with_meta = pd.merge(
            opt_df, 
            school_metadata, 
            left_on='school', 
            right_on='school_join_key', 
            how='left'
        )

        # Merge with nutrition data to get Sub-Category
        opt_full = pd.merge(
            opt_with_meta, 
            nutrition_df[['Food_Name', 'Sub-Category']], 
            left_on='food_item', 
            right_on='Food_Name', 
            how='left'
        )
        
        # Fill missing sub-categories
        opt_full['Sub-Category'] = opt_full['Sub-Category'].fillna('Other')
        
        return (bf_combined, ln_combined, bf_leftover, ln_leftover, bf_consumption, ln_consumption, 
                bf_data_map, ln_data_map, opt_full, bf_lr_school, ln_lr_school, bf_nc_school, ln_nc_school)
        
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure the directory structure is correct (e.g., your app is in a 'pages' folder and data is in 'src/data/preprocessed-data/').")
        return None, None, None, None, None, None, None, None, None, None, None, None, None

(bf_combined, ln_combined, bf_leftover, ln_leftover, bf_consumption, ln_consumption, bf_data_map, ln_data_map, opt_data, bf_lr_school, ln_lr_school, bf_nc_school, ln_nc_school) = load_data()

# --- Optimization and Mapping Functions ---
def generate_fcps_region_choropleth(regional_map_df, geojson_path, columns, initial_column):
    """
    Generates and displays a choropleth map of FCPS regions with a dropdown to select different data columns.
    """
    try:
        with open(geojson_path) as f:
            gj = json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON file not found at {geojson_path}")
        return

    # Create the map centered on Fairfax County
    m = folium.Map(location=[38.8, -77.3], zoom_start=10, tiles='CartoDB positron')

    # Create the choropleth layer
    choropleth = folium.Choropleth(
        geo_data=gj,
        data=regional_map_df,
        columns=['REGION_KEY', initial_column],
        key_on='feature.properties.REGION',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=initial_column,
        name=initial_column
    ).add_to(m)

    # Add a tooltip to the GeoJSON layer
    folium.GeoJsonTooltip(['REGION']).add_to(choropleth.geojson)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m

def prepare_map_data_from_coordinates(breakfast_data, lunch_data):
    """Prepares combined breakfast and lunch data with coordinates for mapping."""
    
    # Calculate total leftover cost per school for breakfast
    breakfast_savings = breakfast_data.groupby(['School_Name', 'latitude', 'longitude'])['Left_Over_Cost'].sum().reset_index()
    breakfast_savings.rename(columns={'Left_Over_Cost': 'Breakfast Savings'}, inplace=True)
    
    # Calculate total leftover cost per school for lunch
    lunch_savings = lunch_data.groupby(['School_Name', 'latitude', 'longitude'])['Left_Over_Cost'].sum().reset_index()
    lunch_savings.rename(columns={'Left_Over_Cost': 'Lunch Savings'}, inplace=True)
    
    # Merge breakfast and lunch savings
    combined_savings = pd.merge(breakfast_savings, lunch_savings, on=['School_Name', 'latitude', 'longitude'], how='outer').fillna(0)
    
    # Calculate total savings
    combined_savings['Total Savings'] = combined_savings['Breakfast Savings'] + combined_savings['Lunch Savings']
    
    return combined_savings

if bf_combined is not None and opt_data is not None and bf_lr_school is not None and bf_nc_school is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Meal Period Filter
    selected_meal_period = st.sidebar.selectbox(
        "Select a Meal Period",
        options=["Overall", "Breakfast", "Lunch"]
    )

    # FCPS Region Filter
    all_regions = sorted(pd.concat([bf_data_map['FCPS Region'], ln_data_map['FCPS Region']]).dropna().unique())
    selected_region = st.sidebar.selectbox(
        "Select an FCPS Region",
        options=["All Regions"] + all_regions
    )

    # Distribution Kitchen Filter
    all_dks = sorted(pd.concat([bf_data_map['Distribution Kitchen (DK)'], ln_data_map['Distribution Kitchen (DK)']]).dropna().unique())
    selected_dk = st.sidebar.selectbox(
        "Select a Distribution Kitchen",
        options=["All Distribution Kitchens"] + all_dks
    )

    # Educational Level Filter
    all_levels = ['Elementary', 'Middle', 'High']
    selected_level = st.sidebar.selectbox(
        "Select an Educational Level",
        options=["All Levels"] + all_levels
    )

    # --- Dynamic School Filtering ---
    # Filter the list of schools based on DK and Level filters
    temp_bf_schools = bf_data_map.copy()
    temp_ln_schools = ln_data_map.copy()

    if selected_region != "All Regions":
        temp_bf_schools = temp_bf_schools[temp_bf_schools['FCPS Region'] == selected_region]
        temp_ln_schools = temp_ln_schools[temp_ln_schools['FCPS Region'] == selected_region]

    if selected_dk != "All Distribution Kitchens":
        temp_bf_schools = temp_bf_schools[temp_bf_schools['Distribution Kitchen (DK)'] == selected_dk]
        temp_ln_schools = temp_ln_schools[temp_ln_schools['Distribution Kitchen (DK)'] == selected_dk]

    if selected_level != "All Levels":
        temp_bf_schools = temp_bf_schools[temp_bf_schools['Educational Level'] == selected_level]
        temp_ln_schools = temp_ln_schools[temp_ln_schools['Educational Level'] == selected_level]
    
    # Create the dynamic list of schools
    all_schools = sorted(pd.concat([temp_bf_schools['School_Name'], temp_ln_schools['School_Name']]).unique())
    
    selected_school = st.sidebar.selectbox(
        "Select a School",
        options=["All Schools"] + all_schools,
        help="Filters based on DK and Level. Select 'All Schools' to see aggregate data for your filters."
    )

    # --- Data Preprocessing & Filtering ---
    def preprocess_and_filter(df, school, region, dk, level):
            """
            Applies all filters to the dataframe.
            """
            # Filter by specific school if chosen
            if school != "All Schools":
                df = df[df['School_Name'] == school].copy()
            else:
                # Otherwise, filter by Region, DK and Level
                if region != "All Regions":
                    df = df[df['FCPS Region'] == region].copy()
                if dk != "All Distribution Kitchens":
                    df = df[df['Distribution Kitchen (DK)'] == dk].copy()
                if level != "All Levels":
                    df = df[df['Educational Level'] == level].copy()
            
            cols_to_convert = [
                'Discarded_Cost', 'Subtotal_Cost', 'Left_Over_Cost', 'Production_Cost_Total',
                'Left_Over_Total', 'Offered_Total', 'Served_Total', 'Discarded_Total'
            ]
            
            for col in cols_to_convert:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0)
                    
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df

    # Apply all filters
    bf_filtered = preprocess_and_filter(bf_data_map, selected_school, selected_region, selected_dk, selected_level)
    ln_filtered = preprocess_and_filter(ln_data_map, selected_school, selected_region, selected_dk, selected_level)

    # --- Main Dashboard Tabs ---
    tab_eda, tab_pop, tab_opt, tab_reg, tab_sav = st.tabs([
        "📈 Exploratory Data Analysis", 
        "⭐ Popularity", 
        "⚙️ Optimization", 
        "📊 Regression",
        "💰 Savings/Loss"
    ])

    with tab_eda:
        # --- This content is now dynamic based on the sidebar filter ---
        if selected_meal_period == "Overall":
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

            if selected_school == "All Schools":
                with st.expander("Geographical Analysis of Potential Savings", expanded=True):
                    st.header("🗺️ Geographical Insights")
                    
                    bf_data_map['Left_Over_Cost'] = pd.to_numeric(bf_data_map['Left_Over_Cost'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0)
                    ln_data_map['Left_Over_Cost'] = pd.to_numeric(ln_data_map['Left_Over_Cost'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0)

                    bf_map_filtered = bf_data_map.copy()
                    ln_map_filtered = ln_data_map.copy()

                    if selected_region != "All Regions":
                        bf_map_filtered = bf_map_filtered[bf_map_filtered['FCPS Region'] == selected_region]
                        ln_map_filtered = ln_map_filtered[ln_map_filtered['FCPS Region'] == selected_region]
                    if selected_dk != "All Distribution Kitchens":
                        bf_map_filtered = bf_map_filtered[bf_map_filtered['Distribution Kitchen (DK)'] == selected_dk]
                        ln_map_filtered = ln_map_filtered[ln_map_filtered['Distribution Kitchen (DK)'] == selected_dk]
                    if selected_level != "All Levels":
                        bf_map_filtered = bf_map_filtered[bf_map_filtered['Educational Level'] == selected_level]
                        ln_map_filtered = ln_map_filtered[ln_map_filtered['Educational Level'] == selected_level]
                    
                    base_path = Path(__file__).resolve().parent.parent.parent / 'src' / 'data' / 'preprocessed-data'
                    geojson_path = base_path / "School_Regions.geojson"

                    bf_regional = bf_map_filtered.groupby('FCPS Region')['Left_Over_Cost'].sum().reset_index()
                    ln_regional = ln_map_filtered.groupby('FCPS Region')['Left_Over_Cost'].sum().reset_index()
                    regional_savings = pd.merge(bf_regional, ln_regional, on='FCPS Region', suffixes=('_bf', '_ln'))
                    regional_savings['total_savings'] = regional_savings['Left_Over_Cost_bf'] + regional_savings['Left_Over_Cost_ln']
                    regional_savings['REGION_KEY'] = regional_savings['FCPS Region'].str.extract('(\d+)').astype(int)
                    
                    school_savings = prepare_map_data_from_coordinates(bf_map_filtered, ln_map_filtered)
                    
                    st.subheader("Potential Savings by Region (Choropleth Map)")
                    regional_map = generate_fcps_region_choropleth(
                        regional_savings, geojson_path,
                        columns=['total_savings', 'Left_Over_Cost_bf', 'Left_Over_Cost_ln'],
                        initial_column='total_savings'
                    )
                    st_folium(regional_map, use_container_width=True)

                    st.subheader("Potential Savings by School (Bubble Map)")
                    bubble_map = folium.Map(location=[38.8, -77.3], zoom_start=10, tiles="CartoDB positron")
                    for idx, row in school_savings.iterrows():
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=row['Total Savings']/1000,
                            popup=f"{row['School_Name']}<br>Total Savings: ${row['Total Savings']:.2f}",
                            color='crimson', fill=True, fill_color='crimson'
                        ).add_to(bubble_map)
                    st_folium(bubble_map, use_container_width=True)
        
        else:
            # Show this message if user selects "Breakfast" or "Lunch"
            st.info(f"Displaying {selected_meal_period} data in the '⭐ Popularity' tab.")

    with tab_pop:
        st.header("Food Item Popularity Analysis")
        
        # --- Check if any filters are active ---
        filters_active = (selected_region != "All Regions" or 
                          selected_dk != "All Distribution Kitchens" or 
                          selected_level != "All Levels")
        
        # --- BREAKFAST SECTION ---
        if selected_meal_period == "Breakfast" or selected_meal_period == "Overall":
            st.subheader(f"Breakfast Insights for: {selected_school}")
            
            # Check if "All Schools" or a specific school is selected
            if selected_school == "All Schools":
                b_col1, b_col2 = st.columns(2)
                
                # --- ALL SCHOOLS (Dynamic Data) ---
                if filters_active:
                    st.markdown("Showing aggregate data for all schools matching your filters.")
                    
                    with b_col1:
                        st.subheader("All Items by Leftover Rate")
                        # --- Calculate from filtered data ---
                        bf_lr_agg = bf_filtered.groupby('Name').agg(
                            Left_Over_Total=('Left_Over_Total', 'sum'),
                            Offered_Total=('Offered_Total', 'sum')
                        ).reset_index()
                        bf_lr_agg = bf_lr_agg[bf_lr_agg['Offered_Total'] > 0] # Avoid division by zero
                        bf_lr_agg['Leftover Rate (%)'] = (bf_lr_agg['Left_Over_Total'] / bf_lr_agg['Offered_Total']) * 100
                        bf_leftover_chart = bf_lr_agg.sort_values(by='Leftover Rate (%)', ascending=True)
                        
                        chart_height = max(400, len(bf_leftover_chart) * 20)
                        fig_bf_leftover = px.bar(
                            bf_leftover_chart, x='Leftover Rate (%)', y='Name', orientation='h',
                            title='Breakfast Items by Leftover Rates (Filtered)',
                            labels={'Name': 'Food Item'}, color='Leftover Rate (%)',
                            color_continuous_scale=px.colors.sequential.Reds, height=chart_height
                        )
                        st.plotly_chart(fig_bf_leftover, use_container_width=True)
                        with st.expander("📋 View Filtered Leftover Rate Data"):
                            st.dataframe(bf_leftover_chart.sort_values(by='Leftover Rate (%)', ascending=False))

                    with b_col2:
                        st.subheader("All Items by Net Consumption Rate")
                        # --- Calculate from filtered data ---
                        bf_nc_agg = bf_filtered.groupby('Name').agg(
                            Offered_Total=('Offered_Total', 'sum'),
                            Left_Over_Total=('Left_Over_Total', 'sum')
                        ).reset_index()
                        bf_nc_agg = bf_nc_agg[bf_nc_agg['Offered_Total'] > 0]
                        bf_nc_agg['Net Consumption'] = bf_nc_agg['Offered_Total'] - bf_nc_agg['Left_Over_Total']
                        bf_nc_agg['Net Consumption Rate (%)'] = (bf_nc_agg['Net Consumption'] / bf_nc_agg['Offered_Total']) * 100
                        bf_consumption_chart = bf_nc_agg.sort_values(by='Net Consumption Rate (%)', ascending=True)

                        chart_height = max(400, len(bf_consumption_chart) * 20)
                        fig_bf_consumption = px.bar(
                            bf_consumption_chart, x='Net Consumption Rate (%)', y='Name', orientation='h',
                            title='Breakfast Items by Net Consumption Rate (Filtered)',
                            labels={'Name': 'Food Item'}, color='Net Consumption Rate (%)',
                            color_continuous_scale=px.colors.sequential.Greens, height=chart_height
                        )
                        st.plotly_chart(fig_bf_consumption, use_container_width=True)
                        with st.expander("📋 View Filtered Net Consumption Rate Data"):
                            st.dataframe(bf_consumption_chart.sort_values(by='Net Consumption Rate (%)', ascending=False))

                else:
                    # --- ALL SCHOOLS (Static County-wide Data) ---
                    st.markdown("Showing aggregate data for all schools (county-wide).")
                    with b_col1:
                        st.subheader("All Items by Leftover Rate")
                        bf_leftover_chart = bf_leftover.sort_values(by='Leftover Rate (%)', ascending=True)
                        chart_height = max(400, len(bf_leftover_chart) * 20)
                        fig_bf_leftover = px.bar(
                            bf_leftover_chart, x='Leftover Rate (%)', y='Item Name', orientation='h',
                            title='Breakfast Items by Leftover Rates (County-Wide)',
                            labels={'Item Name': 'Food Item'}, color='Leftover Rate (%)',
                            color_continuous_scale=px.colors.sequential.Reds, height=chart_height
                        )
                        st.plotly_chart(fig_bf_leftover, use_container_width=True)
                        with st.expander("📋 View County-Wide Leftover Rate Data"):
                            st.dataframe(bf_leftover_chart.sort_values(by='Leftover Rate (%)', ascending=False))

                    with b_col2:
                        st.subheader("All Items by Net Consumption")
                        bf_consumption_chart = bf_consumption.sort_values(by='Net Consumption', ascending=True)
                        chart_height = max(400, len(bf_consumption_chart) * 20)
                        fig_bf_consumption = px.bar(
                            bf_consumption_chart, x='Net Consumption', y='Item Name', orientation='h',
                            title='Most Consumed Breakfast Items (County-Wide)',
                            labels={'Item Name': 'Food Item'}, color='Net Consumption',
                            color_continuous_scale=px.colors.sequential.Greens, height=chart_height
                        )
                        st.plotly_chart(fig_bf_consumption, use_container_width=True)
                        with st.expander("📋 View County-Wide Net Consumption Data"):
                            st.dataframe(bf_consumption_chart.sort_values(by='Net Consumption', ascending=False))
            
            else:
                # --- SPECIFIC SCHOOL ---
                st.markdown(f"Showing data for {selected_school} only.")
                if bf_filtered.empty:
                    st.warning("No breakfast data found for this school.")
                else:
                    b_col1, b_col2 = st.columns(2)
                    
                    with b_col1:
                        st.subheader("All Items by Leftover Rate")
                        bf_lr_school_data = bf_lr_school[bf_lr_school['school_name'] == selected_school]
                        bf_lr_chart = bf_lr_school_data.sort_values(by='leftover_rate', ascending=True)
                        
                        if not bf_lr_chart.empty:
                            chart_height = max(400, len(bf_lr_chart) * 20)
                            fig_bf_school_leftover = px.bar(
                                bf_lr_chart, x='leftover_rate', y='name', orientation='h',
                                title='Breakfast Items with Highest Leftover Rates',
                                labels={'name': 'Food Item', 'leftover_rate': 'Leftover Rate (%)'}, 
                                color='leftover_rate',
                                color_continuous_scale=px.colors.sequential.Reds, height=chart_height
                            )
                            st.plotly_chart(fig_bf_school_leftover, use_container_width=True)
                        else:
                            st.info("No pre-computed leftover rate data found for this school.")
                        
                        with st.expander("📋 View Leftover Rate Data"):
                            st.dataframe(bf_lr_chart.sort_values(by='leftover_rate', ascending=False))
                    
                    with b_col2:
                        st.subheader("All Items by Net Consumption Rate")
                        bf_nc_school_data = bf_nc_school[bf_nc_school['school_name'] == selected_school]
                        bf_nc_chart = bf_nc_school_data.sort_values(by='net_consumption_rate', ascending=True)
                        
                        if not bf_nc_chart.empty:
                            chart_height = max(400, len(bf_nc_chart) * 20)
                            fig_bf_school_consumption = px.bar(
                                bf_nc_chart, x='net_consumption_rate', y='name', orientation='h',
                                title='Breakfast Items with Highest Consumption Rates',
                                labels={'name': 'Food Item', 'net_consumption_rate': 'Net Consumption Rate (%)'}, 
                                color='net_consumption_rate',
                                color_continuous_scale=px.colors.sequential.Greens, height=chart_height
                            )
                            st.plotly_chart(fig_bf_school_consumption, use_container_width=True)
                        else:
                            st.info("No pre-computed net consumption data found for this school.")
                        
                        with st.expander("📋 View Net Consumption Rate Data"):
                            st.dataframe(bf_nc_chart.sort_values(by='net_consumption_rate', ascending=False))
                
                with st.expander("📋 View Raw Filtered Breakfast Data"):
                    st.dataframe(bf_filtered)

            st.markdown("---") # Add a separator

        # --- LUNCH SECTION ---
        if selected_meal_period == "Lunch" or selected_meal_period == "Overall":
            st.subheader(f"Lunch Insights for: {selected_school}")
            
            # Check if "All Schools" or a specific school
            if selected_school == "All Schools":
                l_col1, l_col2 = st.columns(2)

                # --- ALL SCHOOLS (Dynamic Data) ---
                if filters_active:
                    st.markdown("Showing aggregate data for all schools matching your filters.")
                    
                    with l_col1:
                        st.subheader("All Items by Leftover Rate")
                        # --- Calculate from filtered data ---
                        ln_lr_agg = ln_filtered.groupby('Name').agg(
                            Left_Over_Total=('Left_Over_Total', 'sum'),
                            Offered_Total=('Offered_Total', 'sum')
                        ).reset_index()
                        ln_lr_agg = ln_lr_agg[ln_lr_agg['Offered_Total'] > 0]
                        ln_lr_agg['Leftover Rate (%)'] = (ln_lr_agg['Left_Over_Total'] / ln_lr_agg['Offered_Total']) * 100
                        ln_leftover_chart = ln_lr_agg.sort_values(by='Leftover Rate (%)', ascending=True)
                        
                        chart_height = max(400, len(ln_leftover_chart) * 20)
                        fig_ln_leftover = px.bar(
                            ln_leftover_chart, x='Leftover Rate (%)', y='Name', orientation='h',
                            title='Lunch Items by Leftover Rates (Filtered)',
                            labels={'Name': 'Food Item'}, color='Leftover Rate (%)',
                            color_continuous_scale=px.colors.sequential.Reds, height=chart_height
                        )
                        st.plotly_chart(fig_ln_leftover, use_container_width=True)
                        with st.expander("📋 View Filtered Leftover Rate Data"):
                            st.dataframe(ln_leftover_chart.sort_values(by='Leftover Rate (%)', ascending=False))

                    with l_col2:
                        st.subheader("All Items by Net Consumption Rate")
                        # --- Calculate from filtered data ---
                        ln_nc_agg = ln_filtered.groupby('Name').agg(
                            Offered_Total=('Offered_Total', 'sum'),
                            Left_Over_Total=('Left_Over_Total', 'sum')
                        ).reset_index()
                        ln_nc_agg = ln_nc_agg[ln_nc_agg['Offered_Total'] > 0]
                        ln_nc_agg['Net Consumption'] = ln_nc_agg['Offered_Total'] - ln_nc_agg['Left_Over_Total']
                        ln_nc_agg['Net Consumption Rate (%)'] = (ln_nc_agg['Net Consumption'] / ln_nc_agg['Offered_Total']) * 100
                        ln_consumption_chart = ln_nc_agg.sort_values(by='Net Consumption Rate (%)', ascending=True)

                        chart_height = max(400, len(ln_consumption_chart) * 20)
                        fig_ln_consumption = px.bar(
                            ln_consumption_chart, x='Net Consumption Rate (%)', y='Name', orientation='h',
                            title='Most Consumed Lunch Items (Filtered)',
                            labels={'Name': 'Food Item'}, color='Net Consumption Rate (%)',
                            color_continuous_scale=px.colors.sequential.Greens, height=chart_height
                        )
                        st.plotly_chart(fig_ln_consumption, use_container_width=True)
                        with st.expander("📋 View Filtered Net Consumption Rate Data"):
                            st.dataframe(ln_consumption_chart.sort_values(by='Net Consumption Rate (%)', ascending=False))

                else:
                    # --- ALL SCHOOLS (Static County-wide Data) ---
                    st.markdown("Showing aggregate data for all schools (county-wide).")
                    with l_col1:
                        st.subheader("All Items by Leftover Rate")
                        ln_leftover_chart = ln_leftover.sort_values(by='Leftover Rate (%)', ascending=True)
                        chart_height = max(400, len(ln_leftover_chart) * 20)
                        fig_ln_leftover = px.bar(
                            ln_leftover_chart, x='Leftover Rate (%)', y='Item Name', orientation='h',
                            title='Lunch Items by Leftover Rates (County-Wide)',
                            labels={'Item Name': 'Food Item'}, color='Leftover Rate (%)',
                            color_continuous_scale=px.colors.sequential.Reds, height=chart_height
                        )
                        st.plotly_chart(fig_ln_leftover, use_container_width=True)
                        with st.expander("📋 View County-Wide Leftover Rate Data"):
                            st.dataframe(ln_leftover_chart.sort_values(by='Leftover Rate (%)', ascending=False))

                    with l_col2:
                        st.subheader("All Items by Net Consumption")
                        ln_consumption_chart = ln_consumption.sort_values(by='Net Consumption', ascending=True)
                        chart_height = max(400, len(ln_consumption_chart) * 20)
                        fig_ln_consumption = px.bar(
                            ln_consumption_chart, x='Net Consumption', y='Item Name', orientation='h',
                            title='Most Consumed Lunch Items (County-Wide)',
                            labels={'Item Name': 'Food Item'}, color='Net Consumption',
                            color_continuous_scale=px.colors.sequential.Greens, height=chart_height
                        )
                        st.plotly_chart(fig_ln_consumption, use_container_width=True)
                        with st.expander("📋 View County-Wide Net Consumption Data"):
                            st.dataframe(ln_consumption_chart.sort_values(by='Net Consumption', ascending=False))
            
            else:
                # --- SPECIFIC SCHOOL ---
                st.markdown(f"Showing data for {selected_school} only.")
                if ln_filtered.empty:
                    st.warning("No lunch data found for this school.")
                else:
                    l_col1, l_col2 = st.columns(2)
                    
                    with l_col1:
                        st.subheader("All Items by Leftover Rate")
                        ln_lr_school_data = ln_lr_school[ln_lr_school['school_name'] == selected_school]
                        ln_lr_chart = ln_lr_school_data.sort_values(by='leftover_rate', ascending=True)
                        
                        if not ln_lr_chart.empty:
                            chart_height = max(400, len(ln_lr_chart) * 20)
                            fig_ln_school_leftover = px.bar(
                                ln_lr_chart, x='leftover_rate', y='name', orientation='h',
                                title='Lunch Items by Leftover Rates',
                                labels={'name': 'Food Item', 'leftover_rate': 'Leftover Rate (%)'}, 
                                color='leftover_rate',
                                color_continuous_scale=px.colors.sequential.Reds, height=chart_height
                            )
                            st.plotly_chart(fig_ln_school_leftover, use_container_width=True)
                        else:
                            st.info("No pre-computed leftover rate data found for this school.")
                        
                        with st.expander("📋 View Leftover Rate Data"):
                            st.dataframe(ln_lr_chart.sort_values(by='leftover_rate', ascending=False))
                    
                    with l_col2:
                        st.subheader("All Items by Net Consumption Rate")
                        ln_nc_school_data = ln_nc_school[ln_nc_school['school_name'] == selected_school]
                        ln_nc_chart = ln_nc_school_data.sort_values(by='net_consumption_rate', ascending=True)
                        
                        if not ln_nc_chart.empty:
                            chart_height = max(400, len(ln_nc_chart) * 20)
                            fig_ln_school_consumption = px.bar(
                                ln_nc_chart, x='net_consumption_rate', y='name', orientation='h',
                                title='Lunch Items with Highest Consumption Rates',
                                labels={'name': 'Food Item', 'net_consumption_rate': 'Net Consumption Rate (%)'}, 
                                color='net_consumption_rate',
                                color_continuous_scale=px.colors.sequential.Greens, height=chart_height
                            )
                            st.plotly_chart(fig_ln_school_consumption, use_container_width=True)
                        else:
                            st.info("No pre-computed net consumption data found for this school.")
                        
                        with st.expander("📋 View Net Consumption Rate Data"):
                            st.dataframe(ln_nc_chart.sort_values(by='net_consumption_rate', ascending=False))
                
                with st.expander("📋 View Raw Filtered Lunch Data"):
                    st.dataframe(ln_filtered)

        # Handle case where no meal period is selected
        if selected_meal_period not in ["Breakfast", "Lunch", "Overall"]:
            st.info("Select a meal period (Breakfast, Lunch, or Overall) from the sidebar to see popularity data.")
                
    with tab_opt:
        st.header("Optimization Recommendations")

        # A specific school is selected
        if selected_school != "All Schools":
            st.subheader(f"Recommendations for: {selected_school}")
            
            # Filter optimization data for the selected school
            school_opt_data = opt_data[opt_data['School_Name'] == selected_school].copy()

            if school_opt_data.empty:
                st.warning("No optimization data found for this school.")
            else:
                # Calculate and display total items
                total_items = int(school_opt_data['recommended_quantity'].sum())
                st.metric("Total Recommended Weekly Items", f"{total_items:,}")

                # Create sub-category breakdown
                st.subheader("Breakdown by Sub-Category")
                subcat_totals = school_opt_data.groupby('Sub-Category')['recommended_quantity'].sum().reset_index().sort_values(by='recommended_quantity', ascending=False)
                
                # --- Create columns for charts ---
                c1, c2 = st.columns(2)
                
                with c1:
                    fig_opt_subcat = px.bar(
                        subcat_totals,
                        x='recommended_quantity',
                        y='Sub-Category',
                        orientation='h',
                        title=f"Sub-Category Breakdown for {selected_school}",
                        labels={'recommended_quantity': 'Recommended Weekly Quantity', 'Sub-Category': 'Sub-Category'}
                    )
                    fig_opt_subcat.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_opt_subcat, use_container_width=True)
                
                with c2:
                    fig_opt_pie = px.pie(
                        subcat_totals,
                        values='recommended_quantity',
                        names='Sub-Category',
                        title=f"Sub-Category Proportions for {selected_school}"
                    )
                    fig_opt_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_opt_pie, use_container_width=True)

                st.markdown("---") # Add a separator
                st.subheader("Recommended Item List")
                
                # Prepare the table for display
                display_table = school_opt_data[['food_item', 'Sub-Category', 'recommended_quantity']].copy()
                display_table.rename(columns={
                    'food_item': 'Food Item',
                    'Sub-Category': 'Category',
                    'recommended_quantity': 'Recommended Weekly Quantity'
                }, inplace=True)
                
                # Sort by quantity for readability
                display_table = display_table.sort_values(by='Recommended Weekly Quantity', ascending=False)
                
                # Use st.dataframe to display
                st.dataframe(
                    display_table,
                    use_container_width=True,
                    hide_index=True # Hides the pandas index for a cleaner look
                )

        # "All Schools" is selected (aggregate view)
        else:
            # Start with a copy of all optimization data
            filtered_opt_data = opt_data.copy()
            
            # Build the title based on filters
            title = "Aggregated Recommendations"
            filters_applied = []
            if selected_region != "All Regions":
                filtered_opt_data = filtered_opt_data[filtered_opt_data['FCPS Region'] == selected_region]
                filters_applied.append(selected_region)
            if selected_dk != "All Distribution Kitchens":
                filtered_opt_data = filtered_opt_data[filtered_opt_data['Distribution Kitchen (DK)'] == selected_dk]
                filters_applied.append(selected_dk)
            if selected_level != "All Levels":
                filtered_opt_data = filtered_opt_data[filtered_opt_data['Educational Level'] == selected_level]
                filters_applied.append(selected_level)

            if filters_applied:
                title = f"Aggregated Recommendations for: {', '.join(filters_applied)}"
            else:
                title = "County-Wide Recommendations"
            
            st.subheader(title)

            if filtered_opt_data.empty:
                st.warning("No optimization data found for the selected filters.")
            else:
                # Calculate and display total items
                total_items = int(filtered_opt_data['recommended_quantity'].sum())
                st.metric("Total Recommended Weekly Items", f"{total_items:,}")

                # Create sub-category breakdown
                st.subheader("Breakdown by Sub-Category")
                subcat_totals = filtered_opt_data.groupby('Sub-Category')['recommended_quantity'].sum().reset_index().sort_values(by='recommended_quantity', ascending=False)
                
                # --- Create columns for charts ---
                c1, c2 = st.columns(2)

                with c1:
                    fig_opt_subcat = px.bar(
                        subcat_totals,
                        x='recommended_quantity',
                        y='Sub-Category',
                        orientation='h',
                        title=f"Sub-Category Breakdown (Aggregate)",
                        labels={'recommended_quantity': 'Recommended Weekly Quantity', 'Sub-Category': 'Sub-Category'}
                    )
                    fig_opt_subcat.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_opt_subcat, use_container_width=True)

                with c2:
                    fig_opt_pie = px.pie(
                        subcat_totals,
                        values='recommended_quantity',
                        names='Sub-Category',
                        title="Sub-Category Proportions (Aggregate)"
                    )
                    fig_opt_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_opt_pie, use_container_width=True)

    with tab_reg:
        st.header("Regression Analysis")
        st.markdown("Content to be added here.")
        pass

    with tab_sav:
        st.header("Savings/Loss from Optimization")
        st.markdown("Content to be added here.")
        pass

else:
    st.warning("⚠️ Could not load data. Please check file paths and availability.")

