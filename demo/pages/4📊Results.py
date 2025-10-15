import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from pathlib import Path
import base64 # Used for image encoding

# Set page config
st.set_page_config(
    page_title="FCPS Meal Production Analysis",
    page_icon="📊",
    layout="wide"
)

# Define the container for the widgets first. The CSS will style this container.
nav_container = st.container()
with nav_container:
    col1, col2 = st.columns([1, 2])
    with col1:
        meal_type = st.radio(
            "🍽️ Meal Program",
            options=["Breakfast 🍳", "Lunch 🥪"],
            index=0,
            horizontal=True
        )

    @st.cache_data
    def load_optimization_data():
        base_path = Path.cwd().parent.parent / 'src' / 'data' / 'preprocessed-data'
        file_path = base_path / 'school_food_item_optimization.csv'
        df = pd.read_csv(file_path)
        return df
    optimization_df = load_optimization_data()
    school_list = sorted(optimization_df['school'].unique())
    view_options = ["Overall Dashboard"] + [school.title() for school in school_list]

    with col2:
        selected_view = st.selectbox(
            "🏫 Change View",
            options=view_options
        )

# This CSS makes the container a sticky nav bar and styles the image/text overlay.
st.markdown(
    """
    <style>
        /* This selector specifically targets the container holding our nav widgets */
        .main .block-container > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {
            position: fixed;
            top: 2.8rem;
            left: 0;
            width: 100%;
            background-color: #07677F;
            padding: 20px 2rem;
            z-index: 999;
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
        }

        .main .block-container {
            padding-top: 9rem;
        }
        
        .main .block-container > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) * {
            color: white;
        }
        .main .block-container > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) .stRadio > label,
        .main .block-container > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) .stSelectbox > label {
            color: white;
            font-weight: bold;
        }
        .main .block-container > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) div[data-baseweb="select"] > div {
            background-color: #0B80A0;
            color: white;
        }

        .image-container {
            position: relative;
            width: 100%;
            margin-bottom: 20px;
        }
        .image-overlay-text {
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 10;
        }
        .image-overlay-text h2, .image-overlay-text p {
            color: white !important; 
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }
        .image-overlay-text h1 {
            font-size: 3.5rem;
            margin: 0;
            padding: 0;
        }
        .image-overlay-text p {
            font-size: 1.2rem;
            margin: 0;
            padding: 0;
        }
        .image-overlay-text-right {
            position: absolute;
            bottom: 20px;
            right: 20px;
            text-align: right;
            z-index: 10;
        }
        .image-overlay-text-left h1, .image-overlay-text-left p,
        .image-overlay-text-right h2 {
            color: white !important; 
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }
        .image-overlay-text-left h1 { font-size: 3.5rem; margin: 0; padding: 0; }
        .image-overlay-text-left p { font-size: 1.2rem; margin: 0; padding: 0; }
        .image-overlay-text-right h2 { font-size: 1.8rem; margin: 0; padding: 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DATA LOADING FUNCTION (for main content) ---
@st.cache_data
def load_meal_data(meal_program):
    base_path = Path.cwd().parent.parent / 'src' / 'data' / 'preprocessed-data'
    file_name = 'data_breakfast_with_coordinates.csv' if "Breakfast" in meal_program else 'data_lunch_with_coordinates.csv'
    file_path = base_path / file_name
    df = pd.read_csv(file_path)
    cost_columns = ['Production_Cost_Total', 'Left_Over_Cost']
    for col in cost_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)
    if 'Left_Over_Percent_of_Offered' in df.columns and df['Left_Over_Percent_of_Offered'].dtype == 'object':
        df['Left_Over_Percent_of_Offered'] = df['Left_Over_Percent_of_Offered'].str.replace('%', '', regex=False).astype(float)
    return df

data = load_meal_data(meal_type)

# --- CONDITIONAL PAGE DISPLAY ---

if selected_view != "Overall Dashboard":
    st.title("📊 FCPS Meal Production Analysis")
    st.header(f"Optimized Production for: {selected_view}")
    
    selected_school_lower = selected_view.lower()
    meal_type_simple = "Breakfast" if "Breakfast" in meal_type else "Lunch"
    school_specific_data = optimization_df[
        (optimization_df['school'] == selected_school_lower) &
        (optimization_df['meal_type'] == meal_type_simple)
    ]

    st.subheader(f"Recommended Monthly Quantities for {meal_type_simple}")
    if not school_specific_data.empty:
        display_data = school_specific_data[['food_item', 'recommended_quantity']].rename(
            columns={'food_item': 'Food Item', 'recommended_quantity': 'Recommended Quantity'}
        )
        st.dataframe(display_data, use_container_width=True, hide_index=True, height=600)
    else:
        st.warning(f"No optimization data available for {selected_view} for the selected meal program.")
else:
    image_path = Path.cwd().parent / 'images' / 'Dashboard Image.JPG'
    
    # Encode the image to base64 for direct embedding in HTML
    import base64
    def get_image_as_base64(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    img_base64 = get_image_as_base64(image_path)

    st.markdown(
        f"""
        <div class="image-container">
            <img src="data:image/jpeg;base64,{img_base64}" alt="Dashboard Header" style="width:100%; height:auto; display:block;">
            <div class="image-overlay-text">
                <h2>FCPS Meal Production Analysis</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("An exploratory data analysis of the FCPS meal production data.")

    # --- Key Metrics ---
    st.header("Key Metrics for May 2025")
    total_production_cost = data['Production_Cost_Total'].sum()
    total_waste_cost = data['Left_Over_Cost'].sum()
    overall_waste_rate = (total_waste_cost / total_production_cost) * 100 if total_production_cost > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Production Cost 💰", f"${total_production_cost:,.2f}")
    col2.metric("Total Waste Cost 🗑️", f"${total_waste_cost:,.2f}")
    col3.metric("Overall Waste Rate 📉", f"{overall_waste_rate:.2f}%")

    # --- Data Analysis & Visualization ---
    st.header("Data Analysis & Visualization")
    st.subheader("Top 10 Items by Production Cost")
    top_10_production = data.groupby('Name')['Production_Cost_Total'].sum().nlargest(10).reset_index()
    fig_prod = px.bar(top_10_production, x='Production_Cost_Total', y='Name', orientation='h', title='Top 10 Items by Production Cost', labels={'Name': 'Item', 'Production_Cost_Total': 'Total Production Cost ($)'})
    st.plotly_chart(fig_prod, use_container_width=True)

    st.subheader("Top 10 Items by Waste Cost")
    top_10_waste = data.groupby('Name')['Left_Over_Cost'].sum().nlargest(10).reset_index()
    fig_waste = px.bar(top_10_waste, x='Left_Over_Cost', y='Name', orientation='h', title='Top 10 Items by Waste Cost', labels={'Name': 'Item', 'Left_Over_Cost': 'Total Waste Cost ($)'})
    st.plotly_chart(fig_waste, use_container_width=True)

    st.subheader("Top 10 Most Wasted Items (%)")
    top_10_waste_percent = data.groupby('Name')['Left_Over_Percent_of_Offered'].mean().nlargest(10).reset_index()
    fig_waste_percent = px.bar(top_10_waste_percent, x='Left_Over_Percent_of_Offered', y='Name', orientation='h', title='Top 10 Most Wasted Items (by % of offered)', labels={'Name': 'Item', 'Left_Over_Percent_of_Offered': 'Average Waste Rate (%)'})
    st.plotly_chart(fig_waste_percent, use_container_width=True)

    # --- Footer ---
    st.markdown("---")

"""
st.tab
"""