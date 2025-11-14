import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re
from datetime import datetime
from pathlib import Path
import traceback
from PIL import Image 

# ==================================================================
# --- CORRECTED FILE PATHS ---
# ==================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "src" / "data"
OPTIMIZATION_DATA_DIR = DATA_DIR / "optimization-data"
CLEANED_DATA_DIR = DATA_DIR / "clean-data"
RESULTS_DIR = DATA_DIR / "results"
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed-data'

# --- NEW PIPELINE FILES ---
#OPTIMIZATION_FILE = OPTIMIZATION_DATA_DIR / 'monthly_items_breakdown_baseline.csv' 
#FINANCIAL_FILE = OPTIMIZATION_DATA_DIR / 'annual_school_breakdown_baseline.csv'

# --- HISTORICAL COST & ID MAPPING FILES ---
HISTORICAL_BF_FILE = CLEANED_DATA_DIR / "data_breakfast.csv"
HISTORICAL_LN_FILE = CLEANED_DATA_DIR / "data_lunch.csv"

# --- COST FILE ---
COST_FILE = PREPROCESSED_DATA_DIR / 'unit_costs.csv' 

# Logo path
LOGO_FILE = PROJECT_ROOT / 'demo' / 'images' / 'fcps_logo2.png'
# ==================================================================


def generate_pdf(school_to_test, scenario_suffix, scenario_name, meal_type='Both'):
    """
    Generates the food order recommendation PDF for a given school.
    Reads from the NEW optimization pipeline output files.
    """
    try:
        # Load All NEW Data
        opt_file_path = OPTIMIZATION_DATA_DIR / f'monthly_items_breakdown{scenario_suffix}.csv'
        fin_file_path = OPTIMIZATION_DATA_DIR / f'annual_school_breakdown{scenario_suffix}.csv'

        # Load All NEW Data
        df_opt = pd.read_csv(opt_file_path)
        df_financial = pd.read_csv(fin_file_path)
        df_costs = pd.read_csv(COST_FILE)
        
        # Load historical data for COST calculation
        # We need the 'name' column to map identifiers
        hist_cols = ['school_name', 'production_cost_total', 'name', 'identifier']
        df_breakfast = pd.read_csv(HISTORICAL_BF_FILE, low_memory=False, usecols=hist_cols, encoding='latin-1')
        df_lunch = pd.read_csv(HISTORICAL_LN_FILE, low_memory=False, usecols=hist_cols, encoding='latin-1')
        
        # --- Data Cleaning ---
        # Clean financial data
        for col in ['proportional_annual_budget', 'annual_food_cost', 'remaining_annual_balance']:
            if col in df_financial.columns:
                df_financial[col] = (
                    df_financial[col].astype(str)
                                     .str.replace(r'[$,]', '', regex=True)
                                     .str.strip()
                )
                df_financial[col] = pd.to_numeric(df_financial[col], errors='coerce').fillna(0)
        
        # Clean historical cost data
        for df in [df_breakfast, df_lunch]:
            df['school_name'] = df['school_name'].astype(str).str.lower().str.strip()
            if 'production_cost_total' in df.columns:
                df['production_cost_total'] = (
                    df['production_cost_total'].astype(str)
                                             .str.replace(r'[$,]', '', regex=True)
                                             .str.strip()
                )
                df['production_cost_total'] = pd.to_numeric(
                    df['production_cost_total'], errors='coerce'
                ).fillna(0)

        # --- Get Specific Info for the Selected School ---
        school_clean_name = school_to_test.lower().strip()
        
        # Get financial data
        try:
            school_financial_data = df_financial[
                df_financial['school'].str.lower() == school_clean_name
            ].iloc[0]
            
            financial_data = {
                "Allocated Annual Budget": school_financial_data['proportional_annual_budget'],
                "Optimized Annual Food Cost": school_financial_data['annual_food_cost'],
                "Remaining Budget Balance": school_financial_data['remaining_annual_balance']
            }
        except IndexError:
            print(f"Error: No financial data found for school '{school_to_test}'")
            return None

        # Get historical cost
        bf_costs = df_breakfast.groupby('school_name')['production_cost_total'].sum()
        ln_costs = df_lunch.groupby('school_name')['production_cost_total'].sum()
        total_costs_dict = (bf_costs.add(ln_costs, fill_value=0) * 10).to_dict()
        school_actual_cost = total_costs_dict.get(school_clean_name, 0)
        
        # Calculate savings
        total_savings = school_actual_cost - financial_data["Optimized Annual Food Cost"]
        percent_savings = (total_savings / school_actual_cost) * 100 if school_actual_cost > 0 else 0

        savings_data = {
            "Actual Historical Cost": school_actual_cost,
            "Total Annual Savings": total_savings,
            "Savings Percentage": f"{percent_savings:.2f}%"
        }

        # ==================================================================
        # --- FINAL FIX: Direct Name-to-Cost Mapping (No Identifiers) ---
        # ==================================================================
        
        def clean_name_series(name_series):
            """Applies a consistent set of cleaning rules to a pandas Series."""
            # 1. Convert to string, lowercase, and strip whitespace
            cleaned = name_series.astype(str).str.strip().str.lower()
            # 2. Standardize quotes and dashes
            cleaned = cleaned.str.replace("’", "'").str.replace('”', '"')
            cleaned = cleaned.str.replace("–", "-").str.replace("—", "-")
            # 3. Normalize whitespace: remove space *before* parentheses
            cleaned = cleaned.str.replace(r'\s+\(', '(', regex=True)
            # 4. Normalize all other whitespace to a single space
            cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
            return cleaned.str.strip() # Final strip

        # 1. Create ONE simple map from 'unit_costs.csv'
        #    This maps a CLEAN NAME -> SPECIFIC UNIT COST
        df_costs['unit_cost'] = pd.to_numeric(df_costs['unit_cost'], errors='coerce').fillna(0)
        df_costs['name_clean'] = clean_name_series(df_costs['name'])
        
        # We group by the clean name and take the first cost.
        # This gives us the specific cost for each unique item name.
        cost_map_specific = df_costs.groupby('name_clean')['unit_cost'].first()

        # 2. Filter optimization data for the school
        df_school = df_opt[
            df_opt['school'].str.lower() == school_clean_name
        ].copy()
        
        if meal_type in ['Breakfast', 'Lunch']:
            df_school = df_school[df_school['meal_type'].str.lower() == meal_type.lower()]

        if df_school.empty:
            print(f"Error: No optimization data found for school '{school_to_test}' and meal type '{meal_type}'")
            return None
            
        # 3. Map costs using the simple, direct map
        
        #    Step 3a: Apply the SAME cleaning function to the optimization file names
        df_school['food_item_clean'] = clean_name_series(df_school['food_item'])

        #    Step 3b: Map the clean name directly to its specific cost
        df_school['Cost'] = df_school['food_item_clean'].map(cost_map_specific)
        
        # 4. Debugging: Check for items that are TRULY unmatched (NaN)
        #    (This is better than checking for == 0)
        unmatched_items = df_school[
            df_school['Cost'].isna()
        ]['food_item_clean'].unique()
        
        if len(unmatched_items) > 0:
            print("="*20 + " DEBUG: UNMATCHED ITEMS " + "="*20)
            print(f"Found {len(unmatched_items)} items that could not be matched:")
            for item in unmatched_items[:20]: # Print top 20
                print(f"  - {item}")
            print("="*60)

        # 5. Final Calculation and Table Prep
        df_school['Cost'] = df_school['Cost'].fillna(0) # Now fill NaNs with 0
        df_school['Total'] = df_school['recommended_quantity'] * df_school['Cost']
        
        df_final_report = df_school[
            ['meal_type', 'food_item', 'Cost', 'recommended_quantity', 'Total']
        ].copy()
        df_final_report.columns = ['Meal Type', 'Food Item', 'Cost', 'Qty', 'Total']
        
        # ==================================================================

        # ==================================================================
        # --- PDF GENERATION CLASS ---
        # ==================================================================
        class PDF(FPDF):
            def __init__(self, orientation='P', unit='mm', format='Letter', school_name='N/A'):
                super().__init__(orientation, unit, format)
                self.school_name = school_name
                self.set_auto_page_break(auto=True, margin=15)

            def header(self):
                # Suppress header for the first page
                if self.page_no() == 1:
                    return

                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'School Food Optimization Report', 0, 1, 'C')

                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

            def chapter_title(self, title):
                self.set_font('Arial', 'B', 14)
                self.cell(0, 10, title, 0, 1, 'C')
                self.ln(5)

            def add_metric_table(self, title, data_dict):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, title, 0, 1, 'L')
                self.ln(2)

                # Draw the table
                self.set_font('Arial', '', 12)
                self.set_line_width(0.5)
                self.set_draw_color(150, 150, 150) # Light gray border

                # Calculate total height for the box
                total_height = 0
                line_height = 10
                for value in data_dict.values():
                    if value == "---":
                        total_height += 4 # Space for separator
                    else:
                        total_height += line_height

                # Draw outer box
                start_x = self.get_x()
                start_y = self.get_y()
                self.cell(0, total_height, '', 1, 1, 'L')
                self.set_xy(start_x, start_y)

                col_width = 90

                for name, value in data_dict.items():
                    # --- Handle Separator ---
                    if value == "---":
                        self.set_font('Arial', '', 4)
                        self.set_x(start_x + 5) # Indent the line
                        self.cell(self.w - start_x*2 - 10, 2, '', 'T', 1, 'L') 
                        self.ln(2)
                        continue # Skip the rest of the loop

                    self.set_font('Arial', 'B', 12)
                    self.set_x(start_x + 5) # Add padding

                    if isinstance(value, (int, float)):
                        value_str = f"${value:,.2f}"
                        is_positive = value >= 0
                    else:
                        value_str = str(value)
                        is_positive = "Savings" in name and "%" in value_str and not value_str.startswith('-')

                    self.cell(col_width, line_height, name, 0, 0, 'L')

                    if is_positive:
                        self.set_text_color(34, 139, 34)  # Green
                    else:
                        self.set_text_color(220, 20, 60)   # Red

                    self.set_font('Arial', '', 12)

                    # Set X for the value, right-aligned
                    value_x = start_x + self.w - self.l_margin - self.r_margin - 5
                    self.set_x(value_x) 

                    self.cell(0, line_height, value_str, 0, 1, 'R')
                    self.set_text_color(0, 0, 0) # Reset

                self.ln(5)

            def add_item_table(self, data, optimized_monthly_cost):
                self.set_font('Arial', 'B', 10)
                self.set_fill_color(220, 220, 220)
                
                # Headers
                col_widths = {'Meal Type': 25, 'Food Item': 95, 'Cost': 20, 'Qty': 20, 'Total': 30}
                self.cell(col_widths['Meal Type'], 7, "Meal Type", 1, 0, 'C', fill=True)
                self.cell(col_widths['Food Item'], 7, "Food Item", 1, 0, 'C', fill=True)
                self.cell(col_widths['Cost'], 7, "Cost", 1, 0, 'C', fill=True)
                self.cell(col_widths['Qty'], 7, "Qty", 1, 0, 'C', fill=True)
                self.cell(col_widths['Total'], 7, "Total", 1, 1, 'C', fill=True)
                
                # Set base line height
                line_h = 6 
                self.set_font('Arial', '', 9)

                for _, row in data.iterrows():
                    # --- 1. Prepare cell content ---
                    item_name = str(row['Food Item']).replace("’", "'").replace("–", "-").replace("—", "-")
                    meal_type_str = str(row['Meal Type'])
                    cost_str = f"${row['Cost']:.2f}"
                    qty_str = str(row['Qty'])
                    total_str = f"${row['Total']:.2f}"

                    # --- 2. Calculate max height of row ---
                    # Use split_only=True to find how many lines the text needs
                    item_lines = self.multi_cell(col_widths['Food Item'], line_h, item_name, 0, 'L', split_only=True)
                    num_lines = len(item_lines)
                    
                    # Also check meal_type_str, just in case
                    meal_lines = self.multi_cell(col_widths['Meal Type'], line_h, meal_type_str, 0, 'L', split_only=True)
                    num_lines = max(num_lines, len(meal_lines))

                    # Row height is number of lines * line height (min 1 line)
                    row_height = max(line_h, num_lines * line_h)

                    # --- 3. Check for page break ---
                    if self.get_y() + row_height > self.h - 15:
                        self.add_page()
                        self.set_font('Arial', 'B', 10)
                        # Redraw headers
                        self.cell(col_widths['Meal Type'], 7, "Meal Type", 1, 0, 'C', fill=True)
                        self.cell(col_widths['Food Item'], 7, "Food Item", 1, 0, 'C', fill=True)
                        self.cell(col_widths['Cost'], 7, "Cost", 1, 0, 'C', fill=True)
                        self.cell(col_widths['Qty'], 7, "Qty", 1, 0, 'C', fill=True)
                        self.cell(col_widths['Total'], 7, "Total", 1, 1, 'C', fill=True)
                        self.set_font('Arial', '', 9)

                    # --- 4. Draw all cells, managing X/Y manually ---
                    start_y = self.get_y()
                    start_x = self.get_x()

                    # --- FIX ---
                    # Draw the "Meal Type" cell with the full row_height
                    self.cell(col_widths['Meal Type'], row_height, meal_type_str, 1, 0, 'L')
                    
                    # Record the x-position for the *next* cell
                    food_item_x = self.get_x() 
                    
                    # Draw the single-line cells from right-to-left
                    # Set X to the start of the 'Total' column
                    self.set_xy(start_x + col_widths['Meal Type'] + col_widths['Food Item'] + col_widths['Cost'] + col_widths['Qty'], start_y)
                    self.cell(col_widths['Total'], row_height, total_str, 1, 0, 'R')
                    
                    self.set_xy(start_x + col_widths['Meal Type'] + col_widths['Food Item'] + col_widths['Cost'], start_y)
                    self.cell(col_widths['Qty'], row_height, qty_str, 1, 0, 'R')
                    
                    self.set_xy(start_x + col_widths['Meal Type'] + col_widths['Food Item'], start_y)
                    self.cell(col_widths['Cost'], row_height, cost_str, 1, 0, 'R')

                    # Now, draw the wrapping "Food Item" cell in its reserved space
                    self.set_xy(food_item_x, start_y)
                    
                    # --- FIX ---
                    # Use multi_cell for wrapping, and add the border '1'
                    self.multi_cell(col_widths['Food Item'], line_h, item_name, 1, 'L')
                    
                    # Move cursor to the next line, aligned with the bottom of the tallest cell
                    self.set_y(start_y + row_height)
                
                # Footer rows for Subtotal, Expenses, and Total
                self.set_font('Arial', 'B', 10)

                # Get the total cost from the optimization file
                total_cost = optimized_monthly_cost 

                # Define widths for labels and values
                label_width = col_widths['Meal Type'] + col_widths['Food Item'] + col_widths['Cost'] + col_widths['Qty']
                value_width = col_widths['Total']

                # Final "Total Production Cost" row
                self.set_font('Arial', 'B', 10) # Bold for the final total
                self.cell(label_width, 7, "Total Production Cost", 1, 0, 'R', fill=True)
                self.cell(value_width, 7, f"${total_cost:,.2f}", 1, 1, 'R', fill=True)

        # --- Generate PDF ---
        pdf = PDF('P', 'mm', 'Letter', school_to_test)
        
        # Page 1: Title & Summary
        pdf.add_page()

        # --- 1. Add Logo to Top Right ---
        try:
            if LOGO_FILE.exists():
                page_width = pdf.w
                logo_width, right_margin = 30, 10
                logo_x = page_width - logo_width - right_margin
                pdf.image(str(LOGO_FILE), x=logo_x, y=8, w=logo_width)
        except Exception as e: 
            print(f"Error adding logo to Page 1: {e}.")

        # --- 2. Add Left-Aligned Title Block ---
        pdf.set_y(20) # Move down to start content

        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 15, 'Optimization Recommendation', 0, 1, 'L') # Left align

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f"School: {school_to_test.title()}", 0, 1, 'L')

        pdf.set_font('Arial', 'I', 14)
        pdf.cell(0, 10, f"Scenario: {scenario_name}", 0, 1, 'L')
        pdf.ln(10) # Add space before the table

        # --- 3. Combine financial data into one dict ---
        combined_financial_data = {
            "Allocated Annual Budget": financial_data["Allocated Annual Budget"],
            "Optimized Annual Food Cost": financial_data["Optimized Annual Food Cost"],
            "Remaining Budget Balance": financial_data["Remaining Budget Balance"],
            "---": "---", # Simple separator
            "Actual Historical Cost": savings_data["Actual Historical Cost"],
            "Total Annual Savings": savings_data["Total Annual Savings"],
            "Savings Percentage": savings_data["Savings Percentage"]
        }

        # --- 4. Call the metric table function ONCE ---
        pdf.add_metric_table("Financial Summary", combined_financial_data)

        # Page 2: Item Breakdown
        pdf.add_page()
        pdf.chapter_title(f"Recommended Monthly Production for {school_to_test.title()}")

        # Calculate the monthly target cost from the annual data
        opt_monthly_cost = financial_data["Optimized Annual Food Cost"] / 10

        pdf.add_item_table(df_final_report, opt_monthly_cost)

        # Return as bytes
        return pdf.output(dest='S')
    
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")
        traceback.print_exc()
        return None