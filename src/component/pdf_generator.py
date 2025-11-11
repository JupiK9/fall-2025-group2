import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re
from datetime import datetime
from pathlib import Path
import traceback

# Define paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = SCRIPT_DIR.parent / 'data'
PREPROCESSED_DATA_DIR = BASE_DATA_DIR / 'preprocessed-data'

OPTIMIZATION_FILE = BASE_DATA_DIR / 'school_food_item_optimization.csv'
COST_FILE = BASE_DATA_DIR / 'unit_costs.csv'
COORDINATES_FILE = PREPROCESSED_DATA_DIR / 'data_breakfast_with_coordinates.csv'
LOGO_FILE = SCRIPT_DIR.parent.parent / 'demo' / 'fcps_logo2.png'
BREAKFAST_FILE = PREPROCESSED_DATA_DIR / 'breakfast_combined.csv'
LUNCH_FILE = PREPROCESSED_DATA_DIR / 'lunch_combined.csv'


def generate_pdf(school_to_test, meal_type='Both'):
    """
    Generates the food order recommendation PDF for a given school and meal type.

    Args:
        school_to_test (str): The name of the school.
        meal_type (str): The meal type to filter for ('Breakfast', 'Lunch', or 'Both').

    Returns:
        bytes: The generated PDF content as bytes, or None if an error occurs.
    """
    try:
        # Load All Data
        df_opt = pd.read_csv(OPTIMIZATION_FILE)
        df_costs = pd.read_csv(COST_FILE)
        df_coords = pd.read_csv(COORDINATES_FILE)
        df_breakfast = pd.read_csv(BREAKFAST_FILE, usecols=['Name', 'Identifier'], encoding='latin-1', dtype={'Identifier': str})
        df_lunch = pd.read_csv(LUNCH_FILE, usecols=['Name', 'Identifier'], encoding='latin-1', dtype={'Identifier': str})

        # Filter by Meal Type
        if meal_type in ['Breakfast', 'Lunch']:
            df_opt = df_opt[df_opt['meal_type'].str.lower() == meal_type.lower()]

        # Create Identifier and Averaged Cost Lookup Tables
        df_ids = pd.concat([df_breakfast, df_lunch])
        df_ids['food_item_clean'] = df_ids['Name'].str.strip().str.lower()
        df_ids['Identifier'] = df_ids['Identifier'].astype(str)
        df_ids = df_ids.drop_duplicates(subset=['food_item_clean'])
        df_ids = df_ids[['food_item_clean', 'Identifier']]

        id_override_map = {'fat free white milk (each)': '20001'}
        def apply_override(row):
            return id_override_map.get(row['food_item_clean'], row['Identifier'])
        df_ids['Identifier'] = df_ids.apply(apply_override, axis=1)
        df_ids_cost_map = df_ids.set_index('food_item_clean')['Identifier']

        df_costs['Name_clean'] = df_costs['Name'].str.strip().str.lower()
        df_costs['Identifier'] = df_costs['Name_clean'].map(df_ids_cost_map)
        df_costs_with_ids = df_costs.dropna(subset=['Identifier'])
        df_costs_avg = df_costs_with_ids.groupby('Identifier', as_index=False)['Unit_Cost'].mean()

        # Create a School Information Lookup Table
        school_info_cols = ['Normalized_School_Name_excel', 'address']
        df_school_info = df_coords[school_info_cols].copy()
        df_school_info.columns = ['school_clean', 'address']
        df_school_info = df_school_info.drop_duplicates(subset=['school_clean']).set_index('school_clean')

        # Clean and Prepare Data
        def normalize_school_for_lookup(name):
            if not isinstance(name, str): return name
            name = re.sub(r'\s+', ' ', name).strip().lower()
            replacements = {' elementary': ' es', ' middle': ' ms', ' high': ' hs', ' secondary': ' ss', ' school': ''}
            for old, new in replacements.items(): name = name.replace(old, new)
            return name

        df_opt['school_clean'] = df_opt['school'].apply(normalize_school_for_lookup)
        df_opt['food_item_clean'] = df_opt['food_item'].str.strip().str.lower()

        # Get Specific Info for the Selected School
        school_clean_name = normalize_school_for_lookup(school_to_test)
        try:
            school_address = df_school_info.loc[school_clean_name, 'address']
        except KeyError:
            school_address = "Address Not Found"

        # Filter, Aggregate, and Merge
        df_school = df_opt[df_opt['school_clean'] == school_clean_name].copy()
        if df_school.empty:
            print(f"Error: No optimization data found for school '{school_to_test}' and meal type '{meal_type}'")
            return None

        df_school_with_ids = pd.merge(df_school, df_ids, on='food_item_clean', how='left')
        df_mapped = df_school_with_ids.dropna(subset=['Identifier'])
        df_unmapped = df_school_with_ids[df_school_with_ids['Identifier'].isna()]

        id_variation_counts = df_mapped.groupby('Identifier')['food_item_clean'].nunique()
        df_agg_mapped = df_mapped.groupby('Identifier', as_index=False).agg(
            food_item=('food_item', 'first'), recommended_quantity=('recommended_quantity', 'sum')
        )
        df_agg_mapped = pd.merge(df_agg_mapped, id_variation_counts.rename('variation_count'), on='Identifier')
        def clean_name_conditionally(row):
            if row['variation_count'] > 1: return re.sub(r'\s*\(.*\)', '', row['food_item']).strip()
            else: return row['food_item']
        df_agg_mapped['food_item'] = df_agg_mapped.apply(clean_name_conditionally, axis=1)
        df_agg_mapped = df_agg_mapped.drop(columns=['variation_count'])

        df_agg_unmapped = df_unmapped.groupby('food_item_clean', as_index=False).agg(
             food_item=('food_item', 'first'), recommended_quantity=('recommended_quantity', 'sum'), Identifier=('Identifier', 'first')
        )
        df_school_agg = pd.concat([df_agg_mapped, df_agg_unmapped], ignore_index=True)
        df_order_list = pd.merge(df_school_agg, df_costs_avg, on='Identifier', how='left')
        df_order_list['Identifier'] = df_order_list['Identifier'].fillna('N/A')

        # Calculate Total Cost & Format
        df_order_list['Unit_Cost'] = df_order_list['Unit_Cost'].fillna(0)
        df_order_list['Total_Cost'] = df_order_list['recommended_quantity'] * df_order_list['Unit_Cost']
        df_final_report = df_order_list[['Identifier', 'food_item', 'Unit_Cost', 'recommended_quantity', 'Total_Cost']]
        df_final_report = df_final_report.rename(columns={
            'Identifier': 'ID', 'food_item': 'Food Item', 'Unit_Cost': 'Cost', 'recommended_quantity': 'Qty', 'Total_Cost': 'Total'
        })
        df_final_report = df_final_report.sort_values(by='Food Item').reset_index(drop=True)
        df_final_report['Food Item'] = df_final_report['Food Item'].str.replace("’", "'").str.replace("–", "-").str.replace("—", "-")
        production_cost_total = df_final_report['Total'].sum()

        # Generate PDF with Professional Layout
        class PDF(FPDF):
            def __init__(self, orientation='P', unit='mm', format='Letter', school_address='N/A', school_name='N/A'):
                super().__init__(orientation, unit, format)
                self.school_address, self.school_name = school_address, school_name
                self.set_auto_page_break(auto=True, margin=15)
                self.column_widths, self.table_headers, self.start_x_table = [], [], self.l_margin
            def header(self):
                if self.page_no() == 1:
                    self.set_font('Times', 'B', 16); self.set_xy(10, 15); self.cell(120, 10, 'Food Order Recommendation Form', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                    try:
                        page_width, logo_width, right_margin = self.w, 30, 10; logo_x = page_width - logo_width - right_margin
                        if LOGO_FILE.exists(): self.image(str(LOGO_FILE), x=logo_x, y=8, w=logo_width)
                        else: self.rect(logo_x, 8, logo_width, 20, 'D'); self.set_xy(logo_x, 15); self.cell(logo_width, 5, "No Logo", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
                    except Exception as e: print(f"Error adding logo: {e}.")
                    box_y, box_x_left, box_w_left = 25, 10, 90; self.set_font('Times', 'B', 10); self.set_fill_color(220, 220, 220); self.set_xy(box_x_left, box_y); self.cell(box_w_left, 7, 'Recommendation for', border='TLR', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', fill=True)
                    self.set_font('Times', '', 9)
                    try: parts = self.school_address.split(',', 1); recommendation_text = f"  {self.school_name.title()}\n  {parts[0].strip()}\n  {parts[1].strip()}"
                    except Exception: recommendation_text = f"  {self.school_name.title()}\n  {self.school_address}"
                    self.set_x(box_x_left); self.multi_cell(box_w_left, 5, recommendation_text, border='LRB', align='L'); y_box_end_left = self.get_y()
                    box_x_right, box_w_right = 105, 45; self.set_font('Times', 'B', 10); self.set_fill_color(220, 220, 220); self.set_xy(box_x_right, box_y); self.cell(box_w_right, 7, 'Generated', border='TLR', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', fill=True)
                    self.set_font('Times', '', 9); self.set_x(box_x_right); self.multi_cell(box_w_right, 5, f"  {datetime.now().strftime('%B %d, %Y')}", border='LRB', align='L'); y_box_end_right = self.get_y()
                    self.set_y(max(y_box_end_left, y_box_end_right) + 8)
                elif self.page_no() > 1:
                    self.set_font('Times', 'B', 10); self.set_fill_color(200, 220, 255); self.set_x(self.start_x_table)
                    for i, header in enumerate(self.table_headers): self.cell(self.column_widths[i], 7, header, 1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True)
                    self.ln()
            def footer(self): self.set_y(-15); self.set_font('Times', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
            def create_table(self, data, column_widths, production_total):
                self.column_widths, self.table_headers, self.start_x_table = column_widths, list(data.columns), self.l_margin; table_width = sum(column_widths)
                self.set_font('Times', 'B', 10); self.set_fill_color(200, 220, 255); self.set_x(self.start_x_table)
                for i, header in enumerate(self.table_headers): self.cell(self.column_widths[i], 7, header, 1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True)
                self.ln()
                self.set_font('Times', '', 9); self.set_fill_color(255); fill = False
                for _, row in data.iterrows():
                    if self.get_y() + 6 > self.h - 15: self.set_x(self.start_x_table); self.cell(table_width, 0, '', 'T'); self.add_page(self.cur_orientation); self.set_font('Times', '', 9); self.set_fill_color(255); fill = False
                    self.set_x(self.start_x_table); start_x, start_y = self.get_x(), self.get_y(); food_item_x = start_x + column_widths[0]
                    self.set_xy(food_item_x, start_y); self.multi_cell(column_widths[1], 6, str(row['Food Item']), 'LR', 'L', fill=fill); row_height = self.get_y() - start_y
                    self.set_xy(start_x, start_y); self.cell(column_widths[0], row_height, str(row['ID']), 'LR', new_x=XPos.RIGHT, new_y=YPos.TOP, align='L', fill=fill)
                    self.set_xy(food_item_x + column_widths[1], start_y); self.cell(column_widths[2], row_height, f"${row['Cost']:,.2f}", 'LR', new_x=XPos.RIGHT, new_y=YPos.TOP, align='R', fill=fill); self.cell(column_widths[3], row_height, str(int(row['Qty'])), 'LR', new_x=XPos.RIGHT, new_y=YPos.TOP, align='R', fill=fill); self.cell(column_widths[4], row_height, f"${row['Total']:,.2f}", 'LR', new_x=XPos.RIGHT, new_y=YPos.TOP, align='R', fill=fill)
                    self.set_y(start_y + row_height); fill = not fill
                self.set_x(self.start_x_table); self.cell(table_width, 0, '', 'T'); self.ln(2)
                label_width = sum(column_widths[:4]); value_width = column_widths[4]; self.set_x(self.start_x_table); self.set_font('Times', 'B', 10); self.cell(label_width, 7, 'Production Cost Total', 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='R'); self.set_fill_color(240, 240, 240); self.cell(value_width, 7, f"${production_total:,.2f}", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R', fill=True)

        pdf = PDF('P', 'mm', 'Letter', school_address, school_to_test)
        pdf.add_page()
        col_widths = [20, 120, 20, 15, 21]
        pdf.create_table(df_final_report, col_widths, production_cost_total)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return pdf_bytes
    
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")
        traceback.print_exc()
        return None