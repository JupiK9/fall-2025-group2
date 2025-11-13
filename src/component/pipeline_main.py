import sys, os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/main_code
SRC_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA = ROOT / "data"

from component.data_transformer_html import generate_csvs_from_folder
from component.csv_combiner import combine_csvs_from_folder
from component.pdf_processor import process_all_pdfs
from component.popularity import (
    prepare_popularity_data, clean_numeric,
    get_leftover_rate_by_school, get_net_consumption_by_school
)
from component.optimization import (
    run_all_optimizations
)

def html_csv_pipeline():
    """
    HTML → CSV transformers + combiners. Skips if clean outputs exist.
    """

    print("\n[HTML/CSV] Starting HTML to CSV Processing...")
    bf_clean = DATA / "clean-data" / "data_breakfast.csv"
    ln_clean = DATA / "clean-data" / "data_lunch.csv"

    if bf_clean.exists() and ln_clean.exists():
        print("[HTML/CSV] Clean breakfast & lunch already exist. Skipping HTML transform + CSV combine.")
        return

    # Breakfast HTML -> CSVs -> combined
    bf_html_folder = DATA / "FairfaxCounty" / "May 2025 Breakfast production records" / "May 2025 Breakfast production records"
    bf_out_folder  = DATA / "preprocessed-data" / "Breakfast production"
    print("[HTML/CSV] Generating breakfast CSVs...")
    generate_csvs_from_folder(bf_html_folder, bf_out_folder)

    print("[HTML/CSV] Combining breakfast CSVs...")
    combine_csvs_from_folder(bf_out_folder, bf_clean, sort_columns=['school_name', 'date'])

    # Lunch HTML -> CSVs -> combined
    ln_html_folder = DATA / "FairfaxCounty" / "May 2025 Lunch production records" / "May 2025 Lunch production records"
    ln_out_folder  = DATA / "preprocessed-data" / "Lunch production"
    print("[HTML/CSV] Generating lunch CSVs...")
    generate_csvs_from_folder(ln_html_folder, ln_out_folder)

    print("[HTML/CSV] Combining lunch CSVs...")
    combine_csvs_from_folder(ln_out_folder, ln_clean, sort_columns=['school_name', 'date'])


def pdf_pipeline():
    """
    PDF → CSV. Skips if sales.csv exists.
    """

    print("\n[PDF] Starting PDF to CSV Processing...")
    sales_csv = DATA / "clean-data" / "sales.csv"
    if sales_csv.exists():
        print("[PDF] sales.csv already exists. Skipping PDF processing.")
        return
    pdf_folder = DATA / "FairFaxCounty" / "Item Sales Reports - Mar May 2025" / "Item Sales Reports - Mar May 2025"
    print("[PDF] Processing PDFs into sales.csv...")
    process_all_pdfs(pdf_folder, sales_csv)


def popularity_pipeline():
    """
    Popularity analysis. Skips if popularity data exists.
    """

    breakfast_file = DATA / "clean-data" / "data_breakfast.csv"
    lunch_file     = DATA / "clean-data" / "data_lunch.csv"
    sales_file     = DATA / "clean-data" / "sales.csv"

    bf_df, l_df, s_df = prepare_popularity_data(breakfast_file, lunch_file, sales_file)

    num_cols = [
        "served_non-reimbursable", "discarded_total", "discarded_cost",
        "subtotal_cost", "left_over_percent_of_offered", "left_over_cost",
        "left_over_total", "production_cost_total"
    ]
    if bf_df is not None:
        bf_df = clean_numeric(bf_df, num_cols)
    if l_df is not None:
        l_df = clean_numeric(l_df, num_cols)

    print("\n[Popularity] Starting net consumption by school...")
    if bf_df is not None: get_net_consumption_by_school(bf_df, "Breakfast")
    if l_df  is not None: get_net_consumption_by_school(l_df,  "Lunch")

    print("\n[Popularity] Exporting leftover rate by school...")
    if bf_df is not None: get_leftover_rate_by_school(bf_df, "Breakfast")
    if l_df  is not None: get_leftover_rate_by_school(l_df,  "Lunch")


def optimization_pipeline():
    """
    Optimization analysis + exports all charts/maps. 
    """

    print("\n[Optimization] Initializing Optimization Analysis...")

    BF_PATH = DATA / "clean-data" / "data_breakfast.csv"
    LN_PATH = DATA / "clean-data" / "data_lunch.csv"
    SC_PATH = DATA / "preprocessed-data" / "2022-2025 Fairfax County School Student Count.csv"
    UNIT_COSTS_PATH = DATA / "preprocessed-data" / "unit_costs.csv"
    COORDINATES_PATH = DATA / "preprocessed-data" / "data_breakfast_with_coordinates.csv"
    GEOJSON_PATH = DATA / "preprocessed-data" / "School_Regions.geojson"

    out = run_all_optimizations(
        breakfast_file=BF_PATH,
        lunch_file=LN_PATH,
        student_counts_file=SC_PATH,
        unit_costs_file=UNIT_COSTS_PATH,
        coordinates_file=COORDINATES_PATH,
        geojson_file=GEOJSON_PATH,
        total_budget=139144760
    )

    if not (out and out.get("monthly_ilp") is not None):
        print("[Optimization] Monthly ILP results missing; charts skipped.")
        return
    
    