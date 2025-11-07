import sys, os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/tests
SRC_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from component.data_transformer_html import generate_csvs_from_folder
from component.csv_combiner import combine_csvs_from_folder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# HTML Transformer - Breakfast
folder_path = DATA / "FairfaxCounty/May 2025 Breakfast production records/May 2025 Breakfast production records"
output_dir = DATA / "preprocessed-data/Breakfast production"
generate_csvs_from_folder(folder_path, output_dir)

# HTML Transformer - Lunch
folder_path = DATA / "FairfaxCounty/May 2025 Lunch production records/May 2025 Lunch production records"
output_dir = DATA / "preprocessed-data/Lunch production"
generate_csvs_from_folder(folder_path, output_dir)

# CSV Combiner - Breakfast
input_dir = DATA / "preprocessed-data" / "Breakfast production"
output_file = DATA / "clean-data" / "data_breakfast.csv"
sort_columns = ['school_name', 'date']

combine_csvs_from_folder(input_dir, output_file, sort_columns)

# CSV Combiner - Lunch
input_dir = DATA / "preprocessed-data" / "Lunch production"
output_file = DATA / "clean-data" / "data_lunch.csv"
sort_columns = ['school_name', 'date']

combine_csvs_from_folder(input_dir, output_file, sort_columns)