import sys, os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/tests
SRC_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

from component.popularity import (prepare_popularity_data, clean_numeric, food_popularity, net_consumption, leftover_rate, get_leftover_rate_by_school, get_net_consumption_by_school)

breakfast_file = DATA / "clean-data" / "data_breakfast.csv"
lunch_file = DATA / "clean-data" / "data_lunch.csv"
sales_file = DATA / "clean-data" / "sales.csv"

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

get_net_consumption_by_school(bf_df, "Breakfast")
get_net_consumption_by_school(l_df, "Lunch")

get_leftover_rate_by_school(bf_df, "Breakfast")
get_leftover_rate_by_school(l_df, "Lunch")