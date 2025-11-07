import sys, os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/tests
SRC_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from component.optimization import run_all_optimizations, generate_savings_analysis_chart, generate_overall_savings_bar_chart, generate_savings_by_size_charts, generate_savings_map, generate_savings_maps_by_level, generate_all_region_choropleths

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

out = run_all_optimizations(
    DATA / "clean-data" / "data_breakfast.csv",
    DATA / "clean-data" / "data_lunch.csv",
    DATA / "preprocessed-data" / "2022-2025 Fairfax County School Student Count.csv",
    total_budget=139144760
)

if out and out.get("monthly_ilp") is not None:
    generate_savings_analysis_chart(out["opt_data"], out["monthly_ilp"], out["monthly_meal_costs"])
    generate_overall_savings_bar_chart(out["opt_data"], out["monthly_ilp"], out["monthly_meal_costs"])
    generate_savings_by_size_charts(out["opt_data"], out["monthly_ilp"], out["monthly_school_budgets"], out["monthly_meal_costs"]),
    generate_savings_map(out["opt_data"], out["monthly_ilp"], out["monthly_meal_costs"], DATA / "preprocessed-data" / "data_breakfast_with_coordinates.csv"),
    generate_savings_maps_by_level(out["opt_data"], out["monthly_ilp"], out["monthly_meal_costs"], DATA / "preprocessed-data" / "data_breakfast_with_coordinates.csv"),
    generate_all_region_choropleths(out["opt_data"], out["monthly_ilp"], out["monthly_meal_costs"], DATA / "preprocessed-data" / "data_breakfast_with_coordinates.csv", DATA / "preprocessed-data" / "School_Regions.geojson")