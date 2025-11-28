"""
EDA.py
Clean, pipeline-safe EDA module.
All analysis is performed inside run_eda(), so importing this file has no side effects.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path

sns.set(style="whitegrid")

# ---------------------------------------------------------------------
# Helper to clean numeric columns
# ---------------------------------------------------------------------
def clean_numeric_columns(df, cols):
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



# ---------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------
def run_eda(dfb, dfl, dfs, output_folder):
    """
    Runs all EDA charts + exports them to output_folder.
    INPUTS:
        dfb – breakfast dataframe
        dfl – lunch dataframe
        dfs – sales dataframe
        output_folder – path to save plots
    """
    # ------------------------------------------------------------
    # Merge coordinate/metadata files (required for fcps region)
    # ------------------------------------------------------------
    coord_bf = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "preprocessed-data" / "data_breakfast_with_coordinates.csv", low_memory=False)
    coord_lf = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "preprocessed-data" / "data_lunch_with_coordinates.csv", low_memory=False)


    coord_bf.columns = coord_bf.columns.str.lower()
    coord_lf.columns = coord_lf.columns.str.lower()

    keep_cols = ["school_name", "school id", "level", "fcps region",
                 "address", "zipcode", "latitude", "longitude"]

    coord_bf = coord_bf[keep_cols].drop_duplicates(subset=["school_name"])
    coord_lf = coord_lf[keep_cols].drop_duplicates(subset=["school_name"])

    dfb = dfb.merge(coord_bf, on="school_name", how="left")
    dfl = dfl.merge(coord_lf, on="school_name", how="left")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"[EDA] Saving plots to {output_folder}")

    # Clean numeric columns
    num_cols = [
        "production_cost_total", "served_cost", "discarded_percent_of_offered",
        "discarded_cost", "subtotal_cost", "left_over_percent_of_offered",
        "left_over_cost",
    ]

    dfb = clean_numeric_columns(dfb, num_cols)
    dfl = clean_numeric_columns(dfl, num_cols)

    # Standard region + level order
    region_order = [f"Region {i}" for i in range(1, 7)]
    level_order  = ["ES", "MS", "HS"]

    # Combine meal type
    dfb["meal"] = "breakfast"
    dfl["meal"] = "lunch"
    df = pd.concat([dfb, dfl], ignore_index=True)

    # ------------------------------------------------------------------
    # Helper: bar plot
    # ------------------------------------------------------------------
    def bar_plot(df_in, group_col, value_col, title, fname):
        grouped = df_in.groupby(group_col)[value_col].sum().reindex(
            region_order if group_col == "fcps region" else level_order,
            fill_value=0
        )

        plt.figure(figsize=(10,6))
        bars = plt.bar(grouped.index, grouped.values,
                       color="skyblue", edgecolor="black")
        plt.title(title)
        plt.ylabel("Cost ($)")

        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x,_: f"${x:,.0f}")
        )

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height(),
                     f"${bar.get_height():,.0f}",
                     ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_folder / fname, dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Helper: bar plot for AVERAGES
    # ------------------------------------------------------------------
    def bar_plot_avg(df_in, group_col, value_col, title, fname):
        grouped = df_in.groupby(group_col)[value_col].mean().reindex(
            region_order if group_col == "fcps region" else level_order,
            fill_value=0
        )

        plt.figure(figsize=(10,6))
        bars = plt.bar(grouped.index, grouped.values,
                       color="lightgreen", edgecolor="black")
        plt.title(title)
        plt.ylabel("Average Cost ($)")

        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x,_: f"${x:,.0f}")
        )

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height(),
                     f"${bar.get_height():,.0f}",
                     ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_folder / fname, dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Breakfast EDA
    # ------------------------------------------------------------------
    print("[EDA] Breakfast charts...")
    bar_plot(dfb, "fcps region", "production_cost_total",
             "Total Production Cost by Region - Breakfast",
             "bf_production_cost_by_region.png")

    bar_plot(dfb, "level", "production_cost_total",
             "Total Production Cost by School Level - Breakfast",
             "bf_production_cost_by_level.png")

    # NEW AVERAGE PLOTS
    bar_plot_avg(dfb, "fcps region", "production_cost_total",
                 "Average Production Cost by Region - Breakfast",
                 "bf_avg_production_cost_by_region.png")

    bar_plot_avg(dfb, "level", "production_cost_total",
                 "Average Production Cost by School Level - Breakfast",
                 "bf_avg_production_cost_by_level.png")
    
    # ------------------------------------------------------------------
    # Lunch EDA
    # ------------------------------------------------------------------
    print("[EDA] Lunch charts...")
    bar_plot(dfl, "fcps region", "production_cost_total",
             "Total Production Cost by Region - Lunch",
             "lunch_production_cost_by_region.png")

    bar_plot(dfl, "level", "production_cost_total",
             "Total Production Cost by School Level - Lunch",
             "lunch_production_cost_by_level.png")
    
    # NEW AVERAGE PLOTS
    bar_plot_avg(dfl, "fcps region", "production_cost_total",
                 "Average Production Cost by Region - Lunch",
                 "lunch_avg_production_cost_by_region.png")

    bar_plot_avg(dfl, "level", "production_cost_total",
                 "Average Production Cost by School Level - Lunch",
                 "lunch_avg_production_cost_by_level.png")

    # ------------------------------------------------------------------
    # Combined EDA
    # ------------------------------------------------------------------
    print("[EDA] Combined charts...")
    bar_plot(df, "fcps region", "production_cost_total",
             "Total Production Cost by Region",
             "production_cost_by_region.png")

    bar_plot(df, "level", "production_cost_total",
             "Total Production Cost by Level",
             "production_cost_by_level.png")
    
     # NEW AVERAGE PLOTS
    bar_plot_avg(df, "fcps region", "production_cost_total",
                 "Average Production Cost by Region",
                 "avg_production_cost_by_region.png")

    bar_plot_avg(df, "level", "production_cost_total",
                 "Average Production Cost by Level",
                 "avg_production_cost_by_level.png")

    # ------------------------------------------------------------------
    # Sales time series
    # ------------------------------------------------------------------
    print("[EDA] Sales time series...")
    dfs = dfs.copy()
    dfs["date"] = pd.to_datetime(dfs["date"], format='mixed')
    dfs.sort_values("date", inplace=True)

    plt.figure(figsize=(14,5))
    plt.plot(dfs["date"], dfs["total"])
    plt.title("Sales Time Series – Total Items")
    plt.xlabel("Date")
    plt.ylabel("Total")
    plt.tight_layout()
    plt.savefig(output_folder / "timeseries_total_food.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("[EDA] Completed successfully.")
