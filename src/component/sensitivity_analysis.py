import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from optimization import prepare_optimization_data

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "data" / "clean-data"
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "src" / "data" / "preprocessed-data"

# New Output Directory
OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "sensitivity-data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data Files
BREAKFAST_FILE = DATA_DIR / "data_breakfast.csv"
LUNCH_FILE = DATA_DIR / "data_lunch.csv"
SIZES_FILE = PREPROCESSED_DATA_DIR / "2022-2025 Fairfax County School Student Count.csv"

def run_quantity_sensitivity():
    print(f"Loading data from {DATA_DIR}...")
    dfb = pd.read_csv(BREAKFAST_FILE)
    dfl = pd.read_csv(LUNCH_FILE)
    sz = pd.read_csv(SIZES_FILE)
    
    # Prepare the data
    opt_data = prepare_optimization_data(dfb, dfl, sz)
    
    # Calculate metrics
    def clean_currency(df, col):
        return pd.to_numeric(df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)

    avg_bf_c = clean_currency(dfb, 'production_cost_total').sum() / clean_currency(dfb, 'served_reimbursable').sum()
    avg_ln_c = clean_currency(dfl, 'production_cost_total').sum() / clean_currency(dfl, 'served_reimbursable').sum()
    
    bf_w_rate = clean_currency(dfb, 'left_over_total').sum() / clean_currency(dfb, 'offered').sum()
    ln_w_rate = clean_currency(dfl, 'left_over_total').sum() / clean_currency(dfl, 'offered').sum()

    # Total monthly demand across all schools
    total_monthly_demand_bf = sum(d[0] for d in opt_data['demand'].values()) * 20
    total_monthly_demand_ln = sum(d[1] for d in opt_data['demand'].values()) * 20

    # Trade-off Parameter: Unserved Meal Penalty
    penalty_unserved = 0.55 

    lambda_values = np.linspace(0.01, 2.0, 20)
    results = []

    for l_val in lambda_values:
        # Objective logic: (Cost + Penalty) vs (Unserved Penalty)
        coeff_bf = avg_bf_c + (l_val * bf_w_rate * avg_bf_c) - penalty_unserved
        coeff_ln = avg_ln_c + (l_val * ln_w_rate * avg_ln_c) - penalty_unserved
        
        # Decide quantities (Upper bound 1.15 if beneficial to serve, Lower bound 0.85 if waste is too costly)
        qty_bf = 1.15 * total_monthly_demand_bf if coeff_bf < 0 else 0.85 * total_monthly_demand_bf
        qty_ln = 1.15 * total_monthly_demand_ln if coeff_ln < 0 else 0.85 * total_monthly_demand_ln
        
        total_qty = qty_bf + qty_ln
        total_prod_cost = (qty_bf * avg_bf_c) + (qty_ln * avg_ln_c)
        
        results.append({
            "Lambda": l_val,
            "Quantity": total_qty,
            "Cost": total_prod_cost
        })

    # Save CSV to the new folder
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_DIR / "refined_sensitivity_results.csv", index=False)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(summary["Lambda"], summary["Quantity"], color='#1f77b4', linewidth=3, label="Optimal Production (All Schools)")
    
    # Highlight chosen Lambda
    ax1.axvline(x=0.1, color='green', linestyle='--', alpha=0.6, label="Chosen λ (0.1)")
    
    # Identify the "Economic Flip Point" where waste cost > service value
    flip_point = (penalty_unserved - avg_bf_c) / (bf_w_rate * avg_bf_c)
    ax1.axvline(x=flip_point, color='red', linestyle=':', label=f"Critical Threshold (λ ≈ {flip_point:.2f})")

    ax1.set_xlabel("Waste Penalty Weight ($\lambda$)", fontsize=12)
    ax1.set_ylabel("Total Monthly Production (Units)", fontsize=12, color='#1f77b4')
    ax1.ticklabel_format(style='plain', axis='y') # Remove scientific notation
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')

    plt.title("Sensitivity Analysis: Impact of Waste Penalty on Production Strategy", fontsize=14)
    
    # Save Plots to the new folder
    plt.savefig(OUTPUT_DIR / "sensitivity_plot.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "sensitivity_plot.pdf")
    plt.savefig(OUTPUT_DIR / "sensitivity_plot.eps", format='eps')
    
    print(f"\nSuccess! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_quantity_sensitivity()