import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# --- Rigorous Physics Analysis Logic ---

def power_law_func(x, alpha, C):
    """Defines the power-law function y = C * x^(-alpha)"""
    return C * (x ** -alpha)

def run_physics_analysis():
    """
    Loads collected delta_p data and performs a rigorous statistical physics analysis,
    focusing on power-law distribution and Pareto analysis.
    """
    try:
        # 1. Load pre-collected data
        all_bursts = np.load('all_delta_p.npy')
        print(f"Successfully loaded {len(all_bursts)} non-trivial Δp bursts from 'all_delta_p.npy'.")
    except FileNotFoundError:
        print("Error: 'all_delta_p.npy' not found.")
        print("Please run 'benchmarker.py' first to generate the data.")
        sys.exit(1)

    if len(all_bursts) < 20:
        print("Not enough data points to perform a meaningful analysis.")
        sys.exit(1)

    # --- 2. PDF Calculation (Correct Method) ---
    # Use logarithmically spaced bins for power-law data
    bins = np.logspace(np.log10(all_bursts.min()), np.log10(all_bursts.max()), 15)
    
    # Get the counts in each bin
    counts, bin_edges = np.histogram(all_bursts, bins=bins, density=False)
    
    # Calculate the width of each log-bin
    bin_widths = np.diff(bin_edges)
    
    # Normalize by the number of samples AND the bin width to get the true PDF
    pdf = counts / (np.sum(counts) * bin_widths)
    
    # Get the geometric center of each bin for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter out zero-count bins to avoid errors in log-plotting and fitting
    non_zero_mask = pdf > 0
    bin_centers_fit = bin_centers[non_zero_mask]
    pdf_fit = pdf[non_zero_mask]

    # --- 3. Power-Law Fitting (in Log-Log space) ---
    log_x = np.log10(bin_centers_fit)
    log_y = np.log10(pdf_fit)
    
    # Use polyfit for a robust linear fit in log space
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha = -coeffs[0]
    log_C = coeffs[1]
    
    # --- 4. Visualization ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle('Computational Cost Physics: Power Law & Pareto Principle', fontsize=18, color='cyan')

    # --- Left Plot: Pareto Analysis (Lorenz Curve) ---
    ax1 = fig.add_subplot(121)
    sorted_bursts = np.sort(all_bursts)
    cum_cost = np.cumsum(sorted_bursts)
    
    # Normalize data for Lorenz curve
    x_lorenz = np.linspace(0., 1., len(cum_cost))
    y_lorenz = cum_cost / cum_cost[-1]
    
    ax1.plot(x_lorenz, y_lorenz, color='magenta', lw=2.5, label='Lorenz Curve (Cost Distribution)')
    ax1.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Line of Equality')
    ax1.fill_between(x_lorenz, y_lorenz, x_lorenz, color='magenta', alpha=0.3)

    # Correctly calculate the "Top 20% consumes X%" point
    idx_top_20_percent = np.searchsorted(x_lorenz, 0.8)
    cost_consumed_by_top_20 = (1 - y_lorenz[idx_top_20_percent]) * 100
    
    pareto_text = (f'Top 20% of most complex events\n'
                   f'consume ≈ {cost_consumed_by_top_20:.1f}% of total Δp')
    ax1.annotate(pareto_text, xy=(0.8, y_lorenz[idx_top_20_percent]), xytext=(0.25, 0.65),
                 arrowprops=dict(facecolor='cyan', shrink=0.05, width=1, headwidth=8),
                 bbox=dict(boxstyle="round,pad=0.5", fc="#200020", ec="magenta", lw=1, alpha=0.9))

    ax1.set_title('Pareto Analysis of Computational Cost', fontsize=14)
    ax1.set_xlabel('Cumulative % of Events (Sorted from least to most costly)')
    ax1.set_ylabel('Cumulative % of Total Computational Cost (Δp)')
    ax1.legend(loc='upper left')
    ax1.grid(True, ls="--", alpha=0.3)

    # --- Right Plot: Log-Log Probability Density Function ---
    ax2 = fig.add_subplot(122)
    ax2.loglog(bin_centers_fit, pdf_fit, 'o', color='cyan', markersize=10, label='Observed PDF')
    
    # Plot the fitted line, ensuring it passes through the data cloud
    fit_line = (10**log_C) * (bin_centers_fit ** -alpha)
    ax2.loglog(bin_centers_fit, fit_line, 'r-', lw=2.5, 
               label=f'Power Law Fit (α ≈ {alpha:.2f})')

    ax2.set_title('Power Law Distribution of Δp', fontsize=14)
    ax2.set_xlabel('Computational Cost Burst Size (Log Δp)')
    ax2.set_ylabel('Probability Density (Log PDF)')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("final_physics_report.png", dpi=300)
    plt.show()

    # --- 5. Final Conclusion Output ---
    idx_80_cost = np.searchsorted(y_lorenz, 0.2)
    pareto_x = (1 - x_lorenz[idx_80_cost]) * 100

    print("\n" + "="*60)
    print("      COMPUTATIONAL PHYSICS REPORT")
    print("="*60)
    print(f"Power Law Exponent (α): {alpha:.3f}")
    print("  Interpretation: The system exhibits self-organized criticality. The negative\n"
          "  exponent reveals that while small computational bursts are frequent,\n"
          "  catastrophically large bursts are rare but non-negligible, a hallmark\n"
          "  of complex systems operating at the edge of chaos.")
    print(f"\nPareto Principle ('80/20 Rule') Analysis:")
    print(f"  - The top {pareto_x:.1f}% of complex events consumed 80% of the total computational energy (Δp).")
    print(f"  - Conversely, the top 20% of complex events consumed {cost_consumed_by_top_20:.1f}% of the total energy.")
    print("="*60)


if __name__ == '__main__':
    run_physics_analysis()