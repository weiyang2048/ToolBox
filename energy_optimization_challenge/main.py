#!/usr/bin/env python3
"""
Main script for energy market optimization.

This script loads data, runs optimization, and outputs the optimal price-quantity
tuples for each hour of the day.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from energy_optimization_challenge.data_loader import PriceGenData
from energy_optimization_challenge.optimizer import OptimalBiddingStrategy


def load_data():
    """
    Load energy market data.

    Returns:
        PriceGenData: An object containing the loaded data
    """
    print("📊 Loading energy market data...")
    data = PriceGenData()
    print(f"✅ Loaded data with {data.num_samples} samples and {data.num_hours} hours.")
    data.visualize_PriceGen_data()
    print("✅ Data visualization complete!")
    return data


def run_optimization(data: PriceGenData, risk_aversion: float = 0.25) -> pd.DataFrame:
    """
    Run the optimization model for all hours.

    Args:
        data: The data object containing prices and wind power
        risk_aversion: Risk aversion parameter (0=risk-neutral, 1=risk-averse)

    Returns:
        pd.DataFrame: Results for all hours
    """
    print(f"🧮 Running optimization with risk aversion = {risk_aversion}...")
    model = OptimalBiddingStrategy(
        data.da_prices, data.rt_prices, data.wind_power, risk_aversion=risk_aversion
    )

    results = model.optimize_all_hours()
    print("✅ Optimization complete!")
    return results


def extract_price_quantity_tuples(results: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Extract price-quantity tuples from the optimization results.

    Args:
        results: DataFrame containing optimization results

    Returns:
        List of tuples containing (price, quantity) pairs for each hour
    """
    tuples = []
    action_type = []
    for idx, row in results.iterrows():
        price = row["p"]
        # The effective quantity is either q_offer or q_bid depending on which is non-zero
        if row["q_offer"] > 0:
            quantity = row["q_offer"]
            action_type.append("Offer")
        else:
            quantity = row["q_bid"]
            action_type.append("Bid")
        tuples.append((price, quantity))

    return tuples, action_type


def plot_bid_offer_results(
    price_quantity_tuples: List[Tuple[float, float]], action_type: List[str], risk_aversion: float = 0.25
) -> str:
    """
    Create a simple plot showing bid/offer prices and quantities.

    Args:
        price_quantity_tuples: List of (price, quantity) tuples for each hour
        action_type: List indicating whether each hour uses an offer or bid
        risk_aversion: Risk aversion parameter used in optimization

    Returns:
        str: Path to the saved plot file
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(price_quantity_tuples, columns=["Price", "Quantity"])
    df["Type"] = action_type
    df["Hour"] = df.index

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Add risk aversion parameter to the main title
    fig.suptitle(f"Bid/Offer Results (Risk Aversion {risk_aversion:.2f})\n", fontsize=12, y=0.95)

    # Plot prices
    ax1.plot(df["Hour"], df["Price"], "o-", color="blue", linewidth=2)
    ax1.set_title("Bid/Offer Prices by Hour")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(24))

    # Plot quantities with color by type
    offer_mask = df["Type"] == "Offer"
    bid_mask = df["Type"] == "Bid"

    if offer_mask.any():
        ax2.bar(
            df.loc[offer_mask, "Hour"],
            df.loc[offer_mask, "Quantity"],
            color="green",
            alpha=0.7,
            label="Offer",
        )

    if bid_mask.any():
        ax2.bar(
            df.loc[bid_mask, "Hour"],
            df.loc[bid_mask, "Quantity"],
            color="red",
            alpha=0.7,
            label="Bid",
        )

    ax2.set_title("Bid/Offer Quantities by Hour")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Quantity (MW)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(24))

    plt.tight_layout()

    # Save the figure
    results_dir = os.path.join(".", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_path = os.path.join(results_dir, f"bid_offer_plot.png")
    plt.savefig(file_path, dpi=300)

    print(f"📊 Bid/offer plot saved to {file_path}")
    return file_path


def plot_revenue_components(results: pd.DataFrame, risk_aversion: float = 0.25) -> str:
    """
    Create a plot showing revenue components and CVaR by hour.

    Args:
        results: DataFrame containing optimization results with revenue components
        risk_aversion: Risk aversion parameter used in optimization

    Returns:
        str: Path to the saved plot file
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Add risk aversion parameter to the main title
    fig.suptitle(f"Revenue Components and CVaR (Risk Aversion {risk_aversion:.2f})\n", fontsize=12, y=0.95)

    # Plot revenue components
    hours = range(24)
    bar_width = 0.25
    
    # Plot separate bars for each revenue component
    components = ['mu_wind_revenue', 'mu_offer_revenue', 'mu_bid_revenue']
    colors = ['blue', 'green', 'red']
    labels = ['Wind Revenue', 'Offer Revenue', 'Bid Revenue']
    
    for i, (comp, color, label) in enumerate(zip(components, colors, labels)):
        values = results[comp].values
        x = np.array(hours) + i * bar_width
        ax1.bar(x, values, width=bar_width, color=color, alpha=0.7, label=label)

    ax1.set_title("Expected Revenue Components by Hour")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Revenue ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.array(hours) + bar_width)
    ax1.set_xticklabels(hours)

    # Plot total revenue and CVaR
    ax2.plot(hours, results['mu_revenue'], 'o-', color='purple', label='Total Revenue', linewidth=2)
    ax2.plot(hours, results['cvar'], 'o-', color='orange', label='CVaR', linewidth=2)
    
    ax2.set_title("Total Revenue and CVaR by Hour")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Value ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(24))

    plt.tight_layout()

    # Save the figure
    results_dir = os.path.join(".", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_path = os.path.join(results_dir, f"revenue_components.png")
    plt.savefig(file_path, dpi=300)

    print(f"💰 Revenue components plot saved to {file_path}")
    return file_path


def main(risk_aversion: float = 0.25):
    """
    Main function to run the energy market optimization.
    """
    # Load data
    data = load_data()

    # Run optimization with default risk aversion
    results = run_optimization(data, risk_aversion)

    # Extract price-quantity tuples
    price_quantity_tuples, action_type = extract_price_quantity_tuples(results)

    # Print output
    print("\n📈 Optimal Price-Quantity Tuples for Each Hour 📉")
    print("-" * 50)
    print(f"{'Hour':<10}{'Price ($)':<15}{'Quantity (MW)':<15}{'Type':<10}")
    print("-" * 50)

    for hour, (price, quantity) in enumerate(price_quantity_tuples):
        print(f"{hour:<10}{price:<15.2f}{quantity:<15.2f}{action_type[hour]:<10}")

    # Create visualizations
    plot_bid_offer_results(price_quantity_tuples, action_type, risk_aversion)
    plot_revenue_components(results, risk_aversion)

    # Save results to CSV
    results_dir = os.path.join(".", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Simple price-quantity results with type information
    tuples_df = pd.DataFrame(price_quantity_tuples, columns=["Price", "Quantity"])
    tuples_df["Type"] = action_type
    tuples_df.index.name = "Hour"
    tuples_df.to_csv(os.path.join(results_dir, "price_quantity_tuples.csv"))
    print("📅 Price-quantity results saved to price_quantity_tuples.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run energy market optimization")
    parser.add_argument(
        "--risk-aversion", type=float, default=0.25, help="Risk Aversion parameter"
    )
    args = parser.parse_args()
    main(args.risk_aversion)
