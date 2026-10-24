"""
Data loader for the energy market optimization challenge.

This module provides utilities for loading, preprocessing and visualizing sample data.
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import logging

# Import sample_data_generator - this is an internal utility module
# that should be in the same directory or installed in the environment
# If missing, create this module with data generation capabilities
from energy_optimization_challenge.sample_data_generator import (
    generate_sample_data,
    save_sample_data,
    load_sample_data,
    display_sample,
)


logger = logging.getLogger(__name__)


class PriceGenData:
    """Class for loading and managing energy market data."""

    def __init__(self):
        """
        Initialize the data loader and load the data.
        """
        self.data = None
        self.target_names = None
        self.num_samples = 0
        self.num_hours = 0
        self.da_prices = None
        self.rt_prices = None
        self.wind_power = None
        # Load the data
        self.load_data()

    def generate_new_data(self) -> None:
        """Generate new sample data."""
        logger.info("Generating new data...")
        self.data, self.target_names = generate_sample_data()
        save_sample_data(self.data, self.target_names)
        logger.info("New data generated and saved.")

    def load_data(self) -> None:
        """Load the energy market data from the file."""
        try:
            self.data, self.target_names = load_sample_data()
            self.num_samples, self.num_hours, _ = self.data.shape
            self.da_prices = self.data[:, :, 0]
            self.rt_prices = self.data[:, :, 1]
            self.wind_power = self.data[:, :, 2]
            logger.info(
                f"Loaded data with {self.num_samples} samples and {self.num_hours} hours."
            )
        except FileNotFoundError:
            logger.info("No existing data found.")
            self.generate_new_data()
            self.load_data()

    def calculate_hourly_price_spreads(self) -> np.ndarray:
        """
        Calculate the hourly DA-RT price spreads for all samples.

        Returns:
        --------
        np.ndarray
            Array of hourly price spreads with shape (num_samples, num_hours)
        """
        da_prices = self.data[:, :, 0]
        rt_prices = self.data[:, :, 1]

        return da_prices - rt_prices

    def visualize_PriceGen_data(self) -> None:
        """
        Visualize both price spreads and generation volumes in a single 2x1 plot.

        Creates a figure with two subplots:
        - Top: DA-RT price spreads by hour
        - Bottom: Generation volumes by hour

        Both plots show mean values with 10-90% quantile ranges.
        """
        # Calculate price spreads
        spreads = self.calculate_hourly_price_spreads()
        mean_spreads = np.mean(spreads, axis=0)
        spread_q10 = np.quantile(spreads, 0.1, axis=0)
        spread_q90 = np.quantile(spreads, 0.9, axis=0)

        # Get generation volumes
        volumes = self.data[:, :, 2]
        mean_volumes = np.mean(volumes, axis=0)
        volume_q10 = np.quantile(volumes, 0.1, axis=0)
        volume_q90 = np.quantile(volumes, 0.9, axis=0)

        # Create figure explicitly first
        fig = plt.figure(figsize=(12, 10))
        axs = [plt.subplot(2, 1, 1), plt.subplot(2, 1, 2)]

        # Price spread subplot (top)
        axs[0].plot(
            range(24),
            mean_spreads,
            "o-",
            linewidth=2,
            label="Mean Spread",
            color="blue",
        )
        axs[0].fill_between(
            range(24),
            spread_q10,
            spread_q90,
            alpha=0.3,
            color="blue",
            label="10-90% Quantiles",
        )
        axs[0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axs[0].set_title("DA-RT Price Spreads by Hour")
        axs[0].set_ylabel("Price Spread (DA - RT)")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # Generation volume subplot (bottom)
        axs[1].plot(
            range(24),
            mean_volumes,
            "o-",
            linewidth=2,
            label="Mean Volume",
            color="green",
        )
        axs[1].fill_between(
            range(24),
            volume_q10,
            volume_q90,
            alpha=0.3,
            color="green",
            label="10-90% Quantiles",
        )
        axs[1].set_title("Wind Generation Volumes by Hour")
        axs[1].set_xlabel("Hour")
        axs[1].set_ylabel("Volume (MW)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        axs[1].set_xticks(range(24))

        # Adjust layout
        plt.tight_layout()

        # Save figure
        results_dir = os.path.join(".", "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        path = os.path.join(results_dir, "price_gen_data.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"PriceGen data visualization saved to {path}")
