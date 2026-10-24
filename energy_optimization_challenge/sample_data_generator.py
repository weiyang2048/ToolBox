#!/usr/bin/env python3
"""
Sample Data Generator for Energy Market Optimization Challenge

This script generates synthetic energy market data for a single node with 100 ensemble forecasts for the SAME day:
- Day-Ahead Locational Marginal Prices (DALMP)
- Real-Time Locational Marginal Prices (RTLMP)
- Wind Power Generation

Each ensemble sample represents a possible scenario for the same 24-hour period.
The output is a numpy array of shape (num_samples, num_hours, num_targets).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Tuple, Dict, List, Optional, Union, Any


def generate_sample_data(
    num_samples: int = 100,
    num_hours: int = 24,
    wind_capacity_mw: float = 150.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic energy market data for a single node with ensemble forecasts for the same day.
    
    Parameters:
    -----------
    num_samples : int
        Number of ensemble samples/scenarios to generate (all for the same day)
    num_hours : int
        Number of hours in the day
    wind_capacity_mw : float
        Wind farm capacity in MW
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        - 3D numpy array of shape (num_samples, num_hours, num_targets)
        - List of target names ['dalmp', 'rtlmp', 'wind_power_mw']
    """
    # Define target names
    target_names = ['dalmp', 'rtlmp', 'wind_power_mw']
    num_targets = len(target_names)
    
    # Initialize output array
    output_data = np.zeros((num_samples, num_hours, num_targets))
    
    for sample_idx in range(num_samples):
        # Base profiles with some randomness
        base_demand = 20 + 15 * np.sin(np.linspace(0, 2*np.pi, num_hours)) + np.random.normal(0, 2, num_hours)
        
        # Price volatility factor for the day
        daily_volatility = np.random.uniform(0.8, 1.5)
        price_correlation = np.random.uniform(0.7, 0.98)  # Correlation between DA and RT prices
        
        # Node-specific factors
        node_factor = np.random.uniform(0.85, 1.15)
        congestion_factor = np.random.uniform(0.9, 1.2)
        
        # Generate wind profile first - to allow inverse relationship with prices
        base_wind_profile = np.clip(0.4 + 0.3 * np.sin(np.linspace(0, 4*np.pi, num_hours)) + np.random.normal(0, 0.15, num_hours), 0, 1)
        wind_capacity_factor = base_wind_profile.copy()
            
        for hour in range(num_hours):
            # Generate wind power first
            wind_power_mw = wind_capacity_mw * wind_capacity_factor[hour]
            
            # Base price with moderate inverse relation to wind for DA
            da_wind_factor = 1.0 - (0.15 * wind_capacity_factor[hour])  # Lower impact of wind on DA prices
            base_price = (30 + 10 * base_demand[hour] / np.max(base_demand)) * node_factor
            
            # Add time patterns (higher prices during peak hours)
            time_factor = 1.0
            if 7 <= hour < 10:  # Morning peak
                time_factor = 1.2
            elif 17 <= hour < 21:  # Evening peak
                time_factor = 1.3
            
            # Calculate DA LMP with a premium to make it higher than RT on average
            da_premium = np.random.uniform(1.05, 1.15)  # DA prices 5-15% higher on average
            da_lmp = base_price * time_factor * congestion_factor * da_premium * da_wind_factor
            
            # Add some random variations
            da_lmp *= np.random.uniform(0.9, 1.1)
            
            # Allow negative prices (about 10% chance when wind is high)
            if wind_capacity_factor[hour] > 0.7 and np.random.random() < 0.1:
                da_lmp = da_lmp * -0.5 - np.random.uniform(1, 10)
            
            # RT LMP has correlation with DA but more volatility and stronger inverse relation to wind
            rt_wind_factor = 1.0 - (0.5 * wind_capacity_factor[hour])  # Stronger inverse relationship for RT
            rt_noise = np.random.normal(0, daily_volatility * 5)
            rt_lmp = da_lmp * (price_correlation + (1 - price_correlation) * np.random.normal(0, 1)) * rt_wind_factor + rt_noise
            
            # Allow for more negative prices in RT market (about 15% chance when wind is high)
            if wind_capacity_factor[hour] > 0.5 and np.random.random() < 0.15:
                rt_lmp = rt_lmp * -0.8 - np.random.uniform(5, 20)
            
            # Occasionally add price spikes to RT when wind is low
            if wind_capacity_factor[hour] < 0.3:
                # Rare extreme spikes up to $5000 (0.5% of the time)
                if np.random.random() < 0.005:
                    rt_lmp = np.random.uniform(3000, 5000)
                # More common moderate spikes (about 5% of hours)
                elif np.random.random() < 0.05:
                    spike_factor = np.random.uniform(1.5, 5.0)
                    rt_lmp *= spike_factor
            
            # Store values in output array
            output_data[sample_idx, hour, 0] = round(da_lmp, 2)  # DALMP
            output_data[sample_idx, hour, 1] = round(rt_lmp, 2)  # RTLMP
            output_data[sample_idx, hour, 2] = round(wind_power_mw, 2)  # Wind Generation
    
    return output_data, target_names


def save_sample_data(
    data: np.ndarray, 
    target_names: List[str],
    output_dir: str = "./data", 
    filename: str = "energy_market_samples.npz"
) -> None:
    """
    Save the generated data to a numpy npz file.
    
    Parameters:
    -----------
    data : np.ndarray
        3D array containing the generated data
    target_names : List[str]
        List of target names
    output_dir : str
        Directory to save the output file
    filename : str
        Name of the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to output file
    output_path = os.path.join(output_dir, filename)
    
    # Save to npz file with target names
    np.savez(
        output_path, 
        data=data, 
        target_names=target_names
    )
    
    print(f"Sample data saved to {output_path}")
    print(f"Dataset shape: {data.shape} (samples, hours, targets)")
    
    # Print some basic statistics
    print("\nData Summary:")
    print(f"Number of ensemble samples (scenarios): {data.shape[0]}")
    print(f"Number of hours in the day: {data.shape[1]}")
    print(f"Number of targets: {data.shape[2]}")
    
    for i, target in enumerate(target_names):
        print(f"{target} range: {data[:,:,i].min():.2f} - {data[:,:,i].max():.2f}")
    
    # Count negative price occurrences
    da_neg_prices = np.sum(data[:,:,0] < 0)
    rt_neg_prices = np.sum(data[:,:,1] < 0)
    total_prices = data.shape[0] * data.shape[1]
    
    # Count extreme price occurrences
    rt_extreme_prices = np.sum(data[:,:,1] > 1000)
    
    # Calculate average prices
    avg_da = np.mean(data[:,:,0])
    avg_rt = np.mean(data[:,:,1])
    
    # Calculate average correlations across all samples
    all_wind_da_corrs = []
    all_wind_rt_corrs = []
    
    for i in range(data.shape[0]):
        wind = data[i, :, 2]
        da = data[i, :, 0]
        rt = data[i, :, 1]
        
        wind_da_corr = np.corrcoef(wind, da)[0, 1]
        wind_rt_corr = np.corrcoef(wind, rt)[0, 1]
        
        all_wind_da_corrs.append(wind_da_corr)
        all_wind_rt_corrs.append(wind_rt_corr)
    
    avg_wind_da_corr = np.mean(all_wind_da_corrs)
    avg_wind_rt_corr = np.mean(all_wind_rt_corrs)
    
    print(f"\nPrice statistics:")
    print(f"Average DA price: ${avg_da:.2f}")
    print(f"Average RT price: ${avg_rt:.2f}")
    print(f"DA/RT ratio: {avg_da/avg_rt:.2f}")
    
    print(f"\nAverage correlation statistics:")
    print(f"Avg Wind-DA Price Correlation: {avg_wind_da_corr:.4f}")
    print(f"Avg Wind-RT Price Correlation: {avg_wind_rt_corr:.4f}")
    print(f"RT has {abs(avg_wind_rt_corr)/abs(avg_wind_da_corr):.2f}x stronger correlation with wind than DA")
    print(f"\nNegative price statistics:")
    print(f"DA negative prices: {da_neg_prices} ({da_neg_prices/total_prices*100:.2f}% of all hours)")
    print(f"RT negative prices: {rt_neg_prices} ({rt_neg_prices/total_prices*100:.2f}% of all hours)")
    print(f"\nExtreme price statistics:")
    print(f"RT prices > $1000: {rt_extreme_prices} ({rt_extreme_prices/total_prices*100:.2f}% of all hours)")


def load_sample_data(
    filepath: str = "./data/energy_market_samples.npz"
) -> Tuple[np.ndarray, List[str]]:
    """
    Load the sample data from a numpy npz file.
    
    Parameters:
    -----------
    filepath : str
        Path to the npz file
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        - 3D numpy array of sample data
        - List of target names
    """
    loaded = np.load(filepath, allow_pickle=True)
    data = loaded['data']
    target_names = loaded['target_names'].tolist()
    
    return data, target_names


def display_sample(
    data: np.ndarray, 
    target_names: List[str], 
    sample_idx: int = 0
) -> None:
    """
    Display a single sample from the generated data.
    
    Parameters:
    -----------
    data : np.ndarray
        3D array containing the generated data
    target_names : List[str]
        List of target names
    sample_idx : int
        Index of the sample to display
    """
    print(f"\nSample {sample_idx} data:")
    
    # Create a sample dataframe for display
    sample_df = pd.DataFrame({
        'hour': range(data.shape[1]),
    })
    
    for i, target in enumerate(target_names):
        sample_df[target] = data[sample_idx, :, i]
    
    print(sample_df)
    
    # Display correlation between wind and prices for this sample
    wind_data = data[sample_idx, :, 2]
    da_lmp_data = data[sample_idx, :, 0]
    rt_lmp_data = data[sample_idx, :, 1]
    
    wind_da_corr = np.corrcoef(wind_data, da_lmp_data)[0, 1]
    wind_rt_corr = np.corrcoef(wind_data, rt_lmp_data)[0, 1]
    
    print(f"\nCorrelations for sample {sample_idx}:")
    print(f"Wind-DA Price Correlation: {wind_da_corr:.4f}")
    print(f"Wind-RT Price Correlation: {wind_rt_corr:.4f}")
    print(f"RT has {abs(wind_rt_corr)/abs(wind_da_corr):.2f}x stronger correlation with wind than DA")


if __name__ == "__main__":
    # Generate sample data
    print("Generating synthetic energy market data...")
    data, target_names = generate_sample_data(num_samples=100, num_hours=24)
    
    # Save the data
    save_sample_data(data, target_names)
    
    # Display a sample
    display_sample(data, target_names, sample_idx=0)
    
    print("\nData generation complete!")