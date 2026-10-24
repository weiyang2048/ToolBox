"""
Tests for the optimizer module.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import pyomo.environ as pe

from energy_optimization_challenge.optimizer import OptimalBiddingStrategy


class TestOptimalBiddingStrategy(unittest.TestCase):
    """Test cases for OptimalBiddingStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test data
        self.num_samples = 3
        self.num_hours = 4
        
        # Create sample price and wind power data
        self.da_prices = np.array([
            [50, 60, 70, 80],  # Sample 0
            [55, 65, 75, 85],  # Sample 1
            [45, 55, 65, 75]   # Sample 2
        ])
        
        self.rt_prices = np.array([
            [45, 65, 65, 75],  # Sample 0
            [50, 70, 70, 80],  # Sample 1
            [40, 60, 60, 70]   # Sample 2
        ])
        
        self.wind_power = np.array([
            [10, 15, 20, 25],  # Sample 0
            [12, 18, 22, 28],  # Sample 1
            [8, 12, 18, 22]    # Sample 2
        ])
        
        # Create optimizer instance
        self.optimizer = OptimalBiddingStrategy(
            self.da_prices,
            self.rt_prices,
            self.wind_power,
            risk_aversion=0.25,
            alpha=0.95
        )
        
    @patch('energy_optimization_challenge.optimizer.pe.SolverFactory')
    def test_initialization(self, mock_solver_factory):
        """Test initialization of the optimizer."""
        # Check that data was stored correctly
        np.testing.assert_array_equal(self.optimizer.da_prices, self.da_prices)
        np.testing.assert_array_equal(self.optimizer.rt_prices, self.rt_prices)
        np.testing.assert_array_equal(self.optimizer.wind_power, self.wind_power)
        
        self.assertEqual(self.optimizer.risk_aversion, 0.25)
        self.assertEqual(self.optimizer.alpha, 0.95)
        self.assertEqual(self.optimizer.num_samples, self.num_samples)
        self.assertEqual(self.optimizer.num_hours, self.num_hours)
    
    @patch('energy_optimization_challenge.optimizer.pe.SolverFactory')
    def test_optimize_hour(self, mock_solver_factory):
        """Test optimization for a single hour."""
        # Configure mock solver
        mock_solver = MagicMock()
        mock_solver_factory.return_value = mock_solver
        
        # Configure solver results to be successful
        mock_results = MagicMock()
        mock_results.solver.status = pe.SolverStatus.ok
        mock_solver.solve.return_value = mock_results
        
        # Call optimize_hour - set up a simple case that should prefer offering when DA > RT
        hour = 0  # First hour: DA > RT for all scenarios
        
        # Mock the optimization result
        result = self.optimizer.optimize_hour(hour)
        
        # Check that the solver was called with the correct model
        mock_solver_factory.assert_called_once_with('glpk')
        mock_solver.solve.assert_called_once()
        
        # Check that the result contains the expected keys
        expected_keys = [
            'mu_da_p', 'mu_rt_p', 'q_offer', 'q_bid', 'p',
            'mu_offer_revenue', 'mu_bid_revenue', 'mu_wind_revenue',
            'mu_revenue', 'cvar'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that mean price values match expected values
        self.assertAlmostEqual(result['mu_da_p'], np.mean(self.da_prices[:, hour]))
        self.assertAlmostEqual(result['mu_rt_p'], np.mean(self.rt_prices[:, hour]))
    
    @patch('energy_optimization_challenge.optimizer.OptimalBiddingStrategy.optimize_hour')
    def test_optimize_all_hours(self, mock_optimize_hour):
        """Test optimization for all hours."""
        # Configure mock to return predetermined results
        mock_results = [
            {'hour': i, 'q_offer': 10.0, 'q_bid': 0.0} for i in range(self.num_hours)
        ]
        mock_optimize_hour.side_effect = mock_results
        
        # Call optimize_all_hours
        results_df = self.optimizer.optimize_all_hours()
        
        # Check that optimize_hour was called for each hour
        self.assertEqual(mock_optimize_hour.call_count, self.num_hours)
        
        # Check that the result is a DataFrame with the expected number of rows
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), self.num_hours)


if __name__ == '__main__':
    unittest.main() 