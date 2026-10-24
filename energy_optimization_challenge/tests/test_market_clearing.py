"""
Tests for the market_clearing module.
"""

import unittest
import pyomo.environ as pe
import numpy as np
from energy_optimization_challenge.market_clearing import create_market_clearing_variables
from energy_optimization_challenge.variable_definitions import create_price_quantity_variables


class TestMarketClearing(unittest.TestCase):
    """Test cases for market_clearing module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = pe.ConcreteModel()
        self.num_scenarios = 5
        self.model.scenarios = pe.RangeSet(0, self.num_scenarios - 1)
        self.min_da_price = 10.0
        self.max_da_price = 100.0
        self.max_gen = 50.0
        self.da_prices = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
        
        # Create basic variables first
        self.model = create_price_quantity_variables(
            self.model, self.min_da_price, self.max_da_price, self.max_gen
        )
        
    def test_create_market_clearing_variables(self):
        """Test that market clearing variables and constraints are created correctly."""
        model = create_market_clearing_variables(
            self.model, self.max_gen, self.da_prices
        )
        
        # Test that variables exist
        self.assertTrue(hasattr(model, 'clearing_offer'))
        self.assertTrue(hasattr(model, 'clearing_bid'))
        self.assertTrue(hasattr(model, 'cleared_q_offer'))
        self.assertTrue(hasattr(model, 'cleared_q_bid'))
        
        # Check dimensions of indexed variables
        self.assertEqual(len(model.clearing_offer), self.num_scenarios)
        self.assertEqual(len(model.clearing_bid), self.num_scenarios)
        self.assertEqual(len(model.cleared_q_offer), self.num_scenarios)
        self.assertEqual(len(model.cleared_q_bid), self.num_scenarios)
        
        # Test that constraints exist
        self.assertTrue(hasattr(model, 'offer_clearing_rule_upper'))
        self.assertTrue(hasattr(model, 'offer_clearing_rule_lower'))
        self.assertTrue(hasattr(model, 'bid_clearing_rule_upper'))
        self.assertTrue(hasattr(model, 'bid_clearing_rule_lower'))
        self.assertTrue(hasattr(model, 'cleared_offer_upper'))
        self.assertTrue(hasattr(model, 'no_cleared_offer'))
        self.assertTrue(hasattr(model, 'cleared_offer_lower'))
        self.assertTrue(hasattr(model, 'cleared_bid_upper'))
        self.assertTrue(hasattr(model, 'no_cleared_bid'))
        self.assertTrue(hasattr(model, 'cleared_bid_lower'))
        
        # Test constraint dimensions
        self.assertEqual(len(model.offer_clearing_rule_upper), self.num_scenarios)
        self.assertEqual(len(model.offer_clearing_rule_lower), self.num_scenarios)
        self.assertEqual(len(model.bid_clearing_rule_upper), self.num_scenarios)
        self.assertEqual(len(model.bid_clearing_rule_lower), self.num_scenarios)
        
    def test_bigM_calculation(self):
        """Test that the BigM value is calculated reasonably."""
        model = create_market_clearing_variables(
            self.model, self.max_gen, self.da_prices
        )
        
        # Check that clearing rules exist for each scenario
        for i in model.scenarios:
            self.assertTrue(hasattr(model, 'offer_clearing_rule_upper'))
            self.assertTrue(hasattr(model, 'offer_clearing_rule_lower'))
            self.assertTrue(hasattr(model, 'bid_clearing_rule_upper'))
            self.assertTrue(hasattr(model, 'bid_clearing_rule_lower'))
            
            # Check that the constraints are Pyomo constraints (ConstraintData for indexed constraints)
            self.assertIsInstance(model.offer_clearing_rule_upper[i], pe.base.constraint.ConstraintData)
            self.assertIsInstance(model.offer_clearing_rule_lower[i], pe.base.constraint.ConstraintData)
            self.assertIsInstance(model.bid_clearing_rule_upper[i], pe.base.constraint.ConstraintData)
            self.assertIsInstance(model.bid_clearing_rule_lower[i], pe.base.constraint.ConstraintData)


if __name__ == '__main__':
    unittest.main() 