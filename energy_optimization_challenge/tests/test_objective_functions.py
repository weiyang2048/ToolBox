"""
Tests for the objective_functions module.
"""

import unittest
import pyomo.environ as pe
import numpy as np
from energy_optimization_challenge.objective_functions import (
    calculate_scenario_revenue,
    create_expected_revenue_rule,
    create_objective_rule
)


class TestObjectiveFunctions(unittest.TestCase):
    """Test cases for objective_functions module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = pe.ConcreteModel()
        self.num_scenarios = 3
        self.model.scenarios = pe.RangeSet(0, self.num_scenarios - 1)
        
        # Sample data
        self.da_prices = np.array([50.0, 60.0, 70.0])
        self.rt_prices = np.array([45.0, 65.0, 55.0])
        self.wind_power = np.array([10.0, 15.0, 20.0])
        
        # Set up model variables with fixed values (not Pyomo variables)
        self.model.cleared_q_offer = {}
        self.model.cleared_q_offer[0] = 5.0
        self.model.cleared_q_offer[1] = 0.0
        self.model.cleared_q_offer[2] = 10.0
        
        self.model.cleared_q_bid = {}
        self.model.cleared_q_bid[0] = 0.0
        self.model.cleared_q_bid[1] = 8.0
        self.model.cleared_q_bid[2] = 0.0
        
        self.model.cvar = 1000.0
        
    def test_calculate_scenario_revenue(self):
        """Test calculation of revenue for a scenario using a mock model."""
        # Create a test model with regular Python variables (not Pyomo variables)
        class TestModel:
            def __init__(self):
                self.cleared_q_offer = {}
                self.cleared_q_bid = {}
        
        test_model = TestModel()
        test_model.cleared_q_offer[0] = 5.0
        test_model.cleared_q_bid[0] = 0.0
        
        # Test scenario 0: offer, no bid
        revenue_0 = calculate_scenario_revenue(
            test_model, 0, self.da_prices, self.rt_prices, self.wind_power
        )
        
        # Expected revenue components
        expected_offer_rev_0 = test_model.cleared_q_offer[0] * (self.da_prices[0] - self.rt_prices[0])
        expected_bid_rev_0 = test_model.cleared_q_bid[0] * (self.rt_prices[0] - self.da_prices[0])
        expected_wind_rev_0 = self.wind_power[0] * self.rt_prices[0]
        expected_revenue_0 = expected_offer_rev_0 + expected_bid_rev_0 + expected_wind_rev_0
        
        # Convert PyomoExpression to value if needed
        try:
            revenue_0_val = pe.value(revenue_0)
        except:
            revenue_0_val = revenue_0
            
        self.assertAlmostEqual(revenue_0_val, expected_revenue_0)
        
        # Test scenario 1: bid, no offer
        test_model.cleared_q_offer[1] = 0.0
        test_model.cleared_q_bid[1] = 8.0
        
        revenue_1 = calculate_scenario_revenue(
            test_model, 1, self.da_prices, self.rt_prices, self.wind_power
        )
        
        # Expected revenue components
        expected_offer_rev_1 = test_model.cleared_q_offer[1] * (self.da_prices[1] - self.rt_prices[1])
        expected_bid_rev_1 = test_model.cleared_q_bid[1] * (self.rt_prices[1] - self.da_prices[1])
        expected_wind_rev_1 = self.wind_power[1] * self.rt_prices[1]
        expected_revenue_1 = expected_offer_rev_1 + expected_bid_rev_1 + expected_wind_rev_1
        
        # Convert PyomoExpression to value if needed
        try:
            revenue_1_val = pe.value(revenue_1)
        except:
            revenue_1_val = revenue_1
            
        self.assertAlmostEqual(revenue_1_val, expected_revenue_1)
        
    def test_create_expected_revenue_rule(self):
        """Test creation of expected revenue rule."""
        # Create a simple scenario revenue function for testing
        # that returns a numeric value, not a Pyomo expression
        def test_scenario_revenue(model, i):
            # Return a simple test value based on the scenario
            return i * 100
        
        expected_revenue_rule = create_expected_revenue_rule(
            test_scenario_revenue, self.num_scenarios
        )
        
        # Create a test model that doesn't use Pyomo expressions
        class TestModel:
            def __init__(self):
                self.scenarios = range(3)  # Add a scenarios attribute
        
        test_model = TestModel()
        
        # Expected value = (0*100 + 1*100 + 2*100) / 3 = 100
        expected_value = 100.0
        
        self.assertAlmostEqual(expected_revenue_rule(test_model), expected_value)
        
    def test_create_objective_rule(self):
        """Test creation of objective rule with risk aversion."""
        # Create a simple expected revenue rule for testing
        # that returns a numeric value, not a Pyomo expression
        def test_expected_revenue_rule(model):
            return 500.0  # Constant expected revenue
        
        # Create a test model with a regular Python value for cvar
        class TestModel:
            def __init__(self):
                self.cvar = 1000.0
        
        test_model = TestModel()
        
        # Test with risk_aversion = 0 (risk-neutral)
        obj_rule_risk_neutral = create_objective_rule(
            test_expected_revenue_rule, 0.0, "cvar"  # Pass attribute name instead of Pyomo variable
        )
        
        result_risk_neutral = obj_rule_risk_neutral(test_model)
        # Convert PyomoExpression to value if needed
        try:
            result_risk_neutral_val = pe.value(result_risk_neutral)
        except:
            result_risk_neutral_val = result_risk_neutral
            
        expected_risk_neutral = 500.0  # Should just be the expected revenue
        self.assertAlmostEqual(result_risk_neutral_val, expected_risk_neutral)
        
        # Test with risk_aversion = 1 (completely risk-averse)
        obj_rule_risk_averse = create_objective_rule(
            test_expected_revenue_rule, 1.0, "cvar"
        )
        
        result_risk_averse = obj_rule_risk_averse(test_model)
        # Convert PyomoExpression to value if needed
        try:
            result_risk_averse_val = pe.value(result_risk_averse)
        except:
            result_risk_averse_val = result_risk_averse
            
        expected_risk_averse = test_model.cvar  # Should just be the CVaR value
        self.assertAlmostEqual(result_risk_averse_val, expected_risk_averse)
        
        # Test with risk_aversion = 0.25 (mixed)
        obj_rule_mixed = create_objective_rule(
            test_expected_revenue_rule, 0.25, "cvar"
        )
        
        result_mixed = obj_rule_mixed(test_model)
        # Convert PyomoExpression to value if needed
        try:
            result_mixed_val = pe.value(result_mixed)
        except:
            result_mixed_val = result_mixed
            
        expected_mixed = 0.75 * 500.0 + 0.25 * test_model.cvar
        self.assertAlmostEqual(result_mixed_val, expected_mixed)


if __name__ == '__main__':
    unittest.main() 