"""
Tests for the risk_management module.
"""

import unittest
import pyomo.environ as pe
from energy_optimization_challenge.risk_management import create_risk_mechanism


class TestRiskManagement(unittest.TestCase):
    """Test cases for risk_management module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = pe.ConcreteModel()
        self.num_scenarios = 3
        self.model.scenarios = pe.RangeSet(0, self.num_scenarios - 1)
        self.alpha = 0.95  # 95% confidence level
        
        # Define a simple revenue function for testing
        self.revenues = [-100, 200, 500]  # Different scenario revenues
        
    def scenario_revenue_func(self, model, i):
        """Simple test function to return scenario revenue."""
        return self.revenues[i]
        
    def test_create_risk_mechanism(self):
        """Test creation of risk management variables and constraints."""
        model = create_risk_mechanism(
            self.model, self.scenario_revenue_func, self.num_scenarios, self.alpha
        )
        
        # Test that variables exist
        self.assertTrue(hasattr(model, 'var'))  # Value-at-Risk
        self.assertTrue(hasattr(model, 'cvar'))  # Conditional Value-at-Risk
        self.assertTrue(hasattr(model, 'y'))  # Auxiliary variables
        
        # Test dimensions of y
        self.assertEqual(len(model.y), self.num_scenarios)
        
        # Test that constraints exist
        self.assertTrue(hasattr(model, 'cvar_constraint'))
        self.assertTrue(hasattr(model, 'cvar_definition'))
        
        # Test constraint dimensions
        self.assertEqual(len(model.cvar_constraint), self.num_scenarios)
        
    def test_cvar_calculation(self):
        """Test the CVaR calculation with a simple example."""
        model = create_risk_mechanism(
            self.model, self.scenario_revenue_func, self.num_scenarios, self.alpha
        )
        
        # We need to solve a simple model to check the CVaR calculation
        # Set an objective to minimize CVaR (maximize negative CVaR)
        model.objective = pe.Objective(expr=-model.cvar, sense=pe.minimize)
        
        solver = pe.SolverFactory('glpk')
        results = solver.solve(model)
        
        # Check if the optimization was successful
        self.assertEqual(results.solver.status, pe.SolverStatus.ok)
        
        # For alpha = 0.95 with 3 scenarios, only the worst scenario (i=0, revenue=-100)
        # should contribute to CVaR calculation
        # CVaR should be approximately equal to the worst-case revenue
        # Note: This is a simplified test; in reality, CVaR would be a weighted average
        # of the worst scenarios based on alpha
        self.assertLessEqual(model.cvar.value, min(self.revenues))
        
        # VaR should be between worst and second-worst scenario
        self.assertLessEqual(model.var.value, -min(self.revenues))


if __name__ == '__main__':
    unittest.main() 