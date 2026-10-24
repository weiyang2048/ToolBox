"""
Tests for the variable_definitions module.
"""

import unittest
import pyomo.environ as pe
from energy_optimization_challenge.variable_definitions import (
    create_price_quantity_variables,
)


class TestVariableDefinitions(unittest.TestCase):
    """Test cases for variable_definitions module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = pe.ConcreteModel()
        self.min_da_price = 10.0
        self.max_da_price = 100.0
        self.max_gen = 50.0

    def test_create_price_quantity_variables(self):
        """Test that variables and constraints are created correctly."""
        model = create_price_quantity_variables(
            self.model, self.min_da_price, self.max_da_price, self.max_gen
        )

        # Test that variables exist
        self.assertTrue(hasattr(model, "q_offer"))
        self.assertTrue(hasattr(model, "q_bid"))
        self.assertTrue(hasattr(model, "p"))
        self.assertTrue(hasattr(model, "z"))

        # Test variable bounds
        self.assertEqual(model.q_offer.bounds, (0, self.max_gen))
        self.assertEqual(model.q_bid.bounds, (0, self.max_gen))
        self.assertEqual(
            model.p.bounds, (self.min_da_price - 10, self.max_da_price + 10)
        )

        # Test that constraints exist
        self.assertTrue(hasattr(model, "offer_constraint"))
        self.assertTrue(hasattr(model, "bid_constraint"))

        # Check the offer constraint has the right form
        offer_constr = model.offer_constraint
        self.assertIsInstance(offer_constr, pe.Constraint)

        # Check the bid constraint has the right form
        bid_constr = model.bid_constraint
        self.assertIsInstance(bid_constr, pe.Constraint)


if __name__ == "__main__":
    unittest.main()
