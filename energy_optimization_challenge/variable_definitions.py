"""
Variable definitions for energy market optimization.

This module provides functions to create decision variables and constraints for prices and quantities.
"""

import pyomo.environ as pe


def create_price_quantity_variables(model, min_da_price, max_da_price, max_gen):
    """
    Create decision variables and constraints for price and quantities.

    Parameters:
    -----------
    model : pyomo.environ.ConcreteModel
        The optimization model
    min_da_price : float
        Minimum day-ahead price
    max_da_price : float
        Maximum day-ahead price
    max_gen : float
        Maximum generation capacity
    """
    # Decision variables
    model.q_offer = pe.Var(
        domain=pe.Reals, bounds=(0, max_gen), initialize=0
    )  # Offer quantity
    model.q_bid = pe.Var(
        domain=pe.Reals, bounds=(0, max_gen), initialize=0
    )  # Bid quantity
    model.p = pe.Var(
        domain=pe.Reals, bounds=(min_da_price - 10, max_da_price + 10)
    )  # Single price for both bid and offer

    # only one of the two quantities can be non-zero at a time
    # Using binary variable and big-M method to keep the model linear
    model.z = pe.Var(
        domain=pe.Binary
    )  # Binary variable: 1 if using offer, 0 if using bid

    M = 10**4  # Big M value

    # Assume that the bid and offer are not both non-zero at the same time
    # If z = 1, q_bid must be 0; if z = 0, q_offer must be 0
    model.offer_constraint = pe.Constraint(expr=model.q_offer <= M * model.z)
    model.bid_constraint = pe.Constraint(expr=model.q_bid <= M * (1 - model.z))

    return model
