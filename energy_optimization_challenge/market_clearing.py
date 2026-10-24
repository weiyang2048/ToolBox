"""
Market clearing mechanism for energy market optimization.

This module provides functions to create variables and constraints for market clearing.
"""

import pyomo.environ as pe


def create_market_clearing_variables(model, max_gen, da_prices):
    """
    Create binary variables and constraints for market clearing mechanism.

    Parameters:
    -----------
    model : pyomo.environ.ConcreteModel
        The optimization model
    max_gen : float
        Maximum generation capacity for this hour
    da_prices : np.ndarray
        Array of day-ahead prices for each scenario

    Returns:
    --------
    model : pyomo.environ.ConcreteModel
        The updated optimization model with clearing variables and constraints
    """
    # M should be large enough to make constraints non-binding when binary vars are set appropriately
    M = 10**5

    # Clearing variables
    model.clearing_offer = pe.Var(model.scenarios, initialize=0, domain=pe.Binary)
    model.clearing_bid = pe.Var(model.scenarios, initialize=0, domain=pe.Binary)

    # Auxiliary variables for cleared quantities (to keep the model linear)
    model.cleared_q_offer = pe.Var(
        model.scenarios, domain=pe.Reals, bounds=(0, max_gen)
    )
    model.cleared_q_bid = pe.Var(model.scenarios, domain=pe.Reals, bounds=(0, max_gen))

    # Offer Clearing Rule: p >= da_prices[i]
    @model.Constraint(model.scenarios)
    def offer_clearing_rule_upper(model, i):
        # If clearing_offer[i] = 0, then p < da_prices[i]
        return model.p - da_prices[i] <= M * model.clearing_offer[i]

    @model.Constraint(model.scenarios)
    def offer_clearing_rule_lower(model, i):
        # If clearing_offer[i] = 1, then p >= da_prices[i]
        return model.p - da_prices[i] >= -M * (1 - model.clearing_offer[i])

    # Bid Clearing Rule: p <= da_prices[i]
    @model.Constraint(model.scenarios)
    def bid_clearing_rule_upper(model, i):
        # If clearing_bid[i] = 0, then p > da_prices[i]
        return da_prices[i] - model.p <= M * model.clearing_bid[i]

    @model.Constraint(model.scenarios)
    def bid_clearing_rule_lower(model, i):
        # If clearing_bid[i] = 1, then p <= da_prices[i]
        return da_prices[i] - model.p >= -M * (1 - model.clearing_bid[i])

    # Constraints to set cleared quantities
    @model.Constraint(model.scenarios)
    def cleared_offer_upper(model, i):
        return model.cleared_q_offer[i] <= model.q_offer

    @model.Constraint(model.scenarios)
    def no_cleared_offer(model, i):
        # if clearing_offer[i] = 0, then cleared_q_offer[i] <= 0
        return model.cleared_q_offer[i] <= M * model.clearing_offer[i]

    @model.Constraint(model.scenarios)
    def cleared_offer_lower(model, i):
        # if clearing_offer[i] = 1, then cleared_q_offer[i] >= q_offer
        # together with the upper bound, it becomes cleared_q_offer[i] = q_offer
        return model.cleared_q_offer[i] >= model.q_offer - M * (
            1 - model.clearing_offer[i]
        )

    @model.Constraint(model.scenarios)
    def cleared_bid_upper(model, i):
        # if clearing_bid[i] = 0, then cleared_q_bid[i] <= 0
        return model.cleared_q_bid[i] <= model.q_bid

    @model.Constraint(model.scenarios)
    def no_cleared_bid(model, i):
        return model.cleared_q_bid[i] <= M * model.clearing_bid[i]

    @model.Constraint(model.scenarios)
    def cleared_bid_lower(model, i):
        return model.cleared_q_bid[i] >= model.q_bid - M * (1 - model.clearing_bid[i])

    return model
