"""
Objective functions for energy market optimization.

This module provides functions to calculate revenues and define objective functions.
"""


def calculate_scenario_revenue(model, i, da_prices, rt_prices, wind_power):
    """
    Calculate revenue for a specific scenario.
    
    Parameters:
    -----------
    model : pyomo.environ.ConcreteModel
        The optimization model
    i : int
        Scenario index
    da_prices : np.ndarray
        Array of day-ahead prices
    rt_prices : np.ndarray
        Array of real-time prices
    wind_power : np.ndarray
        Array of wind power generation
    
    Returns:
    --------
    float
        Total revenue for the scenario
    """
    # Revenue from offer only if clearing_offer is 1
    offer_revenue = model.cleared_q_offer[i] * (da_prices[i] - rt_prices[i])

    # Revenue from bid only if clearing_bid is 1
    bid_revenue = model.cleared_q_bid[i] * (rt_prices[i] - da_prices[i])

    # Revenue from wind
    wind_revenue = wind_power[i] * rt_prices[i]

    return bid_revenue + offer_revenue + wind_revenue


def create_expected_revenue_rule(scenario_revenue_func, num_samples):
    """
    Create a rule for calculating expected revenue across all scenarios.
    
    Parameters:
    -----------
    scenario_revenue_func : callable
        Function to calculate revenue for each scenario
    num_samples : int
        Number of scenarios
        
    Returns:
    --------
    callable
        Function that calculates expected revenue for the model
    """
    def expected_revenue_rule(model):
        return sum(scenario_revenue_func(model, i) for i in model.scenarios) / num_samples
    
    return expected_revenue_rule


def create_objective_rule(expected_revenue_rule, risk_aversion, model_cvar):
    """
    Create a rule for the objective function that balances expected revenue and CVaR.
    
    Parameters:
    -----------
    expected_revenue_rule : callable
        Function that calculates expected revenue
    risk_aversion : float
        Risk aversion parameter between 0 and 1
    model_cvar : pe.Var or str
        Either a CVaR variable from the model or a string attribute name
        
    Returns:
    --------
    callable
        Function that calculates the objective value for the model
    """
    def objective_rule(model):
        expected_revenue = expected_revenue_rule(model)
        
        # Get CVaR value - handle both Pyomo variable and string attribute name
        if isinstance(model_cvar, str):
            cvar_value = getattr(model, model_cvar)
        else:
            cvar_value = model_cvar
            
        # Risk-adjusted objective using CVaR
        # When risk_aversion = 0, this simplifies to expected_revenue (risk-neutral)
        return (1 - risk_aversion) * expected_revenue + risk_aversion * cvar_value
    
    return objective_rule 