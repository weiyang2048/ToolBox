"""
Optimization model for energy market bidding and offering.

This module provides classes for optimizing bids and offers in day-ahead and real-time markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import pyomo.environ as pe

from energy_optimization_challenge.variable_definitions import (
    create_price_quantity_variables,
)
from energy_optimization_challenge.objective_functions import (
    calculate_scenario_revenue,
    create_expected_revenue_rule,
    create_objective_rule,
)
from energy_optimization_challenge.market_clearing import (
    create_market_clearing_variables,
)
from energy_optimization_challenge.risk_management import (
    create_risk_mechanism,
)


class OptimalBiddingStrategy:
    """
    Class for optimization of bidding and offering strategy.
    """

    def __init__(
        self,
        da_prices: np.ndarray,
        rt_prices: np.ndarray,
        wind_power: np.ndarray,
        risk_aversion: float = 0.25,
        alpha: float = 0.95,
    ):
        """
        Initialize the optimization model.

        Parameters:
        -----------
        da_prices : np.ndarray
            Array with shape (num_samples, num_hours) containing DA prices
        rt_prices : np.ndarray
            Array with shape (num_samples, num_hours) containing RT prices
        wind_power : np.ndarray
            Array with shape (num_samples, num_hours) containing wind generation
        risk_aversion : float
            Risk aversion parameter between 0 and 1
            0 = risk-neutral (maximize expected profit)
            1 = risk-averse (minimize worst-case scenarios)
        alpha : float
            Confidence level for CVaR calculation, typically 0.95 (95%)
        """
        self.da_prices = da_prices
        self.rt_prices = rt_prices
        self.wind_power = wind_power
        self.risk_aversion = risk_aversion
        self.alpha = alpha
        # Extract dimensions
        self.num_samples, self.num_hours = wind_power.shape

    def optimize_hour(self, hour: int) -> Dict[str, Any]:
        """
        Optimize bids and offers for a given hour.
        goal : maximize the expected revenue while managing risk using CVaR
        Parameters:
        -----------
        hour : int
            The hour to optimize bids and offers for.
        """
        da_prices = self.da_prices[:, hour]
        rt_prices = self.rt_prices[:, hour]
        wind_power = self.wind_power[:, hour]

        # Hour-specific parameters
        # Regulatory assumption: Maximum bid and offer quantities are limited by available wind power
        # This reflects real-world grid regulations that prevent offering more than can be generated
        # by the participant's assets, limiting market manipulation possibilities
        max_gen = np.max(wind_power)
        max_da_price = np.max(da_prices)
        min_da_price = np.min(da_prices)

        model = pe.ConcreteModel()

        # scenario variables, make it 0 indexed for uniformity
        model.scenarios = pe.RangeSet(0, self.num_samples - 1)

        # Create decision variables and constraints for price and quantities
        model = create_price_quantity_variables(
            model, min_da_price, max_da_price, max_gen
        )

        # Create clearing variables and constraints
        model = create_market_clearing_variables(model, max_gen, da_prices)

        # Create scenario revenue function
        def scenario_revenue(model, i):
            return calculate_scenario_revenue(
                model, i, da_prices, rt_prices, wind_power
            )

        # Add VaR and CVaR variables and constraints
        model = create_risk_mechanism(
            model, scenario_revenue, self.num_samples, self.alpha
        )

        # Create expected revenue rule and objective function
        expected_revenue_rule = create_expected_revenue_rule(
            scenario_revenue, self.num_samples
        )
        objective_rule = create_objective_rule(
            expected_revenue_rule=expected_revenue_rule,
            model_cvar=model.cvar,
            risk_aversion=self.risk_aversion,
        )

        # Set objective
        model.objective = pe.Objective(expr=objective_rule, sense=pe.maximize)

        opt = pe.SolverFactory("glpk")

        results = opt.solve(model, tee=False)
        # check if optimization is successful
        if results.solver.status != pe.SolverStatus.ok:
            raise ValueError("Optimization failed")

        # Pre-initialize variables if they're not populated from the solution
        for i in model.scenarios:
            if model.cleared_q_offer[i].value is None:
                model.cleared_q_offer[i].value = 0.0
            if model.cleared_q_bid[i].value is None:
                model.cleared_q_bid[i].value = 0.0

        # Calculate expected revenue safely
        try:
            expected_rev_values = []
            for i in model.scenarios:
                try:
                    rev_i = pe.value(scenario_revenue(model, i))
                    expected_rev_values.append(rev_i)
                except:
                    expected_rev_values.append(0.0)
            expected_rev_value = sum(expected_rev_values) / self.num_samples
        except Exception as e:
            print(f"Error calculating expected revenue: {e}")
            expected_rev_value = 0.0

        # Calculate offer and bid revenues
        offer_revenues = []
        bid_revenues = []
        for i in model.scenarios:
            offer_q = model.cleared_q_offer[i].value
            bid_q = model.cleared_q_bid[i].value

            # Handle None values in optimization results
            if offer_q is not None:
                offer_revenues.append(offer_q * (da_prices[i] - rt_prices[i]))
            else:
                offer_revenues.append(0.0)

            if bid_q is not None:
                bid_revenues.append(bid_q * (rt_prices[i] - da_prices[i]))
            else:
                bid_revenues.append(0.0)

        result_dict = {
            "mu_da_p": np.mean(da_prices),
            "mu_rt_p": np.mean(rt_prices),
            "q_offer": model.q_offer.value,
            "q_bid": model.q_bid.value,
            "p": model.p.value,
            "mu_offer_revenue": np.mean(offer_revenues),
            "mu_bid_revenue": np.mean(bid_revenues),
            "mu_wind_revenue": np.mean(wind_power * rt_prices),
            "mu_revenue": expected_rev_value,
            "cvar": model.cvar.value,
        }

        return result_dict

    def optimize_all_hours(self) -> pd.DataFrame:
        """
        Optimize bids and offers for all hours.
        """
        return pd.DataFrame(
            [self.optimize_hour(hour) for hour in range(self.num_hours)]
        )
