"""
Energy Market Optimization Challenge.

This package contains a solution for optimizing bids and offers in electricity markets.
"""

__version__ = "0.1.0"

import logging

# Combined logging configuration with both level and format
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s - %(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Import main components for easier access
from energy_optimization_challenge.variable_definitions import create_price_quantity_variables
from energy_optimization_challenge.objective_functions import (
    calculate_scenario_revenue,
    create_expected_revenue_rule,
    create_objective_rule,
)
from energy_optimization_challenge.market_clearing import create_market_clearing_variables
