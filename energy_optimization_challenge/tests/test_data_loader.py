"""
Tests for the data_loader module.
"""

import unittest
from unittest.mock import patch

import numpy as np

from energy_optimization_challenge.data_loader import PriceGenData


class TestPriceGenData(unittest.TestCase):
    """Test cases for the PriceGenData data loader."""

    def setUp(self):
        """Set up a known three-channel data set."""
        self.target_names = ["dalmp", "rtlmp", "wind_power_mw"]
        self.data = np.array(
            [
                [
                    [10.0, 7.0, 100.0],
                    [20.0, 15.0, 110.0],
                    [30.0, 35.0, 120.0],
                ],
                [
                    [40.0, 30.0, 200.0],
                    [50.0, 60.0, 210.0],
                    [60.0, 55.0, 220.0],
                ],
            ]
        )
        self.expected_da_prices = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ]
        )
        self.expected_rt_prices = np.array(
            [
                [7.0, 15.0, 35.0],
                [30.0, 60.0, 55.0],
            ]
        )
        self.expected_wind_power = np.array(
            [
                [100.0, 110.0, 120.0],
                [200.0, 210.0, 220.0],
            ]
        )

    @patch("energy_optimization_challenge.data_loader.save_sample_data")
    @patch("energy_optimization_challenge.data_loader.generate_sample_data")
    @patch("energy_optimization_challenge.data_loader.load_sample_data")
    def test_load_data_populates_dimensions_and_channels(
        self, mock_load, mock_generate, mock_save
    ):
        """A successful load stores the data and splits its three channels."""
        mock_load.return_value = (self.data, self.target_names)

        loader = PriceGenData()

        mock_load.assert_called_once_with()
        mock_generate.assert_not_called()
        mock_save.assert_not_called()
        np.testing.assert_array_equal(loader.data, self.data)
        self.assertEqual(loader.target_names, self.target_names)
        self.assertEqual(loader.num_samples, 2)
        self.assertEqual(loader.num_hours, 3)
        np.testing.assert_array_equal(loader.da_prices, self.expected_da_prices)
        np.testing.assert_array_equal(loader.rt_prices, self.expected_rt_prices)
        np.testing.assert_array_equal(loader.wind_power, self.expected_wind_power)

    @patch("energy_optimization_challenge.data_loader.load_sample_data")
    def test_calculate_hourly_price_spreads(self, mock_load):
        """Hourly spreads subtract the RT channel from the DA channel."""
        mock_load.return_value = (self.data, self.target_names)

        loader = PriceGenData()
        expected_spreads = np.array(
            [
                [3.0, 5.0, -5.0],
                [10.0, -10.0, 5.0],
            ]
        )

        np.testing.assert_array_equal(
            loader.calculate_hourly_price_spreads(), expected_spreads
        )

    @patch("energy_optimization_challenge.data_loader.save_sample_data")
    @patch("energy_optimization_challenge.data_loader.generate_sample_data")
    @patch("energy_optimization_challenge.data_loader.load_sample_data")
    def test_missing_file_generates_saves_and_reloads(
        self, mock_load, mock_generate, mock_save
    ):
        """A missing data file engages generation, saving, and a second load."""
        generated_data = np.full((1, 2, 3), 9.0)
        mock_load.side_effect = [
            FileNotFoundError("data file missing"),
            (self.data, self.target_names),
        ]
        mock_generate.return_value = (generated_data, self.target_names)

        loader = PriceGenData()

        self.assertEqual(mock_load.call_count, 2)
        mock_generate.assert_called_once_with()
        self.assertEqual(mock_save.call_count, 1)
        saved_data, saved_target_names = mock_save.call_args.args
        np.testing.assert_array_equal(saved_data, generated_data)
        self.assertEqual(saved_target_names, self.target_names)

        # The recursive load, not generate_new_data alone, populates the
        # dimensions and channel views on PriceGenData.
        np.testing.assert_array_equal(loader.data, self.data)
        self.assertEqual(loader.target_names, self.target_names)
        self.assertEqual(loader.num_samples, 2)
        self.assertEqual(loader.num_hours, 3)
        np.testing.assert_array_equal(loader.da_prices, self.expected_da_prices)
        np.testing.assert_array_equal(loader.rt_prices, self.expected_rt_prices)
        np.testing.assert_array_equal(loader.wind_power, self.expected_wind_power)

    @patch("energy_optimization_challenge.data_loader.save_sample_data")
    @patch("energy_optimization_challenge.data_loader.generate_sample_data")
    def test_generate_new_data_stores_and_saves_generated_data(
        self, mock_generate, mock_save
    ):
        """generate_new_data stores the generator result and asks to save it."""
        generated_data = np.arange(18, dtype=float).reshape(2, 3, 3)
        mock_generate.return_value = (generated_data, self.target_names)
        loader = PriceGenData.__new__(PriceGenData)

        loader.generate_new_data()

        mock_generate.assert_called_once_with()
        self.assertEqual(mock_save.call_count, 1)
        saved_data, saved_target_names = mock_save.call_args.args
        np.testing.assert_array_equal(saved_data, generated_data)
        self.assertEqual(saved_target_names, self.target_names)
        np.testing.assert_array_equal(loader.data, generated_data)
        self.assertEqual(loader.target_names, self.target_names)

    @patch("energy_optimization_challenge.data_loader.save_sample_data")
    @patch("energy_optimization_challenge.data_loader.generate_sample_data")
    @patch("energy_optimization_challenge.data_loader.load_sample_data")
    def test_non_missing_file_load_error_propagates_without_generation(
        self, mock_load, mock_generate, mock_save
    ):
        """Only FileNotFoundError engages the generate-and-reload fallback."""
        mock_load.side_effect = ValueError("invalid sample data")

        with self.assertRaisesRegex(ValueError, "invalid sample data"):
            PriceGenData()

        mock_load.assert_called_once_with()
        mock_generate.assert_not_called()
        mock_save.assert_not_called()

    @patch("energy_optimization_challenge.data_loader.save_sample_data")
    @patch("energy_optimization_challenge.data_loader.generate_sample_data")
    @patch("energy_optimization_challenge.data_loader.load_sample_data")
    def test_load_data_rejects_non_three_dimensional_data(
        self, mock_load, mock_generate, mock_save
    ):
        """Loaded data must unpack into samples, hours, and channels."""
        mock_load.return_value = (np.zeros((2, 3)), self.target_names)

        with self.assertRaisesRegex(ValueError, "not enough values to unpack"):
            PriceGenData()

        mock_load.assert_called_once_with()
        mock_generate.assert_not_called()
        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
