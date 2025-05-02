"""
Comprehensive Test Suite for Doghouse Sales Forecasting System

This test suite validates key components of the forecasting system:
1. DataLoader functionality
2. Model implementation features
3. Results processing and output
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path so we can import our modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import the modules to test
import doghouse_predictor2
import doghouse_predictor5
from doghouse_sales_predictor_toggle_v2 import ForecastAppToggle


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality"""
    
    def setUp(self):
        """Create temporary test files and directories for testing"""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create valid test data with minimum required columns
        self.valid_dates = pd.date_range(start='2021-01-01', periods=24, freq='MS')
        self.valid_data = pd.DataFrame({
            'Month': self.valid_dates,
            'Net items sold': np.random.randint(100, 500, 24)
        })
        self.valid_file = os.path.join(self.temp_dir.name, 'valid_data.csv')
        self.valid_data.to_csv(self.valid_file, index=False)
        
        # Create dissertation format data with extended columns
        self.diss_data = self.valid_data.copy()
        self.diss_data['weather_score'] = np.random.random(24)
        self.diss_data['web_traffic'] = np.random.randint(1000, 5000, 24)
        self.diss_file = os.path.join(self.temp_dir.name, 'dissertation_data.csv')
        self.diss_data.to_csv(self.diss_file, index=False)
        
        # Create malformed file with incorrect date format
        self.malformed_data = self.valid_data.copy()
        self.malformed_data['Month'] = ['Invalid-' + str(i) for i in range(24)]
        self.malformed_file = os.path.join(self.temp_dir.name, 'malformed_data.csv')
        self.malformed_data.to_csv(self.malformed_file, index=False)
        
        # Create file missing required columns
        self.missing_columns_data = pd.DataFrame({
            'Month': self.valid_dates,
            'Some_other_column': np.random.random(24)
        })
        self.missing_columns_file = os.path.join(self.temp_dir.name, 'missing_columns.csv')
        self.missing_columns_data.to_csv(self.missing_columns_file, index=False)
    
    def tearDown(self):
        """Clean up temporary files after testing"""
        self.temp_dir.cleanup()
    
    def test_load_valid_data_company_version(self):
        """Test loading valid data in the company version"""
        df = doghouse_predictor2.load_data(self.valid_file)
        self.assertEqual(len(df), 24)
        self.assertTrue('Net items sold' in df.columns)
        
    def test_load_valid_data_dissertation_version(self):
        """Test loading valid data with extended columns in dissertation version"""
        df = doghouse_predictor5.load_data(self.diss_file)
        self.assertEqual(len(df), 24)
        self.assertTrue('Net items sold' in df.columns)
        self.assertTrue('weather_score' in df.columns)
        self.assertTrue('web_traffic' in df.columns)
    
    def test_missing_target_column(self):
        """Test handling of CSV missing the target column"""
        with self.assertRaises(ValueError):
            doghouse_predictor2.load_data(self.missing_columns_file)
    
    def test_date_parsing(self):
        """Test date parsing functionality"""
        df = doghouse_predictor2.load_data(self.valid_file)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        self.assertEqual(df.index[0], pd.Timestamp('2021-01-01'))
        
    def test_auto_fill_missing_features_company(self):
        """Test auto-generation of missing features in company version"""
        df = doghouse_predictor2.load_data(self.valid_file)
        df = doghouse_predictor2.ensure_schema(df)
        
        # Check if missing features were added
        self.assertTrue('discount_amount' in df.columns)
        self.assertTrue('discount_flag' in df.columns)
        self.assertTrue('sin_month' in df.columns)
        self.assertTrue('cos_month' in df.columns)


class TestModelImplementation(unittest.TestCase):
    """Test cases for model implementation functionality"""

    def setUp(self):
        """Create test data for model implementation tests"""
        # Generate synthetic dataset with seasonal patterns
        dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
        
        # Create synthetic sales with seasonality and trend
        t = np.arange(36)
        base = 500 + t * 2  # Upward trend
        seasonal = 100 * np.sin(2 * np.pi * t / 12)  # Seasonal pattern
        noise = np.random.normal(0, 20, 36)  # Random noise
        sales = base + seasonal + noise
        
        # Create DataFrame with minimum required columns
        self.test_df = pd.DataFrame({
            'Month': dates,
            'Net items sold': sales
        })
        self.test_df.set_index('Month', inplace=True)
        
        # Add required features
        self.test_df['sin_month'] = np.sin(2 * np.pi * (self.test_df.index.month - 1) / 12)
        self.test_df['cos_month'] = np.cos(2 * np.pi * (self.test_df.index.month - 1) / 12)
        self.test_df['discount_flag'] = 0
        self.test_df['discount_amount'] = 0
        self.test_df['discount_pct'] = 0
        
    def test_sequence_creation_multicol(self):
        """Test creating sequences for multivariate LSTM"""
        data = np.random.random((50, 5))  # 50 samples, 5 features
        lookback = 12
        
        X, y = doghouse_predictor2.create_sequences_multicol(data, lookback=lookback)
        
        # Check shapes: X should be (n_samples, lookback, n_features-1)
        expected_samples = 50 - lookback
        self.assertEqual(X.shape, (expected_samples, lookback, 4))
        self.assertEqual(y.shape, (expected_samples,))
    
    def test_sequence_creation_singlecol(self):
        """Test creating sequences for univariate LSTM"""
        data = np.random.random((50, 1))  # 50 samples, 1 feature
        lookback = 12
        
        X, y = doghouse_predictor2.create_sequences_singlecol(data, lookback=lookback)
        
        # Check shapes: X should be (n_samples, lookback, 1)
        expected_samples = 50 - lookback
        self.assertEqual(X.shape, (expected_samples, lookback, 1))
        self.assertEqual(y.shape, (expected_samples,))
    
    def test_sarima_in_sample(self):
        """Test SARIMA in-sample prediction"""
        in_sample, fit_res, train_size = doghouse_predictor2.run_sarima_in_sample(self.test_df)
        
        # Check length of predictions and if it made reasonable values
        self.assertEqual(len(in_sample), len(self.test_df))
        self.assertTrue(all(~np.isnan(in_sample)))
        self.assertTrue(all(~np.isinf(in_sample)))
    
    def test_sarima_future_forecast(self):
        """Test SARIMA future forecasting"""
        forecast = doghouse_predictor2.forecast_sarima_future(self.test_df, steps=12)
        
        # Check forecast length and datetime index
        self.assertEqual(len(forecast), 12)
        self.assertTrue(isinstance(forecast.index, pd.DatetimeIndex))
        
        # Check that forecast values are reasonable
        self.assertTrue(all(~np.isnan(forecast)))
        self.assertTrue(all(~np.isinf(forecast)))
        
        # Check that forecast dates are consecutive months
        date_diffs = np.diff([d.timestamp() for d in forecast.index])
        self.assertTrue(all(np.isclose(date_diffs, 30*24*3600, atol=3*24*3600)))  # Approximately 30 days
    
    def test_lstm_sequence_dimensions(self):
        """Test sequence dimensions for LSTM"""
        # Prepare data
        df_pure = self.test_df[['Net items sold']].dropna()
        
        # Run LSTM in-sample
        pure_preds, pure_actual, _, _, _ = doghouse_predictor2.run_lstm_in_sample(df_pure, 'Net items sold', lookback=12)
        
        # Check results
        self.assertGreater(len(pure_preds), 0)
        self.assertEqual(len(pure_preds), len(pure_actual))
        
        # Check that predictions are reasonable
        self.assertTrue(all(~np.isnan(pure_preds)))
        self.assertTrue(all(~np.isinf(pure_preds)))


class TestResultsProcessing(unittest.TestCase):
    """Test cases for results processing functionality"""
    
    def setUp(self):
        """Set up test data for results processing tests"""
        # Create directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plots_dir = os.path.join(self.temp_dir.name, 'final_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store original plots directory
        self.original_plots_dir = doghouse_predictor2.plots_dir
        
        # Set plots directory to temporary directory
        doghouse_predictor2.plots_dir = self.plots_dir
        
        # Create test DataFrame with historical data
        dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
        sales = 500 + np.random.normal(0, 50, 36)
        
        # Create a test CSV file
        self.test_df = pd.DataFrame({
            'Month': dates,
            'Net items sold': sales
        })
        self.test_file = os.path.join(self.temp_dir.name, 'test_sales.csv')
        self.test_df.to_csv(self.test_file, index=False)
        
    def tearDown(self):
        """Clean up after tests"""
        # Restore original plots directory
        doghouse_predictor2.plots_dir = self.original_plots_dir
        self.temp_dir.cleanup()
    
    def test_metric_calculation(self):
        """Test metric calculation accuracy"""
        # Create sample actual and predicted values
        actual = np.array([100, 150, 200, 250, 300])
        predicted = np.array([110, 160, 190, 260, 290])
        
        # Calculate metrics manually
        manual_rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        manual_mae = np.mean(np.abs(actual - predicted))
        
        # Calculate using functions
        rmse = np.sqrt(doghouse_predictor2.mean_squared_error(actual, predicted))
        mae = doghouse_predictor2.mean_absolute_error(actual, predicted)
        
        # Compare results
        self.assertAlmostEqual(rmse, manual_rmse, places=6)
        self.assertAlmostEqual(mae, manual_mae, places=6)
    
    def test_file_output_creation(self):
        """Test file output creation"""
        # Mock the log function
        mock_log = lambda msg: None
        
        # Run analysis
        results = doghouse_predictor2.analyze_csv(self.test_file, log=mock_log)
        
        # Check if output files were created
        in_sample_plot = os.path.join(self.plots_dir, f"{os.path.basename(self.test_file)}_in_sample.png")
        future_plot = os.path.join(self.plots_dir, f"{os.path.basename(self.test_file)}_future.png")
        metrics_file = os.path.join(self.plots_dir, "performance_summary.csv")
        forecast_file = os.path.join(self.plots_dir, "forecast_values.csv")
        
        self.assertTrue(os.path.exists(in_sample_plot))
        self.assertTrue(os.path.exists(future_plot))
        self.assertTrue(os.path.exists(metrics_file))
        self.assertTrue(os.path.exists(forecast_file))
    
    def test_metrics_calculation_in_results(self):
        """Test metrics in the results dictionary"""
        # Mock the log function
        mock_log = lambda msg: None
        
        # Run analysis
        results = doghouse_predictor2.analyze_csv(self.test_file, log=mock_log)
        
        # Check if metrics are in results
        self.assertIn('SARIMA_RMSE', results)
        self.assertIn('SARIMA_MAE', results)
        self.assertIn('Pure_LSTM_RMSE', results)
        self.assertIn('Pure_LSTM_MAE', results)
        self.assertIn('Comb_LSTM_RMSE', results)
        self.assertIn('Comb_LSTM_MAE', results)
        
        # Check that metrics are valid numbers
        self.assertGreater(results['SARIMA_RMSE'], 0)
        self.assertGreater(results['Pure_LSTM_RMSE'], 0)
        self.assertGreater(results['Comb_LSTM_RMSE'], 0)
        
    def test_forecast_csv_structure(self):
        """Test the structure of the forecast CSV file"""
        # Mock the log function
        mock_log = lambda msg: None
        
        # Run analysis
        doghouse_predictor2.analyze_csv(self.test_file, log=mock_log)
        
        # Read the forecast CSV
        forecast_file = os.path.join(self.plots_dir, "forecast_values.csv")
        forecast_df = pd.read_csv(forecast_file)
        
        # Check structure
        self.assertIn('Month', forecast_df.columns)
        self.assertIn('SARIMA', forecast_df.columns)
        self.assertIn('Pure_LSTM', forecast_df.columns)
        self.assertIn('Combined_LSTM', forecast_df.columns)
        
        # Check length
        self.assertEqual(len(forecast_df), 12)  # 12-month forecast


class TestGUIFunctionality(unittest.TestCase):
    """Test cases for GUI functionality"""
    
    @unittest.skip("GUI tests require Tkinter interaction - run manually")
    def test_version_toggling(self):
        """Test version toggling functionality"""
        app = ForecastAppToggle()
        
        # Test initial state
        self.assertEqual(app.version_var.get(), "dissertation")
        
        # Test toggling to company version
        app.version_var.set("company")
        self.assertEqual(app.version_var.get(), "company")
        
    @unittest.skip("GUI tests require Tkinter interaction - run manually")
    def test_csv_validation(self):
        """Test CSV validation functionality"""
        # These tests need to be run manually or with a GUI testing framework
        pass


if __name__ == '__main__':
    unittest.main()