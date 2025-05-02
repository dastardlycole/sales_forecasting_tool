"""
Integration Tests for Doghouse Sales Forecasting System

These tests validate the end-to-end functionality of the forecasting system by running
complete workflows and checking that all components work together correctly.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
import doghouse_predictor2
import doghouse_predictor5
from doghouse_sales_predictor_toggle_v2 import ForecastAppToggle


class TestEndToEndForecasting(unittest.TestCase):
    """End-to-end tests for the entire forecasting workflow"""
    
    def setUp(self):
        """Create test directories and synthetic datasets"""
        # Create temporary directories
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate synthetic data for company version (basic requirements)
        self.company_data = self._generate_realistic_sales_data(
            start_date="2020-01-01",
            periods=36,
            with_extended_features=False
        )
        
        # Generate synthetic data for dissertation version (extended requirements)
        self.diss_data = self._generate_realistic_sales_data(
            start_date="2020-01-01", 
            periods=36,
            with_extended_features=True
        )
        
        # Save to temporary files
        self.company_file = os.path.join(self.temp_dir.name, 'test_company.csv')
        self.diss_file = os.path.join(self.temp_dir.name, 'test_dissertation.csv')
        
        self.company_data.to_csv(self.company_file, index=False)
        self.diss_data.to_csv(self.diss_file, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        self.temp_dir.cleanup()
    
    def _generate_realistic_sales_data(self, start_date, periods, with_extended_features=False):
        """Generate realistic looking sales data with seasonal patterns"""
        dates = pd.date_range(start=start_date, periods=periods, freq='MS')
        
        # Create time components for patterns
        t = np.arange(len(dates))
        
        # Base sales with trend and seasonality
        base = 500 + t * 2  # Upward trend
        seasonal = 100 * np.sin(2 * np.pi * t / 12)  # Yearly seasonal pattern
        noise = np.random.normal(0, 25, len(dates))
        sales = np.maximum(0, base + seasonal + noise).astype(int)
        
        # Create DataFrame with basic columns
        df = pd.DataFrame({
            'Month': dates.strftime('%Y-%m-%d'),
            'Net items sold': sales,
            'Gross sales': sales * 30,  # $30 per item
        })
        
        # Add discount column (discounts in July and December)
        discounts = np.zeros(len(dates))
        discount_months = [7, 12]  # July and December
        for i, date in enumerate(dates):
            if date.month in discount_months:
                discounts[i] = -sales[i] * 30 * 0.15  # 15% discount
        
        df['Discounts'] = discounts
        
        # Add extended features for dissertation version
        if with_extended_features:
            # Weather score - higher in summer, lower in winter
            weather_cycle = 0.5 + 0.4 * np.sin(2 * np.pi * (t - 6) / 12)  # Phase shift to peak in summer
            df['weather_score'] = weather_cycle
            
            # Web traffic - higher before sales seasons
            web_traffic_base = 2000 + 800 * np.sin(2 * np.pi * (t - 2) / 12)  # Phase shift to lead sales
            web_traffic_noise = np.random.normal(0, 200, len(dates))
            df['web_traffic'] = np.maximum(500, web_traffic_base + web_traffic_noise).astype(int)
        
        return df
    
    def test_company_end_to_end(self):
        """Test complete forecasting process using company version"""
        # Run the full analysis with company model
        metrics = doghouse_predictor2.analyze_csv(self.company_file, log=lambda msg: None)
        
        # Validate metrics are computed and reasonable
        self.assertIn('SARIMA_RMSE', metrics)
        self.assertIn('Pure_LSTM_RMSE', metrics)
        self.assertIn('Comb_LSTM_RMSE', metrics)
        
        # Check reasonable metric values
        self.assertGreater(metrics['SARIMA_RMSE'], 0)
        self.assertLess(metrics['SARIMA_RMSE'], 200)  # Expected range for our synthetic data
    
    def test_dissertation_data_compatibility(self):
        """Test that company model works with dissertation-formatted data"""
        try:
            company_with_diss = doghouse_predictor2.analyze_csv(self.diss_file, log=lambda msg: None)
            self.assertIn('SARIMA_RMSE', company_with_diss)
        except Exception as e:
            self.fail(f"Company model failed with dissertation data: {e}")


class TestGUIIntegration(unittest.TestCase):
    """Tests for the GUI integration with the forecasting models"""
    
    def setUp(self):
        """Set up test data and directories"""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate and save a small company-compatible dataset
        dates = pd.date_range(start='2020-01-01', periods=24, freq='MS')
        sales = 100 + 50 * np.sin(np.arange(24) * 2 * np.pi / 12) + np.random.normal(0, 10, 24)
        self.test_data = pd.DataFrame({
            'Month': dates.strftime('%Y-%m-%d'),
            'Net items sold': sales.astype(int),
        })
        
        # Save test file
        self.test_file = os.path.join(self.temp_dir.name, 'gui_test.csv')
        self.test_data.to_csv(self.test_file, index=False)
        
    def tearDown(self):
        """Clean up test files"""
        self.temp_dir.cleanup()
    
    @unittest.skip("GUI tests require Tkinter interaction - run manually")
    def test_gui_forecast_validation(self):
        """Test that the GUI properly validates user inputs"""
        app = ForecastAppToggle()
        app.file_var.set(self.test_file)
        
        # Validation should pass with basic data on company version
        app.version_var.set("company")
        app.validate_csv()
        self.assertTrue(app.csv_validated)
        
        # Validation should fail with basic data on dissertation version
        app.version_var.set("dissertation")
        app.validate_csv()
        self.assertFalse(app.csv_validated)


if __name__ == '__main__':
    unittest.main()