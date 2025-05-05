# Doghouse Sales Forecasting Tool

A powerful time series analysis tool for predicting doghouse sales using advanced forecasting models.

## Overview

This application provides sales forecasting capabilities for doghouse retail businesses using two sophisticated forecasting approaches:

- **Statistical Forecasting (SARIMA)**: Traditional time series analysis with seasonal components
- **Machine Learning Forecasting (LSTM)**: Neural network-based deep learning approach
- **Combined Approach**: Leverages both methods for improved accuracy

The tool offers two operational modes:
- **Dissertation (Extended Features)**: Advanced analysis with enhanced visualization and metrics
- **Company (Standard)**: Streamlined forecasting for business use

  ![main_window](https://github.com/user-attachments/assets/3105d95f-5df1-49ee-82ed-d8d6da081cad)


## Installation

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Setup

1. **Clone or download the repository**

```bash
git clone https://github.com/dastardlycole/sales_forecasting_tool.git
# or download and extract the ZIP file
```

2. **Create a virtual environment**

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

1. **Launch the application**

```bash
python doghouse_sales_predictor_toggle_v2.py
```

2. **Select your preferred version** using the radio buttons:
   - Dissertation (Extended Features)
   - Company (Standard)

3. **Load your sales data**:
   - Click "Browse" to select your CSV file
   - Click "Validate" to verify your data format
   - The application will confirm when validation is successful

4. **Run the forecast**:
   - Configure forecast parameters if needed
   - Click "Generate Forecast"
   - View the results in the visualization panel

## Data Format Requirements

The application expects a CSV file with:

- A `Month` column with date values (YYYY-MM-DD format)
- A `Net items sold` column with numeric sales values

Optional columns for enhanced forecasting accuracy:
- `Discounts`
- `promotion`
- `discount_flag`
- `web_traffic`
- `weather_score`

Example CSV format:
```
Month,Net items sold,Discounts,promotion
2020-01-01,150,-25,1
2020-02-01,125,0,0
2020-03-01,175,-30,0
```

## Features

- **Time Series Visualization**: View historical sales patterns and forecasts
- **Dual Forecasting Models**: Compare SARIMA and LSTM predictions
- **Performance Metrics**: Evaluate model accuracy with RMSE, MAE, and SMAPE
- **Scenario Planning**: Test different forecast horizons and model parameters
- **Export Results**: Save forecasts and visualizations to the `final_plots` directory

## Model Details

### SARIMA (Statistical Model)
- Incorporates seasonality, trend, and residual components
- Handles monthly patterns in sales data
- Order (1,1,1) with seasonal order (1,1,1,12)

### LSTM (Deep Learning Model)
- Captures complex non-linear patterns
- Two variations:
  - Pure LSTM: Using only sales data
  - Combined LSTM: Incorporating SARIMA outputs and exogenous variables

## Troubleshooting

- **Data validation errors**: Ensure your CSV follows the required format with proper date formatting
- **TensorFlow issues**: Check compatible versions for your system
- **Missing visualizations**: Verify matplotlib is installed correctly



## Contributors
Ifemide Cole
