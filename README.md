# Portfolio Factor Analytics Dashboard

This repository contains a Streamlit-based dashboard that performs multi-factor analysis and portfolio diagnostics for a portfolio of equity indices. The app downloads historical price data from Yahoo Finance, stores and processes the data in SQLite databases, and computes various performance metrics and factor exposures. The results are visualized using interactive Plotly charts.

## Overview

The project is organized into the following parts:

- **Data Fetching & Processing:**  
  The `Data` class downloads historical data, processes it (combining, cleaning, and computing returns), and stores it in a SQLite database.

- **Factor Analysis:**  
  The `FactorAnalysis` class performs rolling regressions, rolling correlations, and principal component analysis (PCA) on the portfolio and factor returns to understand the impact of various factors.

- **Performance Diagnostics:**  
  Diagnostic functions calculate key metrics such as annualized return, annualized volatility, Sharpe Ratio, and maximum drawdown.

- **Streamlit Dashboard:**  
  The main file (`streamlit_app.py`) sets up a user interface where you can adjust portfolio allocations, choose analysis dates, and view interactive charts and metrics.

## Code Structure

The code is divided into several parts in `streamlit_app.py`. Below are some key sections with examples of how code is included in the README.

### 1. Data Class

The `Data` class handles fetching, processing, and storing financial data. For example:

```python
class Data:
    def __init__(self, indices, start_date, end_date, interval, db_name):
        self.fetch_data(indices, start_date, end_date, interval)
        self.combine_data()
        self.setup_database(db_name)
        self.populate_database()
        self.fetch_and_process_data()
    # ...
