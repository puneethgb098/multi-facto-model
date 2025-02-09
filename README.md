# Portfolio Factor Analytics Dashboard

This repository contains a Streamlit-based dashboard that performs multi-factor analysis and portfolio diagnostics for a portfolio of equity indices. The app downloads historical price data from Yahoo Finance, stores and processes the data in SQLite databases, and computes various performance metrics and factor exposures. The results are visualized using interactive Plotly charts.

## Overview

The project is organized into the following parts:

- **Data Fetching & Processing:**  
  The `Data` class downloads historical data, processes it (combining, cleaning, and computing returns), and stores it in a SQLite database.

  ```python
  class Data:
      def __init__(self, indices, start_date, end_date, interval, db_name):
          self.fetch_data(indices, start_date, end_date, interval)
          self.combine_data()
          self.setup_database(db_name)
          self.populate_database()
          self.fetch_and_process_data()
    # ...

- **Factor Analysis:**  
  The `FactorAnalysis` class performs rolling regressions, rolling correlations, and principal component analysis (PCA) on the portfolio and factor returns to understand the impact of various factors.

  ```python
  class FactorAnalysis:
      def __init__(self, portfolio_returns, factor_returns):
          self._validate_initial_inputs(portfolio_returns, factor_returns)
          self.portfolio, self.factors = self._align_and_preprocess(portfolio_returns, factor_returns)
          self._post_alignment_validation()
    # ...

- **Performance Diagnostics:**  
  Diagnostic functions calculate key metrics such as annualized return, annualized volatility, Sharpe Ratio, and maximum drawdown.

- **Streamlit Dashboard:**  
  The main file (`streamlit_app.py`) sets up a user interface where you can adjust portfolio allocations, choose analysis dates, and view interactive charts and metrics.

    ```python
  def main():
      st.set_page_config(layout="wide", page_title="Portfolio Analytics Dashboard")
      st.title("Portfolio Factor Analytics Dashboard")
    
      if "run_dashboard" not in st.session_state:
          st.session_state.run_dashboard = False

      def reset_run_dashboard():
          st.session_state.run_dashboard = False
    # ...

## Code Structure

The code is divided into several parts in `streamlit_app.py`. Below are some key sections with examples of how code is included in the README.




