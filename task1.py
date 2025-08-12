
     import yfinance as yf
     import pandas as pd
     import numpy as np
     import matplotlib.pyplot as plt
     import seaborn as sns
     from statsmodels.tsa.stattools import adfuller

     # Set plot style
     plt.style.use('ggplot')
     sns.set_style('whitegrid')

     # Define assets and date range
     assets = ['TSLA', 'BND', 'SPY']
     start_date = '2015-07-01'
     end_date = '2025-07-31'

     # Fetch data
     data = yf.download(assets, start=start_date, end=end_date)
     prices = data['Close'].copy()

     # Clean data
     print("Missing values before cleaning:\n", prices.isna().sum())
     prices = prices.fillna(method='ffill')
     print("Missing values after cleaning:\n", prices.isna().sum())

     # Basic statistics
     print("\nBasic Statistics for Close Prices:\n", prices.describe())

     # Daily returns
     returns = prices.pct_change().dropna()

     # Rolling volatility (30-day, annualized)
     rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)

     # ADF test for stationarity
     print("\nADF Test Results:")
     for asset in assets:
         adf_price = adfuller(prices[asset], autolag='AIC')
         adf_return = adfuller(returns[asset], autolag='AIC')
         print(f"{asset} - Prices: ADF Stat={adf_price[0]:.2f}, p-value={adf_price[1]:.4f}")
         print(f"{asset} - Returns: ADF Stat={adf_return[0]:.2f}, p-value={adf_return[1]:.4f}")

     # Value at Risk (95% confidence)
     var_95 = returns.quantile(0.05)
     print("\nValue at Risk (95% Confidence):\n", var_95)

     # Sharpe Ratio (risk-free rate = 0)
     sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
     print("\nSharpe Ratio:\n", sharpe)

     # Save plots for report
     plt.figure(figsize=(12, 6))
     prices.plot(title='Closing Prices (TSLA, BND, SPY)')
     plt.xlabel('Date')
     plt.ylabel('Price (USD)')
     plt.savefig('closing_prices.png')
     plt.close()

     plt.figure(figsize=(12, 6))
     returns.plot(title='Daily Returns')
     plt.xlabel('Date')
     plt.ylabel('Daily Return')
     plt.savefig('daily_returns.png')
     plt.close()

     plt.figure(figsize=(12, 6))
     rolling_vol.plot(title='30-Day Rolling Volatility (Annualized)')
     plt.xlabel('Date')
     plt.ylabel('Volatility')
     plt.savefig('rolling_vol.png')
     plt.close()

     print("EDA complete. Plots saved as PNGs for report.")
     
