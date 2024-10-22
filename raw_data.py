import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy

end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Use 1 year of historical data
symbols = ['AAPL']

data = yf.download(symbols, start=start_date, end=end_date)

# Save stock data to CSV
stock_data_filename = f'{symbols[0]}_stock_data.csv'
data.to_csv(stock_data_filename)
print(f"Stock data saved to {stock_data_filename}")
