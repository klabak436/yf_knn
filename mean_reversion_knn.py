import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


def download_price_data(tickers, start, end):
    all_tickers = tickers + ["^GSPC"]
    data = yf.download(all_tickers, start=start, end=end)
    return data['Close']


def calculate_daily_returns(close_prices):
    return close_prices.pct_change().dropna()


def calculate_rolling_beta(daily_returns, tickers, benchmark_ticker, window=60):
    beta_values = {}
    for ticker in tickers:
        cov = daily_returns[benchmark_ticker].rolling(window).cov(daily_returns[ticker])
        var = daily_returns[benchmark_ticker].rolling(window).var()
        beta_values[ticker] = cov / var
    return pd.DataFrame(beta_values)


def add_momentum(close_prices, tickers, window=60):
    for ticker in tickers:
        close_prices[f"{ticker}_Mom_{window}d"] = close_prices[ticker].pct_change(periods=window)
    return close_prices


def add_zscore(close_prices, tickers, window=60):
    for ticker in tickers:
        mean = close_prices[ticker].rolling(window).mean()
        std = close_prices[ticker].rolling(window).std()
        close_prices[f"{ticker}_ZScore_{window}d"] = (close_prices[ticker] - mean) / std
    return close_prices


def add_skew_kurtosis(daily_returns, tickers, window=60):
    for ticker in tickers:
        daily_returns[f"{ticker}_Skew_{window}d"] = daily_returns[ticker].rolling(window).skew()
        daily_returns[f"{ticker}_Kurt_{window}d"] = daily_returns[ticker].rolling(window).kurt()
    return daily_returns


def main():
    tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK-A', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'NVDA', 'LLY', 'AVGO', 'COST',
        'PFE', 'MRK', 'ABT', 'TMO', 'CRM', 'CMCSA', 'VZ', 'ADBE', 'KO', 'PEP',
        'NKE', 'DIS', 'CSCO', 'INTC', 'ORCL', 'IBM', 'TXN', 'QCOM', 'AMGN', 'MDLZ',
        'SBUX', 'GE', 'LOW', 'RTX', 'CAT', 'BA', 'HON', 'DE', 'MMM', 'AXP'
    ]
    benchmark_ticker = '^GSPC'

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    close_prices = download_price_data(tickers, start_date, end_date)
    daily_returns = calculate_daily_returns(close_prices)

    rolling_beta_df = calculate_rolling_beta(daily_returns, tickers, benchmark_ticker)
    print("\nRolling Beta (60 days):")
    print(rolling_beta_df.tail())

    close_prices = add_momentum(close_prices, tickers)
    print(f"\nMomentum (60 days):\n{close_prices.filter(like='_Mom_').tail()}")

    close_prices = add_zscore(close_prices, tickers)
    print(f"\nNormalized Price (Z-Score over 60 days):\n{close_prices.filter(like='_ZScore_').tail()}")

    daily_returns = add_skew_kurtosis(daily_returns, tickers)
    print(f"\nSkewness and Kurtosis of Daily Returns (60 days):\n{daily_returns.filter(like='_Skew_').join(daily_returns.filter(like='_Kurt_')).tail()}")


if __name__ == "__main__":
    main()

