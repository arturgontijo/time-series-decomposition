import time

import pandas as pd
from pandas_datareader import data as pd_data

import numpy as np

from statsmodels.tsa.seasonal import STL

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()


def get_ticker_data(ticker, start_date, end_date):
    retry_cnt, max_num_retry = 0, 3
    while retry_cnt < max_num_retry:
        try:
            return pd_data.DataReader(ticker, "yahoo", start_date, end_date)
        except Exception as e:
            print(e)
            retry_cnt += 1
            time.sleep(np.random.randint(1, 10))
    print("yahoo is not reachable")
    return pd.DataFrame()


def get_stl(ticker, start_date, end_date):
    df = get_ticker_data(ticker, start_date, end_date)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.resample('W').last()
    return df, STL(df["Close"]).fit()


def main():
    ticker = input("Ticker: ")
    start_date = input("Start Date: ")
    end_date = input("End Date: ")
    df, stl = get_stl(ticker, start_date, end_date)

    plt.rcParams["figure.figsize"] = [12, 9]

    stl.plot()
    plt.show()


if __name__ == '__main__':
    main()
