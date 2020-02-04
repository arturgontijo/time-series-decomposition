import time

import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()

from pandas_datareader import data as pd_data

from fbprophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


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


def get_ticker_stl(ticker, start_date, end_date, period):
    df = get_ticker_data(ticker, start_date, end_date)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.resample('W').last()
    return df, STL(df["Close"], period=period).fit()


def analysis(file_name="",
             ds_col="Date",
             target_col="Price",
             points=365,
             stl_period=5,
             invert=True,
             normalize=False,
             norm_log=False):
    fn = file_name if file_name else input("CSV File: ")
    df = pd.read_csv(fn)
    if invert:
        df = df.iloc[::-1]
    df[ds_col] = pd.to_datetime(df[ds_col])
    if normalize:
        max_value = max(df[target_col])
        normalized_series = [x / max_value for x in df[target_col]]
        normalized_series = np.array(normalized_series, dtype=np.float32)
        df[target_col] = normalized_series
    elif norm_log:
        df[target_col] = np.log(df[target_col])
    # STL
    df2 = df[[target_col]]
    df2.index = df[ds_col]
    stl = STL(df2, period=stl_period).fit()
    # Prophet
    df = df.reset_index()
    df = df[[ds_col, target_col]]
    df = df.rename(columns={ds_col: "ds", target_col: "y"})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=points)
    forecast = m.predict(future)
    # Plotting Charts
    name = fn.split("/")[-1].replace(".csv", "")
    stl.plot()
    plt.savefig("{}_STL.png".format(name))
    fig1 = m.plot(forecast)
    fig1.savefig("{}_forecast.png".format(name))
    fig2 = m.plot_components(forecast)
    fig2.savefig("{}_forecast_components.png".format(name))
