import pandas as pd
import numpy as np
from pandas_datareader import data as pd_data
import pandas_market_calendars as mcal

from fbprophet import Prophet
from statsmodels.tsa.seasonal import STL

import time
import datetime
import logging

pd.plotting.register_matplotlib_converters()


logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("series_decomposition")


class DecompositionForecast:
    def __init__(self, output_msg):
        self.output_msg = output_msg

    @staticmethod
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

    def get_ticker_stl(self, ticker, start_date="2015-01-01", end_date=None, period=5):
        if not end_date:
            end_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
        df = self.get_ticker_data(ticker, start_date, end_date)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df, STL(df["Close"], period=period).fit()

    def run(self, ds, y, period=5, points=365):
        # Financial Series, first element of ds must by the Ticker
        if len(ds) == 1:
            ticker = ds[0]
            df, stl = self.get_ticker_stl(ticker, period=period)
            df = df.reset_index()
            ds = df["Date"].values
            y = df["Close"].values
            financial = True
        else:
            stl = STL(y, period=period).fit()
            financial = False
        log.info("Forecasting...")
        # Prophet
        df = pd.DataFrame(data={"ds": ds, "y": y})
        df["ds"] = pd.to_datetime(df["ds"])
        m = Prophet()
        m.fit(df)
        if financial:
            nyse = mcal.get_calendar('NYSE')
            start_date = datetime.datetime.today()
            end_date = start_date + datetime.timedelta(days=points)
            valid_days = nyse.valid_days(start_date=start_date, end_date=end_date)
            future = pd.DataFrame(data={"ds": [v.date() for v in valid_days]})
            future = pd.DataFrame(data={"ds": df["ds"].append(future["ds"], ignore_index=True)})
        else:
            future = m.make_future_dataframe(periods=points)
        forecast = m.predict(future)
        forecast_df = []
        for dt in forecast["ds"].values:
            ts = pd.to_datetime(dt)
            forecast_df.append(ts.strftime('%Y-%m-%d'))
        return self.output_msg(observed=stl.observed,
                               trend=stl.trend,
                               seasonal=stl.seasonal,
                               forecast=forecast["yhat"].values,
                               forecast_ds=forecast_df,
                               forecast_lower=forecast["yhat_lower"].values,
                               forecast_upper=forecast["yhat_upper"].values)
