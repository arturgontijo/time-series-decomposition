import pandas as pd

from fbprophet import Prophet
from statsmodels.tsa.seasonal import STL

import logging


logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("series_decomposition")


class DecompositionForecast:
    def __init__(self, output_msg):
        self.output_msg = output_msg

    def run(self, ds, y, period, points=365):
        stl = STL(y, period=period).fit()
        log.info("Forecasting...")
        # Prophet
        df = pd.DataFrame(data={"ds": ds, "y": y})
        df["ds"] = pd.to_datetime(df["ds"])
        m = Prophet()
        m.fit(df)
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
