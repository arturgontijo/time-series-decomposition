import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()

from fbprophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


def analysis(file_name="",
             ds_col="Date",
             target_col="Price",
             points=365,
             stl_period=5,
             invert=True,
             normalize=False):

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
