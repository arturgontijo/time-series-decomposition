import time
import json

import pandas as pd
from pandas_datareader import data as pd_data

import numpy as np

from statsmodels.tsa.seasonal import STL

# import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

from cntk_v2 import Trainer
from cntk_basic import lstm_basic, generate_data, forecast


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
    epochs = 1000
    batch_size = 100
    input_dim = 5
    output_dim = 5
    x, y = generate_data(stl.seasonal.values, time_steps=input_dim, time_shift=output_dim)
    model, input_seq = lstm_basic(x, y,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  input_dim=input_dim)

    result = forecast(model, input_seq, x, y, batch_size)
    with open("result_{}_epochs.json".format(epochs), "w") as fd:
        json.dump(result, fd, indent=4)

    model.save("LSTM_{}_epochs.model".format(epochs))

    return df, stl, model, result
    
    # t = Trainer(series=stl.seasonal,
    #             epochs=1000,
    #             h_dims=4,
    #             batch_size=500)
    # model = t.train()
    #
    # return model, t.forecast(model, 4)
    # plt.rcParams["figure.figsize"] = [12, 9]
    #
    # stl.plot()
    # plt.show()


if __name__ == '__main__':
    d, s, z, r = main()
