import cntk

import numpy as np
import pandas as pd
import time

from statsmodels.tsa.seasonal import STL

import logging


logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("series_decomposition")


class DecompositionForecast:
    def __init__(self, output_msg):
        self.output_msg = output_msg

    @staticmethod
    def split_data(data, val_size=0.1, test_size=0.1):
        """
        splits np.array into training, validation and test
        """
        pos_test = int(len(data) * (1 - test_size))
        pos_val = int(len(data[:pos_test]) * (1 - val_size))
        train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]
        return {"train": train, "val": val, "test": test}

    def generate_data(self, data, time_steps, time_shift):
        """
        generate sequences to feed to rnn for fct(x)
        """
        df = pd.DataFrame(dict(a=data[0:len(data) - time_shift], b=data[time_shift:]))

        rnn_x = []
        for i in range(len(df) - time_steps + 1):
            rnn_x.append(df['a'].iloc[i: i + time_steps].values)
        rnn_x = np.array(rnn_x, dtype=np.float32)
        rnn_x = rnn_x.reshape(rnn_x.shape + (1,))
    
        rnn_y = np.array(df['b'].values, dtype=np.float32)
        rnn_y = rnn_y[time_steps - 1:]
        rnn_y = rnn_y.reshape(rnn_y.shape + (1,))
        return self.split_data(rnn_x), self.split_data(rnn_y)

    @staticmethod
    def create_model(x, input_dim):
        """Create the model for time series prediction"""
        with cntk.layers.default_options(initial_state=0.1):
            m = cntk.layers.Recurrence(cntk.layers.LSTM(input_dim))(x)
            m = cntk.sequence.last(m)
            m = cntk.layers.Dropout(0.2, seed=1)(m)
            m = cntk.layers.Dense(1)(m)
            return m

    @staticmethod
    def next_batch(x, y, ds, batch_size):
        """get the next batch to process"""
        def as_batch(data, start, count):
            part = []
            for j in range(start, start + count):
                part.append(data[j])
            return np.array(part)
        for i in range(0, len(x[ds]) - batch_size, batch_size):
            yield as_batch(x[ds], i, batch_size), as_batch(y[ds], i, batch_size)

    def run(self, series, period, epochs=1000, batch_size=100, input_dim=5):
    
        max_value = max(series)
        min_value = min(series)
        max_abs = max_value if max_value >= abs(min_value) else abs(min_value)
        normalized_series = [x/(max_abs*10) for x in series]
        normalized_series = np.array(normalized_series, dtype=np.float32)
        stl = STL(normalized_series, period=period).fit()

        x, y = self.generate_data(stl.seasonal, time_steps=input_dim, time_shift=input_dim)
        x_axes = [cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()]
        cntk.input_variable(1, dynamic_axes=x_axes)
    
        # create the model
        input_seq = cntk.sequence.input_variable(1)
        z = self.create_model(input_seq, input_dim)
        lb = cntk.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")
        learning_rate = 0.02
        lr_schedule = cntk.learning_parameter_schedule(learning_rate)
        loss = cntk.squared_error(z, lb)
        error = cntk.squared_error(z, lb)

        momentum_schedule = cntk.momentum_schedule(0.9, minibatch_size=batch_size)
        learner = cntk.fsadagrad(z.parameters,
                                 lr=lr_schedule,
                                 momentum=momentum_schedule,
                                 unit_gain=True)
    
        trainer = cntk.Trainer(z, (loss, error), [learner])
    
        # train
        loss_summary = []
        start = time.time()
        for epoch in range(0, epochs):
            for x1, y1 in self.next_batch(x, y, "train", batch_size):
                trainer.train_minibatch({input_seq: x1, lb: y1})
            if epoch % (epochs / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                log.info("epoch: {}, loss: {:.4f} [time: {:.1f}s]".format(epoch,
                                                                          training_loss,
                                                                          time.time() - start))
        log.info("training took {0:.1f} sec".format(time.time() - start))

        log.info("Forecasting...")
        res_dict = {"train": [], "val": [], "test": []}
        for j, ds in enumerate(["train", "val", "test"]):
            res_dict[ds] = []
            for x1, y1 in self.next_batch(x, y, ds, batch_size=input_dim):
                pred = z.eval({input_seq: x1})
                res_dict[ds].extend(pred[:, 0])

        return self.output_msg(observed=stl.observed,
                               trend=stl.trend,
                               seasonal=stl.seasonal,
                               forecast=res_dict["test"])
