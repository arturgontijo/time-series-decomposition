import cntk

import numpy as np
import time


class Trainer:
    def __init__(self, series, epochs, h_dims, batch_size):
        self.series = series
        self.epochs = epochs
        self.batch_size = batch_size
        self.h_dims = h_dims
        self.input_node = cntk.sequence.input_variable(1)
        
    @staticmethod
    def next_batch(x, y, ds, batch_size):
        """get the next batch for training"""
        
        def as_batch(p_data, start, count):
            return p_data[start:start + count]
        
        for i in range(0, len(x[ds]), batch_size):
            yield as_batch(x[ds], i, batch_size), as_batch(y[ds], i, batch_size)
    
    @staticmethod
    def create_model(x_local, h_dims):
        """Create the model for time series prediction"""
        with cntk.layers.default_options(initial_state=0.1):
            m = cntk.layers.Recurrence(cntk.layers.LSTM(h_dims))(x_local)
            m = cntk.sequence.last(m)
            m = cntk.layers.Dropout(0.2)(m)
            m = cntk.layers.Dense(1)(m)
            return m

    def train(self):
        tmp_d = {"x": [], "y": []}
        num_list = []
        count = 0
        for idx, value in enumerate(self.series):
            if idx % self.h_dims == 0:
                num_list = []
                count += 1
                if (self.h_dims * count) > len(self.series):
                    break
            num_list.append(np.float32(value))
            increment_list = []
            for num in num_list:
                increment_list.append(num)
                tmp_d["x"].append(np.array(increment_list))
                tmp_d["y"].append(np.array([np.float32(self.series[self.h_dims * count])]))

        x = {"train": tmp_d["x"]}
        y = {"train": np.array(tmp_d["y"])}

        z = self.create_model(self.input_node, self.h_dims)
        var_l = cntk.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")
        learning_rate = 0.005
        lr_schedule = cntk.learning_parameter_schedule(learning_rate)
        loss = cntk.squared_error(z, var_l)
        error = cntk.squared_error(z, var_l)
        momentum_schedule = cntk.momentum_schedule(0.9, minibatch_size=self.batch_size)
        learner = cntk.fsadagrad(z.parameters,
                                 lr=lr_schedule,
                                 momentum=momentum_schedule)
        trainer = cntk.Trainer(z, (loss, error), [learner])

        # training
        loss_summary = []

        start = time.time()
        for epoch in range(0, self.epochs):
            for x_batch, l_batch in self.next_batch(x, y, "train", self.batch_size):
                trainer.train_minibatch({self.input_node: x_batch, var_l: l_batch})
            if epoch % (self.epochs / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                print("epoch: {}, loss: {:.4f} [time: {:.1f}s]".format(epoch,
                                                                       training_loss,
                                                                       time.time() - start))
        return z

    def forecast(self, model, n):
        _series = [np.float32(v) for v in self.series[-self.h_dims:]]
        results = []
        for p in range(n):
            x = {"pred": []}
            y = {"pred": []}
            num_list = [v for v in _series[-self.h_dims:]]
            increment_list = []
            for num in num_list:
                increment_list.append(num)
                x["pred"].append(np.array(increment_list))
            r = []
            for x_batch, _ in self.next_batch(x, y, "pred", self.h_dims):
                pred = model.eval({self.input_node: x_batch})
                r.extend(pred[:, 0])
            results.append(r[-1])
            _series.append(r[-1])
        return results
