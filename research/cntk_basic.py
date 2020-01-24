import numpy as np
import pandas as pd
import time

import cntk as C


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits np.array into training, validation and test
    """
    pos_test = int(len(data) * (1 - test_size))
    pos_val = int(len(data[:pos_test]) * (1 - val_size))
    
    train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]
    
    return {"train": train, "val": val, "test": test}


def generate_data(data, time_steps, time_shift):
    """
    generate sequences to feed to rnn for fct(x)
    """
    # data = fct(x)
    # if not isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(dict(a=data[0:len(data) - time_shift],
    #                              b=data[time_shift:]))
    
    df = pd.DataFrame(dict(a=data[0:len(data) - time_shift], b=data[time_shift:]))
    
    rnn_x = []
    for i in range(len(df) - time_steps + 1):
        rnn_x.append(df['a'].iloc[i: i + time_steps].values)
    rnn_x = np.array(rnn_x, dtype=np.float32)
    
    # Reshape or rearrange the data from row to columns
    # to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
    rnn_x = rnn_x.reshape(rnn_x.shape + (1,))
    
    rnn_y = np.array(df['b'].values, dtype=np.float32)
    rnn_y = rnn_y[time_steps - 1:]
    
    # Reshape or rearrange the data from row to columns
    # to match the input shape
    rnn_y = rnn_y.reshape(rnn_y.shape + (1,))
    
    return split_data(rnn_x), split_data(rnn_y)


def create_model(x, input_dim):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state=0.1):
        m = C.layers.Recurrence(C.layers.LSTM(input_dim))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2, seed=1)(m)
        m = C.layers.Dense(1)(m)
        return m


def next_batch(x, y, ds, batch_size):
    """get the next batch to process"""
    
    def as_batch(data, start, count):
        part = []
        for j in range(start, start + count):
            part.append(data[j])
        return np.array(part)
    
    for i in range(0, len(x[ds]) - batch_size, batch_size):
        yield as_batch(x[ds], i, batch_size), as_batch(y[ds], i, batch_size)


def lstm_basic(x, y, epochs=1000, batch_size=100, input_dim=5):

    x_axes = [C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()]
    C.input_variable(1, dynamic_axes=x_axes)
    
    # input sequences
    input_seq = C.sequence.input_variable(1)
    
    # create the model
    z = create_model(input_seq, input_dim)
    
    # expected output (label), also the dynamic axes of the model output
    # is specified as the model of the label input
    lb = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")
    
    # the learning rate
    learning_rate = 0.02
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    
    # loss function
    loss = C.squared_error(z, lb)
    
    # use squared error to determine error for now
    error = C.squared_error(z, lb)
    
    # use fsadagrad optimizer
    momentum_schedule = C.momentum_schedule(0.9, minibatch_size=batch_size)
    learner = C.fsadagrad(z.parameters,
                          lr=lr_schedule,
                          momentum=momentum_schedule,
                          unit_gain=True)
    
    trainer = C.Trainer(z, (loss, error), [learner])
    
    # train
    loss_summary = []
    start = time.time()
    for epoch in range(0, epochs):
        for x1, y1 in next_batch(x, y, "train", batch_size):
            trainer.train_minibatch({input_seq: x1, lb: y1})
        if epoch % (epochs / 10) == 0:
            training_loss = trainer.previous_minibatch_loss_average
            loss_summary.append(training_loss)
            print("epoch: {}, loss: {:.4f} [time: {:.1f}s]".format(epoch,
                                                                   training_loss,
                                                                   time.time() - start))
    print("training took {0:.1f} sec".format(time.time() - start))
    
    return z, trainer, input_seq


def forecast(z, input_seq, x, y, batch_size):
    # predict
    res_dict = {"train": [], "val": [], "test": []}
    for j, ds in enumerate(["train", "val", "test"]):
        res_dict[ds] = []
        for x1, y1 in next_batch(x, y, ds, batch_size):
            pred = z.eval({input_seq: x1})
            res_dict[ds].extend(pred[:, 0])
    return res_dict
