import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import os


class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


# Metrics calculation
def metrics_eval(y, predictions, res=True):
    # Import library for metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Mean absolute error (MAE)
    mae = mean_absolute_error(y, predictions)

    # Mean squared error (MSE)
    mse = mean_squared_error(y, predictions)

    # SMAPE is an alternative for MAPE when there are zeros in the testing data. It
    # scales the absolute percentage by the sum of forecast and observed values
    SMAPE = np.mean(np.abs((y - predictions) / ((y + predictions) / 2))) * 100

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y, predictions))

    # Calculate the Mean Absolute Percentage Error
    # y, predictions = check_array(y, predictions)
    MAPE = np.mean(np.abs((y - predictions) / y)) * 100

    if res:
        print('Mean Absolute Error:', round(mae, 3))
        print('Mean Squared Error:', round(mse, 3))
        print('Root Mean Squared Error:', round(rmse, 3))
        print('Mean absolute percentage error:', round(MAPE, 3))
        print('Scaled Mean absolute percentage error:', round(SMAPE, 3))

    return [mae, mse, rmse, MAPE, SMAPE]


# Custom metric of
def trend_accuracy(y_true, y_pred, last_train_value):
    res_list = []
    last_true_value = last_train_value
    last_pred_value = last_train_value

    for i in range(len(y_true)):
        if (y_true[i] - last_true_value == 0 and y_pred[i] - last_pred_value == 0) or \
                (y_true[i] - last_true_value > 0 and y_pred[i] - last_pred_value > 0) or \
                (y_true[i] - last_true_value < 0 and y_pred[i] - last_pred_value < 0):
            res_list.append(1)
        else:
            res_list.append(0)

        last_true_value = y_true[i]
        last_pred_value = y_pred[i]
    return sum(res_list) / len(res_list)


# Plot train/test/predicted
def train_test_pred_plot(train, test, predicted):
    plt.plot(range(len(train)), train, label="Train")
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")
    plt.legend()
    save_image("train_test_predicted")
    plt.show()

    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")
    plt.legend()
    save_image("test_predicted")
    plt.show()

# Saving images
def save_image(name, fmt="png"):
    pwd = os.getcwd()
    iPath = './pictures/'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)

def set_seed(seed_value):

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)