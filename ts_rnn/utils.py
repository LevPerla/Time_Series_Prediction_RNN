#################################           Load libs                      #############################################

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product
import datetime as dt
import numpy as np
import random
import os

#################################           Timer class                    #############################################
class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))

#################################          Metrics calculation             #############################################
def metrics_eval(y_true, y_pred, print_result=True, save_dir=None):
    """
    Evaluate MAE, MSE, SMAPE, RMSE, MAPE metrics
    :param y_true: np.array of true values
    :param y_pred: np.array of predicted values
    :param res: bool, printing results
    :return: list of metrics
    """
    # Mean absolute error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # SMAPE is an alternative for MAPE when there are zeros in the testing data. It
    # scales the absolute percentage by the sum of forecast and observed values
    SMAPE = np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2))) * 100

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate the Mean Absolute Percentage Error
    # y, predictions = check_array(y, predictions)
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if print_result:
        print('Mean Absolute Error:', round(mae, 3))
        print('Mean Squared Error:', round(mse, 3))
        print('Root Mean Squared Error:', round(rmse, 3))
        print('Mean absolute percentage error:', round(MAPE, 3))
        print('Scaled Mean absolute percentage error:', round(SMAPE, 3))

    if save_dir is not None:
        with open(save_dir + '/metrics.txt', 'w') as record_file:
            record_file.write('Mean Absolute Error:' + str(round(mae, 3)))
            record_file.write('\nMean Squared Error:' + str(round(mse, 3)))
            record_file.write('\nRoot Mean Squared Error:' + str(round(rmse, 3)))
            record_file.write('\nMean absolute percentage error:' + str(round(MAPE, 3)))
            record_file.write('\nScaled Mean absolute percentage error:' + str(round(SMAPE, 3)))
            record_file.close()

    return mae, mse, rmse, MAPE, SMAPE

#################################          Saving images                   #############################################
def save_image(save_dir, name, fmt="png"):
    pwd = os.getcwd()
    os.chdir(save_dir)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)

#################################          Plot train/test/predicted       #############################################
def train_test_pred_plot(train, test, predicted, save_dir=None):
    # train_test_pred_plot
    plt.plot(range(len(train)), train, label="Train")
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, "train_test_predicted")
        plt.show()
    else:
        plt.show()

    # test_pred_plot
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, "test_predicted")
        plt.close()
    else:
        plt.show()

#################################          Satting seed                    #############################################
def set_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    # tf.random.set_seed(seed_value)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)

#################################          config_generator                #############################################
def config_generator(new_config, config):
    '''

    '''

    # Convert dict to [full_key, value]
    # [
    #  [[key11, key12..], value1],
    #  [[key21, key22..], value2],
    # ..]

    dict_to_list = {}

    def read_new_dict(new_dict, parents):
        if not isinstance(new_dict, dict) or len(new_dict) == 0:
            dict_to_list[parents] = new_dict
            return None

        for curr_parent in new_dict:
            read_new_dict(new_dict[curr_parent], parents + (curr_parent,))

    read_new_dict(new_config, ())


    for full_key, value in list(dict_to_list.items()):
        temp_config = config

        for key in full_key[:-1]:
            if key in temp_config:
                temp_config = temp_config[key]
            else:
                temp_config[key] = {}
                temp_config = temp_config[key]

        if isinstance(value, list) or isinstance(value, tuple):
            assert len(value)
            temp_config[full_key[-1]] = value[0]
        else:
            temp_config[full_key[-1]] = value
            del dict_to_list[full_key]


    for values in product(*dict_to_list.values()):
        for ind, full_key in enumerate(dict_to_list):
            temp_config = config

            for key in full_key[:-1]:
                temp_config = temp_config[key]

            temp_config[full_key[-1]] = values[ind]
        yield config


#################################          split_sequence                  #############################################
def split_sequence(data, n_steps_in, n_steps_out, all=False):
    """
    Split sequence to X and y
    :param data: (np.ndarray) Sequences to split
    :param n_steps_in: (int) input size of NN
    :param n_steps_out: (int) output size of NN
    :param all: (bool)  output with factors (used in point-by-point prediction with factors)
    :return: np.array, np.array: X sequence, y sequence
    """
    X, y = list(), list()
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_steps_in + 1

        out_end_ix = end_ix + n_steps_out - 1

        # check if we are beyond the sequence
        if out_end_ix > len(data):
            break

        # gather input and output parts of the pattern
        if all:
            seq_x, seq_y = data[i:end_ix - 1, :], data[end_ix - 1: out_end_ix, :].flatten()
        else:
            seq_x, seq_y = data[i:end_ix - 1, :], data[end_ix - 1: out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


#################################          train_test_split                #############################################
def train_test_split(data, test_len):
    """
    Train/ Test split
    :param data: (np.array) Sequence to split
    :param test_len: (int) length of test sequence
    :return: np.array, np.array: train sequence, test sequence
    """
    # print("[DataProcessor] Train/Test split")
    if test_len == 0:
        return data, None
    return data[:-test_len], data[-test_len:]
