#################################           Load libs                      #############################################

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf
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
def metrics_eval(y_true, y_pred, res=True):
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

    if res:
        print('Mean Absolute Error:', round(mae, 3))
        print('Mean Squared Error:', round(mse, 3))
        print('Root Mean Squared Error:', round(rmse, 3))
        print('Mean absolute percentage error:', round(MAPE, 3))
        print('Scaled Mean absolute percentage error:', round(SMAPE, 3))

    return mae, mse, rmse, MAPE, SMAPE

#################################          Trend accuracy metric           #############################################
def trend_accuracy(y_true, y_pred):
    """
    Evaluate trend accuracy metric
    :param y_true: np.array of true values
    :param y_pred: np.array of predicted values
    :return: int, metric output
    """
    res_list = []
    last_true_value = y_true[0]
    last_pred_value = y_pred[0]

    for i in range(1, len(y_true)):
        if (y_true[i] - last_true_value == 0 and y_pred[i] - last_pred_value == 0) or \
                (y_true[i] - last_true_value > 0 and y_pred[i] - last_pred_value > 0) or \
                (y_true[i] - last_true_value < 0 and y_pred[i] - last_pred_value < 0):
            res_list.append(1)
        else:
            res_list.append(0)

        last_true_value = y_true[i]
        last_pred_value = y_pred[i]
    return sum(res_list) / len(res_list)

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
    plt.close()

    # test_pred_plot
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, "test_predicted")
    plt.close()

#################################          Satting seed                    #############################################
def set_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

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

    # По full_key спускаемся через temp_config до value и меняем его
    # Пока value выступает либо сам элемент, либо первый элемент list/tuple

    for full_key, value in list(dict_to_list.items()):
        temp_config = config

        for key in full_key[:-1]:
            if key in temp_config:
                temp_config = temp_config[key]
            else:
                temp_config[key] = {}
                temp_config = temp_config[key]

        if isinstance(value, list) or isinstance(value, tuple):
            assert len(value), 'Передан пустой список'
            temp_config[full_key[-1]] = value[0]
        else:
            temp_config[full_key[-1]] = value
            del dict_to_list[full_key]

    # Генератор пробегает по всем комбинациям элементов config

    for values in product(*dict_to_list.values()):
        for ind, full_key in enumerate(dict_to_list):
            temp_config = config

            for key in full_key[:-1]:
                temp_config = temp_config[key]

            temp_config[full_key[-1]] = values[ind]
        yield config
