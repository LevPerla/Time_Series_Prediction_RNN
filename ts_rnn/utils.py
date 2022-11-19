#################################           Load libs                      #############################################
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ts_rnn.logger import logger


#################################           timeit                         #############################################
def timeit(f):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        logger.info(f"[Timing] {f.__name__} takes: {round(te - ts, 2)} sec")
        return result

    return timed

#################################              mae                         #############################################
def mean_absolute_error(y_true, y_pred):
    mae = np.absolute(np.subtract(y_true, y_pred)).mean()
    return mae

#################################              mse                         #############################################
def mean_squared_error(y_true, y_pred):
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    return mse

#################################          Metrics calculation             #############################################
def metrics_eval(y_true, y_pred, print_result=True, save_dir=None, name='metrics'):
    """
    Evaluate MAE, MSE, SMAPE, RMSE, MAPE metrics
    :param y_true: np.array of true values
    :param y_pred: np.array of predicted values
    :param res: bool, printing results
    :return: list of metrics
    """

    metrics_dict = {}

    # Mean absolute error (MAE)
    metrics_dict['Mean Absolute Error'] = round(mean_absolute_error(y_true, y_pred), 3)

    # Mean squared error (MSE)
    metrics_dict['Mean Squared Error'] = round(mean_squared_error(y_true, y_pred), 3)

    # SMAPE is an alternative for MAPE when there are zeros in the testing data. It
    # scales the absolute percentage by the sum of forecast and observed values
    SMAPE = np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2))) * 100
    metrics_dict['Symmetric Mean absolute percentage error'] = round(SMAPE, 3)

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics_dict['Root Mean Squared Error'] = round(rmse, 3)

    # Calculate the Mean Absolute Percentage Error
    # y, predictions = check_array(y, predictions)
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics_dict['Mean absolute percentage error'] = round(MAPE, 3)

    if print_result:
        for key in metrics_dict.keys():
            print(f"{key}: {metrics_dict[key]}")

    if save_dir is not None:
        with open(os.path.join(save_dir, name + '.txt'), 'w') as record_file:
            for key in metrics_dict.keys():
                record_file.write(f"{key}: {metrics_dict[key]}\n")

    return metrics_dict


#################################          Saving images                   #############################################
def save_image(save_dir, name, fmt="png"):
    pwd = os.getcwd()
    os.chdir(save_dir)
    plt.savefig('{}.{}'.format(name, fmt))
    os.chdir(pwd)

#################################          Plot train/test/predicted       #############################################
def train_test_pred_plot(train, test, predicted, save_dir=None, show=True):
    # train_test_pred_plot
    plt.plot(range(len(train)), train, label="Train")
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")

    if isinstance(predicted, dict):
        for key in predicted.keys():
            plt.plot(range(len(train), len(train) + len(predicted[key]["predictions"])),
                     predicted[key]["predictions"],
                     label=key)
    else:
        plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")

    plt.title('train_test_predicted')
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, "train_test_predicted")
    if show:
        plt.show()
    plt.close()

    # test_pred_plot
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
    if isinstance(predicted, dict):
        for key in predicted.keys():
            plt.plot(range(len(train), len(train) + len(predicted[key]["predictions"])),
                     predicted[key]["predictions"],
                     label=key)
    else:
        plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Pred")

    plt.title('test_predicted')
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, "test_predicted")
    if show:
        plt.show()
    plt.close()


#################################          Plot train/val/test/val_pred/test_pred       #############################################
def train_val_test_pred_plot(train, val, test, val_pred, test_pred, save_dir=None, show=True, name_add=''):
    # train_test_pred_plot
    plt.plot(train, label="Train")
    plt.plot(val, label="Val")
    plt.plot(test, label="Test")

    if isinstance(val_pred, dict):
        for key in val_pred.keys():
            plt.plot(val_pred[key]["Val_pred"], label=key)
    else:
        plt.plot(val_pred, label="Val_pred")

    if isinstance(test_pred, dict):
        for key in test_pred.keys():
            plt.plot(test_pred[key]["Test_pred"], label=key)
    else:
        plt.plot(test_pred, label="Test_pred")

    plt.title('train_val_test_predicted')
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, f"train_val_test_predicted_{name_add}")
    if show:
        plt.show()
    plt.close()

    # val_test_pred_plot
    if len(train) > 20:
        plt.plot(train[-20:], label="Train")
    plt.plot(val, label="Val")
    plt.plot(test, label="Test")

    if isinstance(val_pred, dict):
        for key in val_pred.keys():
            plt.plot(val_pred[key]["Val_pred"], label=key)
    else:
        plt.plot(val_pred, label="Val_pred")

    if isinstance(test_pred, dict):
        for key in test_pred.keys():
            plt.plot(test_pred[key]["Test_pred"], label=key)
    else:
        plt.plot(test_pred, label="Test_pred")

    plt.title('val_test_predicted')
    plt.legend()
    if save_dir is not None:
        save_image(save_dir, f"val_test_predicted_{name_add}")
    if show:
        plt.show()
    plt.close()

#################################          split_sequence                  #############################################
def split_sequence(data, n_steps_in, n_steps_out, _full_out=False, _i_model=0, _start_ind=0):
    """
    Split sequence to X and y
    :param data: (np.ndarray) Sequences to split
    :param n_steps_in: (int) input size of NN
    :param n_steps_out: (int) output size of NN
    :param _full_out: (bool)  output with factors (used in point-by-point prediction with factors)
    :param _i_model: (bool)  sample of out (used in n_models prediction)
    :return: np.array, np.array: X sequence, y sequence
    """
    X, y = list(), list()
    for i in range(len(data)):
        # set_index
        in_start_ind = i
        in_end_ind = i + n_steps_in

        out_start_ind = in_end_ind + _i_model + _start_ind
        out_end_ind = out_start_ind + n_steps_out

        # check if we are beyond the sequence
        if out_end_ind - 1 >= len(data):
            break

        # gather input and output parts of the pattern
        if _full_out:
            seq_x, seq_y = data[in_start_ind:in_end_ind, :], data[out_start_ind: out_end_ind, :].flatten()
        else:
            seq_x, seq_y = data[in_start_ind:in_end_ind, :], data[out_start_ind: out_end_ind, -1]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    return X, y


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

#################################             history_plot                 #############################################
def history_plot(history, save_dir=None, show=True):
    plt.subplot(212)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.legend(loc="best")
    plt.tight_layout()
    if save_dir is not None and not show:
        save_image(save_dir, "train_val_loss_plot")
        plt.close()
    elif save_dir is not None and show:
        plt.show()
    else:
        plt.show()
    plt.close()
