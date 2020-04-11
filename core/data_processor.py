import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor():
    def __init__(self):
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.scaler_factors = MinMaxScaler(feature_range=(0, 1))

    def scaler_fit_transform(self, data, target=True):
        """
        scaler fitting
        :param data (np.array):  Sequence
        :return: np.array of scalled sequence
        """
        if target:
            print("[DataProcessor] Target scaler fitting")
            return self.scaler_target.fit_transform(data.reshape(-1, 1))
        else:
            print("[DataProcessor] Factors scaler fitting")
            return self.scaler_factors.fit_transform(data.reshape(-1, 1))

    def scaler_inverse_transform(self, data, target=True):
        """
        scaler inverse transform
        :param data (np.array):  Sequence
        :return: np.array of unscalled sequence
        """
        if target:
            print("[DataProcessor] Target scaler inverse transform")
            return self.scaler_target.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            print("[DataProcessor] Factors scaler inverse transform")
            return self.scaler_factors.inverse_transform(data.reshape(-1, 1)).flatten()


    def train_test_split(self, data, test_len):
        """
        Train/ Test split
        :param data (np.array):  Sequence to split
        :return: np.array, np.array: train sequence, test sequence
        """
        print("[DataProcessor] Train/Test split")
        if test_len == 0:
            return data, None
        return data[:-test_len], data[-test_len:]

    def split_sequence(self, data, n_steps_in, n_steps_out, all=False):
        """
        Split sequence to X and y
        :param data (np.ndarray):  Sequences to split
        :param n_steps_in (int): input size of NN
        :param n_steps_out (int):  output size of NN
        :param lag (int):  lag between X and y
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
