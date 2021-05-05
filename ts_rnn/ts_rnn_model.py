#################################           Load libs                      #############################################
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from ts_rnn.utils import Timer, save_image, split_sequence, train_test_split
from sklearn.utils.validation import check_X_y, column_or_1d, _assert_all_finite
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization


#################################           Model Class                    #############################################

class TS_RNN:
    """ A class for an building and inferencing an RNN models  """

    def __init__(self, configs, n_step_in, n_step_out, test_len, n_features=0, loss="mae", optimizer="adam"):
        """
        Attributes:
        model (Keras.Sequential): Keras model
        id (str): unique uuid of model to logging
        n_step_in (int): len of input vector
        n_step_out (int): len of output vector
        n_features (int): number time series in input
        params (dict): dict with all info of model
        last_obs_index (str): date index of last observation showed to model
        score (float): five decades accuracy
        """
        self.model = Sequential()
        self.id = str(uuid.uuid1())
        self.n_step_in = n_step_in
        self.n_step_out = n_step_out
        self.n_features = n_features + 1
        self.params = configs
        self.test_len = test_len
        self.loss = loss
        self.optimizer = optimizer
        self.last_known_target = None
        self.last_known_factors = None

        self._build_model()

    def load_model(self, filepath):
        """
        A method for loading model from h5 file
        :param filepath: (str) path to h5 file with model
        :return: None
        """
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def fit(self, target=None, factors=None,
            epochs=30,
            batch_size=36,
            verbose=1,
            validation_split=0,
            save_dir=None,
            callbacks=None):
        """
        Train model
        :param target: (np.array)
        :param factors: (np.array)
        :param epochs: (int)
        :param batch_size: (int)
        :param verbose: (int) printing fitting process
        :param validation_split: (float) percent of train data used in validation
        :param callbacks: callbacks for EarlyStopping
        :param save_dir: (str) path to saving history plot
        :return: self
        """

        timer = Timer()
        timer.start()
        assert target is not None

        self.last_known_target = target[-self.n_step_in:]
        if factors is not None:
            self.last_known_factors = factors[-self.n_step_in:, :]

        X_train, y_train, X_test, y_test = self._data_process(factors=factors, target=target)

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        if (X_test is None) or (y_test is None):
            validation_data = None
            callbacks = None
        else:
            validation_data = (X_test, y_test)

        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_split=validation_split,
                                 validation_data=validation_data,
                                 batch_size=batch_size,
                                 callbacks=callbacks,
                                 shuffle=False
                                 )

        if save_dir is not None and ((validation_split != 0) or (validation_data is not None)):
            plt.subplot(212)
            plt.plot(history.history["loss"], label="Train")
            plt.plot(history.history["val_loss"], label="Validation")
            plt.legend(loc="best")
            plt.tight_layout()
            save_image(save_dir, "train_val_loss_plot")
            plt.show()
            plt.close()
        timer.stop()
        return self

    def predict(self, target=None, factors=None, prediction_len=None):
        """
        Prediction with auto-set method based by params
        :param factors: np.array
        :param target: np.array
        :param prediction_len: int
        :return: np.array of predictions
        """

        # Some tests
        assert target is not None
        assert prediction_len is not None
        self.prediction_len = prediction_len

        if factors is None:
            target = column_or_1d(target, warn=True)
            _assert_all_finite(target)
            assert len(target) == self.n_step_in
        else:
            factors, target = check_X_y(factors, target)
            assert factors.shape[0] == self.n_step_in
            assert factors.shape[1] == self.n_features - 1

        if factors is not None:
            input_df = np.hstack((factors, target.reshape(-1, 1)))
        else:
            input_df = target.reshape(-1, 1)

        # if multi-step prediction
        if self.n_step_out != 1:  # if multi-step prediction
            predicted = self._predict_multi_step(input_df)

        # if point-by-point prediction
        else:
            predicted = self._predict_point_by_point(input_df, prediction_len=self.prediction_len)

        return predicted

    def _build_model(self):
        """
        A method for building and compiling RNN models
        :return: None
        """
        timer = Timer()
        timer.start()

        for layer in self.params['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_sequences'] if 'return_sequences' in layer else None
            kernel_regularizer = regularizers.l1(layer['kernel_regularizer']) if 'kernel_regularizer' in layer else None
            recurrent_regularizer = regularizers.l1(layer['recurrent_regularizer']) if 'recurrent_regularizer' in layer else None
            bias_regularizer = regularizers.l1(layer['bias_regularizer']) if 'bias_regularizer' in layer else None

            if layer['type'] == 'Dense':
                if self.n_features > 1 and (self.n_step_out == 1):
                    self.model.add(Dense(self.n_features, activation=activation))
                else:
                    self.model.add(Dense(self.n_step_out, activation=activation))
            if layer['type'] == 'LSTM':
                self.model.add(LSTM(neurons,
                                    input_shape=(self.n_step_in, self.n_features),
                                    kernel_initializer="glorot_uniform",
                                    activation=activation,
                                    return_sequences=return_seq,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer))
            if layer['type'] == 'GRU':
                self.model.add(GRU(neurons,
                                   input_shape=(self.n_step_in, self.n_features),
                                   activation=activation,
                                   return_sequences=return_seq,
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   bias_regularizer=bias_regularizer))
            if layer['type'] == "Bidirectional":
                self.model.add(Bidirectional(LSTM(neurons,
                                                  activation=activation,
                                                  return_sequences=return_seq,
                                                  kernel_regularizer=kernel_regularizer,
                                                  recurrent_regularizer=recurrent_regularizer,
                                                  bias_regularizer=bias_regularizer),
                                             input_shape=(self.n_step_in, self.n_features)))
            if layer['type'] == 'Dropout':
                self.model.add(Dropout(dropout_rate))

            if layer['type'] == 'BatchNormalization':
                self.model.add(BatchNormalization(scale=False))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer)

        assert isinstance(self.model.layers[-1], Dense), "last block need to be Dense"

        print('[Model] Model Compiled')
        timer.stop()

    def _predict_point_by_point(self, data, prediction_len):
        """
        Predict only 1 step ahead each time of prediction_len
        :param data: np.array, the last sequence of true data
        :param prediction_len: int, prediction horizon length
        :return: np.array of predictions
        """
        # print('[Model] Predicting Point-by-Point')

        # Check output length
        if self.n_step_out != 1:
            print("Error: Output length needs to be 1. Use method predict_multi_step")
            return None
        if (prediction_len == 0) or (prediction_len is None):
            print("Error: prediction_len is 0")
            return None

        predicted = []
        past_targets = data
        var = np.reshape(past_targets, (1, self.n_step_in, self.n_features))

        for i in range(prediction_len):
            # Prediction RNN for i step
            prediction_point = self.model.predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
            past_targets = np.concatenate((past_targets[1:], prediction_point))
            var = np.reshape(past_targets, (1, self.n_step_in, self.n_features))
        predicted = np.array(predicted)
        return predicted

    def _predict_multi_step(self, data):
        """
        Predict n_step_out steps ahead. Use it if n_step_out != 1
        :param data : np.array, the last sequence of true data
        :return: np.array of predictions
        """
        var = np.reshape(data, (1, self.n_step_in, self.n_features))
        return self.model.predict(var)

    def _data_process(self, factors=None, target=None):
        """
        Process input series
        :param X: np.array, factors
        :param y: np.array,target
        """

        # Some tests
        assert target is not None

        if factors is None:
            y = column_or_1d(target, warn=True)
            _assert_all_finite(target)
        else:
            factors, target = check_X_y(factors, target)

        if factors is not None:
            input_df = np.hstack((factors, target.reshape(-1, 1)))
        else:
            input_df = target.reshape(-1, 1)

        # Train/ Test split
        train, test = train_test_split(input_df, test_len=self.test_len)

        # split into samples
        if factors is not None and (self.n_step_out == 1):
            ALL = True
        else:
            ALL = False

        X_train, y_train = split_sequence(train,
                                           n_steps_in=self.n_step_in,
                                           n_steps_out=self.n_step_out,
                                           all=ALL)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], input_df[0].size))

        # prepare X_test and y_test
        if self.test_len == 0:
            X_test, y_test = None, None
        else:
            X_test, y_test = split_sequence(input_df, n_steps_in=self.n_step_in,
                                            n_steps_out=self.n_step_out, all=ALL)
            X_test = X_test[-len(test):]
            y_test = y_test[-len(test):]
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.n_features))

        return X_train, y_train, X_test, y_test

    def forecast(self, prediction_len):
        predicted = self.predict(factors=self.last_known_factors,
                                 target=self.last_known_target,
                                 prediction_len=prediction_len)
        return predicted
