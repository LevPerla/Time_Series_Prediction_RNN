#################################           Load libs                      #############################################
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model, clone_model
from ts_rnn.utils import Timer, save_image, split_sequence, train_test_split
from sklearn.utils.validation import check_X_y, column_or_1d, _assert_all_finite
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization


#################################           Model Class                    #############################################

class TS_RNN:
    """ A class for an building and inferencing an RNN models  """

    def __init__(self, configs, n_step_in, n_step_out, test_len,
                 n_features=0, loss="mae", optimizer="adam", n_models=False):
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
        self.model = None
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
        self.n_models = n_models

        if self.n_models:
            n_step_out = self.n_step_out
            self.n_step_out = 1
            self._build_model()
            self.n_step_out = n_step_out

            for _ in range(self.n_step_out):
                model = self._build_model()
                if self.model is None:
                    self.model = [model]
                else:
                    self.model.append(model)
        else:
            self.model = self._build_model()

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

        assert target is not None

        self.last_known_target = target[-self.n_step_in:]
        if factors is not None:
            self.last_known_factors = factors[-self.n_step_in:, :]

        for model_id in range(self.n_step_out):
            if not self.n_models:
                model_id = None
            X_train, y_train, X_test, y_test = self._data_process(factors=factors, target=target, _i_model=model_id)

            if verbose != 0:
                print('[Model] Training Started')
                print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

            if (X_test is None) or (y_test is None):
                validation_data = None
                callbacks = None
            else:
                validation_data = (X_test, y_test)

            if not self.n_models:
                model_to_train = self.model
            else:
                model_to_train = self.model[model_id]
            history = model_to_train.fit(X_train,
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
            if not self.n_models:
                break
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
        if self.n_step_out != 1 and (not self.n_models):  # if multi-step prediction
            predicted = self._predict_multi_step(input_df)

        elif self.n_step_out != 1 and self.n_models:  # if n_models prediction
            predicted = self._predict_n_models(input_df)

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

        model = Sequential()

        for layer in self.params['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_sequences'] if 'return_sequences' in layer else None
            kernel_regularizer = regularizers.l1(layer['kernel_regularizer']) if 'kernel_regularizer' in layer else None
            recurrent_regularizer = regularizers.l1(
                layer['recurrent_regularizer']) if 'recurrent_regularizer' in layer else None
            bias_regularizer = regularizers.l1(layer['bias_regularizer']) if 'bias_regularizer' in layer else None

            if layer['type'] == 'Dense':
                if self.n_features > 1 and (self.n_step_out == 1):
                    model.add(Dense(self.n_features, activation=activation))
                else:
                    model.add(Dense(self.n_step_out, activation=activation))
            if layer['type'] == 'LSTM':
                model.add(LSTM(neurons,
                               input_shape=(self.n_step_in, self.n_features),
                               kernel_initializer="glorot_uniform",
                               activation=activation,
                               return_sequences=return_seq,
                               kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer,
                               bias_regularizer=bias_regularizer))
            if layer['type'] == 'GRU':
                model.add(GRU(neurons,
                              input_shape=(self.n_step_in, self.n_features),
                              activation=activation,
                              return_sequences=return_seq,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer))
            if layer['type'] == "Bidirectional":
                model.add(Bidirectional(LSTM(neurons,
                                             activation=activation,
                                             return_sequences=return_seq,
                                             kernel_regularizer=kernel_regularizer,
                                             recurrent_regularizer=recurrent_regularizer,
                                             bias_regularizer=bias_regularizer),
                                        input_shape=(self.n_step_in, self.n_features)))
            if layer['type'] == 'Dropout':
                model.add(Dropout(dropout_rate))

            if layer['type'] == 'BatchNormalization':
                model.add(BatchNormalization(scale=False))

        model.compile(loss=self.loss, optimizer=self.optimizer)

        assert isinstance(model.layers[-1], Dense), "last block need to be Dense"

        print('[Model] Model Compiled')
        return model
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
        assert self.n_step_out == 1, "Error: Output length needs to be 1. Use method predict_multi_step"
        assert (prediction_len != 0) or (prediction_len is not None), "Error: prediction_len is 0"

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

    def _predict_n_models(self, data):
        """
        Predict only 1 step ahead by one model fot each value in prediction_len
        :param data: np.array, the last sequence of true data
        :return: np.array of predictions
        """

        predicted = []
        var = np.reshape(data, (1, self.n_step_in, self.n_features))

        assert len(self.model) == self.n_step_out

        for i in range(len(self.model)):
            # Prediction RNN for i step
            prediction_point = self.model[i].predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
        predicted = np.array(predicted)
        return predicted

    def _data_process(self, factors=None, target=None, _i_model=None):
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
                                          _all=ALL,
                                          _i_model=_i_model)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], input_df[0].size))

        # prepare X_test and y_test
        if self.test_len == 0:
            X_test, y_test = None, None
        else:
            X_test, y_test = split_sequence(input_df, n_steps_in=self.n_step_in,
                                            n_steps_out=self.n_step_out, _all=ALL, _i_model=_i_model)
            X_test = X_test[-len(test):]
            y_test = y_test[-len(test):]
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.n_features))

        return X_train, y_train, X_test, y_test

    def forecast(self, prediction_len):
        predicted = self.predict(factors=self.last_known_factors,
                                 target=self.last_known_target,
                                 prediction_len=prediction_len)
        return predicted
