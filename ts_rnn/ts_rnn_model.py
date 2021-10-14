# Load libs
import os
import numpy as np
import logging
from ts_rnn.logger import logger
from tensorflow.keras.models import Sequential, load_model
from ts_rnn.utils import split_sequence, train_test_split, history_plot, timeit
from sklearn.utils.validation import check_X_y, column_or_1d, _assert_all_finite
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization


class TS_RNN:
    """ A class for an building and inferencing an RNN models for time series prediction"""

    def __init__(self,
                 rnn_arch: dict,
                 n_step_in: int,
                 horizon: int,
                 test_len: int,
                 strategy="MiMo",
                 n_step_out=1,
                 n_features=0,
                 loss="mae",
                 optimizer="adam",
                 save_dir=None):
        """
        :param rnn_arch: dict with layers params
        :param n_step_in: number time series steps in input
        :param horizon: length of prediction horizon
        :param test_len: length of last observations that will be dropped from training
        :param strategy: prediction strategy
        :param n_step_out: number time series steps in out
        :param n_features: number exogeny time series in input
        :param loss: keras loss
        :param optimizer: keras optimizer
        """
        try:
            assert strategy in ["Direct", "Recursive", "MiMo", "DirRec", "DirMo"]
        except AssertionError:
            logger.exception('Use strategy from ["Direct", "Recursive", "MiMo", "DirRec", "DirMo"]')
            raise AssertionError('Use strategy from ["Direct", "Recursive", "MiMo", "DirRec", "DirMo"]')

        if strategy in ["Direct", "Recursive", "DirRec"]:
            assert n_step_out == 1, "For Direct, Recursive and DirRec strategies n_step_out must be 1"

        if strategy == "DirMo":
            possible_n_out = [i + 1 for i in range(10) if 10 % (i + 1) == 0]
            assert horizon % n_step_out == 0, f"For DirMo strategy choose n_step_out from {possible_n_out}"

        self.model_list = None
        self.n_step_in = n_step_in
        self.n_step_out = horizon if strategy == "MiMo" else n_step_out
        self.horizon = horizon
        self.n_features = n_features + 1
        self.params = rnn_arch
        self.test_len = test_len
        self.loss = loss
        self.strategy = strategy
        self.optimizer = optimizer
        self.save_dir = save_dir
        self._last_known_target = None
        self._last_known_factors = None
        self.prediction_len = None

        if save_dir is not None:
            handler = logging.FileHandler(os.path.join(save_dir, "ts_rnn.log"), mode='w')
            logger.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('[%(levelname)s] - %(asctime)s - %(message)s'))
            logger.addHandler(handler)

        # compile model
        self._build_by_stategy()

    def _build_model(self, name):
        """
        A method for building and compiling RNN models
        :return: keras.Sequential()
        """
        _model = Sequential(name=name)

        for layer in self.params['layers']:
            if layer[0] == 'Dense':
                if self.n_features > 1 and (self.n_step_out == 1):
                    _model.add(Dense(self.n_features, **layer[1]))
                else:
                    _model.add(Dense(self.n_step_out, **layer[1]))
            elif layer[0] == 'LSTM':
                _model.add(LSTM(input_shape=(self.n_step_in, self.n_features), **layer[1]))
            elif layer[0] == 'GRU':
                _model.add(GRU(input_shape=(self.n_step_in, self.n_features), **layer[1]))
            elif layer[0] == "Bidirectional":
                _model.add(Bidirectional(LSTM(**layer[1]), input_shape=(self.n_step_in, self.n_features)))
            elif layer[0] == 'Dropout':
                _model.add(Dropout(**layer[1]))
            elif layer[0] == 'BatchNormalization':
                _model.add(BatchNormalization(scale=False, **layer[1]))
            else:
                logger.critical(f"TS_RNN doesn't support layer type {layer[0]}")
                raise AssertionError(f"TS_RNN doesn't support layer type {layer[0]}")

        _model.compile(loss=self.loss, optimizer=self.optimizer)
        assert isinstance(_model.layers[-1], Dense), "last block need to be Dense"

        logger.info(f'[Model building] {name} compiled')
        return _model

    def _build_by_stategy(self):
        """
        Build TS model by prediction strategy
        :return:
        """
        if self.strategy == "Direct":
            for i in range(self.horizon):
                name = f'{self.strategy}_model_{i + 1}'
                _model = self._build_model(name=name)
                if self.model_list is None:
                    self.model_list = [{"model_name": name,
                                        "model": _model}]
                else:
                    self.model_list.append({"model_name": name,
                                            "model": _model})
        if self.strategy in ["Recursive", "MiMo"]:
            name = f"{self.strategy}_model"
            _model = self._build_model(name=name)
            self.model_list = [{"model_name": name,
                                "model": _model}]
        return self

    @timeit
    def fit(self,
            target=None,
            factors=None,
            epochs=30,
            batch_size=36,
            verbose=1,
            validation_split=0,
            callbacks=None,
            **kwargs):
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

        # Tests
        assert target is not None, "Define target argument"

        for model_id in range(len(self.model_list)):

            _X_train, _y_train, _X_test, _y_test = self._data_process(target=target, factors=factors, _i_model=model_id)

            logger.info(f'[Training] Training {self.model_list[model_id]["model_name"]} started on '
                        f'%s epochs, %s batch size' % (epochs, batch_size))

            if (_X_test is None) or (_y_test is None):
                validation_data = None
                callbacks = None
            else:
                validation_data = (_X_test, _y_test)

            history = self.model_list[model_id]["model"].fit(_X_train,
                                                             _y_train,
                                                             epochs=epochs,
                                                             verbose=verbose,
                                                             validation_split=validation_split,
                                                             validation_data=validation_data,
                                                             callbacks=callbacks,
                                                             shuffle=False,
                                                             **kwargs
                                                             )

            if self.save_dir is not None and ((validation_split != 0) or (validation_data is not None)):
                history_plot(history,
                             self.save_dir,
                             show=True if verbose > 0 else False)

        logger.info('[Training] Training ended')
        return self

    @timeit
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

        if factors is not None:
            factors, target = check_X_y(factors, target)
            assert factors.shape[0] == self.n_step_in
            assert factors.shape[1] == self.n_features - 1
        else:
            # target = column_or_1d(target, warn=True)
            # _assert_all_finite(target)
            assert len(target) == self.n_step_in

        # Prepare input
        input_df = target.reshape(-1, 1) if factors is None else np.hstack((factors, target.reshape(-1, 1)))

        logger.info(f'[Prediction] Start predict by {self.strategy} strategy')
        # if Multi input Multi-out prediction
        if self.strategy == "MiMo":
            predicted = self._mimo_pred(input_df)
        # if Direct prediction
        elif self.strategy == "Direct":
            predicted = self._direct_pred(input_df)
        # if Recursive prediction
        else:
            predicted = self._recursive_pred(input_df, prediction_len=self.prediction_len)

        logger.info(f'[Prediction] End predict by {self.strategy} strategy')
        return predicted

    def _recursive_pred(self, data: np.array, prediction_len: int):
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
            prediction_point = self.model_list[0]['model'].predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
            past_targets = np.concatenate((past_targets[1:], prediction_point))
            var = np.reshape(past_targets, (1, self.n_step_in, self.n_features))
        predicted = np.array(predicted)
        return predicted

    def _mimo_pred(self, data: np.array):
        """
        Predict n_step_out steps ahead. Use it if n_step_out != 1
        :param data : np.array, the last sequence of true data
        :return: np.array of predictions
        """
        var = np.reshape(data, (1, self.n_step_in, self.n_features))
        return self.model_list[0]['model'].predict(var)

    def _direct_pred(self, data: np.array):
        """
        Predict only 1 step ahead by one model fot each value in prediction_len
        :param data: np.array, the last sequence of true data
        :return: np.array of predictions
        """
        assert len(self.model_list) == self.horizon, "Num of models != length of prediction horizon"

        predicted = []
        var = np.reshape(data, (1, self.n_step_in, self.n_features))

        for i in range(len(self.model_list)):
            # Prediction RNN for i step
            prediction_point = self.model_list[i]['model'].predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
        predicted = np.array(predicted)
        return predicted

    def _data_process(self, target=None, factors=None, _i_model=None):
        """
        Prepare input to model
        :param target:
        :param factors:
        :param _i_model: number of model from Direct strategy
        """
        if factors is None:
            target = column_or_1d(target, warn=True)
            _assert_all_finite(target)
            pass
        else:
            factors, target = check_X_y(factors, target)

        # Prepare input
        input_df = target.reshape(-1, 1) if (factors is None) else np.hstack((factors, target.reshape(-1, 1)))

        # Train/ Test split
        train, test = train_test_split(input_df, test_len=self.test_len)

        self._last_known_target = train[-self.n_step_in:, -1]
        if factors is not None:
            self._last_known_factors = train[-self.n_step_in:, :-1]

        # split into samples
        _X_train, _y_train = split_sequence(train,
                                            n_steps_in=self.n_step_in,
                                            n_steps_out=self.horizon if (
                                                    self.strategy == "Direct") else self.n_step_out,
                                            _full_out=True if ((factors is not None) and
                                                               (self.strategy == "Recursive")) else False,
                                            _i_model=_i_model if (self.strategy == "Direct") else None)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        _X_train = _X_train.reshape((_X_train.shape[0], _X_train.shape[1], input_df[0].size))

        # prepare X_test and y_test
        if self.test_len == 0:
            _X_test, _y_test = None, None
        else:
            _X_test, _y_test = split_sequence(input_df,
                                              n_steps_in=self.n_step_in,
                                              n_steps_out=self.horizon if (
                                                      self.strategy == "Direct") else self.n_step_out,
                                              _full_out=True if ((factors is not None) and
                                                                 (self.strategy == "Recursive")) else False,
                                              _i_model=_i_model if (self.strategy == "Direct") else None)
            _X_test = _X_test[-len(test):]
            _y_test = _y_test[-len(test):]
            _X_test = _X_test.reshape((_X_test.shape[0], _X_test.shape[1], self.n_features))

        return _X_train, _y_train, _X_test, _y_test

    def forecast(self, prediction_len):
        predicted = self.predict(factors=self._last_known_factors,
                                 target=self._last_known_target,
                                 prediction_len=prediction_len)
        return predicted

    # def load_model(self, filepath):
    #     """
    #     A method for loading model from h5 file
    #     :param filepath: (str) path to h5 file with model
    #     :return: None
    #     """
    #     print('[Model] Loading model from file %s' % filepath)
    #     self.model_list = [{"model_name": f"{self.strategy}_model",
    #                        "model": load_model(filepath)}]

    def summary(self):
        """
        Printing models summary
        :return: None
        """
        for model_id in range(len(self.model_list)):
            self.model_list[model_id]["model"].summary()

    def save(self, save_dir):
        """
        A method for saving model to h5 file
        :param save_dir: (str) path to h5 file with model
        :return: None
        """
        logger.info('[Model Saving] Saving model to file %s' % save_dir)
        for model_id in range(len(self.model_list)):
            self.model_list[model_id]["model"].save(self.model_list[model_id]["model_name"] + ".h5/" + save_dir)
