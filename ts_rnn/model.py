# Load libs
import json
import logging
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization, SimpleRNN
from tensorflow.keras.models import Sequential, load_model

from ts_rnn import config, utils
from ts_rnn.logger import logger


class TS_RNN:
    """ A class for an building and inferencing an RNN models for time series prediction"""

    def __init__(self,
                 n_lags=None,
                 horizon=None,
                 rnn_arch=None,
                 strategy="MiMo",
                 n_step_out=1,
                 n_features=0,
                 tuner="BayesianOptimization",
                 tuner_hp=None,
                 loss="mae",
                 optimizer="adam",
                 save_dir=None,
                 **kwargs):
        """
        :param rnn_arch: dict with layers params
        :param n_lags: number time series steps in input
        :param horizon: length of prediction horizon
        :param strategy: prediction strategy
        :param n_step_out: number time series steps in out
        :param n_features: number exogeny time series in input
        :param tuner: keras tuner name
        :param tuner_hp: keras_tuner.HyperParameters class
        :param loss: keras loss
        :param optimizer: keras optimizer
        :param save_dir: (str) path to saving history plot
        """
        # Set logger
        # Delete old handlers
        for hdle in logger.handlers[:]:
            if isinstance(hdle, logging.FileHandler):
                hdle.close()
                logger.removeHandler(hdle)
        logger.setLevel(logging.DEBUG)

        # Set new handler
        if save_dir is not None:
            handler = logging.FileHandler(os.path.join(save_dir, "ts_rnn.log"), mode='w')
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(levelname)s] - %(asctime)s - %(message)s'))
        logger.addHandler(handler)
        self.logger = logger

        # Set model arch
        if rnn_arch is None:
            self.logger.warning(f'rnn_arch is not defined. Model will be compiled with default architecture')
            self.rnn_arch = config.DEFAULT_ARCH
            self.hp = config.DEFAULT_HP
        elif (rnn_arch is not None) and (tuner_hp is None):
            self.logger.warning(f'tuner_hp is not defined. Model will be trained without tuning architecture')
            self.rnn_arch = rnn_arch
            self.hp = None
        else:
            self.rnn_arch = rnn_arch
            self.hp = tuner_hp

        self.n_lags = n_lags
        self.n_step_out = horizon if strategy == "MiMo" else n_step_out
        self.horizon = horizon
        self.n_features = n_features + 1
        self.loss = loss
        self.strategy = strategy
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.tuner = tuner
        self._last_known_target = None
        self._last_known_factors = None
        self.model_list = None
        self.kwargs = kwargs

        if 'load' not in self.kwargs.keys():
            self._assert_init_params()
            # compile model
            self._build_by_stategy()

    def _assert_init_params(self):
        try:
            assert (self.n_lags is not None) and (self.horizon is not None)
        except AssertionError:
            self.logger.exception('Define n_lags and horizon')
            raise AssertionError('Define n_lags and horizon')
        try:
            assert self.strategy in config.MODEL_STRATEGIES
        except AssertionError:
            self.logger.exception(f'Use strategy from {config.MODEL_STRATEGIES}')
            raise AssertionError(f'Use strategy from {config.MODEL_STRATEGIES}')

        try:
            assert self.tuner in ["RandomSearch", "BayesianOptimization", "Hyperband"]
        except AssertionError:
            self.logger.exception('Use tuner from ["RandomSearch", "BayesianOptimization", "Hyperband"]')
            raise AssertionError('Use tuner from ["RandomSearch", "BayesianOptimization", "Hyperband"]')

        if self.strategy in ["Direct", "Recursive", "DirRec"]:
            assert self.n_step_out == 1, "For Direct, Recursive and DirRec strategies n_step_out must be 1"

        if self.strategy == "DirMo" and (self.n_step_out == 1):
            self.logger.warning(f'n_step_out == 1. Strategy DirMo equal to Direct. Please set n_step_out')

    def _build_by_stategy(self):
        """
        Build TS model by prediction strategy
        :return:
        """

        if self.strategy == "Direct":
            for i in range(self.horizon):
                name = f'{self.strategy}_model_{i + 1}'
                self._build_model_or_tuner(name)

        elif self.strategy in ["Recursive", "MiMo"]:
            name = f"{self.strategy}_model"
            self._build_model_or_tuner(name)

        elif self.strategy == "DirRec":
            true_n_lags = self.n_lags
            for i in range(self.horizon):
                self.n_lags = true_n_lags + i
                name = f'{self.strategy}_model_{i + 1}'
                self._build_model_or_tuner(name)
            self.n_lags = true_n_lags

        elif self.strategy == "DirMo":
            n_models_need = math.ceil(self.horizon / self.n_step_out)
            for i in range(n_models_need):
                name = f'{self.strategy}_model_{i + 1}'
                self._build_model_or_tuner(name)

        return self

    def _build_model_or_tuner(self, name):
        """
        function to switch between defining model or tuner (if hp exist)
        :param name: name of model
        :return: self
        """
        if self.hp is None:
            _model = self._build(hp=None)
            if self.model_list is None:
                self.model_list = [{"model_name": name,
                                    "tuner": None,
                                    "model": _model}]
            else:
                self.model_list.append({"model_name": name,
                                        "tuner": None,
                                        "model": _model})
        else:
            tuner_kwargs = {"hypermodel": self._build,
                            "objective": "val_loss",
                            "max_trials": 10,
                            "max_epochs": 100,
                            "project_name": "TS_RNN_tuner_log",
                            "directory": self.save_dir,
                            "overwrite": True,
                            "hyperparameters": self.hp}
            tuner_kwargs.update(self.kwargs)

            if self.tuner == "RandomSearch":
                del tuner_kwargs['max_epochs']
                _tuner = RandomSearch(**tuner_kwargs)
            elif self.tuner == "Hyperband":
                del tuner_kwargs['max_trials']
                _tuner = Hyperband(**tuner_kwargs)
            elif self.tuner == "BayesianOptimization":
                del tuner_kwargs['max_epochs']
                _tuner = BayesianOptimization(**tuner_kwargs)
            else:
                _tuner = BayesianOptimization(**tuner_kwargs)

            if self.model_list is None:
                self.model_list = [{"model_name": name,
                                    "tuner": _tuner,
                                    "model": None}]
            else:
                self.model_list.append({"model_name": name,
                                        "tuner": _tuner,
                                        "model": None})
        return self

    def _build(self, hp):
        """
        A method for building and compiling RNN models
        :return: keras.Sequential()
        """
        _model = Sequential()

        for layer in self.rnn_arch['layers']:
            if layer[0] == 'Dense':
                if self.n_features > 1 and (self.n_step_out == 1):
                    _model.add(Dense(self.n_features, **layer[1]))
                else:
                    _model.add(Dense(self.n_step_out, **layer[1]))
            elif layer[0] == 'LSTM':
                _model.add(LSTM(input_shape=(self.n_lags, self.n_features), **layer[1]))
            elif layer[0] == 'GRU':
                _model.add(GRU(input_shape=(self.n_lags, self.n_features), **layer[1]))
            elif layer[0] == 'SimpleRNN':
                _model.add(SimpleRNN(input_shape=(self.n_lags, self.n_features), **layer[1]))
            elif layer[0] == "Bidirectional":
                _model.add(Bidirectional(LSTM(**layer[1]), input_shape=(self.n_lags, self.n_features)))
            elif layer[0] == 'Dropout':
                _model.add(Dropout(**layer[1]))
            elif layer[0] == 'BatchNormalization':
                _model.add(BatchNormalization(scale=False, **layer[1]))
            else:
                self.logger.critical(f"TS_RNN doesn't support layer type {layer[0]}")
                raise AssertionError(f"TS_RNN doesn't support layer type {layer[0]}")

        _model.compile(loss=self.loss, optimizer=self.optimizer)
        assert isinstance(_model.layers[-1], Dense), "last block need to be Dense"

        return _model

    @utils.timeit
    def fit(self,
            target_train,
            target_val=None,
            factors_train=None,
            factors_val=None,
            epochs=30,
            batch_size=36,
            verbose=1,
            validation_split=0,
            **kwargs):
        """
        Train model
        :param target_train: (np.array)
        :param factors_train: (np.array)
        :param epochs: (int)
        :param batch_size: (int)
        :param verbose: (int) printing fitting process
        :param validation_split: (float) percent of train data used in validation
        :param callbacks: callbacks for EarlyStopping
        :return: self
        """
        # Check input
        target_train, target_val, factors_train, factors_val = self._assert_fit_input(target_train, target_val,
                                                                                      factors_train, factors_val)
        true_n_lags = self.n_lags
        for model_id in range(len(self.model_list)):
            if target_train.shape[1] > 1:
                self.logger.info(f'[Training] Target is a multiple time series, preparing global model')
            else:
                self.logger.info(f'[Training] target is a single time series, preparing local model')

            _X_train, _y_train, _X_val, _y_val = None, None, None, None
            for i in range(target_train.shape[1]):
                _X_train_i, _y_train_i, _X_val_i, _y_val_i = self._data_process(
                    target_train=target_train.iloc[:, i].to_frame(),
                    target_val=target_val.iloc[:, i].to_frame(),
                    factors_train=factors_train,
                    factors_val=factors_val,
                    _i_model=model_id)
                if _X_train is None:
                    _X_train = _X_train_i
                    _y_train = _y_train_i
                    _X_val = _X_val_i
                    _y_val = _y_val_i
                else:
                    _X_train = np.vstack((_X_train, _X_train_i))
                    _y_train = np.vstack((_y_train, _y_train_i))
                    _X_val = np.vstack((_X_val, _X_val_i))
                    _y_val = np.vstack((_y_val, _y_val_i))

            self.logger.info(f'[Training] Training {self.model_list[model_id]["model_name"]} started on '
                             f'%s epochs, %s batch size' % (epochs, batch_size))

            if (_X_val is None) or (_y_val is None):
                validation_data = None
            else:
                validation_data = (_X_val, _y_val)

            if self.model_list[model_id]["tuner"] is None:
                history = self.model_list[model_id]["model"].fit(_X_train,
                                                                 _y_train,
                                                                 epochs=epochs,
                                                                 verbose=verbose,
                                                                 batch_size=batch_size,
                                                                 validation_split=validation_split,
                                                                 validation_data=validation_data,
                                                                 shuffle=False,
                                                                 **kwargs
                                                                 )

                if self.save_dir is not None and ((validation_split != 0) or (validation_data is not None)):
                    utils.history_plot(history,
                                       self.save_dir,
                                       show=True if verbose > 0 else False)
            else:
                if self.strategy == "DirRec":
                    self.n_lags = true_n_lags + model_id  # needed to train DirRec strategy
                self.model_list[model_id]["tuner"].search(_X_train,
                                                          _y_train,
                                                          epochs=epochs,
                                                          verbose=verbose,
                                                          batch_size=batch_size,
                                                          validation_split=validation_split,
                                                          validation_data=validation_data,
                                                          shuffle=False,
                                                          **kwargs
                                                          )
                self.model_list[model_id]["model"] = self.model_list[model_id]["tuner"].get_best_models(num_models=1)[0]
                if self.strategy == "DirRec":
                    self.n_lags = true_n_lags

        if self.save_dir is not None:
            shutil.rmtree(os.path.join(self.save_dir, 'TS_RNN_tuner_log'), ignore_errors=True)
        self.logger.info('[Training] Training ended')
        return self

    def _assert_fit_input(self, target_train, target_val, factors_train, factors_val):
        assert isinstance(target_train, (pd.Series, pd.DataFrame)), 'target_train need to be pd.Series or pd.DataFrame'
        assert isinstance(target_val, (
            pd.Series, pd.DataFrame)) or target_val is None, 'target_val need to be pd.Series or pd.DataFrame'

        assert isinstance(factors_train, (
            pd.Series, pd.DataFrame)) or factors_train is None, 'factors_train need to be pd.Series or pd.DataFrame'
        assert isinstance(factors_val, (
            pd.Series, pd.DataFrame)) or factors_val is None, 'factors_val need to be pd.Series or pd.DataFrame'

        if isinstance(target_train, pd.Series):
            target_train = target_train.to_frame()
        if isinstance(target_val, pd.Series):
            target_val = target_val.to_frame()
        if isinstance(factors_train, pd.Series):
            factors_train = factors_train.to_frame()
        if isinstance(factors_val, pd.Series):
            factors_val = factors_val.to_frame()

        if target_val is not None:
            assert list(target_train.columns) == list(target_val.columns), 'names of target_train != target_val'

        self.target_names = list(target_train.columns)
        self.train_index = target_train.index

        if factors_train is not None:
            assert list(factors_train.columns) == list(factors_val.columns), 'names of factors_train != factors_val'
            assert list(factors_train.index) == list(target_train.index), 'index of target_train != factors_train'
            self.factors_names = list(factors_train.columns)
        else:
            self.factors_names = None

        if (factors_val is not None) and (target_val is not None):
            assert list(target_val.index) == list(factors_val.index), 'index of target_val != factors_val'

        return target_train, target_val, factors_train, factors_val

    @utils.timeit
    def predict(self, target, prediction_len, factors=None):
        """
        Prediction with auto-set method based by rnn_arch
        :param factors: np.array
        :param target: np.array
        :param prediction_len: int
        :return: np.array of predictions
        """
        assert ((factors is not None) and (self.factors_names is not None)) or \
               ((factors is None) and (
                       self.factors_names is None)), f"model was fitted with range of factors: {self.factors_names}. Add it to factors attribute"

        if factors is not None:
            assert factors.shape[
                       0] == self.n_lags, f'factors to predict need to have model.n_lags lengths: {self.n_lags}'
            assert factors.shape[1] == self.n_features - 1, f'factors length need to be {self.n_features - 1}'
            assert list(factors.columns) == self.factors_names, f'factors names need to be {self.factors_names}'
        assert len(target) == self.n_lags, f'target to predict need to have model.n_lags lengths: {self.n_lags}'

        # Prepare input
        if isinstance(target, pd.Series):
            target = target.values.reshape(-1, 1)
        elif isinstance(target, pd.DataFrame):
            target = target.values
        input_df = target if factors is None else np.hstack((factors, target))

        self.logger.info(f'[Prediction] Start predict by {self.strategy} strategy')
        # if Multi input Multi-out prediction
        if self.strategy == "MiMo":
            predicted = self._mimo_pred(input_df)
        # if Direct prediction
        elif self.strategy == "Direct":
            predicted = self._direct_pred(input_df)
        # if Recursive prediction
        elif self.strategy == "Recursive":
            predicted = self._recursive_pred(input_df, prediction_len=prediction_len)
        # if DirRec prediction
        elif self.strategy == 'DirRec':
            predicted = self._dirrec_pred(input_df)
        # if DirMo prediction
        elif self.strategy == 'DirMo':
            predicted = self._dirmo_pred(input_df)
        else:
            self.logger.critical(f'Use strategy from {config.MODEL_STRATEGIES}')
            raise AssertionError(f'Use strategy from {config.MODEL_STRATEGIES}')

        self.logger.info(f'[Prediction] End predict by {self.strategy} strategy')
        return predicted.flatten()

    def _recursive_pred(self, data: np.array, prediction_len: int):
        """
        Predict only 1 step ahead each time of prediction_len
        :param data: np.array, the last sequence of true data
        :param prediction_len: int, prediction horizon length
        :return: np.array of predictions
        """

        # Check output length
        assert self.n_step_out == 1, "Error: Output length needs to be 1. Use method predict_multi_step"
        assert (prediction_len != 0) or (prediction_len is not None), "Error: prediction_len is 0"

        predicted = []
        past_targets = data
        var = np.reshape(past_targets, (1, self.n_lags, self.n_features))

        for i in range(prediction_len):
            # Prediction RNN for i step
            prediction_point = self.model_list[0]['model'].predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
            past_targets = np.concatenate((past_targets[1:], prediction_point))
            var = np.reshape(past_targets, (1, self.n_lags, self.n_features))
        predicted = np.array(predicted)
        return predicted

    def _mimo_pred(self, data: np.array):
        """
        Predict n_step_out steps ahead. Use it if n_step_out != 1
        :param data : np.array, the last sequence of true data
        :return: np.array of predictions
        """
        var = np.reshape(data, (1, self.n_lags, self.n_features))
        return self.model_list[0]['model'].predict(var)

    def _direct_pred(self, data: np.array):
        """
        Predict only 1 step ahead by one model fot each value in prediction_len
        :param data: np.array, the last sequence of true data
        :return: np.array of predictions
        """
        assert len(self.model_list) == self.horizon, "Num of models != length of prediction horizon"

        predicted = []
        var = np.reshape(data, (1, self.n_lags, self.n_features))

        for i in range(len(self.model_list)):
            # Prediction RNN for i step
            prediction_point = self.model_list[i]['model'].predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
        predicted = np.array(predicted)
        return predicted

    def _dirrec_pred(self, data: np.array):
        """
        Predict only 1 step ahead each time of prediction_len
        :param data: np.array, the last sequence of true data
        :param prediction_len: int, prediction horizon length
        :return: np.array of predictions
        """

        # Check output length
        assert self.n_step_out == 1, "Error: Output length needs to be 1"

        predicted = []
        past_targets = data
        var = np.reshape(past_targets, (1, self.n_lags, self.n_features))

        for i in range(len(self.model_list)):
            # Prediction RNN for i step
            prediction_point = self.model_list[i]['model'].predict(var)
            predicted.append(prediction_point[0][-1])
            # Preparation of sequence
            past_targets = np.concatenate((past_targets, prediction_point))
            var = np.reshape(past_targets, (1, self.n_lags + i + 1, self.n_features))
        predicted = np.array(predicted)
        return predicted

    def _dirmo_pred(self, data: np.array):
        """
        Predict only 1 step ahead by one model fot each value in prediction_len
        :param data: np.array, the last sequence of true data
        :return: np.array of predictions
        """
        assert len(self.model_list) == math.ceil(self.horizon / self.n_step_out), "Num of models != horizon/s_step_out"

        predicted = []
        var = np.reshape(data, (1, self.n_lags, self.n_features))

        for i in range(len(self.model_list)):
            # Prediction RNN for i step
            prediction_point = self.model_list[i]['model'].predict(var)
            predicted.append(prediction_point[0])
            # Preparation of sequence
        predicted = np.array(predicted).flatten()[:self.horizon]
        return predicted

    def _data_process(self,
                      target_train=None,
                      target_val=None,
                      factors_train=None,
                      factors_val=None,
                      _i_model=None):
        """
        Prepare input to model
        :param target:
        :param factors:
        :param _i_model: number of model from Direct strategy
        """
        if (factors_train is None) and (factors_val is not None):
            self.logger.critical(f'factors_train is not defined, but factors_val exists')
            raise AssertionError(f'factors_train is not defined, but factors_val exists')

        if (target_train is None) and (target_val is not None):
            self.logger.critical(f'target_train is not defined, but target_val exists')
            raise AssertionError(f'target_train is not defined, but target_val exists')

        # if validation with factors
        if (target_val is not None) and (factors_val is not None):
            target = pd.concat([target_train, target_val], axis=0)
            factors = pd.concat([factors_train, factors_val], axis=0)
            val_len = target_val.shape[0]
        # if validation without factors
        elif (target_val is not None) and (factors_val is None) and (factors_train is None):
            target = pd.concat([target_train, target_val], axis=0)
            factors = None
            val_len = target_val.shape[0]
        else:
            self.logger.warning(f'Validation target or factors is not defined. Validation will not be used')
            target = target_train
            factors = factors_train
            val_len = 0

        # Prepare input
        input_df = target if factors_train is None else pd.concat([factors, target], axis=1)

        try:
            assert input_df.shape[
                       1] == self.n_features, 'Shape of input != model.n_features. Please, check factors frame'
        except AssertionError:
            self.logger.exception(
                f'Define n_features in init of TS_RNN, factors given: {input_df.shape[1]}, n_features: {self.n_features}')
            raise AssertionError(
                f'Define n_features in init of TS_RNN, factors given: {input_df.shape[1]}, n_features: {self.n_features}')

        self._last_known_target = target_train.iloc[-self.n_lags:, :]
        if factors_train is not None:
            self._last_known_factors = factors_train.iloc[-self.n_lags:, :]

        # split into samples
        _X_train, _y_train = utils.split_sequence(input_df.values,
                                                  n_steps_in=self.n_lags + _i_model if (
                                                          self.strategy == "DirRec") else self.n_lags,
                                                  n_steps_out=self.n_step_out,
                                                  _full_out=True if ((factors is not None) and (
                                                          self.strategy in ["Recursive", "DirRec"])) else False,
                                                  _i_model=_i_model if (self.strategy in ["Direct", 'DirMo']) else 0,
                                                  _start_ind=_i_model * self.n_step_out - _i_model if (
                                                          self.strategy == "DirMo") else 0)
        if val_len == 0:
            _X_test, _y_test = None, None
        else:
            _X_test = _X_train[-val_len:]
            _y_test = _y_train[-val_len:]
            _X_test = _X_test.reshape((_X_test.shape[0], _X_test.shape[1], self.n_features))

            _X_train = _X_train[:-val_len]
            _y_train = _y_train[:-val_len]

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        _X_train = _X_train.reshape((_X_train.shape[0], _X_train.shape[1], self.n_features))

        return _X_train, _y_train, _X_test, _y_test

    def forecast(self, prediction_len):
        predicted = self.predict(
            factors=self._last_known_factors if self._last_known_factors is not None else None,
            target=self._last_known_target,
            prediction_len=prediction_len)
        return predicted

    def summary(self):
        """
        Printing models summary
        :return: None
        """
        for model_id in range(len(self.model_list)):
            self.model_list[model_id]["model"].summary()

    def save(self, model_name='tsrnn_model', save_dir='.'):
        """
        A method for saving model to h5 file
        :param model_name: (str) name for your model
        :param save_dir: (str) path to h5 file with model
        :return: None
        """
        models_folder_path = os.path.join(save_dir, model_name)
        if not os.path.exists(models_folder_path):
            os.makedirs(models_folder_path)
        self.logger.info('[Model Saving] Saving model to file %s' % models_folder_path)
        with open(os.path.join(models_folder_path, 'ts_rnn.json'), 'w') as fp:
            json.dump({key: value for key, value in self.__dict__.items() if
                       key not in ['hp', 'model_list', 'logger', '_last_known_target', '_last_known_factors',
                                   'train_index']}, fp)

        for model_id in range(len(self.model_list)):
            self.model_list[model_id]["model"].save(os.path.join(models_folder_path,
                                                                 self.model_list[model_id]["model_name"] + ".h5"))


def load_ts_rnn(path):
    with open(os.path.join(path, f'ts_rnn.json')) as json_file:
        params = json.load(json_file)
        params.update({'model_list': []})
    ts_rnn = TS_RNN(load=True)
    ts_rnn.__dict__ = params
    ts_rnn.logger = logger

    for i, model_name in enumerate(sorted(os.listdir(path))):
        if '.h5' in model_name:
            _model = load_model(os.path.join(path, model_name))
            ts_rnn.model_list.append({"model_name": model_name, "model": _model, 'tuner': None})
    return ts_rnn
