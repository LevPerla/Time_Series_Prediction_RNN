#################################           Load libs                      #############################################
import os
import uuid
import numpy as np
from src.utils import Timer, save_image
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization


#################################           Model Class                    #############################################

class Model:
    """ A class for an building and inferencing an RNN models  """

    def __init__(self):
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
        self.id = -1
        self.n_step_in = 0
        self.n_step_out = 0
        self.n_features = 0
        self.params = {}
        self.last_obs_index = "0"
        self.score = 0
        self.factors_names = "none"

    def load_model(self, filepath):
        """
        A method for loading model from h5 file
        :param filepath: (str) path to h5 file with model
        :return: None
        """
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        """
        A method for building and compiling RNN models
        :param configs: (dict) dict with params of model to build
        :return: None
        """
        timer = Timer()
        timer.start()

        self.id = str(uuid.uuid1())
        self.n_step_out = configs['model']['n_step_out']
        self.params = configs

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_sequences'] if 'return_sequences' in layer else None
            n_step_in = layer['n_step_in'] if 'n_step_in' in layer else None
            kernel_regularizer = regularizers.l1(layer['kernel_regularizer']) if 'kernel_regularizer' in layer else None
            recurrent_regularizer = regularizers.l1(
                layer['recurrent_regularizer']) if 'recurrent_regularizer' in layer else None
            bias_regularizer = regularizers.l1(layer['bias_regularizer']) if 'bias_regularizer' in layer else None

            if layer['type'] == 'Dense':
                if configs["factors"] and (configs["model"]["n_step_out"] == 1):
                    self.model.add(Dense(self.n_features, activation=activation))
                else:
                    self.model.add(Dense(self.n_step_out, activation=activation))
            if layer['type'] == 'LSTM':
                self.model.add(LSTM(neurons,
                                    input_shape=(n_step_in, self.n_features),
                                    kernel_initializer="glorot_uniform",
                                    activation=activation,
                                    return_sequences=return_seq,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer))
                self.n_step_in = n_step_in
            if layer['type'] == 'GRU':
                self.model.add(GRU(neurons,
                                   input_shape=(n_step_in, self.n_features),
                                   activation=activation,
                                   return_sequences=return_seq,
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   bias_regularizer=bias_regularizer))
                self.n_step_in = n_step_in
            if layer['type'] == "Bidirectional":
                self.model.add(Bidirectional(LSTM(neurons,
                                                  activation=activation,
                                                  return_sequences=return_seq,
                                                  kernel_regularizer=kernel_regularizer,
                                                  recurrent_regularizer=recurrent_regularizer,
                                                  bias_regularizer=bias_regularizer),
                                             input_shape=(n_step_in, self.n_features)))
                self.n_step_in = n_step_in
            if layer['type'] == 'Dropout':
                self.model.add(Dropout(dropout_rate))

            if layer['type'] == 'BatchNormalization':
                self.model.add(BatchNormalization(scale=False))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def fit(self, x_train, y_train, epochs, batch_size=36, verbose=1,
            validation_split=0, save_dir=None, x_test=None, y_test=None,
            callbacks=None):
        """
        Train model
        :param x_train: (np.array)
        :param y_train: (np.array)
        :param epochs: (int)
        :param batch_size: (int)
        :param verbose: (int) printing fitting process
        :param validation_split: (float) percent of train data used in validation
        :param y_test: (np.array)
        :param x_test: (np.array)
        :param callbacks: callbacks for EarlyStopping
        :param save_dir: (str) path to saving history plot
        :return: None
        """

        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        if (x_test is None) or (y_test is None):
            validation_data = None
            callbacks = None
        else:
            validation_data = (x_test, y_test)

        history = self.model.fit(x_train,
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

    def save(self, save_dir, name="_", folder_num=False):
        """
        :param save_dir: str, path to save folder
        :param name: str, name of saving model
        :param folder_num: bool, prepend number of file in folder
        :return: None
        """
        if folder_num:
            pwd = os.getcwd()
            os.chdir(save_dir)
            num = str(len(os.listdir(os.getcwd())))
            os.chdir(pwd)
            save_name = os.path.join(save_dir, num + "_" + name)
        else:
            save_name = os.path.join(save_dir, name)
        self.model.save(save_name)

    def predict_point_by_point(self, data, prediction_len):
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

    def predict_multi_step(self, data):
        """
        Predict n_step_out steps ahead. Use it if n_step_out != 1
        :param data : np.array, the last sequence of true data
        :return: np.array of predictions
        """
        var = np.reshape(data, (1, self.n_step_in, self.n_features))
        return self.model.predict(var)

    def predict(self, data):
        """
        Prediction with auto-set method based by params
        :param data : np.array, the last sequence of true data
        :return: np.array of predictions
        """
        # if multi-step prediction
        if self.n_step_out != 1:  # if multi-step prediction
            predicted = self.predict_multi_step(data)

        # if point-by-point prediction
        else:
            # without test set
            if self.params["test_len"] == 0:
                predicted = self.predict_point_by_point(data,
                                                        prediction_len=self.params["prediction_len"])
            # with test set
            else:
                predicted = self.predict_point_by_point(data,
                                                        prediction_len=self.params["prediction_len"])
        return predicted
