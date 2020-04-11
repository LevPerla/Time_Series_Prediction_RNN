import os
import numpy as np
import datetime as dt
from core.utils import Timer, save_image
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model():
    """A class for an building and inferencing an LSTM model"""

    def __init__(self):
        self.model = Sequential()
        self.n_step_in = 0
        self.n_step_out = 0
        self.n_features = 0

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()
        self.n_step_out = configs['model']['n_step_out']

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            n_step_in = layer['n_step_in'] if 'n_step_in' in layer else None

            if layer['type'] == 'Dense':
                if configs["factors"] and (configs["model"]["n_step_out"] == 1):
                    self.model.add(Dense(self.n_features, activation=activation))
                else:
                    self.model.add(Dense(self.n_step_out, activation=activation))
            if layer['type'] == 'LSTM':
                self.model.add(LSTM(neurons,
                                    input_shape=(n_step_in, self.n_features),
                                    return_sequences=return_seq))
                self.n_step_in = n_step_in
            if layer['type'] == 'GRU':
                self.model.add(GRU(neurons,
                                    input_shape=(n_step_in, self.n_features),
                                    return_sequences=return_seq))
                self.n_step_in = n_step_in
            if layer['type'] == "Bidirectional":
                self.model.add(Bidirectional(LSTM(neurons, activation=activation),
                                             input_shape=(n_step_in, self.n_features),
                                             return_sequences=return_seq))
                self.n_step_in = n_step_in
            if layer['type'] == 'Dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x_train, y_train, epochs, batch_size, save_dir, verbose, validation_data):
        '''
        Train model
        :param x_train (np.array):
        :param y_train (np.array):
        :param epochs (int):
        :param batch_size (int):
        :param prediction_len (int): prediction horizon length
        :return: np.array of predictions
        '''
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        # callbacks = [
        #     EarlyStopping(monitor='val_loss', min_delta=0, patience=0),
        #     ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        # ]

        # Set validation indicator
        if validation_data[0] is None:
            val_data = None
        else:
            val_data = validation_data

        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_data=val_data,
                                 batch_size=batch_size,
                                 # callbacks=callbacks,
                                 shuffle=False
        )
        if val_data is not None:
            plt.subplot(212)
            plt.plot(history.history["loss"], label="Train")
            plt.plot(history.history["val_loss"], label="Validation")
            plt.legend(loc="best")
            plt.tight_layout()
            save_image("loss_by_training")
            plt.show()


        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_point_by_point(self, data, prediction_len):
        '''
        Predict only 1 step ahead each time of prediction_len
        :param data (np.array): the last sequence of true data
        :param prediction_len (int): prediction horizon length
        :return: np.array of predictions
        '''
        print('[Model] Predicting Point-by-Point')

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
        '''
        Predict n_step_out steps ahead. Use it if n_step_out != 1
        :param data (np.array): the last sequence of true data
        :return: np.array of predictions
        '''
        var = np.reshape(data, (1, self.n_step_in, self.n_features))
        return self.model.predict(var)



