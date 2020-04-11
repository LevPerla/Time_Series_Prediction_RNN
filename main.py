#################################           Load libs                      #############################################

from core.utils import Timer, train_test_pred_plot, metrics_eval, trend_accuracy, save_image, set_seed
from core.data_processor import DataProcessor
from core.model import Model
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt


#################################           Load data and set params       #############################################

# Set dir
os.chdir("/Users/levperla/PycharmProjects/TS_RNN/Time_Series_RNN")
SAVE_DIR = "/Users/levperla/PycharmProjects/TS_RNN/Time_Series_RNN/Models"

# Load description of experiment
configs = json.load(open('config.json', 'r'))

# Import targets
# target = pd.read_csv("data/series_g.csv", sep=";").series_g.values
target = pd.read_csv("data/monthly-beer-production-in-austr.csv", sep=",").iloc[:, -1].values
# target = pd.read_csv("data/Electric_Production.csv", sep=",").iloc[:, -1].values
# target = pd.read_csv("data/Chicago_hotels.csv", sep="\t").iloc[:, 3].values
# target = pd.read_csv("data/PJME_hourly.csv", sep=",").PJME_MW.values

# Import factors
f1 = np.arange(1,len(target)+1).reshape(-1,1)
f2 = (np.arange(1,len(target)+1)**2).reshape(-1,1)
factors = f1
# factors = np.hstack((f1, f2))


# Set classes
# set_seed(42)
data_processor = DataProcessor()
model = Model()

# scaler fitting
target_std = data_processor.scaler_fit_transform(target, target=True)
factors_std = data_processor.scaler_fit_transform(factors, target=False)

# Making input
if configs["factors"]:
    input = np.hstack((factors_std, target_std))
else:
    input = target_std

# Set number of features
model.n_features = input[0].size

# Building model
model.build_model(configs)
model.model.summary()


#################################           Data Pricessing                #############################################

# Train/ Test split
train, test = data_processor.train_test_split(input, test_len=configs["test_len"])

# split into samples
if configs["factors"] and (configs["model"]["n_step_out"] == 1):
    ALL = True
else:
    ALL = False

X_train, y_train = data_processor.split_sequence(train,
                                                 n_steps_in=model.n_step_in,
                                                 n_steps_out=configs["model"]["n_step_out"],
                                                 all=ALL)

# Making validation_data for training plot
if configs["test_len"] == 0:
    X_test, y_test = None, None
else:
    X_test, y_test = data_processor.split_sequence(input,
                                                       n_steps_in=model.n_step_in,
                                                       n_steps_out=configs["model"]["n_step_out"], all=ALL)
    X_test = X_test[-len(test):]
    y_test = y_test[-len(test):]
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], model.n_features))


# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], model.n_features))

#################################           Model Training                 #############################################

model.train(x_train=X_train,
            y_train=y_train,
            epochs=configs["training"]["epochs"],
            batch_size=configs["training"]["batch_size"],
            save_dir=SAVE_DIR,
            validation_data=(X_test, y_test),
            verbose=1)

#################################           Prediction                    ##############################################

#if multi-step prediction
if model.n_step_out != 1: # if multi-step prediction
    predicted = model.predict_multi_step(train[-model.n_step_in:])

# if point-by-point prediction
else:
    # without test set
    if configs["test_len"] == 0:
        predicted = model.predict_point_by_point(train[-model.n_step_in:],
                                                 prediction_len=configs["prediction_len"])
    # with test set
    else:
        predicted = model.predict_point_by_point(train[-model.n_step_in:],
                                                 prediction_len=configs["test_len"] + configs["prediction_len"])

#################################     Printing results and plots        ################################################

# inverse transform
predicted = data_processor.scaler_inverse_transform(predicted)
train = data_processor.scaler_inverse_transform(train[:,-1])

print(predicted)


if configs["test_len"] == 0:
    plt.plot(range(len(train)), train, label="Train")
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Predicted")
    plt.legend()
    save_image("train_predicted")
    plt.show()
else:
    test = data_processor.scaler_inverse_transform(test[:, -1])
    train_test_pred_plot(train, test, predicted)
    print("\n")
    metrics_eval(test, predicted[:len(test)])
    print("accuracy:", trend_accuracy(test, predicted[:len(test)], train[-1]))