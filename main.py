#################################           Load libs                      #############################################
import os
import json
import pandas as pd
from pprint import pprint
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg') # to fix problem on Mac OS
import matplotlib.pyplot as plt

from src.utils import metrics_eval, Timer, save_image
from src.data.data_processor import DataProcessor
from src.model.model import Model

#################################           Load data and set params       #############################################

main_dir = os.getcwd()
timer = Timer()

# Load description of experiment
configs = json.load(open(main_dir + '/src/config.json', 'r'))

# Import targets
target = pd.read_csv("data/series_g.csv", sep=";").series_g.values
# target = pd.read_csv("data/monthly-beer-production-in-austr.csv", sep=",").iloc[:, -1].values
# target = pd.read_csv("data/Electric_Production.csv", sep=",").iloc[:, -1].values
# target = pd.read_csv("data/Chicago_hotels.csv", sep="\t").iloc[:, 3].values
# target = pd.read_csv("data/PJME_hourly.csv", sep=",").PJME_MW.values

# Import factors
f1 = np.arange(1, len(target) + 1).reshape(-1, 1)
f2 = (np.arange(1, len(target) + 1) ** 2).reshape(-1, 1)
# factors = f1
factors = np.hstack((f1, f2))

#################################           Model training                ##############################################
#Set classes
model = Model()
data_processor = DataProcessor()

# scaler fitting
target_std = data_processor.scaler_fit_transform(target, target=True)

# Making input
if configs["factors"]:
    factors_std = data_processor.scaler_fit_transform(factors, target=False)
    input = np.hstack((factors_std, target_std))
else:
    input = target_std

# Set number of features
model.n_features = input[0].size

# Building model
model.build_model(configs)
pprint(model.params)
model.model.summary()

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

# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], model.n_features))

# Making validation_data for training plot

if configs["test_len"] == 0:
    X_test, y_test = None, None

    model.fit(x_train=X_train,
              y_train=y_train,
              epochs=configs["training"]["epochs"],
              batch_size=configs["training"]["batch_size"],
              verbose=4)

else:
    X_test, y_test = data_processor.split_sequence(input,
                                                    n_steps_in=model.n_step_in,
                                                    n_steps_out=configs["model"]["n_step_out"], all=ALL)
    X_test = X_test[-len(test):]
    y_test = y_test[-len(test):]
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], model.n_features))

    model.fit(x_train=X_train,
              y_train=y_train,
              epochs=configs["training"]["epochs"],
              batch_size=configs["training"]["batch_size"],
              validation_data=(X_test, y_test),
              verbose=4)

#################################   Making new folder with results        ##############################################

path = main_dir + "/reports"

try:
    os.chdir(path)
except:
    os.makedirs(path)
    os.chdir(path)
new_folder_num = str(len(os.listdir()) + 1)
new_folder = "%s/%s" % (path, new_folder_num)
os.mkdir(new_folder)


#################################           Prediction                    ##############################################

timer.start()
predicted = model.predict(train[-model.n_step_in:])
print("\nTraining time:")
timer.stop()
print()

#################################     Printing results and plots        ################################################

# inverse transform
predicted = data_processor.scaler_inverse_transform(predicted, target=True)
train = data_processor.scaler_inverse_transform(train[:, -1], target=True)
if configs["test_len"] != 0:
    test = data_processor.scaler_inverse_transform(test[:, -1], target=True)

plt.plot(range(len(train)), train, label="Train")
if configs["test_len"] != 0:
    plt.plot(range(len(train), len(train) + len(test)), test, label="Test")
plt.plot(range(len(train), len(train) + len(predicted)), predicted, label="Predicted")
plt.legend()
save_image(new_folder, "train_test_predicted", fmt="png")
plt.close()

if configs["test_len"] != 0 and len(test) >= len(predicted):
    mae, mse, rmse, MAPE, SMAPE = metrics_eval(test[:len(predicted)], predicted)
    with open(new_folder + '/metrics_%s.txt' % (model.id), 'w') as record_file:
        record_file.write('Mean Absolute Error:' + str(round(mae, 3)))
        record_file.write('\nMean Squared Error:' + str(round(mse, 3)))
        record_file.write('\nRoot Mean Squared Error:' + str(round(rmse, 3)))
        record_file.write('\nMean absolute percentage error:' + str(round(MAPE, 3)))
        record_file.write('\nScaled Mean absolute percentage error:' + str(round(SMAPE, 3)))
        record_file.close()


#################################            Saving                     ################################################

# Save classes
with open(new_folder + '/model.pickle', 'wb') as record_file:
    pickle.dump(model, record_file)
    record_file.close()

with open(new_folder + '/data_processor.pickle', 'wb') as record_file:
    pickle.dump(data_processor, record_file)
    record_file.close()

# Config saving
with open(new_folder + "/configs_%s.json" % model.id, "w") as record_file:
    json.dump(model.params, record_file)


