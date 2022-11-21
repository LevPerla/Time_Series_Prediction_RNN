# Time_Series_Prediction_RNN

![code-size][code-size]
![license][license]

[code-size]: https://img.shields.io/github/languages/code-size/LevPerla/Time_Series_Prediction_RNN

[license]: https://img.shields.io/badge/license-MIT-green

## Requirements

ts_rnn requires the following to run:

* [Python](node) 3.7.3+

## Installation

### From pip

You could install the latest version from PyPi:

```sh
pip install ts-rnn
```

### From Github

You could install the latest version directly from Github:

```sh
pip install https://github.com/LevPerla/Time_Series_Prediction_RNN/archive/master.zip
```

### From source
Download the source code by cloning the repository or by pressing ['Download ZIP'](https://github.com/pandas-profiling/pandas-profiling/archive/master.zip) on this page. 

Install by navigating to the proper directory and running:

```sh
python setup.py install
```
## Example
[Example](https://github.com/LevPerla/Time_Series_Prediction_RNN/blob/master/notebooks/Example.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LevPerla/Time_Series_Prediction_RNN/blob/master/notebooks/Example.ipynb)


## Documentation

The full documentation haven't ready yet. I hope, it will show later.

### Getting started
To import TS_RNN model run

```python
from ts_rnn.model import TS_RNN
```

First of all, we need to set architecture of RNN in config in the way like this:
```python
rnn_arch = {"layers": [
                        ["LSTM", {"units": 64,
                                  "return_sequences": False,
                                  "kernel_initializer": "glorot_uniform",
                                  "activation": "linear"}],
                        ["Dropout", {"rate": 0.2}],
                        ["Dense", {"activation": "linear"}]
                    ]}
```

> **WARNING**: *Last RNN block need to gave return_sequences: False, another - True*


#### To set the architecture of RNN you can use some of this blocks:
```python
# LSTM block
["LSTM", {Keras layer params}],
["GRU", {Keras layer params}],
["SimpleRNN", {Keras layer params}],
["Bidirectional", {Keras layer params}],
["Dropout", {Keras layer params}],
["Dense", {Keras layer params}]
```

### TS_RNN class has 7 attributes:

<ol>
<li>n_lags - length of the input vector;</li>
<li>horizon - length of prediction horizon;</li>
<li>rnn_arch - description of the model's parameters in Python dictionary format;</li>
<li>strategy - prediction strategy: "Direct", "Recursive", "MiMo", "DirRec", "DirMo"</li>
<li>tuner - tupe of Keras.tuner: "RandomSearch", "BayesianOptimization", "Hyperband"</li>
<li>tuner_hp - keras_tuner.HyperParameters class</li>
<li>n_step_out - length of the output vector (Need to define only for DirMo strategy);</li>
<li>loss - Keras loss to train model;</li>
<li>optimizer - Keras optimizer to train model.</li>
<li>n_features - number of time series in the input (only for factors forecasting);</li>
<li>save_dir - dir to save logs</li>
</ol>

#### You can set model this way:
```python
model = TS_RNN(rnn_arch=rnn_arch,  # dict with model architecture
               n_lags=12,  # length of the input vector
               horizon=TEST_LEN,  # length of prediction horizon
               strategy="MiMo",  # Prediction strategy from "Direct", "Recursive", "MiMo", "DirRec", "DirMo"
               loss="mae",  # Keras loss
               optimizer="adam",  # Keras optimizer
               n_features=X_train.shape[1]  # also you need to define this if use factors
               )
```

### TS_RNN supports 5 methods:

<ol>
<li>fit - train the neural network;</li>
<li>predict - predict by the neural network by input;</li>
<li>forecast - predict by the neural network by last train values;</li>
<li>summary - print NNs architecture</li>
<li>save - save model files to dict</li>
</ol>

FIT

```python
my_callbacks = [callbacks.EarlyStopping(patience=30, monitor='val_loss')]

model.fit(factors_train=factors_val,  # pd.DataFrame with factors time series
          target_train=target_val,  # pd.DataFrame or pd.Series with target time series
          factors_val=factors_val,  # pd.DataFrame with factors time series
          target_val=target_val,  # pd.DataFrame or pd.Series with target time series
          epochs=100,  # num epoch to train
          batch_size=12,  # batch_size
          callbacks=my_callbacks,  # Keras callbacks
          save_dir="../your_folder",  # folder to image save 
          verbose=2)  # verbose
```

PREDICT

```python
predicted = model.predict(factors=factors_to_pred,
                          target=target_to_pred,
                          prediction_len=len(y_test))
```

FORECAST

```python
predicted = model.forecast(prediction_len=HORIZON)
```

SUMMARY

```python
model.summary()
```

SAVE

```python
model.save(model_name='tsrnn_model', save_dir='path')
```

Also you may load TS_RNN model from folder

```python
from ts_rnn.model import load_ts_rnn

model = load_ts_rnn(os.path.join('path', 'tsrnn_model'))
```

### Simple example of usage:

> **Info**: For better performance use MinMaxScaler and Deseasonalizer before fitting

```python
from sklearn.model_selection import train_test_split
from ts_rnn.model import TS_RNN
import pandas as pd

HORIZON = 12

data_url = "https://raw.githubusercontent.com/LevPerla/Time_Series_Prediction_RNN/master/data/series_g.csv"
target = pd.read_csv(data_url, sep=";").series_g
target_train, target_test = train_test_split(target, test_size=HORIZON, shuffle=False)

model = TS_RNN(n_lags=12, horizon=HORIZON)
model.fit(target_train=target_train,
          target_val=target_test,
          epochs=40,
          batch_size=12,
          verbose=1)

model.summary()
predicted = model.predict(target=target_train[-model.n_lags:], prediction_len=HORIZON)
```