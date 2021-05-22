# Time_Series_Prediction_RNN

## Installation

### From Github
you could install the latest version directly from Github:  
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
The library includes one main module ts_rnn: 

### Getting started
To import TS_RNN model run
```python
from ts_rnn.ts_rnn_model import TS_RNN
```


First of all, we need to set architecture of RNN in config in the way like this:
```python
configs ={"model": {
            "layers": [
                {
                "type": "LSTM",
                "neurons": 128,
                "return_sequences": False
                },
                {
                "type": "Dropout",
                "rate": 0.2
                },
                {
                "type": "Dense",
                "activation": "linear"
                }
            ]
        }}
```
::: warning  
*Last block always need to be Dense*  
*Last RNN block need to gave return_sequences: False, another - True*  
:::


#### To set the architecture of RNN you can use some of this blocks:
```python
# LSTM block
{
"type": "LSTM",
"neurons": 128,
"return_sequences": False,
"activation":"linear",
"kernel_regularizer": None,
"recurrent_regularizer": None,
"bias_regularizer": None
},

# GRU block
{
"type": "GRU",
"neurons": 128,
"return_sequences": False,
"activation":"linear",
"kernel_regularizer": None,
"recurrent_regularizer": None,
"bias_regularizer": None
},

# Bidirectional block
{
"type": "Bidirectional",
"neurons": 128,
"return_sequences": False,
"activation":"linear",
"kernel_regularizer": None,
"recurrent_regularizer": None,
"bias_regularizer": None
},

# Dropout block
{
"type": "Dropout",
"rate": 0.2
},

# Dense block
{
"type": "Dense",
"activation": "linear"
}
```

### TS_RNN class has 7 attributes:
1. model – a Sequential neural network model (internal class of the Keras library);
2. id – model Number in uuid format, required for saving experiment logs;
3. n_step_in - length of the input vector;
4. n_step_out - length of the output vector;
5. n_features - number of time series in the input;
6. params - description of the model's parameters in Python dictionary format;
7. test_len - number of data that will be replaced from train data;
8. loss - Keras loss to train model;
9. optimizer - Keras optimizer to train model.

#### You can set model this way:
```python
model = TS_RNN(configs=configs,            # configs with model architecture
              n_step_in=12,                # length of the input vector
              n_step_out=len(y_test),      # length of the output vector
              test_len=len(y_test),        # number of data that will be removed from train data;
              loss="mae",                  # Keras loss
              optimizer="adam",            # Keras optimizer
              n_features=X_train.shape[1]  # also you need to define this if use factors
             )
```
### TS_RNN supports 5 methods:
1. load_model – load weights of a neural network from a file format h5;
2. fit - train the neural network;
```python
my_callbacks = [callbacks.EarlyStopping(patience=30, monitor='val_loss')]

model.fit(factors=factors_std,          # np.array with factors time series
          target=target,                # np.array with target time series
          epochs=100,                   # num epoch to train
          batch_size=12,                # batch_size
          callbacks=my_callbacks,       # Keras callbacks
          save_dir="../your_folder",    # folder to image save 
          verbose=2)                    # verbose
```
3. predict - predict using a neural network, using two methods:  
```python
predicted = model.predict(factors=factors_to_pred,
                          target=target_to_pred,
                          prediction_len=len(y_test))
```

## Predictions methods

There are two main methods to predict with RNN:  
* Point-by-point prediction - the output of the neural network is the one number. 
Forecasting is performed by adding the received forecast to the input of the model cyclically
```python
model = TS_RNN(...,
               n_step_out=1,      # length of the output vector
               ...
             )
```
For univariate time series  
![Point-by-point_uniivariate](https://raw.githubusercontent.com/LevPerla/Time_Series_Prediction_RNN/master/image/Point-by-point_univariate.png)

For multivariate time series
![Point-by-point_multivariate](https://raw.githubusercontent.com/LevPerla/Time_Series_Prediction_RNN/master/image/Point-by-point_multivariate.png)

* Multi-step prediction - the output of the neural network is a vector of the length of the forecast period.
```python
model = TS_RNN(...,
               n_step_out=len(y_test),      # length of the output vector
               ...
             )
```

For univariate time series  
![Multi-step_uniivariate](https://raw.githubusercontent.com/LevPerla/Time_Series_Prediction_RNN/master/image/Multi-step_univariate.png)

For multivariate time series
![Multi-step_multivariate](https://raw.githubusercontent.com/LevPerla/Time_Series_Prediction_RNN/master/image/Multi-step_multivariate.png)
