# Time_Series_Prediction_RNN

## Getting Started

Installation:  
<code>$ pip3 install -r requirements.txt</code>

## Example
You could find it on notebooks/Example.ipynb

## Discription
The library includes one main module:

### Model
#### Model class has 7 attributes:
1. model – a Sequential neural network model (internal class of the Keras library);
2. id – model Number in uuid format, required for saving experiment logs;
3. n_step_in - length of the input vector;
4. n_step_out - length of the output vector;
5. n_features - number of time series in the input;
6. params - description of the model's parameters in Python dictionary format;
7. factors_names - names of features that used in training of model;
8. test_len - number of data that will be replaced from train data;
9. loss - Keras loss to train model;
10. optimizer - Keras optimizer to train model.

#### Model supports 5 methods:
1. load_model – load weights of a neural network from a file format h5;
2. fit - train the neural network;
3. predict - predict using a neural network, using two methods:
  a. _predict_point_by_point - cyclical forecast for one value forward;
  b. _predict_multi_step - forecast vector on the forecast horizon.
4. _build_model - build a neural network according to the parameters specified in the configs;
5. _data_process - process input data;
  a. scaler_fit - training the scaler;
  b. scaler_fit_transform - training the scaler and convert values to a range from 0 to 1;
  c. scale_transform - convert values to a range from 0 to 1 based on an already trained scaler;
  d. scaler_inverse_transform – inverse transformation of the forecast values.
