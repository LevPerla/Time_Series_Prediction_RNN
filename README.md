# Time_Series_Prediction_RNN

## Getting Started

Installation:  
<code>$ pip3 install -r requirements.txt</code>

## Example
You could find it on notebooks/Example.ipynb

## Discription
The library includes two main modules:
* Data_processor - class that is responsible for converting data;
* Model - class that represents a model of a recurrent neural network with its attributes.

### Data_processor
The Data_processor class includes the main data processing methods that need to be performed when predicting using recurrent neural networks.

The Data_processor class has 2 attributes: <em>scaler_target</em> and <em>scale_factors</em>, which store scaler settings (classes that convert values to a range from 0 to 1).

#### Data_processor supports 6 methods:
1. scaler_fit - training the scaler;
2. scaler_fit_transform - training the scaler and convert values to a range from 0 to 1;
3. scale_transform - convert values to a range from 0 to 1 based on an already trained scaler;
4. scaler_inverse_transform – inverse transformation of the forecast values;
5. train_test_split - splitting the time series into training and test samples;
6. split_sequence – transform of the time series under the input format of the neural network.

### Model
#### Model class has 7 attributes:
1. model – a Sequential neural network model (internal class of the Keras library);
2. id – model Number in uuid format, required for saving experiment logs;
3. n_step_in - length of the input vector;
4. n_step_out - length of the output vector;
5. n_features - number of time series in the input;
6. params - description of the model's hyperparameters in Python dictionary format.
7. factors_names - names of features that used in training of model

#### Model supports 5 methods:
1. load_model – load weights of a neural network from a file format h5;
2. build_model - build a neural network according to the parameters specified in the configs file.json;
3. fit - train the neural network;
4. save - save the model weights in h5 format along the specified path;
5. predict - predict using a neural network, using two methods:
a. predict_point_by_point - cyclical forecast for one value forward;
b. predict_multi_step - forecast vector on the forecast horizon.
