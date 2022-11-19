from keras_tuner import HyperParameters

MODEL_STRATEGIES = ["Direct", "Recursive", "MiMo", 'DirRec', "DirMo"]

DEFAULT_HP = HyperParameters()
DEFAULT_ARCH = {"layers": [
    ["LSTM", {"units": DEFAULT_HP.Int(name='units',
                                      min_value=32,
                                      max_value=128,
                                      step=32,
                                      default=64
                                      ),
              "return_sequences": False,
              "kernel_initializer": "glorot_uniform",
              "activation": DEFAULT_HP.Choice(name='LSTM_1_activation',
                                              values=['relu', 'tanh', 'sigmoid', "linear"],
                                              default='relu'),
              }],
    ["Dropout", {"rate": DEFAULT_HP.Float(name='dropout',
                                          min_value=0.0,
                                          max_value=0.5,
                                          default=0.2,
                                          step=0.05)
                 }],
    ["Dense", {"activation": "linear"}]
]}
