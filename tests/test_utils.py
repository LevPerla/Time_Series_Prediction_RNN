from ts_rnn.utils import metrics_eval, split_sequence
import numpy as np
import math
import json


def test_metrics():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([11, 12, 13, 14])
    metrics = metrics_eval(y_true, y_pred, print_result=False, save_dir=None)
    assert metrics['Mean Absolute Error'] == 10


def test_split_sequence():
    real_res = json.load(open('test_real_outs.json'))

    n_lags = 3
    horizon = 6
    factors = False
    train = np.arange(1, 12).reshape(-1, 1)

    for strategy in ["Direct", "Recursive", "MiMo", "DirRec",
                     "DirMo"]:  # "Direct", "Recursive", "MiMo", "DirRec", "DirMo"
        res_dict = {}
        n_step_out = 2 if (strategy == "DirMo") else horizon if (strategy == "MiMo") else 1
        n_models = math.ceil(horizon / n_step_out)

        for _i_model in range(
                n_models if (strategy == "DirMo") else 1 if (strategy in ["MiMo", 'Recursive']) else horizon):
            res_dict[f'model №{_i_model}'] = {}
            x, y = split_sequence(train,
                                  n_steps_in=n_lags + _i_model if (strategy == "DirRec") else n_lags,
                                  n_steps_out=n_step_out,
                                  _full_out=True if ((factors is not None) and (
                                          strategy in ["Recursive", "DirRec"])) else False,
                                  _i_model=_i_model if (strategy in ["Direct", 'DirMo']) else 0,
                                  _start_ind=_i_model * n_step_out - _i_model if (strategy == "DirMo") else 0)
            res_dict[f'model №{_i_model}']['Input'] = x.reshape(-1, x.shape[-2]).tolist()
            res_dict[f'model №{_i_model}']['Output'] = y.tolist()
        assert res_dict == real_res[strategy]
