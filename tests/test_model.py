from ts_rnn.model import TS_RNN
import numpy as np
import shutil


def test_model_fit():
    PRED_LEN = 4
    target = np.arange(1, 12).reshape(-1, 1)
    f1 = (np.arange(1, len(target) + 1) * 10).reshape(-1, 1)
    f2 = (np.arange(1, len(target) + 1) ** 2).reshape(-1, 1)
    factors_df = np.hstack((f1, f2))

    for strategy in ["Direct", "Recursive", "MiMo", 'DirRec', "DirMo"]:
        for factors in [True, False]:
            model = TS_RNN(
                n_lags=3,
                horizon=PRED_LEN,
                tuner="BayesianOptimization",  # "RandomSearch", "BayesianOptimization", "Hyperband"
                strategy=strategy,
                n_step_out=2 if (strategy == "DirMo") else PRED_LEN if (strategy == "MiMo") else 1,
                max_trials=1,
                n_features=factors_df.shape[1] if factors else 0,
                loss="mae",
                optimizer="adam")

            model.fit(target_train=target,
                      factors_train=factors_df if factors else None,
                      target_val=target,
                      factors_val=factors_df if factors else None,
                      epochs=10,
                      batch_size=12,
                      verbose=0)

            predicted = model.predict(
                factors=factors_df[-model.n_lags:] if factors else None,
                target=target[-model.n_lags:],
                prediction_len=PRED_LEN).flatten()

            shutil.rmtree('TS_RNN_tuner_log', ignore_errors=True)
            assert len(predicted) == PRED_LEN
