import collections
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ts_rnn.utils import save_image, mean_squared_error, mean_absolute_error


####################################         feature_importance         ################################################

def feature_importance(target_train, target_val,
                       factors_train,
                       model,
                       ratio=0.2, metric="mae",
                       plot=False, save_dir=None, seed=None, max_iter=None):
    print("\nMaking feature importance")
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Get factors names
    col_names = np.array(model.factors_names)
    res_dict = collections.Counter()

    # Add stopper
    if (max_iter is None) or (max_iter > len(factors_train)):
        max_iter = len(factors_train)

    for _ in tqdm(range(max_iter)):

        real_pred = model.predict(factors=factors_train[-model.n_lags:],
                                  target=target_train[-model.n_lags:],
                                  prediction_len=len(target_val))

        if metric == "mse":
            base_score = mean_squared_error(target_val, real_pred)
        elif metric == "mae":
            base_score = mean_absolute_error(target_val, real_pred)

        final_score = []
        shuff_pred = []

        for i, col in enumerate(col_names):

            # shuffle column
            shuff_train = factors_train.copy()
            shuff_train.iloc[:, i] = np.random.permutation(shuff_train.iloc[:, i])

            prediction = model.predict(factors=shuff_train[-model.n_lags:],
                                       target=target_train[-model.n_lags:],
                                       prediction_len=len(target_val))

            if metric == "mse":
                score = mean_squared_error(target_val, prediction)
            elif metric == "mae":
                score = mean_absolute_error(target_val, prediction)
            else:
                model.logger.critical(f'Use metric from ["mse", "mae"]')
                raise AssertionError(f'Use metric from ["mse", "mae"]')

            shuff_pred.append(prediction)
            final_score.append(score)

        final_score = np.asarray(final_score)

        scores = (final_score - base_score) / base_score * 100

        for feature in col_names[scores >= scores.mean()]:
            res_dict[feature] += 1

    plt.bar(list(map(lambda x: x[0], res_dict.most_common())),
            list(map(lambda x: x[1] / max_iter, res_dict.most_common())))
    plt.subplots_adjust(bottom=0.5, top=1)
    plt.tight_layout()

    if save_dir is not None:
        save_image(save_dir, name="factors_importance", fmt="png")
    if plot:
        plt.show()
    plt.close()
    return [row[0] for row in res_dict.most_common() if (row[1] / max_iter) > ratio]
