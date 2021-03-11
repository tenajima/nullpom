import pickle
import math
import os
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_null_importance(
    params, output_dir="", n_runs=100, *, X_train, X_valid, y_train, y_valid
):
    experiment = Experiment(
        params,
        n_runs=n_runs,
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )
    result = experiment.execute()
    if output_dir == "":
        output_dir = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    result.save(output_dir)
    return result


class NullImportanceResult:
    def __init__(self, actual_importance, actual_model, null_importance):
        self.actual_importance = actual_importance
        self.actual_model = actual_model
        self.null_importance = null_importance

    def save(self, output_dir):
        output_dir = os.path.join("./", output_dir)
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "actual_model.pkl"), "wb") as f:
            pickle.dump(self.actual_model, f)

        self.actual_importance.to_pickle(
            os.path.join(output_dir, "actual_importance.pkl")
        )
        self.null_importance.to_pickle(os.path.join(output_dir, "null_importance.pkl"))

        fig = self.plot_importance()
        fig.savefig(os.path.join(output_dir, "distribution_of_importance.png"))

    @classmethod
    def load(cls, input_dir):
        actual_model = pd.read_pickle(os.path.join(input_dir, "actual_model.pkl"))
        actual_importance = pd.read_pickle(
            os.path.join(input_dir, "actual_importance.pkl")
        )
        null_importance = pd.read_pickle(os.path.join(input_dir, "null_importance.pkl"))
        return cls(actual_importance, actual_model, null_importance)

    def plot_importance(self):
        features = self.actual_importance["feature"].unique().tolist()
        num_features = len(features)
        AX_COUNT_PER_ROW = 4
        col_width = AX_COUNT_PER_ROW * 8
        row_width = np.maximum(1, (num_features // AX_COUNT_PER_ROW)) * 4
        fig = plt.figure(figsize=(col_width, row_width))
        num_of_rows = math.ceil(num_features / AX_COUNT_PER_ROW)
        for i, feature in enumerate(features):
            ax = fig.add_subplot(num_of_rows, AX_COUNT_PER_ROW, i + 1)
            hist_info = ax.hist(
                self.null_importance.query(f"feature == '{feature}'")["importance"],
                label="Null importance",
            )
            ax.vlines(
                x=self.actual_importance.loc[i, "importance"],
                ymin=0,
                ymax=np.max(hist_info[0]),
                color="r",
                linewidth=5,
                label="Real Target",
            )
            ax.legend(loc="upper right")
            ax.set_title(f"Importance of {feature.upper()}", fontweight="bold")
            ax.set_xlabel(f"Null Importance Distribution for {feature.upper()}")
            ax.set_ylabel("Importance")
        return fig


class Experiment:
    def __init__(self, params, n_runs=100, *, X_train, X_valid, y_train, y_valid):
        self.params = params
        self.n_runs = n_runs
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def check_params(self):
        if "early_stopping_rounds" not in self.params.keys():
            self.params["early_stopping_rounds"] = 100

    def get_feature_importance(self, *, shuffle=False):
        if shuffle:
            y_train = np.random.permutation(self.y_train)
            y_valid = np.random.permutation(self.y_valid)
            dtrain = lgb.Dataset(self.X_train, y_train)
            dvalid = lgb.Dataset(self.X_valid, y_valid)
            verbose_eval = False
        else:
            dtrain = lgb.Dataset(self.X_train, self.y_train)
            dvalid = lgb.Dataset(self.X_valid, self.y_valid)
            verbose_eval = 1000

        model = lgb.train(
            self.params, dtrain, valid_sets=[dtrain, dvalid], verbose_eval=verbose_eval
        )

        importance = pd.DataFrame()
        importance["feature"] = model.feature_name()
        importance["importance"] = model.feature_importance("gain")
        importance = importance.sort_values("importance", ascending=False).reset_index(
            drop=True
        )
        return importance, model

    def execute(self):
        self.check_params()

        actual_importance, actual_model = self.get_feature_importance()

        null_importance = pd.DataFrame()

        for _ in range(self.n_runs):
            tmp_importance, _ = self.get_feature_importance(shuffle=True)
            null_importance = pd.concat([null_importance, tmp_importance])

        return NullImportanceResult(actual_importance, actual_model, null_importance)
