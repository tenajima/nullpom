import math
import os
import pickle
import shutil
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class NullImportanceResult:
    def __init__(
        self,
        actual_importance: pd.DataFrame,
        actual_model: lgb.Booster,
        null_importance: pd.DataFrame,
    ):
        self.actual_importance = actual_importance
        self.actual_model = actual_model
        self.null_importance = null_importance

    def save(self, output_dir: str, fig=Optional[Figure]) -> None:
        with open(os.path.join(output_dir, "actual_model.pkl"), "wb") as f:
            pickle.dump(self.actual_model, f)

        self.actual_importance.to_pickle(
            os.path.join(output_dir, "actual_importance.pkl")
        )
        self.null_importance.to_pickle(os.path.join(output_dir, "null_importance.pkl"))

        if fig is not None:
            fig.savefig(os.path.join(output_dir, "distribution_of_importance.png"))

    @classmethod
    def load(cls, input_dir: str):
        actual_model = pd.read_pickle(os.path.join(input_dir, "actual_model.pkl"))
        actual_importance = pd.read_pickle(
            os.path.join(input_dir, "actual_importance.pkl")
        )
        null_importance = pd.read_pickle(os.path.join(input_dir, "null_importance.pkl"))
        return cls(actual_importance, actual_model, null_importance)

    def plot_importance(self) -> Figure:
        features = self.actual_importance["feature"].unique().tolist()
        num_features = len(features)
        AX_COUNT_PER_ROW = 4
        col_width = AX_COUNT_PER_ROW * 8
        row_width = np.maximum(1, (num_features // AX_COUNT_PER_ROW)) * 4
        fig = plt.figure(figsize=(col_width, row_width))
        fig.subplots_adjust(hspace=0.6)
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

    def get_effective_feature(self, threshold: Union[int, float] = 90):
        feature_threshold = (
            self.null_importance.groupby("feature")["importance"]
            .apply(lambda x: np.percentile(x, threshold))
            .to_dict()
        )
        importance = self.actual_importance.copy()
        importance["threshold"] = importance["feature"].map(feature_threshold)
        effective_feature = importance.loc[
            importance["importance"] > importance["threshold"], "feature"
        ].tolist()
        return effective_feature


class Experiment:
    def __init__(self, params, n_runs=100, *, X_train, X_valid, y_train, y_valid):
        self.params = params
        self.n_runs = n_runs
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def check_params(self) -> None:
        if "early_stopping_rounds" not in self.params.keys():
            self.params["early_stopping_rounds"] = 100

    def get_feature_importance(
        self, *, shuffle=False
    ) -> Tuple[pd.DataFrame, lgb.Booster]:
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

    def execute(self) -> NullImportanceResult:
        self.check_params()

        actual_importance, actual_model = self.get_feature_importance()

        null_importance = pd.DataFrame()

        for _ in range(self.n_runs):
            tmp_importance, _ = self.get_feature_importance(shuffle=True)
            null_importance = pd.concat([null_importance, tmp_importance])

        return NullImportanceResult(actual_importance, actual_model, null_importance)


def run_null_importance(
    params: Dict,
    output_dir: str = "output/{time}",
    n_runs: int = 100,
    plot_importance: bool = True,
    if_exists: str = "error",
    *,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> NullImportanceResult:
    experiment = Experiment(
        params,
        n_runs=n_runs,
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )
    output_dir = output_dir.format(time=datetime.now().strftime(r"%Y%m%d_%H%M%S"))

    # Check output_dir
    if os.path.exists(output_dir):
        if if_exists == "error":
            raise ValueError("directory {} already exists.".format(output_dir))
        elif if_exists == "replace":
            warnings.warn(
                "directory {} already exists. \
                    It will be replaced by the new result".format(
                    output_dir
                )
            )
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            raise ValueError("`if_exists` must be `error` or `replace`.")

    os.makedirs(output_dir)

    result = experiment.execute()
    if plot_importance:
        fig = result.plot_importance()
    else:
        fig = None
    result.save(output_dir, fig)
    return result
