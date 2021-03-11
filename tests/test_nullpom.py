import os
import shutil

import pandas as pd
import pytest
from nullpom.main import Experiment, NullImportanceResult, run_null_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=42, test_size=0.3
)

TMP_RESULT_DIR = "./tmp"


@pytest.fixture
def result():
    result = run_null_importance(
        {"objective": "binary", "seed": 42},
        output_dir=TMP_RESULT_DIR,
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )
    yield result
    shutil.rmtree(TMP_RESULT_DIR)


class TestExperiment:
    def test_init(self):
        _ = Experiment(
            {"objective": "binary"},
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
        )
        assert True

    def test_execute(self):
        experiment = Experiment(
            {"objective": "binary"},
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
        )
        experiment.execute()
        assert True


def test_run_null_importance(result):
    assert os.path.exists(os.path.join(TMP_RESULT_DIR, "actual_model.pkl"))
    assert os.path.exists(os.path.join(TMP_RESULT_DIR, "actual_importance.pkl"))
    assert os.path.exists(os.path.join(TMP_RESULT_DIR, "null_importance.pkl"))

    result_loaded = NullImportanceResult.load(TMP_RESULT_DIR)
    assert hasattr(result_loaded, "actual_model")
    assert hasattr(result_loaded, "actual_importance")
    assert hasattr(result_loaded, "null_importance")
