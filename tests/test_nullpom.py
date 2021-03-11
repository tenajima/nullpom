import pandas as pd
from nullpom.main import Experiment, run_null_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=42, test_size=0.3
)


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


def test_run_null_importance():
    run_null_importance(
        {"objective": "binary"},
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )
    assert True
