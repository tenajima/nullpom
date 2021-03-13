[![Test with pytest](https://github.com/tenajima/nullpom/actions/workflows/pytest.yml/badge.svg)](https://github.com/tenajima/nullpom/actions/workflows/pytest.yml)
[![Format with black](https://github.com/tenajima/nullpom/actions/workflows/black.yml/badge.svg)](https://github.com/tenajima/nullpom/actions/workflows/black.yml)

# nullpom

Library to easily run Null Importances.

## About Null Importances

Null Importances is feature selection process using target permutation tests actual importance significance against the distribution of feature importances when fitted to noise (shuffled target).

### Detail
- [paper](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [kaggle notebook](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances)
- [japanese article](https://qiita.com/trapi/items/1d6ede5d492d1a9dc3c9)

# Output
![output](https://raw.githubusercontent.com/tenajima/nullpom/main/img/distribution_of_importance.png)

# Basic usage

```python
from nullpom import run_null_importance

result = run_null_importance(
    {"objective": "binary", "seed": 42},
     X_train=X_train,
    X_valid=X_valid,
    y_train=y_train,
    y_valid=y_valid,
)
```

# Install
```sh
pip install nullpom
```
