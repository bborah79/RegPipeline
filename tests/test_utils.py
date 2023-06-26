import sys
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

sys.path.append("./regression_pipeline")
from utils import Utilities


@pytest.fixture
def instantiate_class():
    utl = Utilities()
    return utl


def test_extract_raw_feature_names(instantiate_class):
    feature_subset = ["x0", "x1", "x2"]
    feature_names_map = {"x0": "Feat1", "x1": "Feat2", "x2": "Feat3"}
    actual_raw_names = instantiate_class.extract_raw_feature_names(
        feature_subset, feature_names_map
    )
    expected = ["Feat1", "Feat2", "Feat3"]

    assert actual_raw_names == expected


def test_filter_features(instantiate_class):
    xmat = np.array(
        [
            [0.78, 0.23, 0.99, 0.23],
            [0.23, 0.33, 0.43, 0.78],
            [0.33, 0.45, 0.89, 0.21],
            [0.11, 0.23, 0.34, 0.78],
        ]
    )

    feature_subset = ["x0", "x1", "x2"]

    xmat_exp = np.array(
        [[0.78, 0.23, 0.99], [0.23, 0.33, 0.43], [0.33, 0.45, 0.89], [0.11, 0.23, 0.34]]
    )

    xmat_actual = instantiate_class.filter_features(xmat, feature_subset)

    assert np.array_equal(xmat_exp, xmat_actual) == True


@patch("utils.variance_inflation_factor")
def test_calculate_VIF(mock_method, instantiate_class):
    xmat = np.array([[0.78, 0.23, 0.78]])

    expec_vifdata = pd.DataFrame(columns=["feature", "VIF"])
    expec_vifdata["feature"] = ["x0", "x1", "x2"]
    expec_vif = [10, 10, 10]
    mock_method.return_value = 10
    expec_vifdata["VIF"] = expec_vif

    reqd_vifdata = instantiate_class.calculate_VIF(xmat)

    assert list(expec_vifdata["VIF"]) == list(reqd_vifdata["VIF"])
    assert mock_method.call_count == 3
