import sys
import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append("./regression_pipeline")
from evalmodels import ModelSelection


@pytest.fixture
@patch("evalmodels.r2_score")
def instantiate_cls(mock_r2):
    all_feature_sets = [
        ["x0", "x1", "x2"],
    ]

    scoring = ["r2"]
    cv = 2
    njobs = -1
    mock_r2.side_effect = [0.92, 0.93]
    attr = {"__name__": "r2_score"}
    mock_r2.configure_mock(**attr)
    modelsel_cls = ModelSelection(all_feature_sets, scoring, [mock_r2], cv, njobs)

    return modelsel_cls


@patch("evalmodels.ttest_ind")
@patch("evalmodels.stats.norm.interval")
@patch("evalmodels.cross_validate")
@patch("sklearn.linear_model.LinearRegression")
def test_evaluate_model_performance(
    mock_model, mock_cv, mock_normint, mock_ttest, instantiate_cls
):
    mock_cv.return_value = {
        "test_r2": np.array([0.91, 0.93, 0.92]),
        "train_r2": np.array([0.91, 0.92, 0.93]),
    }
    mock_model_instance = mock_model()
    mock_model_instance.fit.return_value = None
    mock_model_instance.predict.side_effect = [
        np.array([0.23, 0.34, 0.45, 0.55]),
        np.array([0.22, 0.35, 0.43, 0.56]),
    ]
    mock_attrs = Mock(attribute=3)
    attrs = {
        "xtrain": "fake_data_trn",
        "xtest": "fake_data_tst",
        "ytrain": "fake_ytrain",
        "ytest": "fake_ytest",
        "atbt.return_value": 10,
    }
    mock_attrs.configure_mock(**attrs)
    mock_normint.side_effect = [(2.0, 2.0), (2.5, 2.5)]
    mock_ttest.return_value = (12.0, 0.002)
    actual_output = instantiate_cls.evaluate_model_performance(
        mock_model_instance,
        mock_attrs.xtrain,
        mock_attrs.xtest,
        mock_attrs.ytrain,
        mock_attrs.ytest,
    )

    expected_results = pd.DataFrame()
    expected_results["cv_train_r2"] = [np.mean([0.91, 0.92, 0.93])]
    expected_results["cv_test_r2"] = [np.mean([0.91, 0.93, 0.92])]
    expected_results["pval_r2"] = [0.002]
    expected_results["t-stat_r2"] = [12.0]
    expected_results["train_r2"] = [0.92]
    expected_results["test_r2"] = [0.93]
    expected_results["%diff_r2"] = [(abs(0.92 - 0.93) / 0.92) * 100]
    mock_model_instance.fit.assert_called_once()
    assert mock_model_instance.predict.call_count == 2
    assert mock_cv.call_count == 1
    assert mock_normint.call_count == 2
    assert mock_ttest.call_count == 1
    assert mock_model.call_count == 1
    mock_normint.assert_called_with(
        alpha=0.95,
        loc=np.mean([0.22, 0.35, 0.43, 0.56]),
        scale=np.std([0.22, 0.35, 0.43, 0.56]),
    )
    assert actual_output[0].equals(expected_results) == True


@patch.object(ModelSelection, "evaluate_model_performance")
@patch("evalmodels.Utilities")
@patch("sklearn.linear_model.LinearRegression")
def test_run_model_selection(
    mock_lrmodel, mock_utils, mock_model_perf, instantiate_cls
):
    model_inst = mock_lrmodel()
    models = [model_inst]
    fake_data = Mock()
    fake_attr = {
        "xtrain": "fake_xtrain",
        "xtest": "fake_xtest",
        "ytrain": "fake_ytrain",
        "ytest": "fake_ytest",
    }
    fake_data.configure_mock(**fake_attr)
    feature_names_map = {"x0": "feat1", "x1": "feat2", "x2": "feat3", "x3": "feat4"}
    mock_utils_inst = mock_utils.return_value
    mock_utils_inst.filter_features.side_effect = [fake_data.xtrain, fake_data.xtest]
    mock_utils_inst.extract_raw_feature_names.return_value = ["feat1", "feat2", "feat3"]
    results = pd.DataFrame(index=["MagicMock_set1"])
    results.index.name = "Models"
    results["cv_train_r2"] = [np.mean([0.91, 0.92, 0.93])]
    results["cv_test_r2"] = [np.mean([0.91, 0.93, 0.92])]
    results["pval_r2"] = [0.002]
    results["t-stat_r2"] = [12.0]
    results["train_r2"] = [0.92]
    results["test_r2"] = [0.93]
    results["%diff_r2"] = [(abs(0.92 - 0.93) / 0.92) * 100]
    expected_results = (results, (1.0, 20.0, 1.0), (1.0, 20.0, 1.0))
    mock_model_perf.return_value = expected_results

    actual_result = instantiate_cls.run_model_selection(
        models,
        fake_data.xtrain,
        fake_data.xtest,
        fake_data.ytrain,
        fake_data.ytest,
        feature_names_map,
    )
    assert actual_result[0].equals(expected_results[0])
