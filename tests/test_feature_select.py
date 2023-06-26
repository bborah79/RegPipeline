import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from unittest.mock import Mock, patch

sys.path.append("./regression_pipeline")
from feature_select import FeatureSelect
import feature_select


@pytest.fixture
def instance_fselectclass():
    Xtn = np.random.rand(5, 5)
    Xtt = np.random.rand(3, 5)
    ytr = np.random.rand(5)
    f_names = ["A", "B", "C", "X", "Y"]
    fselect = FeatureSelect(Xtn, Xtt, ytr, f_names)
    return fselect, Xtn, Xtt, ytr, f_names


def test_the_init_method(instance_fselectclass):
    assert instance_fselectclass[0].xtrain.all() == instance_fselectclass[1].all()
    assert instance_fselectclass[0].xtest.all() == instance_fselectclass[2].all()
    assert instance_fselectclass[0].ytrain.all() == instance_fselectclass[3].all()
    assert instance_fselectclass[0].feature_names == instance_fselectclass[4]


def test_create_feature_names_sub(instance_fselectclass):
    instance_fselectclass[0].create_feature_names_sub()
    expected_features = np.array(["x0", "x1", "x2", "x3", "x4"])
    assert (
        np.array_equal(
            instance_fselectclass[0].org_feature_names_sub_, expected_features
        )
        == True
    )


def test_map_orig_feature_names(instance_fselectclass):
    instance_fselectclass[0].map_orig_feature_names()
    expected_map = {"x0": "A", "x1": "B", "x2": "C", "x3": "X", "x4": "Y"}
    assert (instance_fselectclass[0].org_feature_names_map_ == expected_map) == True


@pytest.mark.parametrize(
    "number_dropped_features, dropped_features, expected_feat_sub, expected_map",
    [
        (
            2,
            np.array(["x1", "x4"]),
            np.array(["x0", "x1", "x2"]),
            {"x0": "A", "x1": "C", "x2": "X"},
        ),
        (
            3,
            np.array(["x0", "x2", "x3"]),
            np.array(["x0", "x1"]),
            {"x0": "B", "x1": "Y"},
        ),
        (4, np.array(["x0", "x2", "x3", "x4"]), np.array(["x0"]), {"x0": "B"}),
        (
            0,
            np.zeros(1),
            np.array(["x0", "x1", "x2", "x3", "x4"]),
            {"x0": "A", "x1": "B", "x2": "C", "x3": "X", "x4": "Y"},
        ),
    ],
)
def test_update_org_feature_names_map(
    instance_fselectclass,
    number_dropped_features,
    dropped_features,
    expected_feat_sub,
    expected_map,
):
    instance_fselectclass[0].map_orig_feature_names()
    instance_fselectclass[0].update_org_feature_names_map(
        number_dropped_features, dropped_features
    )

    assert (instance_fselectclass[0].updat_feature_names_map_ == expected_map) == True
    assert (
        np.array_equal(
            instance_fselectclass[0].updat_feature_names_sub_, expected_feat_sub
        )
        == True
    )


@patch("feature_select.RFECV")
def test_rfecv(mock_rfecv, instance_fselectclass):
    mockinstance = mock_rfecv()
    mockinstance.get_feature_names_out.return_value = np.array(["x5", "x1", "x3", "x4"])
    instance_fselectclass[0].select_recursive_feature_elimination(
        LinearRegression(), 2, 1
    )

    assert (
        np.array_equal(
            instance_fselectclass[0].selector1_features,
            np.array(["x5", "x1", "x3", "x4"]),
        )
        == True
    )

    # assert mock_rfecv.call_count == 2
    mockinstance.get_feature_names_out.assert_called_once()


@pytest.mark.parametrize(
    "support_sets, expected_feat_sets",
    [
        (
            np.array([True, False, False, True, True]),
            np.array(["x0", "x3", "x4", 0, 0], dtype=object),
        ),
        (
            np.array([False, True, False, True, True]),
            np.array(["x1", "x3", "x4", 0, 0], dtype=object),
        ),
        (
            np.array([True, True, True, True, True]),
            np.array(["x0", "x1", "x2", "x3", "x4"], dtype=object),
        ),
    ],
)
@patch("feature_select.BorutaPy")
def test_select_borutapy(
    mock_borutapy, instance_fselectclass, support_sets, expected_feat_sets
):
    mockinstance = mock_borutapy()
    attr = {"support_": support_sets}
    mockinstance.configure_mock(**attr)
    instance_fselectclass[0].select_borutapy(RandomForestRegressor(), 1234)

    assert (
        np.array_equal(instance_fselectclass[0].selector2_features, expected_feat_sets)
        == True
    )

    assert mock_borutapy.call_count == 2


@pytest.mark.parametrize(
    "features, expected_feats",
    [
        (
            (np.array([1.0, 7.0, 12.0, 0.8]), np.array([0.003, 0.8, 0.001, 0.02])),
            np.array(["x0", "x2", "x3", 0], dtype=object),
        ),
        (
            ((np.array([1.0, 7.0, 12.0, 0.8]), np.array([0.87, 0.003, 0.1, 0.02]))),
            np.array(["x1", "x3", 0, 0], dtype=object),
        ),
    ],
)
def test_select_fregression(mocker, instance_fselectclass, features, expected_feats):
    mocker.patch("feature_select.f_regression", return_value=features)
    pval = 0.05

    instance_fselectclass[0].select_fregression(pval)

    assert feature_select.f_regression.call_count == 1

    assert (
        np.array_equal(instance_fselectclass[0].selector3_features, expected_feats)
        == True
    )


@patch("feature_select.SequentialFeatureSelector")
def test_select_sequential_feature_selector(mocker_sfs, instance_fselectclass):
    mocksfs_inst = mocker_sfs.return_value
    mocksfs_inst.get_feature_names_out.return_value = np.array(["x5", "x1", "x3", "x4"])
    model = LinearRegression()
    cv = 2
    tolerance = 0.001
    njobs = -1
    instance_fselectclass[0].select_sequential_feature_selector(
        model, cv, tolerance, njobs
    )

    assert (
        np.array_equal(
            instance_fselectclass[0].selector4_features,
            np.array(["x5", "x1", "x3", "x4"]),
        )
        == True
    )

    mocksfs_inst.get_feature_names_out.assert_called_once()
    assert mocker_sfs.call_count == 1


@patch.object(FeatureSelect, "filter_low_variance_features")
@patch.object(FeatureSelect, "select_recursive_feature_elimination")
@patch.object(FeatureSelect, "select_feature_low_vif")
@patch.object(FeatureSelect, "select_borutapy")
@patch.object(FeatureSelect, "select_fregression")
@patch.object(FeatureSelect, "select_sequential_feature_selector")
def test_select_features_vote(
    mock1, mock2, mock3, mock4, mock5, mock6, instance_fselectclass
):
    mock1.return_value = None
    mock2.return_value = None
    mock3.return_value = None
    mock4.return_value = None
    mock5.return_value = None
    mock6.return_value = None

    instance_fselectclass[0].selector0_features = np.array(
        ["x0", "x1", "x2", "x3", "x4"], dtype=object
    )
    instance_fselectclass[0].selector1_features = np.array(
        ["x0", "x2", "x3", "x4"], dtype=object
    )
    instance_fselectclass[0].selector2_features = np.array(
        ["x0", "x1", "x4", 0, 0], dtype=object
    )
    instance_fselectclass[0].selector3_features = np.array(
        ["x0", "x1", "x2", "x4", 0], dtype=object
    )
    instance_fselectclass[0].selector4_features = np.array(
        ["x1", "x2", "x4"], dtype=object
    )
    instance_fselectclass[0].selector5_features = np.array(
        ["x0", "x1", "x4", 0, 0], dtype=object
    )
    instance_fselectclass[0].updat_feature_names_sub_ = np.array(
        ["x0", "x1", "x2", "x3", "x4"], dtype=object
    )

    instance_fselectclass[0].select_features_vote(
        RandomForestRegressor(), LinearRegression(), 2, -1, 0.001, 0.0, 0.05, 123, 1
    )

    expected = ["x0", "x1", "x2", "x4"]

    assert mock1.call_count == 1
    assert mock2.call_count == 1
    assert mock3.call_count == 1
    assert mock4.call_count == 1
    assert mock5.call_count == 1
    assert mock6.call_count == 1
    mock1.assert_called_once()
    mock2.assert_called_once()
    mock3.assert_called_once()
    mock4.assert_called_once()
    mock5.assert_called_once()
    mock6.assert_called_once()
    assert instance_fselectclass[0].selected_features_ == expected
