import sys
import pytest
import pandas as pd
import numpy as np
from scipy import stats

sys.path.append("./regression_pipeline")
from preprocess import PreProcessData


@pytest.fixture
def instantiate_preprocess_cls():
    dframe = pd.DataFrame(columns=["feat1", "feat2", "feat3", "feat4"])
    dframe["feat1"] = [1.0, 2.0, 2.0, 4.0, 5.0, np.NaN]
    dframe["feat2"] = [1.0, 1.0, np.NaN, 9.0, np.NaN, 10.0]
    dframe["feat3"] = [1.0, 9.0, np.NaN, 9.0, np.NaN, 10.0]
    dframe["feat4"] = [1.0, 2.0, np.NaN, 3.0, np.NaN, 1.0]

    preproc = PreProcessData(dframe)
    return preproc


def test_fill_na_mean(instantiate_preprocess_cls):
    expected_df = pd.DataFrame(columns=["feat1", "feat2", "feat3", "feat4"])
    expected_df["feat1"] = [1.0, 2.0, 2.0, 4.0, 5.0, np.mean([1.0, 2.0, 2.0, 4.0, 5.0])]
    expected_df["feat2"] = [
        1.0,
        1.0,
        np.mean([1.0, 1.0, 9.0, 10.0]),
        9.0,
        np.mean([1.0, 1.0, 9.0, 10.0]),
        10.0,
    ]
    expected_df["feat3"] = [
        1.0,
        9.0,
        np.mean([1.0, 9.0, 9.0, 10.0]),
        9.0,
        np.mean([1.0, 9.0, 9.0, 10.0]),
        10.0,
    ]
    expected_df["feat4"] = [
        1.0,
        2.0,
        np.mean([1.0, 2.0, 3.0, 1.0]),
        3.0,
        np.mean([1.0, 2.0, 3.0, 1.0]),
        1.0,
    ]

    instantiate_preprocess_cls.fill_na("mean")
    assert expected_df.equals(instantiate_preprocess_cls.df)


def test_fill_na_median(instantiate_preprocess_cls):
    expected_df1 = pd.DataFrame(columns=["feat1", "feat2", "feat3", "feat4"])
    expected_df1["feat1"] = [
        1.0,
        2.0,
        2.0,
        4.0,
        5.0,
        np.median([1.0, 2.0, 2.0, 4.0, 5.0]),
    ]
    expected_df1["feat2"] = [
        1.0,
        1.0,
        np.median([1.0, 1.0, 9.0, 10.0]),
        9.0,
        np.median([1.0, 1.0, 9.0, 10.0]),
        10.0,
    ]
    expected_df1["feat3"] = [
        1.0,
        9.0,
        np.median([1.0, 9.0, 9.0, 10.0]),
        9.0,
        np.median([1.0, 9.0, 9.0, 10.0]),
        10.0,
    ]
    expected_df1["feat4"] = [
        1.0,
        2.0,
        np.median([1.0, 2.0, 3.0, 1.0]),
        3.0,
        np.median([1.0, 2.0, 3.0, 1.0]),
        1.0,
    ]

    instantiate_preprocess_cls.fill_na("median")
    assert expected_df1.equals(instantiate_preprocess_cls.df)


def test_fill_na_mode(instantiate_preprocess_cls):
    expected_df1 = pd.DataFrame(columns=["feat1", "feat2", "feat3", "feat4"])
    expected_df1["feat1"] = [
        1.0,
        2.0,
        2.0,
        4.0,
        5.0,
        stats.mode([1.0, 2.0, 2.0, 4.0, 5.0]),
    ]
    expected_df1["feat2"] = [
        1.0,
        1.0,
        stats.mode([1.0, 1.0, 9.0, 10.0]),
        9.0,
        stats.mode([1.0, 1.0, 9.0, 10.0]),
        10.0,
    ]
    expected_df1["feat3"] = [
        1.0,
        9.0,
        stats.mode([1.0, 9.0, 9.0, 10.0]),
        9.0,
        stats.mode([1.0, 9.0, 9.0, 10.0]),
        10.0,
    ]
    expected_df1["feat4"] = [
        1.0,
        2.0,
        stats.mode([1.0, 2.0, 3.0, 1.0]),
        3.0,
        stats.mode([1.0, 2.0, 3.0, 1.0]),
        1.0,
    ]

    instantiate_preprocess_cls.fill_na("mode")


def test_split_data(instantiate_preprocess_cls):
    instantiate_preprocess_cls.fill_na("mean")
    instantiate_preprocess_cls.split_data("feat4", "False", None, "False", None)

    expected_Xcols = ["feat1", "feat2", "feat3"]
    expected_Y = np.array(
        [
            1.0,
            2.0,
            np.mean([1.0, 2.0, 3.0, 1.0]),
            3.0,
            np.mean([1.0, 2.0, 3.0, 1.0]),
            1.0,
        ]
    )

    expected_X = np.array(
        [
            [1.0, 2.0, 2.0, 4.0, 5.0, np.mean([1.0, 2.0, 2.0, 4.0, 5.0])],
            [
                1.0,
                1.0,
                np.mean([1.0, 1.0, 9.0, 10.0]),
                9.0,
                np.mean([1.0, 1.0, 9.0, 10.0]),
                10.0,
            ],
            [
                1.0,
                9.0,
                np.mean([1.0, 9.0, 9.0, 10.0]),
                9.0,
                np.mean([1.0, 9.0, 9.0, 10.0]),
                10.0,
            ],
        ]
    )

    assert list(instantiate_preprocess_cls.X.columns) == expected_Xcols
    assert np.array_equal(np.array(instantiate_preprocess_cls.Y), expected_Y) == True
    assert np.array_equal(np.array(instantiate_preprocess_cls.X), expected_X.T) == True
