import re
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Utilities:
    """Performs various utility tasks required during ML model selection run

    Methods
    -------------
    extract_raw_feature_names():
        Extracts the original feature names from the subscripted feature names
    filter_features():
        Filters out the unwanted features from the given training or test
        dataset.
    calculate_VIF():
        Calculates variance inflation factor for the features in the dataset
    """

    @staticmethod
    def extract_raw_feature_names(feature_subset, feature_names_map):
        """Extracts the original feature names in the raw data for the selected feature subsets

        Parameters
        -----------
        feature_subset: an array
                        Selected feature names in the x0,x1,x2.....form

        feature_names_map: A dictionary
                        The original feature names mapped to subscripted feature names

        Returns
        ----------
        feature_names_raw: A list
                        Orginal feature names corresponding to the subscripted features in the feature_susbet
        """

        feature_names_raw = [feature_names_map[val] for val in feature_subset]

        return feature_names_raw

    @staticmethod
    def filter_features(X, feature_subset):
        """Filters out the less important features from the data

        Parameters
        -------------
        X: numpy array of shape (n_samples, n_features)
            Training/test data

        subset_features: an array
            Selected relevant feature names

        Returns
        -----------
        X_filtered: numpy array of shape (n_samples, n_features - k)
            Filtered datasets set
        """

        temp = re.compile("([a-zA-Z]+)([0-9]+)")
        feature_indices = [int(temp.match(val).groups()[1]) for val in feature_subset]
        X_filtered = X[:, feature_indices]

        return X_filtered

    @staticmethod
    def calculate_VIF(X):
        """Calculates the variance inflation factor (VIF) of the features

        Parameters
        --------------
        X: numpy array of shape (n_samples, n_features)
            The numeric features of the training set

        Returns
        -----------
        vif_data: A pandas dataframe
            vif values of each of the features in the training set
        """

        X_numeric_df = pd.DataFrame(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_numeric_df.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_numeric_df.values, i)
            for i in range(len(X_numeric_df.columns))
        ]

        return vif_data
