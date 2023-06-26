import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import *

from math_transformers import MathTransformations


class PreProcessData:
    """Pre-processes the raw data.

    The raw data is processed with different methods to make it ready for the
    algorithm to run on.

    Parameters
    ---------------
    df: a pandas dataframe
        The raw data in a pandas dataframe.
     
    Attributes
    -------------
    X_train_: a pandas dataframe (n_samples, n_features)
        Train data

    X_test_: an pandas dataframe (k_samples, n_features)
        Test data

    y_train_: a pandas series (n_samples,)
        Train target values

    y_test_: a pandas series (k_samples, )
        Test target values

    X_train_scl_: an array of shape (n_samples, n_features)
        Scaled training data

    X_test_scl_: an array of shape (k_samples, n_features)
        Scaled test data

    Methods
    ------------
    fill_na():
        Fills the null values in the data with appropriate values.

    split_data():
        Splits the data and create a training and a test datasets.

    encode_categorical_features():
        Encodes the categorical features with appropriate algorithm.

    scaling_data():
        Scales the data in order to have all features in same scale.
    """

    def __init__(self, df):
        self.df = df

    def fill_na(self, fill_val_def):
        """Fills the missing values (NAs) or the null values.

        The null values in the data need to be treated appropriately so that we
        do not loose on the critical information. This function allows one to
        fill in those null values with the mean, median or the mode of the
        feature. The way of filling in the null vallues depends upon the
        problem and other situations.

        Parameters
        -----------
        fill_val_def: str
            String(i.e. mean, median, or mode) to define the way to fill the NA
            values.
        """

        num_cols = len(self.df.columns)
        for col in range(num_cols):
            if self.df.iloc[:, col].isnull().sum() > 0:
                if fill_val_def == "median":
                    fill_val = self.df.iloc[:, col].median()
                elif fill_val_def == "mode":
                    fill_val = self.df.iloc[:, col].mode()[0]
                elif fill_val_def == "mean":
                    fill_val = self.df.iloc[:, col].mean()

                self.df.iloc[:, col].fillna(fill_val, inplace=True)

    def split_data(
        self,
        Target_col,
        requires_target_transformation,
        target_transformer,
        requires_feature_transformation,
        features_transfromers_dict,
    ):
        """Splits data and creates train and test data.

        This function uses the sklearn train_test_split method to create train
        and test data. Also, transforms the data as per requirement with the
        appropriate math function.

        Parameters
        --------------
        Target_col: string
            The name of the target in the dataf.

        requires_target_transformation: string (either "True" or "False")
            Indicates whether requires math transformation of the target
            values.

        target_transformer: string
            Indicates the type of transformation requires. The available
            transformer keywords are:
            (logtransform, squaretransform, squareroottransform,
            reciprocaltransform, boxcoxtransform,
            yeojohnsontransform, quantiletransform)

        requires_feature_transformation: string (either "True" or "False")
            Indicates whether requires any math transformation of any features
            in the data.

        features_transfromers_dict: a dictionary
            A dictionary of maping the feature name and the corresponding
            required transformation (i.e. {"feat1": "logtransform", "feat2":
            "squaretransform"})
        """

        self.X = self.df.loc[:, self.df.columns != Target_col]
        self.Y = self.df[Target_col].copy()

        if requires_target_transformation == "True":
            mathtrans = MathTransformations()
            func = getattr(mathtrans, target_transformer)
            self.Y = func(self.Y)
        else:
            self.Y = np.array(self.Y)

        if requires_feature_transformation == "True":
            mathtrans = MathTransformations()
            for feature, transformer in features_transfromers_dict.items():
                func = getattr(mathtrans, transformer)
                self.X[feature] = func(np.array(self.X[feature]))

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            self.X, self.Y, random_state=1324
        )

    def encode_categorical_features(
        self, encoders, features_to_encode, ordinal_encode_map=None
    ):
        """Encodes the categorical features with the desired encoder.

        Parameters
        ---------------
        encoders: a list of strings
            The desired encoder key-words to encode the features.

        features_to_encode: a dict
            A dictionary with keys being the encoder key-word and the values
            being the list of features to be encoded.

        ordinal_encode_map: a list of dictionaries
            A list of dictionaries defining the required maps for ordinal
            encoding and the feature names to be encoded.
        """

        for encoder in encoders:
            if encoder == "target_encoding":
                enc_target = TargetEncoder(
                    cols=features_to_encode[encoder],
                    min_samples_leaf=30,
                    smoothing=20,
                )  # Target based
                self.X_train_ = enc_target.fit_transform(self.X_train_, self.y_train_)
                self.X_test_ = enc_target.transform(self.X_test_)
            elif encoder == "catboost_encoding":
                enc_catboost = CatBoostEncoder(
                    cols=features_to_encode[encoder],
                    sigma=10.0,
                    a=5,
                )  # Target based
                self.X_train_ = enc_catboost.fit_transform(self.X_train_, self.y_train_)
                self.X_test_ = enc_catboost.transform(self.X_test_)
            elif encoder == "ordinal_encoding":
                enc_ordinal = OrdinalEncoder(
                    cols=features_to_encode[encoder],
                    mapping=ordinal_encode_map,
                )  # Not Target based
                self.X_train_ = enc_ordinal.fit_transform(self.X_train_)
                self.X_test_ = enc_ordinal.transform(self.X_test_)

    def scaling_data(self, scaler):
        """Scales the features with the desired scaler.

        The features may have very different scales of measurements. In such
        case, the features need to be scaled to have features in the similar
        scale.

        Paramaters
        -------------
        scaler: a scaler instance (i.e. MinMaxScaler)
        """

        self.X_train_scl_ = scaler.fit_transform(self.X_train_)
        self.X_test_scl_ = scaler.transform(self.X_test_)
