import re


class FeatureEngineering:
    """Engineers new features or modifies existing features.

    Feature engineering is crucial to enhance performance at times.
    Methods under this module should be able to engineer new appropriate
    features or modifies exsiting feature to make it more appropriate.
    However, these methods are problem specific and one needs to manually
    add them as the problem at hand demands.

    Methods
    ---------
    engineer_feature:
        Creates new features from existing features

    """

    def engineer_feature(self, df):
        """Creates new features from the existing features.

        Parameters
        -----------
        df : pandas dataframe
            The dataset as a dataframe

        Returns
        ---------
        feature_names : a list
            A list of the feature names in the original dataset

        df : pandas dataframe
            The modified dataframe with the new features

        """

        # Convert the Max_torque and Max_power features to numeric features
        df["Engine_f"] = [int(val.split()[0]) for val in list(df["Engine"])]
        df["Engine_hp"] = [
            float(re.split(r" |@", val)[0]) for val in list(df["Max_power"])
        ]
        df.drop(columns=["Engine", "Max_power", "Max_torque"], inplace=True)
        feature_names = [val for val in df.columns if val != "Price"]

        df["Year"] = df["Year"].apply(int)
        df["Seating_capacity"] = df["Seating_capacity"].apply(str)

        return feature_names, df
