import os
import shutil
import yaml
from yaml.loader import FullLoader

def create_yaml_gen_input():
    '''Creates configurations to run model selection

    This method creates a general configuration file
    that is required to run the model selection.
    
    '''
    cwd = os.getcwd()
    cw_path = os.path.dirname(cwd) + os.sep + 'data/'
    input_data = dict(
        cw_path= cw_path,
        data_file_name="car_details_v4.csv",
        # These are general parameters
        cv=10,  # cross validation method
        random_state=1234,
        njobs=-1,  #  parameter to run jobs in all the processors
        p_val=0.05,  # threshold p-value for accepting or rejecting the null hypothesis
        tolerance=1e-3,  # tolerance for SequentialFeatureSelector
        variance_thershold=0.0,
        step_rfe=1,  # steps key for recursive feature selection method
        regression={
            "regression_flag": "True",
            "models": {
                "rf": "RandomForestRegressor()",
                "linreg": "LinearRegression()",
                "svr": "SVR(kernel='linear')",
            },
            "scoring": ["r2", "neg_mean_squared_error"],
            "score_funcs": ["r2_score", "mean_squared_error"],
        }, # regression models and the model evaluation scoring methods
        # These are problem specific parameters related to encoding
        # Provide below a list of the features in the dataset
        original_col_heads=[
            "Make",
            "Model",
            "Price",
            "Year",
            "Kilometer",
            "Fuel_type",
            "Transmission",
            "Location",
            "Color",
            "Owner",
            "Seller_type",
            "Engine",
            "Max_power",
            "Max_torque",
            "Drivetrain",
            "Length",
            "Width",
            "Height",
            "Seating_capacity",
            "Fuel_tank_capacity",
        ],
        target_feature="Price", # The target column head
        requires_target_transformation="True",
        target_transformer="logtransform",
        requires_feature_transformation="False",
        features_transfromers_dict=None,  # (i.e. {"feat1": "logtransform"})
        data_contains_nullval="True",
        null_fill_procedure="mode",
        requires_feature_engineering="True",
        requires_feature_selection="True",
        ntrials = 100, #Provide the number of trials for Boruta algorithm
        requires_hyperparam_opt="False",
        requires_feature_encoding="True",
        requires_feature_scaling="True",
        run_regularized_models = "True",
        alphas = 'np.logspace(-10, 1, 400)',
        l1_ratio_list = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        scaler="MinMaxScaler()",
        encoding_requires=["catboost_encoding", "ordinal_encoding"],

        # Provide a dictionary with the keys as the encoder and the values
        # as the features that require encoding with that specific encoder
        features_to_encode={
            "catboost_encoding": [
                "Model",
                "Make",
                "Location",
                "Color",
                "Seller_type",
                "Fuel_type",
            ],
            "ordinal_encoding": [
                "Owner",
                "Seating_capacity",
                "Transmission",
                "Drivetrain",
            ],
        },
        #Provide a list of dictionaries for mapping the ordinal encoding of
        #each feature that requires ordinal encoding.
        ordinal_encode_map=[
            {
                "col": "Owner",
                "mapping": {
                    "UnRegistered Car": 0,
                    "First": 1,
                    "Second": 2,
                    "Third": 3,
                    "Fourth": 4,
                    "4 or More": 5,
                },
            },
            {
                "col": "Seating_capacity",
                "mapping": {"2.0": 0, "4.0": 1, "5.0": 2, "6.0": 3, "7.0": 4, "8.0": 5},
            },
            {"col": "Drivetrain", "mapping": {"FWD": 0, "RWD": 1, "AWD": 2}},
            {"col": "Transmission", "mapping": {"Manual": 0, "Automatic": 1}},
        ],
    )

    with open("../config/input_data.yaml", "w") as f:
        yaml.dump(input_data, f, sort_keys=False, default_flow_style=False)


def create_yaml_optuna():
    optuna_data = dict(
        name_of_study="car-price-prediction",
        obj_direction="maximize",
        scoring_obejctive="neg_mean_squared_error",
        ntrials=100,
        models_to_try=["random-forest", "gradient-boosting", "huberreg",
                       "quantilereg"],
        random_forest_params=[
            [
                "n_estimators",
                {"dtype": "int", "max_val": 200, "min_val": 20, "step_size": 10},
            ],
            [
                "criterion",
                {"dtype": "categorical", "vals": ["squared_error", "friedman_mse"]},
            ],
            [
                "max_depth",
                {"dtype": "int", "max_val": 200, "min_val": 50, "step_size": 10},
            ],
        ],
        gradient_boost_params=[
            [
                "n_estimators",
                {"dtype": "int", "max_val": 200, "min_val": 20, "step_size": 10},
            ],
            [
                "criterion",
                {"dtype": "categorical", "vals": ["squared_error", "friedman_mse"]},
            ],
            [
                "max_depth",
                {"dtype": "int", "max_val": 200, "min_val": 50, "step_size": 10},
            ],
        ],

        huberreg_params = [
            [
                "epsilon",
                {"dtype": "float", "max_val": 20, "min_val": 1.1, "step_size":
                 1},
            ],
            [
                "alpha",
                {"dtype": "float", "max_val": 10, "min_val": 0.01, "step_size":
                 0.1},
            ],
        ],
        quantilereg_params = [
            [
                "quantile",
                {"dtype": "float", "max_val": 0.9, "min_val": 0.1, "step_size":
                0.1},
            ],
            [
                "alpha",
                {"dtype": "float", "max_val": 10, "min_val": 0.01, "step_size":
                0.1},
            ],
        ],
    )

    with open("../config/hyperopt_data.yaml", "w") as f:
        yaml.dump(optuna_data, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    config_path = "../config"
    if os.path.exists(config_path):
        shutil.rmtree(config_path)
    os.mkdir(config_path)

    create_yaml_gen_input()
    create_yaml_optuna()
