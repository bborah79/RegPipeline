import sys
import json
import shutil
import os
import argparse
import yaml
from yaml.loader import FullLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
import statsmodels.api as sm
import scipy.stats as scpst
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

sys.path.append("./regression_pipeline")
from preprocess import PreProcessData
from feature_select import FeatureSelect
from evalmodels import ModelSelection
from feature_eng import FeatureEngineering
from vizualization import VisualizationData
from hyperparam_opt import OptunaOpt


def main():
    argparser = argparse.ArgumentParser(description="Model selection process")
    argparser.add_argument(
        "--gen-config",
        required=True,
        dest="genconfig",
        help="General configurations file of the simulation",
    )
    argparser.add_argument(
        "--optuna-config", dest="optunaconfig", help="Optuna configuration file"
    )
    argsp = vars(argparser.parse_args())
    print(argsp)

    try:
        with open(argsp["genconfig"], "r") as f:
            input_data = yaml.load(f, Loader=FullLoader)
    except FileNotFoundError as ferror:
        print(ferror)
    except IOError as ioerror:
        print(ioerror)
    except yaml.YAMLError as exc:
        print(exc)

    cw_path = input_data.get("cw_path")
    data_file_name = input_data.get("data_file_name")
    cv_method = input_data.get("cv")
    random_state = input_data.get("random_state")
    njobs = input_data.get("njobs")
    pval = input_data.get("p_val")
    tolerance = input_data.get("tolerance")
    var_threshold = input_data.get("variance_thershold")
    step_rfe = input_data.get("step_rfe")

    task = input_data.get("regression")
    regression_flag = task.get("regression_flag")
    models = [eval(val) for key, val in task.get("models").items()]
    rf = models[0]
    linreg = models[1]
    scoring = task.get("scoring")
    score_funcs = [eval(val) for val in task.get("score_funcs")]
    original_col_heads = input_data.get("original_col_heads")

    target_feature = input_data.get("target_feature")
    requires_target_transformation = input_data.get("requires_target_transformation")
    target_transformer = input_data.get("target_transformer")
    requires_feature_transformation = input_data.get("requires_feature_transformation")
    features_transfromers_dict = input_data.get("input_data.get")
    data_contains_nullval = input_data.get("data_contains_nullval")
    null_fill_procedure = input_data.get("null_fill_procedure")
    requires_feature_engineering = input_data.get("requires_feature_engineering")
    requires_feature_selection = input_data.get("requires_feature_selection")
    ntrials = input_data.get("ntrials")
    requires_hyperparam_opt = input_data.get("requires_hyperparam_opt")
    requires_feature_encoding = input_data.get("requires_feature_encoding")
    requires_feature_scaling = input_data.get("requires_feature_scaling")
    scaler = eval(input_data.get("scaler"))
    encoders = input_data.get("encoding_requires")
    features_to_encode = input_data.get("features_to_encode")
    ordinal_encode_map = input_data.get("ordinal_encode_map")

    try:
        org_df = pd.read_csv("".join([cw_path, data_file_name]))
    except FileNotFoundError as ferror:
        print(ferror)
    except IOError as ioerror:
        print(ioerror)
    except:
        print("There is some problem with the data file")

    org_df.columns = original_col_heads

    preprocess = PreProcessData(org_df)

    if data_contains_nullval == "True":
        preprocess.fill_na(null_fill_procedure)

    if requires_feature_engineering == "True":
        feat_eng = FeatureEngineering()
        feature_names, preprocess.df = feat_eng.engineer_feature(preprocess.df)
    else:
        feature_names = [val for val in original_col_heads if val != target_feature]

    preprocess.split_data(
        target_feature,
        requires_target_transformation,
        target_transformer,
        requires_feature_transformation,
        features_transfromers_dict,
    )
    y_train = preprocess.y_train_
    y_test = preprocess.y_test_

    # Encode the categorical features here (this is problem specific and needs
    # change for every problem)
    if requires_feature_encoding == "True":
        if "ordinal_encoding" in encoders:
            preprocess.encode_categorical_features(
                encoders, features_to_encode, ordinal_encode_map
            )

        else:
            preprocess.encode_categorical_features(encoders, features_to_encode)

    if requires_feature_scaling == "True":
        preprocess.scaling_data(scaler)
        Xtrain = preprocess.X_train_scl_
        Xtest = preprocess.X_test_scl_
    else:
        Xtrain = np.array(preprocess.X_train_)
        Xtest = np.array(preprocess.X_test_)

    if requires_feature_selection == "True":
        feature_selection = FeatureSelect(Xtrain, Xtest, y_train, feature_names)
        feature_selection.select_features_vote(
            rf,
            linreg,
            cv_method,
            njobs,
            tolerance,
            var_threshold,
            pval,
            random_state,
            step_rfe,
            ntrials,
        )

        Xtrain = feature_selection.xtrain
        Xtest = feature_selection.xtest
        all_selected_features = feature_selection.all_selected_features_
        updat_feature_names_map = feature_selection.updat_feature_names_map_
        updat_feature_names_sub = feature_selection.updat_feature_names_sub_
    else:
        feature_selection = FeatureSelect(Xtrain, Xtest, y_train, feature_names)
        feature_selection.map_orig_feature_names()
        updat_feature_names_map = feature_selection.org_feature_names_map_
        updat_feature_names_sub = feature_selection.org_feature_names_sub_
        all_selected_features = [list(updat_feature_names_sub)]

    if requires_hyperparam_opt == "True":
        print("running optuna opt.....")
        hyperopt = OptunaOpt(cv_method, njobs, scoring, score_funcs)

        results = hyperopt.run_optuna(
            argsp["optunaconfig"],
            Xtrain,
            Xtest,
            y_train,
            y_test,
            all_selected_features,
        )

        return results

    else:
        model_select = ModelSelection(
            all_selected_features, scoring, score_funcs, cv_method, njobs
        )
        results = model_select.run_model_selection(
            models, Xtrain, Xtest, y_train, y_test, updat_feature_names_map
        )
        return results


if __name__ == "__main__":
    all_results = main()
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 3)

    df_cv = abs(all_results[0].filter(regex="cv|pval|t-stat", axis=1))
    df_oth = abs(
        all_results[0].drop(all_results[0].filter(regex="cv|pval|t-stat").columns, axis=1)
    )

    dfAsString_cv = df_cv.to_string(header=True, index=True)
    dfAsString_oth = df_oth.to_string(header=True, index=True)

    out_path = "./output"
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    with open("./output/output_results_cv.dat", "w") as f:
        f.write(dfAsString_cv)

    with open("./output/output_results.dat", "w") as f:
        f.write(dfAsString_oth)

    vz = VisualizationData(all_results[0])
    vz.plot_train_test_scores()
    vz.plot_cv_train_test_scores()
    vz.plot_pvalues()

    with open("./output/feature_sets.json", 'w') as file:
        json_string = json.dumps(all_results[1], default=lambda o: o.__dict__, sort_keys=True, indent=2)
        file.write(json_string)
