import optuna
import pickle
import yaml
import pandas as pd
import numpy as np
from yaml.loader import FullLoader
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from evalmodels import ModelSelection
from utils import Utilities


class OptunaOpt:
    """Optimization of hyperparameters with optuna

    Parameters
    -------------
    cv: int, cross-validated generator or an iterator
        if an int, defines the number cross-validation folds

    njobs: int
        number of jobs to be run in parallel

    scoring: a list of strings
        A list of performance metrics to be evaluated

    score_funcs: a list of strings
        A list of score functions from sklearn

    Methods
    ------------
    set_model_params():
        creates the model parameters with their suggested values as per optuna
        format

    parse_hyperparams_inp():
        Parses the input file and getting the control parameters for optuna

    create_model():
        Creates a trial model with the suggested parameters

    objective():
        Defines the objective function to be optimized

    run_optuna():
        Runs the optuna optimization process
    """

    def __init__(self, cv, njobs, scoring, score_funcs):
        self.cv = cv
        self.njobs = njobs
        self.scoring = scoring
        self.score_funcs = score_funcs

    def set_model_params(self, trial, params_names, params_dic):
        """Sets the necessary hyperparameter values as per optuna

        Parameters
        ------------
        trial: Optuna trial object

        params_names: a list
            A list of parameter names of the model to be tuned

        params_dic: a list of dictionaries
            A list of dictionaries indicating range values for the parameters
            to try
        """

        params = {}
        ii = 0
        for name in params_names:
            if params_dic[ii][1]["dtype"] == "int":
                max_val = params_dic[ii][1]["max_val"]
                min_val = params_dic[ii][1]["min_val"]
                step_size = params_dic[ii][1]["step_size"]
                params[name] = trial.suggest_int(name, min_val, max_val, step_size)
            elif params_dic[ii][1]["dtype"] == "float":
                max_val = params_dic[ii][1]["max_val"]
                min_val = params_dic[ii][1]["min_val"]
                step_size = params_dic[ii][1]["step_size"]
                if params_dic[ii][1]["log_val"] == "True":
                    log_val = True
                else:
                    log_val = False
                params[name] = trial.suggest_float(
                    name, min_val, max_val, step=step_size, log=log_val
                )
            elif params_dic[ii][1]["dtype"] == "categorical":
                vals = params_dic[ii][1]["vals"]
                params[name] = trial.suggest_categorical(name, vals)
            ii = ii + 1

        return params

    def parse_hyperparams_inp(self, config_file):
        """Parses the hyperparameter optimization config file

        Parameters
        ------------
        config_file: string
            Name of the config file
        """

        try:
            with open(config_file, "r") as f:
                params_data = yaml.load(f, Loader=FullLoader)
        except FileNotFoundError as ferror:
            print(ferror)
        except IOError as ioerror:
            print(ioerror)
        except yaml.YAMLError as exc:
            print(exc)

        self.name_of_study = params_data.get("name_of_study")
        self.obj_direction = params_data.get("obj_direction")
        self.scoring_obejctive = params_data.get("scoring_obejctive")
        self.ntrials = params_data.get("ntrials")
        self.models_to_try = params_data.get("models_to_try")
        for modeltype in self.models_to_try:
            if modeltype == "random-forest":
                self.random_forest_params = params_data.get("random_forest_params")
            if modeltype == "gradient-boosting":
                self.gradient_boost_params = params_data.get("gradient_boost_params")

    def create_model(self, trial):
        """Creates a trail model with a subset of the parameters

        Parameters
        ------------
        trial: Optuna trial object

        Returns
        ---------
        model: model instance
            A model instance after setting the parameter values
        """

        model_type = trial.suggest_categorical("model_t", self.models_to_try)

        if model_type == "gradient-boosting":
            gradient_boost_params_names = [
                self.gradient_boost_params[ii][0]
                for ii in range(len(self.gradient_boost_params))
            ]

            params = self.set_model_params(
                trial, gradient_boost_params_names, self.gradient_boost_params
            )

            model = GradientBoostingRegressor()
            model.set_params(**params)

        if model_type == "random-forest":
            random_forest_params_names = [
                self.random_forest_params[ii][0]
                for ii in range(len(self.random_forest_params))
            ]
            params = self.set_model_params(
                trial, random_forest_params_names, self.random_forest_params
            )

            model = RandomForestRegressor()
            model.set_params(**params)

        return model

    def objective(self, trial, xtrain, ytrain):
        """Defines the objective function to be optimized

        Parameters
        ----------------
        trial: Optuna trial object

        xtrain: an array of shape (n_samples, n_features)
            Training data

        ytrain: an array of shape (n_samples, )
            Training target values

        Returns
        -----------
        add_score: float
            The addition of the cross-validated train and test scores
        """

        model_to_try = self.create_model(trial)

        cv_results = cross_validate(
            model_to_try,
            xtrain,
            ytrain,
            scoring=self.scoring_obejctive,
            cv=self.cv,
            n_jobs=self.njobs,
            return_train_score=True,
            return_estimator=True,
        )

        mean_test_score = np.mean(cv_results["test_score"])
        mean_train_score = np.mean(cv_results["train_score"])

        add_score = mean_train_score + mean_test_score

        return add_score

    def run_optuna(
        self, optuna_config_file, xtrain, xtest, ytrain, ytest, all_selected_features
    ):
        """Runs the optimization process

        Parameters
        ------------
        optuna_config_file: str
            The name of the hyperparameter optimization configuration

        xtrain: an array of shape (n_sapmles, n_features)
            Train data

        xtest: an array of shape (k_samples, n_features)
            Test data

        ytrain: an array of shape (n_samples, )
            Train target values

        ytest: an array of shape (k_samples, )
            Test target data

        all_selected_features: a list
            A list of all the selected subsets of features

        Returns
        ----------
        all_results_df: a pandas dataframe
            A dataframe of all the results and scores of the optimized models
            for each subset of features
        """

        ii = 0
        best_models_list = []
        results_ind_df = pd.DataFrame()
        all_results_df = pd.DataFrame()
        results_df_indx = []
        mselect = ModelSelection(
            all_selected_features,
            self.scoring,
            self.score_funcs,
            self.cv,
            self.njobs,
        )
        self.parse_hyperparams_inp(optuna_config_file)

        for features in all_selected_features:
            filtered_X_train = Utilities.filter_features(xtrain, features)
            filtered_X_test = Utilities.filter_features(xtest, features)
            ii = ii + 1

            study_name = "".join([self.name_of_study, str(ii)])
            # storage = "sqlite:///{}.db".format(study_name)
            # with open("sampler.pkl", "wb") as fout:
            #    pickle.dump(study.sampler, fout)
            # restored_sampler = pickle.load(open("sampler.pkl", "rb"))

            study = optuna.create_study(
                study_name=study_name,
                direction=self.obj_direction,
                storage=None,
                load_if_exists=False,
                sampler=None,
            )

            study.optimize(
                lambda trial: self.objective(trial, filtered_X_train, ytrain),
                n_trials=self.ntrials,
            )
            best_model = self.create_model(study.best_trial)
            model_name = "".join([type(best_model).__name__, str(ii)])
            results_df_indx.append(model_name)

            results = mselect.evaluate_model_performance(
                best_model, filtered_X_train, filtered_X_test, ytrain, ytest
            )
            all_results_df = pd.concat([all_results_df, results[0]], ignore_index=True)
            best_models_list.append(best_model)
        all_results_df = all_results_df.set_axis(results_df_indx, axis="index")
        all_results_df.index.name = "Models"

        return all_results_df
