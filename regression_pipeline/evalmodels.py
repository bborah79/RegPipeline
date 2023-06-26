from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from utils import Utilities

# import shap


class ModelSelection:
    """Performs model selection using various feature sets and models.

    This class implements methods in order to figure out the best performing
    models out of all the models under trial along with the best feature set.
    However, the models are the vanila models without any fine tuning of the
    parameters.

    Parameters
    --------------
    all_feature_sets: a list of lists of strings
        a list of various feature sets obtained from feature selection methods

    scoring: a list of strings
        A list of scoring metrics to evaluate the models (i.e.
        neg_mean_squared_error)

    score_funcs: a list of strings
        A list of score functions to evaluate the models (i.e.
        mean_squared_error)

    cv: int, cross-validation generator or an iterator
        if int, defines the number of cross-validated folds

    njobs: int
        Number of jobs to be run in parallel

    Methods
    -------
    evaluate_model_performance(model, xtrain, xtest, ytrain, ytest):
        Evaluates the performance of the given model in terms of the provided
        metrics.

    run_model_selection(models, xtrain, xtest, ytrain, ytest, feature_names_map):
        Runs the model selection process with different feature sets and
        models.

    """

    def __init__(self, all_feature_sets, scoring, score_funcs, cv, njobs):
        self.all_feature_sets = all_feature_sets
        self.scoring = scoring
        self.cv = cv
        self.njobs = njobs
        self.score_funcs = score_funcs

    def evaluate_model_performance(self, model, xtrain, xtest, ytrain, ytest):
        """Evaluates performance of a model on the training and test data.

        The model performance is evaluated in terms of the appropriate metric
        relevant to the problem at hand. Also performs a ttest on the
        cross-validated scores of the model.

        Parameters
        ------------
        model: An instance of the model for which the performance is being
            evaluated

        xtrain: an array of shape (n_samples, n_features)
            Train data

        xtest: an array of shape (m_samples, n_features)
            Test data

        ytrain: an array of shape(n_samples,)
            Target values of train data

        ytest: an array of shape (m_samples, )
            Target values of test data

        Returns
        ---------------
        mean_results_df: a pandas dataframe
            all the performance results w.r.t. the train and test data

        ci_train: a tuple of floats
            the confidence intervals of the predicted mean of the train data

        ci_test: a tuple of floats
            the confidence intervals of the predicted mean of the test data
        """

        mean_results_df = pd.DataFrame()

        cv_results = cross_validate(
            model,
            xtrain,
            ytrain,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.njobs,
            return_train_score=True,
            return_estimator=True,
        )

        mean_test_score = [
            np.mean(cv_results["_".join(["test", score])]) for score in self.scoring
        ]
        mean_train_score = [
            np.mean(cv_results["_".join(["train", score])]) for score in self.scoring
        ]

        train_score_cv = [
            cv_results["_".join(["train", score])] for score in self.scoring
        ]
        test_score_cv = [
            cv_results["_".join(["test", score])] for score in self.scoring
        ]

        score_abbreviation = {
            "neg_mean_squared_error": "mse",
            "neg_mean_absolute_error": "mae",
            "r2": "r2",
            "mean_squared_error": "mse",
            "mean_absolute_error": "mae",
            "r2_score": "r2",
        }

        for ii in range(len(self.scoring)):
            name_score = score_abbreviation[self.scoring[ii]]

            ttest_res = ttest_ind(
                np.array(train_score_cv[ii]), np.array(test_score_cv[ii])
            )

            res_col_head1 = "_".join(["cv_train", name_score])
            res_col_head2 = "_".join(["cv_test", name_score])
            res_col_head3 = "_".join(["pval", name_score])
            res_col_head4 = "_".join(["t-stat", name_score])

            mean_results_df[res_col_head1] = [mean_train_score[ii]]
            mean_results_df[res_col_head2] = [mean_test_score[ii]]

            mean_results_df[res_col_head3] = [ttest_res[1]]
            mean_results_df[res_col_head4] = [ttest_res[0]]

        model.fit(xtrain, ytrain)
        # explainer = shap.Explainer(model, xtrain)
        # shap_values = explainer(xtrain)
        # print(shap_values)

        y_pred_train = model.predict(xtrain)
        y_pred_test = model.predict(xtest)
        mean_y_pred_train = np.mean(y_pred_train)
        sigma_y_pred_train = np.std(y_pred_train)
        mean_y_pred_test = np.mean(y_pred_test)
        sigma_y_pred_test = np.std(y_pred_test)

        low_bound_train, upper_bound_train = stats.norm.interval(
            alpha=0.95, loc=mean_y_pred_train, scale=sigma_y_pred_train
        )  # /np.sqrt(len(y_pred_train)))
        low_bound_test, upper_bound_test = stats.norm.interval(
            alpha=0.95, loc=mean_y_pred_test, scale=sigma_y_pred_test
        )  # /np.sqrt(len(y_pred_test)))
        low_bound_train = low_bound_train - mean_y_pred_train
        upper_bound_train = upper_bound_train - mean_y_pred_train
        low_bound_test = low_bound_test - mean_y_pred_test
        upper_bound_test = upper_bound_test - mean_y_pred_test

        ci_train = (low_bound_train, mean_y_pred_train, upper_bound_train)
        ci_test = (low_bound_test, mean_y_pred_test, upper_bound_test)

        train_score = [score(ytrain, y_pred_train) for score in self.score_funcs]
        test_score = [score(ytest, y_pred_test) for score in self.score_funcs]

        for ii in range(len(self.score_funcs)):
            name_score = score_abbreviation[self.score_funcs[ii].__name__]

            res_col_head5 = "_".join(["train", name_score])
            res_col_head6 = "_".join(["test", name_score])
            res_col_head7 = "_".join(["%diff", name_score])

            mean_results_df[res_col_head5] = [train_score[ii]]
            mean_results_df[res_col_head6] = [test_score[ii]]
            mean_results_df[res_col_head7] = [
                (abs(train_score[ii] - test_score[ii]) / train_score[ii]) * 100
            ]

        return mean_results_df, ci_train, ci_test

    def run_model_selection(
        self, models, xtrain, xtest, ytrain, ytest, feature_names_map
    ):
        """Runs model selection process.

        Parameters
        ------------
        models: a list
            A list of model instances to be tested

        xtrain: an array of shape (n_samples, n_features)
            Training data

        xtest: an array of shape (m_samples, n_features)
            Test data

        ytrain: an array of shape (n_samples, )
            Training target values

        ytest: an array of shape (m_samples, )
            Test target values

        feature_names_map: a dict
            A dictionary mapping the dummy feature names to the original
            feature names

        Returns
        ------------
        all_results_df: a pandas dataframe
            performance results of the models tried in the study

        diffnt_trial_features: a dict
            A dictionary of the vaious feature sets that were tried in terms
            of the orginal feature names

        ci_train_all: a list of tuples
            a list of tuples of the confidence intervals w.r.t. the training
            data for all the models

        ci_test_all: a list of tuples
            A list of tuples of the confidence intervals w.r.t. the test data
            for all the models tried
        """

        all_results_df = pd.DataFrame()
        all_reslts_index = []
        FeatureSet = 0
        diffnt_trial_features = {}
        ci_train_all = []
        ci_test_all = []

        for feature_set_i in self.all_feature_sets:
            FeatureSet = FeatureSet + 1

            filtered_feature_names_org = Utilities.extract_raw_feature_names(
                feature_set_i, feature_names_map
            )
            X_filtered_train = Utilities.filter_features(xtrain, feature_set_i)
            X_filtered_test = Utilities.filter_features(xtest, feature_set_i)
            feature_set_name = "".join(["set", str(FeatureSet)])
            diffnt_trial_features[feature_set_name] = filtered_feature_names_org

            for model_i in models:
                results = self.evaluate_model_performance(
                    model_i, X_filtered_train, X_filtered_test, ytrain, ytest
                )

                model_name = "_".join([type(model_i).__name__, feature_set_name])

                all_reslts_index.append(model_name)

                all_results_df = pd.concat(
                    [all_results_df, results[0]], ignore_index=True
                )

                ci_train_all.append(results[1])
                ci_test_all.append(results[2])

        all_results_df = all_results_df.set_axis(all_reslts_index, axis="index")
        all_results_df.index.name = "Models"

        return all_results_df, diffnt_trial_features, ci_train_all, ci_test_all
