from sklearn.feature_selection import (
    RFECV,
    VarianceThreshold,
    f_regression,
    SequentialFeatureSelector,
)
from genetic_selection import GeneticSelectionCV
from boruta import BorutaPy
from BorutaShap import BorutaShap
import pandas as pd
import numpy as np
import re

from utils import Utilities


class FeatureSelect:
    """Feature selection process

    Variety of feature selection methods are employed in order to obtain
    sets of important features.

    Parameters
    ----------
    xtrain : numpy ndarray
         Training data

    xtest : numpy ndarray
         Test dataset

    ytrain : numpy ndarray
         Training target

    feature_names : a list
         A list of all the feature names.

    Attributes
    ----------
    org_feature_names_sub_ : an array of shape (n_features, )
         An array of dummy feature names in the subscripted form like x0, x1,
         x2...xn.

    org_feature_names_map_ : a dictionary
         A dictionary that maps the dummy features to the original feature names
         of the problem at hand.

    updat_feature_names_sub_ : an array of shape (k_features, )
         An array of the updated dummy features when some features are dropped
         from the data.

    updat_feature_names_map_ : a dictionary
         A dictionary of the updated map of the dummy features to the original
         features.

    all_selected_features_ : a list
         A list of all the feature subsets selected by various methods.

    selected_features_ : a list
         A list of selected feature names selected by majority voting.

    Methods
    -------
    create_feature_names_sub():
         Creates an array of feature names in the form x0, x1, x2,.....,xn.

    map_orig_feature_name():
         Maps the orginal feature names with the feature names in the form
         x0,x1,.....xn.

    update_org_feature_names_map(number_dropped_features, dropped_features):
         Updates feature names and the feature map.

    filter_low_variance_features():
         Drops feature whose variance is low or zero.

    select_recursive_feature_elimination():
         Selects features with recusrsive feature elimination method.

    select_borutapy():
         Selects features with BorutaPy algorithm.

    select_fregression():
         Selects features with correlation significance with the target.

    select_sequential_feature_selector():
         Selects feature with sequential feature selection method.

    select_feature_low_vif():
         Selects feature with low variance inflation factor avoiding
         multicolinierity.

    select_features_BorutaShap():
        Selects features with shap values

    select_features_using_GA():
        Selects features using a genetic algorithm.

    select_features_vote():
         Selects features having majority selection by various methods.

    """

    def __init__(self, xtrain, xtest, ytrain, feature_names):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.feature_names = feature_names

    def create_feature_names_sub(self):
        """Creates an array of dummy feature names

        A set of feature names in the form of x0, x1, x2,....xn are created as
        these will be used while running the machine learning algorithms with
        the data in an array form of shape (n_samples, n_features).
        """

        self.org_feature_names_sub_ = np.array(
            ["".join(["x", str(ii)]) for ii in range(len(self.feature_names))],
            dtype="object",
        )

    def map_orig_feature_names(self):
        """Maps the dummy feature names to the original feature names.

        The original feature names are necessary in order to interpret the
        models. Therefor, this function maps the dummy features (x0, x1,
        x2,...xn) in to the corresponding original feature names of the problem
        of interest. A dictionary is created with the keys being the dummy
        features and the values being the original feature names.
        """

        self.create_feature_names_sub()
        self.org_feature_names_map_ = {
            val1: val2
            for val1, val2 in zip(self.org_feature_names_sub_, self.feature_names)
        }

    def update_org_feature_names_map(self, number_dropped_features, dropped_features):
        """Updates the feature names map in case some features are dropped.

        During the feature selection process some features may get dropped
        completely from the data due to their low variance or other reason. In
        such case, it is necessary to keep a track of the features dropped and
        update the feature name map for book keeping.

        Parameters
        -----------
        number_dropped_features: int
                Number of features dropped.

        dropped_features: A list
                A list of the dropped feature names
        """

        temp = re.compile("([a-zA-Z]+)([0-9]+)")
        cnt = 0
        oth_indx = 0
        init_indx = 0
        self.updat_feature_names_map_ = {}
        updat_shape = len(self.org_feature_names_sub_) - number_dropped_features
        self.updat_feature_names_sub_ = np.empty(updat_shape, dtype=object)

        changed_feature_names_map = dict.copy(self.org_feature_names_map_)

        for key in dropped_features:
            changed_feature_names_map.pop(key, None)

        for k, v in changed_feature_names_map.items():
            cnt = cnt + 1
            if cnt == 1:
                init_indx = int(temp.match(k).groups()[1])
                if init_indx > 0:
                    self.updat_feature_names_sub_[cnt - 1] = "".join(["x", str(0)])
                    self.updat_feature_names_map_["".join(["x", str(0)])] = v
                    init_indx = 0
                elif init_indx == 0:
                    self.updat_feature_names_sub_[cnt - 1] = "".join(
                        ["x", str(init_indx)]
                    )
                    self.updat_feature_names_map_["".join(["x", str(init_indx)])] = v
            else:
                oth_indx = int(temp.match(k).groups()[1])

            diff = oth_indx - init_indx

            if diff == 1:
                self.updat_feature_names_sub_[cnt - 1] = "".join(["x", str(oth_indx)])
                self.updat_feature_names_map_[
                    "".join(["x", str(oth_indx)])
                ] = self.org_feature_names_map_["".join(["x", str(oth_indx)])]
                init_indx = oth_indx
            elif diff > 1:
                self.updat_feature_names_sub_[cnt - 1] = "".join(
                    ["x", str(oth_indx - (diff - 1))]
                )
                self.updat_feature_names_map_[
                    "".join(["x", str(oth_indx - (diff - 1))])
                ] = self.org_feature_names_map_["".join(["x", str(oth_indx)])]
                init_indx = oth_indx - (diff - 1)

    def filter_low_variance_features(self, var_threshold):
        """Filters features that have low variance than the threshold

        Some features in the data may have zero variance or variance much lower
        than the pre-set threshold. Such features are carry not much
        information and hence need to be dropped from the data.

        Parameters
        ------------
        var_threshold: float
            Threshold for the variance for dropping a particular feature from
            the data.
        """

        selector0 = VarianceThreshold(threshold=var_threshold)
        self.xtrain = selector0.fit_transform(self.xtrain)
        self.xtest = selector0.transform(self.xtest)
        self.selector0_features = selector0.get_feature_names_out()

        self.map_orig_feature_names()

        if len(self.selector0_features) < len(self.org_feature_names_sub_):
            num_dropped_features = len(self.org_feature_names_sub_) - len(
                selector0_features
            )
            dropped_features = [
                fet
                for fet in self.org_feature_names_sub_
                if fet not in self.selector0_features
            ]
            self.update_org_feature_names_map(num_dropped_features, dropped_features)
            self.selector0_features = np.copy(self.updat_feature_names_sub_)
        else:
            self.updat_feature_names_map_ = dict.copy(self.org_feature_names_map_)
            self.updat_feature_names_sub_ = np.copy(self.org_feature_names_sub_)

    def select_recursive_feature_elimination(self, model, cv, step_rfe):
        """Selects features using recursive feature elimination method.

        .. _RecFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

        Parameters
        ------------
        model: A model instance (i.e. LinearRegression, SVR, etc.)

        cv: int, cross-validation generator or an iterable
            if integer, then defines the number of cross-validation folds

        step_rfe: int or float
            Please see the sklearn doc for `RecFECV`_.
        """

        selector1 = RFECV(model, step=step_rfe, cv=cv)
        selector1.fit(self.xtrain, self.ytrain)
        self.selector1_features = selector1.get_feature_names_out()

    def select_borutapy(self, rf, random_state, ntrials):
        """Selects features employing BortaPy algorithm

        .. _BorutaPy: https://github.com/scikit-learn-contrib/boruta_py

        More about the BorutaPy algorithm that is used here can be obtained
        in `BorutaPy`_.

        Parameters
        ------------
        rf: A random forest regression instance

        random_state: int
            Seed value to control the random number generator

        ntrials: int
            Number of trials/iterations
        """

        selector2 = BorutaPy(rf, n_estimators="auto", max_iter = ntrials, random_state=random_state)
        selector2.fit(self.xtrain, self.ytrain)
        self.selector2_features = np.zeros(self.xtrain.shape[1], dtype=object)

        ii = 0
        for idx, val in enumerate(selector2.support_):
            if val == True:
                self.selector2_features[ii] = "".join(["x", str(idx)])
                ii = ii + 1

    def select_fregression(self, pval):
        """Selects features based on univariate linear regression tests

        .. _f_regression: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html

        The f_regression algorithm as implemented in sklearn is being used
        here. More can be read from the sklearn `f_regression`_ documentation.

        Parameters
        -------------
        pval: float
            p-value threshold in order to decide if the regression is
            significant or not.
        """

        selector3 = pd.DataFrame(
            f_regression(self.xtrain, self.ytrain), index=["F-static", "p-value"]
        ).T
        self.selector3_features = np.zeros(selector3.shape[0], dtype=object)
        ii = 0
        for idx, val in enumerate(selector3["p-value"]):
            if val < pval:
                self.selector3_features[ii] = "".join(["x", str(idx)])
                ii = ii + 1

    def select_sequential_feature_selector(self, model, cv, tolerance, njobs):
        """Selects features according to sequential feature selection method.

        .. _SeqFS: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html

        The sequential feature selector implemented in sklearn is being used
        here. More of it can be read from the sklearn doocumentation `SeqFS`_.

        Parameters
        -------------
        model: A model instance (i.e. LinearRegression, SVR, etc.)

        cv: int, cross-validation generator or an iterator
            if int, then defines the number corss-validation folds.

        tolerance: float
            If the score is not incremented by at least tol between
            two consecutive feature additions or removals, stop adding or
            removing. More explanation can be read from sklearn doc.

        njobs: int
            Number of jobs to run in parallel.
        """

        selector4 = SequentialFeatureSelector(
            model, n_features_to_select="auto", tol=tolerance, cv=cv, n_jobs=njobs
        )
        selector4.fit(self.xtrain, self.ytrain)
        self.selector4_features = selector4.get_feature_names_out()

    def select_feature_low_vif(self):
        """Selects features with low variance inflation factor (VIF)

        VIF is a key indicator of multicolinearity among the features. if VIF
        is higher, say closer to 10 or higher, then such features are highly
        correlated with other features and need to be dropped. Which features
        need to be dropped is based on the problem at hand and the feasibility
        of the data.
        """

        selector5 = Utilities.calculate_VIF(self.xtrain)
        self.selector5_features = np.zeros(self.xtrain.shape[1], dtype=object)
        ii = 0
        for idx, val in enumerate(selector5["VIF"]):
            if val < 10.0:
                self.selector5_features[ii] = "".join(["x", str(idx)])
                ii = ii + 1

    def select_features_BorutaShap(self, random_state, ntrials):
        """Selects feature set using a shap values from BorutaShap

        .. _BorutaShap: https://github.com/Ekeany/Boruta-Shap

        The BorutaShap method implemented here can be obtained from `BorutaShap`_.

        Parameters
        ------------
        random_state: int
            Seed value to control the random number generator

        ntrials: int
            Number of trials.

        """

        selector6 = BorutaShap(importance_measure="shap", classification=False)
        xcols = ["x" + str(ii) for ii in range(self.xtrain.shape[1])]
        X = pd.DataFrame(self.xtrain, columns=xcols)
        selector6.fit(
            X=X, y=self.ytrain, n_trials=ntrials, verbose=False, random_state=random_state
        )

        self.selector6_features = np.array(selector6.accepted, dtype=object)

    def select_features_using_GA(self, model, cv, njobs):
        """Selects features using genetic algorithm

        .. _GeneticAlgo: https://pypi.org/project/sklearn-genetic/

        The genetic algorithm used here is adopted from `GeneticAlgo`_.
        More about the method can be read `GeneticAlgo`_.

        Parameters
        -------------
        model : a model instance
        cv: int, cross-validation generator or an iterator
            if int, then defines the number of cross-validation folds
        njobs: int
            number of jobs to run in parallel

        """

        estimator = model
        selector7 = GeneticSelectionCV(
            estimator,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_population=100,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=40,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=10,
            caching=True,
            n_jobs=njobs,
        )

        selector7 = selector7.fit(self.xtrain, self.ytrain)

        self.selector7_features = np.zeros(self.xtrain.shape[1], dtype=object)

        ii = 0
        for idx, val in enumerate(selector7.support_):
            if val == True:
                self.selector7_features[ii] = "".join(["x", str(idx)])
                ii = ii + 1

        print(self.selector7_features)

    def select_features_vote(
        self,
        rf,
        model,
        cv,
        njobs,
        tolerance,
        var_threshold,
        pval,
        random_state,
        step_rfe,
        ntrials,
    ):
        """Selects features that are selected by majority of the methods as
        above

        Feature which are selected by majority of the methods implemented here
        are supposed to be important features. This function selects the
        features based on majority voting.

        Parameters
        --------------
        rf: A random forest regression instance

        model: A model instance (i.e. LinearRegression, SVR, etc..)

        cv: int, cross-validation generator or an iterator
            if int, then defines the number of cross-validation folds

        njobs: int
            number of jobs to run in parallel

        tolerance: float
            If the score is not incremented by at least tol between
            two consecutive feature additions or removals, stop adding or
            removing. More explanation can be read from sklearn doc.

        var_threshold: float
            Threshold for the variance for dropping a particular feature from
            the data.

        pval: float
            p-value threshold in order to decide if the regression is
            significant or not.

        random_state: int
            Seed value to control the random number generator

        step_rfe: int or float
            Please see the sklearn doc for RFECV

        ntrials: int
            Number of trials for Boruta algorithm.

        """

        self.filter_low_variance_features(var_threshold)
        self.select_recursive_feature_elimination(model, cv, step_rfe)
        self.select_feature_low_vif()
        self.select_borutapy(rf, random_state, ntrials)
        self.select_fregression(pval)
        self.select_sequential_feature_selector(model, cv, tolerance, njobs)
        self.select_features_BorutaShap(random_state, ntrials)
        self.select_features_using_GA(model, cv, njobs)

        count = np.zeros(len(self.updat_feature_names_sub_), dtype=int)

        for idx, feature in enumerate(self.updat_feature_names_sub_):
            if feature in self.selector1_features:
                count[idx] = count[idx] + 1
            if feature in self.selector2_features:
                count[idx] = count[idx] + 1
            if feature in self.selector3_features:
                count[idx] = count[idx] + 1
            if feature in self.selector4_features:
                count[idx] = count[idx] + 1
            if feature in self.selector5_features:
                count[idx] = count[idx] + 1
            if feature in self.selector6_features:
                count[idx] = count[idx] + 1
            if feature in self.selector7_features:
                count[idx] = count[idx] + 1

        prob = [val / 7.0 for val in count]
        self.selected_features_ = [
            val for val, p in zip(self.updat_feature_names_sub_, prob) if p > 0.5
        ]
        prob_sel_features = [val for val in prob if val > 0.5]

        self.selector2_features = np.trim_zeros(self.selector2_features)
        self.selector3_features = np.trim_zeros(self.selector3_features)
        self.selector5_features = np.trim_zeros(self.selector5_features)
        self.selector7_features = np.trim_zeros(self.selector7_features)

        self.all_selected_features_ = [
            self.selector0_features,
            self.selector1_features,
            self.selector2_features,
            self.selector3_features,
            self.selector4_features,
            self.selector5_features,
            self.selector6_features,
            self.selector7_features,
            np.array(self.selected_features_, dtype=object),
        ]
