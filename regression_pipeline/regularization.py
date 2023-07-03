import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV


class Regularizations:
    '''Runs regularized models

    This class takes care of various regularized models such as Ridge
    regression, Lasso regression, and ElasticNet.

    Parameters
    -----------
    cv: int, cross-validation generator or an iterable
        if integer then specifies the number of folds

    alphas : float
        a list of floats that defines the strength of the penalty on the
        model parameters

    njobs: int
        Number of jobs to be run on parallel

    random_state: int
        Seed value that controls the random number generation

    Methods
    --------
    perform_ridgereg:
        Performs a RidgeCV regression for a set of alpha values

    perform_lassocv:
        Performs a LassoCV regression for a set of alpha values

    perform_elasticnetcv:
        Performs ElasticNet regression
    '''

    def __init__(self, cv, alphas, njobs, random_state):
        self.cv = cv
        self.alphas = alphas
        self.njobs = njobs
        self.random_state = random_state

    def perform_ridgereg(self, xtrain, ytrain, xtest, ytest, scoring):
        '''Performs Ridge regression with varying alpha values

        Parameters
        ------------
        xtrain : array of shape (n_samples, n_features)
            Training data

        ytrain : array of shape (n_samples, )
            Training target values

        xtest : array if shape (m_samples, n_features)
            Test data

        ytest : an array of shape (m_samples, )
            Test target values

        scoring : a string
            metric for evaluation

        Returns
        --------
        r2score_train : float
            R2 score of the training data

        r2score_test : float
            R2 score of the test data

        estimated_alpha : float
            The optimized regularization parameter

        ridge_coefficients : array of shape (n_features, )
            The optimized model parameters

        '''

        ridgereg = RidgeCV(alphas = self.alphas, scoring = scoring,
                          store_cv_values=True)
        ridgereg.fit(xtrain, ytrain)

        r2score_train = ridgereg.score(xtrain, ytrain)
        r2score_test = ridgereg.score(xtest, ytest)
        estimated_alpha = ridgereg.alpha_
        best_score = ridgereg.best_score_
        ridge_coefficients = ridgereg.coef_

        mse_ridge = np.mean(ridgereg.cv_values_, axis = 0)

        plt.plot(self.alphas, mse_ridge)
        plt.xlabel("Alpha values")
        plt.ylabel("MSE")
        plt.savefig("./ridge_alphas_mse.png")

        return r2score_train, r2score_test, estimated_alpha, ridge_coefficients

    def perform_lassocv(self, xtrain, ytrain, xtest, ytest):
        '''Performs a lasso regression for a range of regularization
        parameters.

        Parameters
        --------------
        xtrain : an array of shape (n_samples, n_features)
            Training data

        ytrain : an array of shape (n_samples, )
            Training target values

        xtest :  an array of shape (k_samples, n_features)
            Test data

        ytest : an array of shape (k_samples, )
            Test target values

        Returns
        --------
        r2score_train : float
            R2 score of the training data

        r2score_test : float
            R2 score of the test data

        estimated_alpha : float
            Optimized regularization parameter

        lasso_coefficients : an array of shape (n_features, )
            Optimized model parameters

        '''

        eps = self.alphas[0]/self.alphas[-1]
        lassocv = LassoCV(eps = eps, alphas = self.alphas, cv = self.cv,
                         n_jobs = self.njobs, random_state = self.random_state,
                         n_alphas = len(self.alphas))
        lassocv.fit(xtrain, ytrain)

        r2score_train = lassocv.score(xtrain, ytrain)
        r2score_test = lassocv.score(xtest, ytest)
        estimated_alpha = lassocv.alpha_
        lasso_coefficients = lassocv.coef_
        mse = np.mean(lassocv.mse_path_, axis = 1)
        
        plt.plot(self.alphas, mse)
        plt.xlabel("Alpha values")
        plt.ylabel("MSE")
        plt.savefig("./lasso_alphas_mse.png")

        return r2score_train, r2score_test, estimated_alpha, lasso_coefficients

    def perform_elasticnetcv(self, xtrain, ytrain, xtest, ytest, l1_ratio_list):
        '''Performs ElasticNet regression.

        Parameters
        -----------
        xtrain : an array of shape (n_samples, n_features)
            Training data

        ytrain : an array of shape (n_samples, )
            Training target values

        xtest :  an array of shape (k_samples, n_features)
            Test data

        ytest : an array of shape (k_samples, )
            Test target values

        l1_ratio_list : float or a list of float
            Float between 0 and 1 passed to ElasticNet
            (scaling between l1 and l2 penalties). For
            l1_ratio = 0 the penalty is an L2 penalty. For
            l1_ratio = 1 it is an L1 penalty. For
            0 < l1_ratio < 1, the penalty is a combination of L1 and L2

        Returns
        --------
        r2score_train : float
            R2 score of the training data

        r2score_test : float
            R2 score of the test data

        estimated_alpha : float
            Optimized regularization parameter

        elasticnetcv_coefficients : an array of shape (n_features, )
            Optimized model parameters

        '''

        eps = self.alphas[0]/self.alphas[-1]
        elasticnetcv = ElasticNetCV(l1_ratio = l1_ratio_list, eps = eps, alphas
                                   = self.alphas, cv = self.cv, n_jobs = self.njobs,
                                    random_state = self.random_state)
        elasticnetcv.fit(xtrain, ytrain)

        estimated_alpha = elasticnetcv.alpha_
        r2score_train = elasticnetcv.score(xtrain, ytrain)
        r2score_test = elasticnetcv.score(xtest, ytest)
        elasticnetcv_coefficients = elasticnetcv.coef_
        mse = np.mean(elasticnetcv.mse_path_, axis=0)

        return r2score_train, r2score_test, estimated_alpha, elasticnetcv_coefficients
