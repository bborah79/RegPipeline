import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer


class MathTransformations:
    '''Performs various mathemtaical transformations of the features.

    Methods
    -------
    logtransform(X):
        Performs log transformation.

    squaretransform(X):
        Performs square transformation.

    squareroottransform(X):
        Performs square root transformation.

    reciprocaltransform(X):
        Performs reciprocal transformation.

    boxcoxtransform(X):
        Performs pwer transformation.

    yeojohnsontransform(X):
        Performs power transformation.

    quantiletransform(X):
        Performs quantile transformation.

    '''

    def logtransform(self, X):
        '''Performs a logarithmic (natural log) transformation.

        Parameters
        ----------
        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        return np.log(X)

    def squaretransform(self, X):
        '''Performs a square transformation of the given array.

        Parameters
        ----------
        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        return np.square(X)

    def squareroottransform(self, X):
        '''Performs a square root transformation.

        Parameters
        ----------
        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        return np.sqrt(X)

    def reciprocaltransform(self, X):
        '''Performs a reciprocal transformation of the array provided.

        Parameters
        ----------
        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        return np.reciprocal(X)

    def boxcoxtransform(self, X):
        '''Performs a power transformation using box-cox method.

        Parameters
        ----------

        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        boxcox = PowerTransformer(method="box-cox")
        X_transformed = boxcox.fit_transform(X)
        return X_transformed

    def yeojohnsontransform(self, X):
        '''Performs power transformation using Yeo-Johnson method.

        Parameters
        ----------
        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        yeojohnson = PowerTransformer()
        X_transformed = yeojohnson.fit_transform(X)
        return X_transformed

    def quantiletransform(self, X):
        '''Performs a quantile transformation of the given array.

        Parameters
        ----------
        X : an array
            An array of shape (n_samples, m_features) or an array of shape
            (n_samples, ).

        '''

        quatiletrans = QuantileTransformer(output_distribution="normal")
        X_transformed = quatiletrans.fit_transform(X)
        return X_transformed
