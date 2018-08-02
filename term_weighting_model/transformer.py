# coding: utf-8
# created by deng on 8/2/2018

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict
from sklearn.feature_extraction.text import _document_frequency

import numpy as np
import scipy.sparse as sp


class TfdcTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf-dc representation
    Tf means term-frequency while tf-dc means distributional-concentration
    times term-frequency.

    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_dc : boolean, default=True
        Enable dc reweighting.
    smooth_dc : boolean, default=True
        Smooth dc weights by adding one to f(t,Ci) and f(t). Prevents
        zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    balanced: use bdc instead of dc

    References
    ----------
    .. [2015 ICTAI] `Tao Wang, Yi Cai. Entropy-based TermWeighting Schemes for
                     Text Categorization in VSM`
    """

    def __init__(self, norm='l2', use_dc=True, smooth_dc=True,
                 sublinear_tf=False, balanced=False):
        self.norm = norm
        self.use_dc = use_dc
        self.smooth_dc = smooth_dc
        self.sublinear_tf = sublinear_tf
        self.balanced = balanced

    def fit(self, X, y):
        """Learn the idf vector (global term weights)
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        y : labels of each samples, [n_samples,]
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        y = np.asarray(y)  # pd.Series don't support csr slicing

        if self.use_dc:

            n_samples, n_features = X.shape

            # calculate term frequencies, f(t)
            tfs = np.asarray(X.sum(axis=0)).ravel()
            tps = np.zeros(n_features)  # sum of p(t, Ci) for bdc

            # initialize dc
            dc = np.ones(n_features)

            # all labels
            labels = set(y)
            print(X.shape, y.shape)
            for label in labels:
                label_X = X[y == label]
                label_tfs = np.asarray(label_X.sum(axis=0)).ravel()  # f(t, Ci)

                # calculate sum of p(t, Ci) for bdc
                if self.balanced:
                    tps = tps + label_tfs / label_tfs.sum()  # p(t, Ci) for bdc

                label_h = label_tfs / tfs  # f(t, Ci) / f(t)
                dc += label_h * np.log(label_h) / np.log(len(labels))

            # calculate balanced dc
            if self.balanced:

                # initialize bdc
                dc = np.ones(n_features)
                for label in labels:
                    label_X = X[y == label]  # sliced data of specific label
                    label_tfs = np.asarray(label_X.sum(axis=0)).ravel()  # f(t, Ci) for dc
                    label_tfs = label_tfs / label_tfs.sum()  # p(t, Ci) for bdc
                    label_h = label_tfs / tps
                    dc = dc + label_h * np.log(label_h) / np.log(len(labels))

            self._dc_diag = sp.spdiags(dc, diags=0, m=n_features, n=n_features, format='csr')


    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_dc:
            check_is_fitted(self, '_dc_diag', 'dc vector is not fitted')

            expected_n_features = self._dc_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._dc_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._dc_diag.sum(axis=0))


def main():
    pass


if __name__ == '__main__':
    main()
