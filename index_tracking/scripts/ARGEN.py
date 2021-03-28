import numpy as np
from math import sqrt
import copy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from scipy.stats import special_ortho_group
import inspect
from sklearn.utils.validation import check_is_fitted
from baseconvert import base
from generalized_elastic_net import GeneralizedElasticNetSolver


class FeatureSelectionRegressor(BaseEstimator, RegressorMixin, GeneralizedElasticNetSolver):
    """
    wrapper class based on GeneralizedElasticNetSolver to conduct feature selection through bisection search
    """

    def __init__(self, p, target_number, err_tol=1e-8,
                 verbose='Off', text_fr=200):
        self.lam_1 = None
        self.lam_2 = 0
        self.lowbo = np.zeros(p)
        self.upbo = np.ones(p) * np.inf
        # temp_vec = np.ones(p)
        # zero_index = np.random.RandomState(wvec_random_state).choice(range(p), target_number, False)
        # temp_vec[zero_index] = 0
        # self.wvec = temp_vec / sum(temp_vec)
        self.wvec = np.ones(p) / p
        self.sigma_mat = np.eye(p)
        self.err_tol = err_tol
        self.verbose = verbose
        self.text_fr = text_fr
        self.coef_ = None
        self.target_number = target_number

    def fit(self, X, y=None):
        n, p = X.shape
        zero_threshold = 1e-8
        max_iter = 1000

        lam_1_low = 0
        lam_1_up = 100
        lam_1 = (lam_1_low + lam_1_up) / 2

        iteration = 0

        coef = self.solve(X, y, lam_1, self.lam_2, self.lowbo, self.upbo, self.wvec, self.sigma_mat,
                          self.err_tol, self.verbose, self.text_fr)
        iteration += 1
        non_zero_coef = np.sum(coef > zero_threshold)
        # print(iteration, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        while iteration < max_iter:

            if non_zero_coef > self.target_number:
                lam_1_low = lam_1
            elif non_zero_coef < self.target_number:
                lam_1_up = lam_1
            else:
                break

            lam_1 = (lam_1_low + lam_1_up) / 2
            coef = self.solve(X, y, lam_1, self.lam_2, self.lowbo, self.upbo, self.wvec, self.sigma_mat,
                              self.err_tol, self.verbose, self.text_fr)
            non_zero_coef = np.sum(coef > zero_threshold)
            iteration += 1
            # print(iteration, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)
            coef[coef <= zero_threshold] = 0
        self.lam_1 = lam_1
        self.coef_ = coef

        return self

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")
        return np.dot(X, self.coef_)

    def predict(self, X, y=None):
        return self._decision_function(X)

    def score(self, X, y=None, sample_weight=None):
        return mean_squared_error(y, self.predict(X), sample_weight=sample_weight)


class ARGEN(BaseEstimator, RegressorMixin, GeneralizedElasticNetSolver):
    """
    ARGEN Solver class based on GeneralizedElasticNetSolver
    """

    def __init__(self, p, lam_1, lam_2, lowbo, upbo, wvec_random_state, sigma_random_state, err_tol=1e-8,
                 verbose='Off', text_fr=200):
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lowbo = lowbo
        self.upbo = upbo
        self.p = p
        self.wvec_random_state = wvec_random_state
        self.sigma_random_state = sigma_random_state
        self.wvec = None
        self.sigma_mat = None
        self.err_tol = err_tol
        self.verbose = verbose
        self.text_fr = text_fr
        self.coef_ = None

    def fit(self, X, y=None):
        zero_threshold = 1e-8
        if self.wvec_random_state != 0:
            self.wvec = np.random.RandomState(self.wvec_random_state).choice(range(3), self.p)
            self.wvec = self.wvec / np.sum(self.wvec)
        else:
            self.wvec = np.ones(self.p) / self.p
        if self.sigma_random_state != 0:
            sig_ds = np.random.RandomState(self.sigma_random_state).choice(range(3), self.p)
            ortho_mat = special_ortho_group.rvs(dim=self.p, random_state=self.sigma_random_state)
            self.sigma_mat = ortho_mat @ np.diag(sig_ds) @ ortho_mat.T
        else:
            self.sigma_mat = np.eye(self.p)
        coef = self.solve(X, y, self.lam_1, self.lam_2, self.lowbo, self.upbo, self.wvec, self.sigma_mat,
                          self.err_tol, self.verbose, self.text_fr)
        coef[coef <= zero_threshold] = 0
        self.coef_ = coef

        return self

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")
        return np.dot(X, self.coef_)

    def predict(self, X, y=None):
        return self._decision_function(X)

    def score(self, X, y=None, sample_weight=None):
        return mean_squared_error(y, self.predict(X), sample_weight=sample_weight)
