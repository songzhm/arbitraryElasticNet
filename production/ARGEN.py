import numpy as np
from math import sqrt
import copy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from scipy.stats import special_ortho_group
import inspect
from sklearn.utils.validation import check_is_fitted
from baseconvert import base


class GeneralizedElasticNetSover(object):

    def gmulup_solve(self, Amat, lvec, bvec, dvec, v0, err_tol=1e-8, text='Off', text_fr=200, max_iter=1000):
        A_plus = copy.deepcopy(Amat)
        A_plus[A_plus < 0] = 0

        A_minus = copy.deepcopy(Amat)
        A_minus[A_minus > 0] = 0
        A_minus = abs(A_minus)

        v = np.array([1.0 for x in range(len(bvec))])

        old_v = np.array([0 for x in range(len(bvec))])
        v0 = v0.astype(float)
        v0[np.where(v0 == 0)] = 0.00000001

        updateFactor = np.array([1.0 for x in range(len(bvec))])
        count = 0
        while ((old_v - v) ** 2).sum() > err_tol and count < max_iter:
            v[np.where(v == 0)] = 0.00000001
            updateFactor0 = v0 / v
            updateFactor0[np.where(updateFactor0 == 0)] = 0.00000001
            dFa = np.array(A_plus.dot(v))
            dFb = copy.deepcopy(bvec)
            dFc = np.array(A_minus.dot(v))
            for i in range(len(bvec)):
                if dFa[i] == 0:
                    dFa[i] = 0.00000001
                if dFa[i] * updateFactor0[i] + (dFb[i] - dvec[i]) - dFc[i] / updateFactor0[i] > 0:
                    updateFactor[i] = float(
                        (-(dFb[i] - dvec[i]) + sqrt((dFb[i] - dvec[i]) ** 2 + 4 * dFa[i] * dFc[i])) / (2 * dFa[i]))
                elif dFa[i] * updateFactor0[i] + (dFb[i] + dvec[i]) - dFc[i] / updateFactor0[i] < 0:
                    updateFactor[i] = float(
                        (-(dFb[i] + dvec[i]) + sqrt((dFb[i] + dvec[i]) ** 2 + 4 * dFa[i] * dFc[i])) / (2 * dFa[i]))
                else:
                    updateFactor[i] = updateFactor0[i]
            if np.count_nonzero(~np.isnan(updateFactor)) == len(bvec):
                old_v = copy.deepcopy(v)
                v = np.minimum(lvec, updateFactor * v)
            else:
                break
            if (count % text_fr == 0) & (text == 'On'):
                print(((old_v - v) ** 2).sum())
            count += 1

        return v

    def solve(self, Xmat, Yvec, lam_1, lam_2, lowbo, upbo, wvec, Sigma, err_tol=1e-8, text='Off', text_fr=200,
              max_iter=1000):
        """Xmat, Yvec, lowbo, upbo,  wvec, Sigma,: numpy array;
           lam_1, lam_2: float; """
        p = Xmat.shape[1]
        Amat = Xmat.transpose().dot(Xmat) + lam_2 * Sigma
        bvec = 2 * Amat.dot(lowbo) - 2 * Xmat.transpose().dot(Yvec)
        Amat = 2 * Amat
        dvec = lam_1 * wvec
        v0 = np.maximum(0, -lowbo)
        lvec = upbo - lowbo
        v = self.gmulup_solve(Amat, lvec, bvec, dvec, v0, err_tol, text, text_fr, max_iter)

        beta = v + lowbo
        return beta  # , plo

        # To solve prob: Yvec=Xmat*beta

        # Example of input
        # Xmat = np.random.randn(N,K)
        # Yvec = np.random.randn(N)
        # lam_1 = 0.0034
        # lam_2 = 0
        # Sigma = np.diag([1]*K)
        # wvec = np.ones(K)
        # lowbo = -1*np.ones(N)
        # upbo = 1*np.ones(N)


class FeatureSelectionRegressor(BaseEstimator, RegressorMixin, GeneralizedElasticNetSover):
    """

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


class ARGEN(BaseEstimator, RegressorMixin, GeneralizedElasticNetSover):
    """

    """

    def __init__(self, p, lam_1, lam_2, lowbo, upbo, wvec_random_state, sigma_random_state, err_tol=1e-8,
                 verbose='Off', text_fr=200):
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lowbo = lowbo
        self.upbo = upbo
        self.p = p
        # temp_vec = np.ones(p)
        # zero_index = np.random.RandomState(wvec_random_state).choice(range(p), target_number, False)
        # temp_vec[zero_index] = 0
        # self.wvec = temp_vec / sum(temp_vec)
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
