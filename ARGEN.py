

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

    def gmulup_solve(self, Amat, lvec, bvec, dvec, v0, err_tol=1e-8, text='Off', text_fr=200, max_iter=2000):
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

    def solve(self, Xmat, Yvec, lam_1, lam_2, lowbo, upbo, wvec, Sigma, err_tol=1e-8, text='Off', text_fr=200, max_iter=2000):
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


class GeneralizedElasticNetRegressor(BaseEstimator, RegressorMixin):
    """

    """

    def __init__(self, beta, lam_1=0.0, lam_2=0.0, lowbo=None, upbo=None, ds=None, sigma_ds=None, wvec=None,
                 random_state=None,
                 sigma_choice=0, sigma_choice_base=None, sigma_choice_up=10 ** 5,
                 w_choice=0, w_choice_base=None, w_choice_up=10 ** 5,
                 err_tol=1e-8, verbose=False, text_fr=200, tune_lam_1=False, target_number=None, sigma_mat=None):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        self.p = len(beta)

        for arg, val in values.items():
            setattr(self, arg, val)

        if self.sigma_choice_base is None:
            base = 2
            while base ** self.p <= self.sigma_choice_up:
                base += 1
            self.sigma_choice_base = base - 1

        if self.w_choice_base is None:
            base = 2
            while base ** self.p <= self.w_choice_up:
                base += 1
            self.w_choice_base = base - 1

    def _generate_combination(self, p, k, choices=[0, 1]):
        assert (k < 2 ** p), 'k cannot be bigger than 2**p'
        res = np.empty(p)
        for i in range(p):
            den = 2 ** (i + 1)
            if k >= den:
                num = k - den
            else:
                num = k
            temp = np.floor(num / 2 ** i)
            res[i] = choices[int(np.mod(temp, 2))]
        return res

    @staticmethod
    def decimal2combination(decimal, p, bases):
        # decimal<bases**p
        if decimal >= bases ** p:
            raise ValueError('decimal should less than bases**p')
        comb = base(int(decimal), 10, bases)
        comb = max((p - len(comb)), 0) * [0] + [item for item in comb]
        return np.array(comb).astype('float')

    @staticmethod
    def combination2decimal(comb, bases):
        dec = base("".join(comb.astype(str)), bases, 10, string=True)
        return int(dec)

    def fit(self, X, y=None):
        n, p = X.shape

        solver = GeneralizedElasticNetSover()

        if self.random_state is None:
            self.random_state = 10

        if self.sigma_mat is None:
            if self.sigma_ds is None:
                if self.ds is None:
                    # self.ds=np.ones(p, dtype=np.float)
                    # print(self.sigma_choice, p, self.sigma_choice_base)
                    ds = GeneralizedElasticNetRegressor.decimal2combination(self.sigma_choice, p, self.sigma_choice_base)
                    if sum(ds) != 0:
                        ds = ds / sum(ds)
                    ortho_mat = special_ortho_group.rvs(dim=p, random_state=self.random_state)
                    self.sigma_mat = ortho_mat @ np.diag(ds) @ ortho_mat.T
                else:
                    assert (self.ds.shape[0] == p), 'Please make sure the dimension of the dataset matches!'
                    ortho_mat = special_ortho_group.rvs(dim=p, random_state=self.random_state)
                    self.sigma_mat = ortho_mat @ np.diag(self.ds) @ ortho_mat.T
            else:
                self.sigma_mat = np.diag(self.sigma_ds)


        if self.wvec is None:
            # self.wvec=np.ones(p, dtype=np.float)/p
            self.newwvec = GeneralizedElasticNetRegressor.decimal2combination(self.w_choice, p, self.w_choice_base)
            if sum(self.newwvec) != 0:
                self.newwvec = self.newwvec / sum(self.newwvec)
        else:
            self.newwvec = self.wvec

        if self.lowbo is None:
            self.lowbo = np.repeat(0.000001, p)

        if self.upbo is None:
            self.upbo = np.repeat(float('inf'), p)

        zero_threshold = 1.01e-6

        if self.tune_lam_1:

            max_iter = 1000

            lam_1_low = 0
            lam_1_up = 100
            lam_1 = (lam_1_low + lam_1_up) / 2

            iteration = 0

            coef = solver.solve(X, y, lam_1, self.lam_2, self.lowbo, self.upbo, self.newwvec, self.sigma_mat,
                                self.err_tol, self.verbose, self.text_fr)
            iteration += 1
            non_zero_coef = np.sum(coef >= zero_threshold)
            # print('iter', ',', 'lam_1_low', ',', 'lam_1', ',', 'lam_1_up', ',', 'non_zero_coef')
            # print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

            while iteration < max_iter:

                if non_zero_coef > self.target_number:
                    lam_1_low = lam_1
                elif non_zero_coef < self.target_number:
                    lam_1_up = lam_1
                else:
                    break

                lam_1 = (lam_1_low + lam_1_up) / 2
                coef = solver.solve(X, y, lam_1, self.lam_2, self.lowbo, self.upbo, self.newwvec, self.sigma_mat,
                                    self.err_tol, self.verbose, self.text_fr)
                non_zero_coef = np.sum(coef > zero_threshold)
                iteration += 1
                # print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

            self.lam_1 = lam_1
            self.coef_ = coef

        else:
            self.coef_ = solver.solve(X, y, self.lam_1, self.lam_2, self.lowbo, self.upbo, self.newwvec, self.sigma_mat,
                                  self.err_tol, self.verbose, self.text_fr)

        return self

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")
        return np.dot(X, self.coef_)

    def predict(self, X, y=None):

        return self._decision_function(X)

    def score(self, X, y=None, sample_weight=None):
        return mean_squared_error(X.dot(self.beta), self.predict(X), sample_weight=sample_weight)