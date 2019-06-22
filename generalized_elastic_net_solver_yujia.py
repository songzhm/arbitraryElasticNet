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
    def gmulup_solve(self, Amat, lvec, bvec, dvec, v0, err_tol=1e-8, text='Off', text_fr=200):
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
        while (((old_v - v) ** 2).sum() > err_tol):
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

    def solve(self, Xmat, Yvec, lam_1, lam_2, lowbo, upbo, wvec, Sigma, err_tol=1e-8, text='Off', text_fr=200):
        """Xmat, Yvec, lowbo, upbo,  wvec, Sigma,: numpy array;
           lam_1, lam_2: float; """
        p = Xmat.shape[1]
        Amat = Xmat.transpose().dot(Xmat) + lam_2 * Sigma
        bvec = 2 * Amat.dot(lowbo) - 2 * Xmat.transpose().dot(Yvec)
        Amat = 2 * Amat
        dvec = lam_1 * wvec
        v0 = np.maximum(0, -lowbo)
        lvec = upbo - lowbo
        v = self.gmulup_solve(Amat, lvec, bvec, dvec, v0, err_tol, text, text_fr)

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

    def __init__(self, beta,lam_1=0.0, lam_2=0.0, lowbo=None, upbo=None, ds=None, wvec=None, random_state=None,
                 sigma_choice=0, w_choice=0,
                 err_tol=1e-8, verbose=False, text_fr=200):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

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
        comb = max((p - len(comb)), 0) * [0] + list(comb)
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

        if self.ds is None:
            # self.ds=np.ones(p, dtype=np.float)
            ds = GeneralizedElasticNetRegressor.decimal2combination(self.sigma_choice, p, 2)
            ortho_mat = special_ortho_group.rvs(dim=p, random_state=self.random_state)
            self.sigma_mat = ortho_mat @ np.diag(ds) @ ortho_mat.T
        else:
            assert (self.ds.shape[0] == p), 'Please make sure the dimension of the dataset matches!'
            ortho_mat = special_ortho_group.rvs(dim=p, random_state=self.random_state)
            self.sigma_mat = ortho_mat @ np.diag(self.ds) @ ortho_mat.T

        if self.wvec is None:
            # self.wvec=np.ones(p, dtype=np.float)/p
            self.wvec = GeneralizedElasticNetRegressor.decimal2combination(self.w_choice, p, 2)
            if sum(self.wvec) != 0:
                self.wvec = self.wvec / sum(self.wvec)
                #         else:
                #             wvec = self.wvec

        if self.lowbo is None:
            self.lowbo = np.repeat(0.000001, p)

        if self.upbo is None:
            self.upbo = np.repeat(float('inf'), p)

        self.coef_ = solver.solve(X, y, self.lam_1, self.lam_2, self.lowbo, self.upbo, self.wvec, self.sigma_mat,
                                  self.err_tol, self.verbose, self.text_fr)

        return self

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")
        return np.dot(X, self.coef_)

    def predict(self, X, y=None):

        return self._decision_function(X)

    #     def score(self, X, y=None, sample_weight=None):

    #         return mean_squared_error(y, self.predict(X), sample_weight=sample_weight)

    def score(self, X, y=None, sample_weight=None):
        return mean_squared_error(X.dot(self.beta), self.predict(X), sample_weight=sample_weight)