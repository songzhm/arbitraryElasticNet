# -*- coding: utf-8 -*-
"""
Created by: Yujia Ding
Created on: 2019-05-11 14:59
"""


import numpy as np
from math import sqrt
import copy

class GeneralizedElasticNetSover(object):

    def gmulup_solve(self, Amat, lvec, bvec, dvec, v0, err_tol=1e-8, verbose=False, text_fr=200):
        A_plus = copy.deepcopy(Amat)
        A_plus[A_plus < 0] = 0

        A_minus = copy.deepcopy(Amat)
        A_minus[A_minus > 0] = 0
        A_minus = abs(A_minus)

        v = np.array([1.0 for x in range(len(bvec))])

        old_v = np.array([0 for x in range(len(bvec))])
        v0 = v0.astype(float)
        v0[np.where(v0 == 0)] = 0.00000001
        updateFactor0 = v0 / v
        updateFactor = np.array([1.0 for x in range(len(bvec))])
        count = 0
        while (((old_v - v) ** 2).sum() > err_tol):
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
            if (count % text_fr == 0) & verbose:
                print(((old_v - v) ** 2).sum())
            count += 1
            updateFactor0 = v0 / v
        return v

    def solve(self, Xmat, Yvec, lam_1, lam_2, lowbo, upbo, wvec, Sigma, err_tol=1e-8, verbose=False, text_fr=200):
        """
        Args:
            Xmat: numpy array
            Yvec: numpy array
            lam_1: float
            lam_2: float
            lowbo: numpy array
            upbo: numpy array
            wvec: numpy array
            Sigma: numpy array
            err_tol: float
            verbose: boolean
            text_fr: int

        Returns:
            beta: numpy array

        Examples:
            To solve prob: Yvec=Xmat*beta

            >>> Xmat = np.random.randn(N,K)
            >>> Yvec = np.random.randn(N)
            >>> lam_1 = 0.0034
            >>> lam_2 = 0
            >>> Sigma = np.diag([1]*K)
            >>> wvec = np.ones(K)
            >>> lowbo = -1*np.ones(N)
            >>> upbo = 1*np.ones(N)
            >>> solver = GeneralizedElasticNetSover()
            >>> betas = solver.solve(Xmat, Yvec, lam_1, lam_2, lowbo, upbo, wvec, Sigma)

        """
        p = Xmat.shape[1]
        Amat = Xmat.transpose().dot(Xmat) + lam_2 * Sigma
        bvec = 2 * Amat.dot(lowbo) - 2 * Xmat.transpose().dot(Yvec)
        Amat = 2 * Amat
        dvec = lam_1 * wvec
        v0 = np.maximum(0, -lowbo)
        lvec = upbo - lowbo
        v = self.gmulup_solve(Amat, lvec, bvec, dvec, v0, err_tol, verbose, text_fr)

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
