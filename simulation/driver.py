import os
import pickle

import numpy as np
import scipy.linalg
from generalized_elastic_net import GeneralizedElasticNetRegressor
from sklearn.model_selection import train_test_split
from skopt.space import Integer


class Simulator(object):
    """
    Date set simulation
    """

    def _set_split(self, X, y, Ntrain, Ntval, Ntest, random_state=42):
        N = Ntrain + Ntval + Ntest
        test_size = Ntest / N
        val_size = Ntval / (N - Ntest)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size,
                                                                    random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                          random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def example2_simulator(self):
        state = 50
        beta = np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])
        p = len(beta)
        sigma = 3
        Ntrain = 20
        Ntval = 20
        Ntest = 200
        N = Ntrain + Ntval + Ntest
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = 0.5 ** (abs(i - j))
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': None,
                     'upbo': None}
        for i in range(state):
            eps = np.random.randn(N)
            X = (np.linalg.cholesky(cov)).dot(np.random.randn(p, N))
            X = X.T
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example1_simulator(self):
        state = 50
        beta = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
        p = len(beta)
        sigma = 3
        Ntrain = 20
        Ntval = 20
        Ntest = 200
        N = Ntrain + Ntval + Ntest
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = 0.5 ** (abs(i - j))
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': None,
                     'upbo': None}
        for i in range(state):
            eps = np.random.randn(N)
            X = (np.linalg.cholesky(cov)).dot(np.random.randn(p, N))
            X = X.T
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example3_simulator(self):
        state = 50
        beta = np.array([0] * 10 + [2] * 10 + [0] * 10 + [2] * 10)
        p = len(beta)
        sigma = 15
        Ntrain = 100
        Ntval = 100
        Ntest = 400
        N = Ntrain + Ntval + Ntest
        cov = 0.5 * np.ones((p, p))
        np.fill_diagonal(cov, 1)
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': None,
                     'upbo': None}
        for i in range(state):
            eps = np.random.randn(N)
            X = (np.linalg.cholesky(cov)).dot(np.random.randn(p, N))
            X = X.T
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example4_simulator(self):
        state = 50
        beta = np.array([-3] * 6 + [0] * 9)
        p = len(beta)
        sigma = 15
        Ntrain = 40
        Ntval = 40
        Ntest = 100
        N = Ntrain + Ntval + Ntest
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': None,
                     'upbo': None}
        for i in range(state):
            eps = np.random.randn(N)
            eps_x = np.random.normal(0, 0.01, (N, 6))
            Z_1 = np.random.randn(N, 1)
            Z_2 = np.random.randn(N, 1)
            Z_3 = np.random.randn(N, 1)
            Z = np.hstack([np.repeat(Z_1, 2, axis=1), np.repeat(Z_2, 2, axis=1), np.repeat(Z_3, 2, axis=1)])
            X = Z + eps_x
            X = np.hstack([X, np.random.randn(N, 9)])
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example5_simulator(self):
        state = 50
        beta = np.array([-3, -1.5, 0, 0, 2, 0, 0, 0])
        p = len(beta)
        sigma = 3
        Ntrain = 20
        Ntval = 20
        Ntest = 200
        N = Ntrain + Ntval + Ntest
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = 0.5 ** (abs(i - j))
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': np.array([-1000] * p),
                     'upbo': None}
        for i in range(state):
            eps = np.random.randn(N)
            X = (np.linalg.cholesky(cov)).dot(np.random.randn(p, N))
            X = X.T
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example6_simulator(self):
        state = 50
        beta = np.random.randint(-5, 5, 8)
        p = len(beta)
        sigma = 3
        Ntrain = 20
        Ntval = 20
        Ntest = 200
        N = Ntrain + Ntval + Ntest
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = 0.5 ** (abs(i - j))
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': np.array([-5] * p),
                     'upbo': np.array([5] * p)}
        for i in range(state):
            eps = np.random.randn(N)
            X = (np.linalg.cholesky(cov)).dot(np.random.randn(p, N))
            X = X.T
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example7_simulator(self):
        state = 50
        beta = np.array([-6, -8, 0, 0, 7, 0, 0, 0])
        p = len(beta)
        sigma = 3
        Ntrain = 20
        Ntval = 20
        Ntest = 200
        N = Ntrain + Ntval + Ntest
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = 0.5 ** (abs(i - j))
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': np.array([-5] * p),
                     'upbo': np.array([5] * p)}
        for i in range(state):
            eps = np.random.randn(N)
            X = (np.linalg.cholesky(cov)).dot(np.random.randn(p, N))
            X = X.T
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def example8_simulator(self):
        state = 50
        beta = np.array([-3] * 6 + [0] * 9)
        p = len(beta)
        sigma = 15
        Ntrain = 5
        Ntval = 5
        Ntest = 50
        N = Ntrain + Ntval + Ntest
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': np.array([-1000] * p),
                     'upbo': None}
        for i in range(state):
            eps = np.random.randn(N)
            eps_x = np.random.normal(0, 0.01, (N, 6))
            Z_1 = np.random.randn(N, 1)
            Z_2 = np.random.randn(N, 1)
            Z_3 = np.random.randn(N, 1)
            Z = np.hstack([np.repeat(Z_1, 2, axis=1), np.repeat(Z_2, 2, axis=1), np.repeat(Z_3, 2, axis=1)])
            X = Z + eps_x
            X = np.hstack([X, np.random.randn(N, 9)])
            y = X.dot(beta) + sigma * eps
            data = self._set_split(X, y, Ntrain, Ntval, Ntest, random_state=42)
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def signal_simulator(self):
        state = 1
        # signal length
        N = 4096
        # number of spikes in the signal
        T = 160
        # number of observations to make
        K = 1024
        x = np.zeros((N, 1))
        q = np.random.permutation(N)
        x[q[0:T]] = np.sign(np.random.randn(T, 1))
        beta = x
        p = len(beta)
        data_dict = {'state': state,
                     'beta': beta,
                     'lowbo': np.array([-1] * p),
                     'upbo': np.array([1] * p)}
        for i in range(state):
            # measurement matrix
            A = np.random.randn(K, N)
            A = scipy.linalg.orth(A.transpose()).transpose()
            X = A
            # observations
            y = np.reshape(A.dot(x), (A.dot(x).shape[0],)) + np.random.normal(0, 0.1, K)
            data = [X, X, X, y, y, y]
            data_dict[i] = {}
            add_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
            for j, key in enumerate(add_keys):
                data_dict[i][key] = data[j]
        return data_dict

    def data_simulator(self, example):
        """example = '1', '2', '3', 'Signal', 'Abtrary_Signal'"""
        data_dict_exists = os.path.isfile('./data_dict_eg_' + example + '.pkl')

        if data_dict_exists:
            data_dict = pickle.load(open('data_dict_eg_' + example + '.pkl', "rb"))

        else:
            example_dict = {'1': self.example1_simulator,
                            '2': self.example2_simulator,
                            '3': self.example3_simulator,
                            '4': self.example4_simulator,
                            '5': self.example5_simulator,
                            '6': self.example6_simulator,
                            '7': self.example7_simulator,
                            '8': self.example8_simulator,
                            'Signal': self.signal_simulator,
                            }
            data_dict = example_dict[example]()
            # save
            try:
                pickle.dump(data_dict, open('data_dict_eg_' + example + '.pkl', "wb"))
            except ValueError:
                print('No date_dict')

        return data_dict


class Driver(object):
    def __init__(self, example, method):
        self.example = example
        self.method = method

    def space_generator(self, up_sigma_choice, up_w_choice, p):

        fix_paras = {}
        tune_paras = {}
        space = []
        if self.method in ['NL', 'ARL']:
            space = [Integer(0, 100, name='lam_1'), ]
            fix_paras['lam_2'] = 0
            fix_paras['wvec'] = np.ones(p)
            fix_paras['sigma_ds'] = [1] * p
            tune_paras['lam_1'] = []

        if self.method in ['NR', 'ARR']:
            space = [Integer(0, 100, name='lam_2'), ]
            fix_paras['lam_1'] = 0
            fix_paras['wvec'] = np.ones(p)
            fix_paras['sigma_ds'] = [1] * p
            tune_paras['lam_2'] = []

        if self.method in ['NEN', 'AREN']:
            space = [Integer(0, 100, name='lam_1'),
                     Integer(0, 100, name='lam_2'),
                     ]
            fix_paras['wvec'] = np.ones(p)
            fix_paras['sigma_ds'] = [1] * p
            tune_paras['lam_1'] = []
            tune_paras['lam_2'] = []

        if self.method in ['NGL', 'ARGL']:
            space = [Integer(0, 100, name='lam_1'),
                     Integer(0, up_w_choice, name='w_choice'),
                     ]
            fix_paras['lam_2'] = 0
            fix_paras['sigma_ds'] = [1] * p
            tune_paras['lam_1'] = []
            tune_paras['w_choice'] = []

        if self.method in ['NGR', 'ARGR']:
            space = [Integer(0, 100, name='lam_2'),
                     Integer(0, up_sigma_choice, name='sigma_choice'),
                     ]
            fix_paras['lam_1'] = 0
            fix_paras['wvec'] = np.ones(p)
            tune_paras['lam_2'] = []
            tune_paras['sigma_choice'] = []

        if self.method in ['NLEN', 'ARLEN']:
            space = [
                Integer(0, 100, name='lam_1'),
                Integer(0, 100, name='lam_2'),
                Integer(0, up_w_choice, name='w_choice'),
            ]
            fix_paras['sigma_ds'] = [1] * p
            tune_paras['lam_1'] = []
            tune_paras['lam_2'] = []
            tune_paras['w_choice'] = []
        if self.method in ['NREN', 'ARREN']:
            space = [
                Integer(0, 100, name='lam_1'),
                Integer(0, 100, name='lam_2'),
                Integer(0, up_sigma_choice, name='sigma_choice'),
            ]
            fix_paras['wvec'] = np.ones(p)
            tune_paras['lam_1'] = []
            tune_paras['lam_2'] = []
            tune_paras['sigma_choice'] = []
        if self.method in ['NGEN', 'ARGEN']:
            space = [
                Integer(0, 100, name='lam_1'),
                Integer(0, 100, name='lam_2'),
                Integer(0, up_sigma_choice, name='sigma_choice'),
                Integer(0, up_w_choice, name='w_choice'),
            ]
            tune_paras['lam_1'] = []
            tune_paras['lam_2'] = []
            tune_paras['sigma_choice'] = []
            tune_paras['w_choice'] = []

        return space, fix_paras, tune_paras

    def driver(self):
        sim = Simulator()
        data_dict = sim.data_simulator(self.example)
        self.state = data_dict['state']
        beta = data_dict['beta']
        lowbo = data_dict['lowbo']
        upbo = data_dict['upbo']

        reg = GeneralizedElasticNetRegressor(beta, sigma_choice_base=2,
                                             w_choice_base=2, lowbo=lowbo, upbo=upbo)
        p = reg.p
        up_sigma_choice = min(np.iinfo(np.int64).max, reg.sigma_choice_base ** p - 1)
        up_w_choice = min(np.iinfo(np.int64).max, reg.w_choice_base ** p - 1)
        self.space, self.fix_paras, self.tune_paras = self.space_generator(up_sigma_choice, up_w_choice, p)
        self.reg = reg
        self.data_dict = data_dict
