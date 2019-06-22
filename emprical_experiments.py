# # -*- coding: utf-8 -*-
# """
# Created by: zhengmingsong
# Created on: 2019-05-11 16:13
# """
#
# # %%
# import numpy as np
#
# from generalized_elastic_net_solver import *
#
# solver = GeneralizedElasticNetSover()
#
# # %%
# lam_1 = 0.01
# lam_2 = 0.01
#
# # %%
# # generate dataset
# n = 1000
# p = 4
# np.random.seed(0)
# other_variables = np.random.rand(n, p - 1)
# b = np.array([1 / 6, 1 / 6, 1 / 2])
# last_variable = other_variables @ b + 5 / 6 * np.random.rand(n)
# last_variable = last_variable.reshape((n, 1))
# x = np.concatenate((last_variable, other_variables), axis=1)
#
# betas = np.array([0, 2, 3, 0])
#
# y = x @ betas + np.random.rand(n)
#
# # %%
# lam_1 = 1000
#
# lam_2 = 10
#
# Sigma = np.diag([1] * p)
#
# wvec = np.ones(p)
#
# lowbo = -1 * np.ones(p)
#
# upbo = 5 * np.ones(p)
#
# betas_est = solver.solve(x, y, lam_1, lam_2, lowbo, upbo, wvec, Sigma)
#
# print(betas_est)
#
# genreg = GeneralizedElasticNetRegressor(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, ds=np.diag(Sigma),
#                                         random_state=42)
# genreg.fit(x, y)
#
# # %%
#
# n = 100
# p = 10
# np.random.seed(0)
# x = np.random.rand(n, p)
#
# betas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0, 0])
#
# y = x @ betas + np.random.rand(n)
#
# Sigma = np.diag([1] * p)
#
# wvec = np.ones(p)
#
# lowbo = -1 * np.ones(p)
#
# upbo = 10 * np.ones(p)
#
# lam_1s = [i for i in range(2050)]
#
# lam_2s = [0, 1, 10, 100, 1000]
#
# for lam_2 in lam_2s:
#     betas_est_set = np.zeros((len(lam_1s), p))
#     for ind, lam_1 in enumerate(lam_1s):
#         betas_est = solver.solve(x, y, lam_1, lam_2, lowbo, upbo, wvec, Sigma)
#         print('lam_1={}, lam_2={}'.format(lam_1, lam_2))
#         betas_est_set[ind, :] = betas_est
#
#     print(betas_est_set)
#
#     import pandas as pd
#     import seaborn as sns;
#
#     sns.set()
#     sns.set_style("white")
#     import matplotlib.pyplot as plt
#
#     plot_data = pd.DataFrame(betas_est_set, columns=['beta_{}'.format(i) for i in range(10)])
#     plot_data['lam_1'] = lam_1s
#     ax = sns.lineplot(x='lam_1', y='value', hue='variable',
#                       data=pd.melt(plot_data, ['lam_1']))
#     ax.set_title('lam_2={}, n={}'.format(lam_2, n))
#     plt.show()
#     fig = ax.get_figure()
#     fig.savefig('lam_2_{}_n_{}.png'.format(lam_2, n))

# %%

# https://scikit-optimize.github.io

# try test data

import numpy as np

from generalized_elastic_net_solver import *

import pandas as pd

mydateparser = lambda x: pd.datetime.strptime(x, "%m/%d/%y")
sp500_all = pd.read_csv('sandp500/sp500_pct.csv', index_col='Date', parse_dates=['Date.1'], date_parser=mydateparser)

sp500_all.index = sp500_all['Date.1']

constituents_names = sp500_all.columns.tolist()
constituents_names = [x for x in constituents_names if x not in ['Date', 'Date.1', 'SP500']]
constituents = sp500_all[constituents_names]
sp500 = sp500_all['SP500']

X = constituents.values
y = sp500.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_, p = X_train.shape

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dump, load
from skopt import forest_minimize

HPO_PARAMS = {'n_calls': 10,
              'n_random_starts': 10,
              'base_estimator': 'ET',
              'acq_func': 'EI',
              'xi': 0.02,
              'kappa': 1.96,
              'n_points': 10000,
              "n_jobs": -1,
              "verbose": True
              }

reg = GeneralizedElasticNetRegressor()
up = min(np.iinfo(np.int64).max, 2 ** p)
# define search space and parameters
space = [
    Real(10 ** -5, 10, 'log_uniform', name='lam_1'),
    Real(10 ** -5, 10, 'log_uniform', name='lam_2'),
    Integer(0, up, name='sigma_choice'),
    Integer(0, up, name='w_choice'),
]

# define objective function, I used mse as score function for our solver,
# which can be found inn `generalized_elastic_net_solver.py`
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)
    return np.mean(cross_val_score(reg, X_test, y_test, cv=3, n_jobs=3))


# callback function in each iteration, where I just print out the parameter that was tried in each iter
def monitor(res):
    dump(res, './hyper_optimization_results.pkl')
    print('run_parameters', str(res.x_iters[-1]))
    print('run_score', res.func_vals[-1])


import os

# test if result file is already exist, if yes, continue from the existing results, else, start from beginning
exists = os.path.isfile('./hyper_optimization_results.pkl')
print(exists)
if exists:

    results = load('./hyper_optimization_results.pkl')
    print('previous result exists')
    res_gp = forest_minimize(objective, space, callback=[monitor],
                             x0=results.x_iters,
                             y0=results.func_vals,
                             **HPO_PARAMS
                             )

else:
    res_gp = forest_minimize(objective, space, callback=[monitor],
                             **HPO_PARAMS)

print("Best score=%.4f" % results.fun)

# plot results
import matplotlib
# matplotlib.use('TkAgg')
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

plot_convergence(results)
plt.show()
plot_evaluations(results)
plt.show()
plot_objective(results)
plt.show()
