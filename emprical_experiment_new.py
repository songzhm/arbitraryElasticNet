# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 2019-07-21 07:22
"""
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dump, load
from skopt import forest_minimize, dummy_minimize
import pandas as pd

from generalized_elastic_net_solver_yujia import *

import numpy as np



def calculate_cumulative_return(portfolio_return):
    one_plus_return = 1+portfolio_return
    return np.cumprod(one_plus_return)[-1]-1

def calculate_annual_average_return(portfolio_return):
    average_return = np.average(portfolio_return)
    return (1+average_return)**252 - 1

def calculated_annual_volatility(portfolio_return):
    return np.sqrt(252)*np.std(portfolio_return)

def calculate_daily_tracking_error(portfolio_return, index_return):
    assert len(portfolio_return)==len(index_return), "two vectors need to be the same length"
    res = np.sqrt(np.average(np.power(portfolio_return-index_return,2)))
    return res

def calculate_daily_tracking_error_volatility(portfolio_return, index_return):
    assert len(portfolio_return)==len(index_return), "two vectors need to be the same length"
    return np.std(np.abs(portfolio_return-index_return))

def calculate_monthly_average_turnover(weights,update_frequency='none'):
    _, np = weights.shape
    diff = np.abs(weights[:,1:np] - weights[:,0:(np-1)])
    if update_frequency == 'quarterly':
        f = 3

    elif update_frequency == 'semi-annually':
        f = 6

    elif update_frequency == 'annually':
        f = 12

    else:
        return 0

    return np.sum(diff)/2/f


mydateparser = lambda x: pd.datetime.strptime(x, "%m/%d/%y")
sp500_all = pd.read_csv('sandp500/sp500_pct.csv', index_col='Date', parse_dates=['Date.1'], date_parser=mydateparser)

sp500_all.index = sp500_all['Date.1']

constituents_names = sp500_all.columns.tolist()
constituents_names = [x for x in constituents_names if x not in ['Date', 'Date.1', 'SP500']]
constituents = sp500_all[constituents_names]
sp500 = sp500_all['SP500']

X = constituents.values
y = sp500.values
time_vec = np.array(list(range(1, X.shape[0]+1)))
time_mat = np.array([time_vec,]*X.shape[1]).T
X_t_interaction = X*time_mat
X =np.concatenate((time_vec[...,None],X,X_t_interaction),axis=1)
del time_mat
del X_t_interaction

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)


_, p = X.shape

HPO_PARAMS = {'n_calls': 10,
              'n_random_state': 10,
              'base_estimator': 'ET',
              'acq_func': 'EI',
              'xi': 0.02,
              'kappa': 1.96,
              'n_points': 1000,
              "n_jobs": -1,
              "verbose": True
              }

beta = [0 for i in range(p)]

reg = GeneralizedElasticNetRegressor(beta=beta, sigma_choice_base=2, w_choice_base=2)
up = min(np.iinfo(np.int64).max, 2 ** p-1)
# define search space and parameters
space = [
    Integer(0, 100, name='lam_1'),
    Integer(0, 100, name='lam_2'),
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
    res_gp = dummy_minimize(objective, space, callback=[monitor],
                             x0=results.x_iters,
                             y0=results.func_vals,
                             **HPO_PARAMS
                             )

else:
    res_gp = dummy_minimize(objective, space, callback=[monitor],
                             **HPO_PARAMS)

print("Best score=%.4f" % results.fun)


from skopt.plots import plot_convergence, plot_evaluations, plot_objective

plot_convergence(results)
plt.show()
plot_evaluations(results)
plt.show()
# plot_objective(results)
# plt.show()

reg.set_params(lam_1=results.x[0], lam_2=results.x[1], lowbo=None, upbo=None, ds=None,
               wvec=None, random_state=None, sigma_choice=results.x[2], w_choice=results.x[3],
               err_tol=1e-8, verbose=False, text_fr=200)
coef=(reg.fit(X_train,y_train)).coef_
print(reg.score(X_test))


# utility functions

def get_data_index(time_vec, update_frequency='none'):
    n = len(time_vec)
    training_size = 6*21
    if update_frequency=='quarterly':
        testing_size = 63
    elif update_frequency=='semi-annually':
        testing_size = 126
    elif update_frequency=='annually':
        testing_size = 252
    else:
        testing_size = n - training_size

    total_size = training_size + testing_size
    num_of_updates = int(n / total_size)

    training_indexes = np.array([time_vec[(testing_size * i):(testing_size * i + training_size)] for i in range(num_of_updates)])
    testing_indexes = np.array([time_vec[(testing_size * i + training_size):(testing_size * i + total_size)] for i in range(num_of_updates)])

    return training_indexes, testing_indexes


