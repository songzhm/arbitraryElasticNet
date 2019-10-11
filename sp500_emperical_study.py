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
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams["axes.grid"] = False
# from generalized_elastic_net_solver_yujia import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

mse_score = make_scorer(mean_squared_error, greater_is_better=False)


from ARGEN import *

import numpy as np
import pandas as pd


# utility functions
def get_data_index(time_vec, update_frequency='none', training_month=3):
    n = len(time_vec)
    training_size = training_month * 21

    if update_frequency == 'quarterly':
        testing_size = 63
    elif update_frequency == 'semi-annually':
        testing_size = 126
    elif update_frequency == 'annually':
        testing_size = 252
    else:
        testing_size = n - training_size

    train_start = 0
    train_end = training_size
    test_start = train_end + 1
    test_end = train_end + testing_size

    training_indexes = []
    testing_indexes = []

    while test_start < n:
        # print(train_start, train_end, test_start, test_end)
        training_indexes.append(time_vec[train_start: train_end])
        testing_indexes.append(time_vec[test_start:test_end])

        test_start = test_end
        train_end = test_start - 1
        test_end = min(n, train_end + testing_size)
        # train_start = train_end - training_size

    # num_of_updates = int(n / total_size)
    #
    # training_indexes = np.array(
    #     [time_vec[(testing_size * i):(testing_size * i + training_size)] for i in range(num_of_updates)])
    # testing_indexes = np.array(
    #     [time_vec[(testing_size * i + training_size):(testing_size * i + total_size)] for i in range(num_of_updates)])

    return np.array(training_indexes), np.array(testing_indexes)


def calculate_cumulative_return(portfolio_return):
    one_plus_return = 1 + portfolio_return
    return np.cumprod(one_plus_return)[-1] - 1


def calculate_annual_average_return(portfolio_return):
    average_return = np.average(portfolio_return)
    return (1 + average_return) ** 252 - 1


def calculated_annual_volatility(portfolio_return):
    return np.sqrt(252) * np.std(portfolio_return)


def calculate_daily_tracking_error(portfolio_return, index_return):
    assert len(portfolio_return) == len(index_return), "two vectors need to be the same length"
    res = np.sqrt(np.average(np.power(portfolio_return - index_return, 2)))
    return res


def calculate_daily_tracking_error_volatility(portfolio_return, index_return):
    assert len(portfolio_return) == len(index_return), "two vectors need to be the same length"
    return np.std(np.abs(portfolio_return - index_return))


def calculate_monthly_average_turnover(weights, update_frequency='none'):
    _, np_ = weights.shape
    # diff = np.abs(weights[:, 1:np_] - weights[:, 0:(np_ - 1)])
    diff = np.abs(np.diff(weights, 1))
    if update_frequency == 'quarterly':
        f = 3

    elif update_frequency == 'semi-annually':
        f = 6

    elif update_frequency == 'annually':
        f = 12

    else:
        return 0

    return np.sum(diff) / 2 / f


mydateparser = lambda x: pd.datetime.strptime(x, "%m/%d/%y")
sp500_all = pd.read_csv('sandp500/sp500_pct.csv', index_col='Date', parse_dates=['Date.1'], date_parser=mydateparser)

sp500_all.index = sp500_all['Date.1']

constituents_names = sp500_all.columns.tolist()
constituents_names = [x for x in constituents_names if x not in ['Date', 'Date.1', 'SP500']]
constituents = sp500_all[constituents_names]
sp500 = sp500_all['SP500']

X = constituents.values
y = sp500.values
time_vec = np.array(list(range(1, X.shape[0] + 1))) - 1

# time_mat = np.array([time_vec,]*X.shape[1]).T
# X_t_interaction = X*time_mat
# X =np.concatenate((time_vec[...,None],X,X_t_interaction),axis=1)
# del time_mat
# del X_t_interaction

# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)


_, p = X.shape

beta = np.ones(p)

max_iter = 5000

lowbo = np.repeat(0, p)
upbo = np.repeat(float('inf'), p)
c = 0.005
transaction_cost = np.log((1 - c) / (1 + c))
d = 0.02 / 252
short_cost = -d

# %%
#
# quarterly
# semi-annually
# annually
# none

cost_adjustment_factor = 40

cost_adjustment = (transaction_cost + short_cost) / cost_adjustment_factor
zero_threshold = 1.01e-6
target_number = 45
training_month = 6  # 3, 6, 12

dates = sp500_all['Date.1'][training_month * 21 + 1:]

# print('{num:.02%}'.format(num=-cost_adjustment))

print('c_r, '
      'a_r, '
      'a_v, '
      'd_te, '
      'd_tev',
      'm_tr')

# ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']
# [1,3,6,12]
counter = 0
fig1, ax1 = plt.subplots()
colors = ['r', 'b', 'g', 'y']

for update_frequency in ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']:

    training_index, testing_index = get_data_index(time_vec, update_frequency, training_month)

    nps = len(training_index)

    pred = np.array([])
    true = np.array([])
    coefs = []

    for i in range(nps):
        # print('{} out of {}'.format(i, nps))
        training_ind = training_index[i]
        testing_ind = testing_index[i]

        X_train = X[training_ind, :]
        X_test = X[testing_ind, :]
        y_train = y[training_ind]
        y_test = y[testing_ind]
        X_train = X_train - cost_adjustment
        X_test = X_test - cost_adjustment

        lam_1_low = 0
        lam_1_up = 1
        lam_1 = (lam_1_low + lam_1_up) / 2

        iter = 0

        reg2 = GeneralizedElasticNetRegressor(beta=beta, lam_1=lam_1, wvec=np.ones(p), sigma_ds=np.ones(p))
        reg2.fit(X_train, y_train)
        iter += 1
        non_zero_coef = np.sum(reg2.coef_ >= zero_threshold)
        # print('iter', ',', 'lam_1_low', ',', 'lam_1', ',', 'lam_1_up', ',', 'non_zero_coef')
        # print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        while iter < max_iter:

            if non_zero_coef > target_number:
                lam_1_low = lam_1
            elif non_zero_coef < target_number:
                lam_1_up = lam_1
            else:
                break

            lam_1 = (lam_1_low + lam_1_up) / 2
            reg2 = GeneralizedElasticNetRegressor(beta=beta, lam_1=lam_1, lowbo=lowbo, upbo=upbo, wvec=np.ones(p),
                                                  sigma_ds=np.ones(p))
            reg2.fit(X_train, y_train)
            non_zero_coef = np.sum(reg2.coef_ > zero_threshold)
            iter += 1
            # print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        port = np.where(reg2.coef_ > zero_threshold)[0]
        X_train_ = X_train[:, port.tolist()]
        X_test_ = X_test[:, port.tolist()]
        coefs_ = reg2.coef_
        coefs_[coefs_ <= zero_threshold] = 0
        coefs.append(coefs_ / sum(coefs_))

        # time_vec_ = np.array(list(range(1, X_train.shape[0] + 1)))
        # time_mat_ = np.array([time_vec_, ] * X_train_.shape[1]).T
        # X_t_interaction = X_train_ * time_mat_
        # X_train_ = np.concatenate((time_vec_[..., None], X_train_, X_t_interaction), axis=1)
        # del time_mat_
        # del X_t_interaction
        #
        # time_vec_ = np.array(list(range(1, X_test_.shape[0] + 1)))+X_train.shape[0]
        # time_mat_ = np.array([time_vec_, ] * X_test_.shape[1]).T
        # X_t_interaction = X_test_ * time_mat_
        # X_test_ = np.concatenate((time_vec_[..., None], X_test_, X_t_interaction), axis=1)
        # del time_mat_
        # del X_t_interaction

        # ols_reg = LinearRegression()

        p_ = X_test_.shape[1]
        lowbo_ = np.repeat(0, p_)
        upbo_ = np.repeat(float('inf'), p_)

        ols_reg = GeneralizedElasticNetRegressor(beta=np.ones(p_), lowbo=lowbo_, upbo=upbo_, wvec=np.zeros(p_),
                                                 sigma_ds=np.zeros(p_))
        ols_reg.fit(X_train_, y_train)
        coef = ols_reg.coef_
        coef = coef / np.sum(coef)

        # adjust for transaction cost
        X_test_[0, :] = X_test_[0, :] + transaction_cost
        X_test_[:, coef < 0] = X_test_[:, coef < 0] + short_cost
        pred_ = ols_reg.predict(X_test_)
        # print(port)
        # print(coef)

        pred = np.append(pred, pred_)
        true = np.append(true, y_test)

    coefs = np.array(coefs).T
    fig, handle = plt.subplots()
    im = handle.matshow(coefs, interpolation='none', aspect='auto', cmap=plt.cm.Spectral_r)
    fig.colorbar(im)
    handle.title.set_text('{} update w/ port #: {}'.format(update_frequency, coefs.shape[1]))
    fig.show()
    fig.savefig('training_month_{}_cost_factor_{}_stock_number_{}_update_frequency_{}.png'.format(training_month,
                                                                                                  cost_adjustment_factor,
                                                                                                  target_number,
                                                                                                  update_frequency),
                dpi=800)

    # ols_reg_2 = LinearRegression()
    # ols_reg_2.fit(X_train, y_train)
    # pred_2 = ols_reg_2.predict(X_test)

    print('{num0:.2%},{num1:.2%},{num2:.2%},{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2%}'
          .format(num0=calculate_cumulative_return(pred),
                  num1=calculate_annual_average_return(pred),
                  num2=calculated_annual_volatility(pred),
                  num3=calculate_daily_tracking_error(pred, true),
                  num4=calculate_daily_tracking_error_volatility(pred, true),
                  num5=calculate_monthly_average_turnover(coefs, update_frequency),
                  num6=mean_squared_error(true, pred)
                  )
          )
    # print(calculate_cumulative_return(true),
    #       calculate_annual_average_return(true),
    #       calculated_annual_volatility(true),
    #       calculate_daily_tracking_error(true, true),
    #       calculate_daily_tracking_error_volatility(true, true))

    cum_pred = np.cumprod(1 + pred)
    cum_true = np.cumprod(1 + true)

    ax1.plot(dates, cum_pred, '{}--'.format(colors[counter]),
             label='{t1} update | te={t2:.2%}'.format(t1=update_frequency,
                                                      t2=calculate_daily_tracking_error(pred, true)))
    counter += 1

print('{num0:.2%},{num1:.2%},{num2:.2%},{num3:.3%},{num4:.3%}, {num5:.2%}, {num6: .2%}'
      .format(num0=calculate_cumulative_return(true),
              num1=calculate_annual_average_return(true),
              num2=calculated_annual_volatility(true),
              num3=calculate_daily_tracking_error(true, true),
              num4=calculate_daily_tracking_error_volatility(true, true),
              num5=0.0,
              num6=0.0
              )
      )
ax1.plot(dates, cum_true, 'k', label='sp500')
# ax1.legend(['buy-and-hold', 'annually', 'semi-annually', 'quarterly', 'sp500'])
ax1.legend()
ax1.title.set_text(
    'stock #: {num1}, training size: {num2}, cost adjustment factor: {num3:.02%}'.format(num1=target_number,
                                                                                         num2=training_month,
                                                                                         num3=-cost_adjustment))
fig1.show()
fig1.savefig(
    'cumulative_return_plot_training_month_{}_cost_factor_{}_stock_number_{}.png'.format(training_month,
                                                                                         cost_adjustment_factor,
                                                                                         target_number),
    dpi=800)


# %%

def f(X, y, update_frequency, cost_adjustment, zero_threshold, target_number, training_month):
    training_index, testing_index = get_data_index(time_vec, update_frequency, training_month)

    nps = len(training_index)

    pred = np.array([])
    true = np.array([])

    for i in range(nps):
        training_ind = training_index[i]
        testing_ind = testing_index[i]

        X_train = X[training_ind, :]
        X_test = X[testing_ind, :]
        y_train = y[training_ind]
        y_test = y[testing_ind]
        X_train = X_train - cost_adjustment
        X_test = X_test - cost_adjustment

        lam_1_low = 0
        lam_1_up = 1
        lam_1 = (lam_1_low + lam_1_up) / 2

        iter = 0

        reg2 = GeneralizedElasticNetRegressor(beta=beta, lam_1=lam_1, wvec=np.ones(p), sigma_ds=np.ones(p))
        reg2.fit(X_train, y_train)
        iter += 1
        non_zero_coef = np.sum(reg2.coef_ >= zero_threshold)
        # print('iter', ',', 'lam_1_low', ',', 'lam_1', ',', 'lam_1_up', ',', 'non_zero_coef')
        # print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        while iter < max_iter:

            if non_zero_coef > target_number:
                lam_1_low = lam_1
            elif non_zero_coef < target_number:
                lam_1_up = lam_1
            else:
                break

            lam_1 = (lam_1_low + lam_1_up) / 2
            reg2 = GeneralizedElasticNetRegressor(beta=beta, lam_1=lam_1, lowbo=lowbo, upbo=upbo, wvec=np.ones(p),
                                                  sigma_ds=np.ones(p))
            reg2.fit(X_train, y_train)
            non_zero_coef = np.sum(reg2.coef_ > zero_threshold)
            iter += 1
            # print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        port = np.where(reg2.coef_ > zero_threshold)[0]
        X_train_ = X_train[:, port.tolist()]
        X_test_ = X_test[:, port.tolist()]
        # time_vec_ = np.array(list(range(1, X_train.shape[0] + 1)))
        # time_mat_ = np.array([time_vec_, ] * X_train_.shape[1]).T
        # X_t_interaction = X_train_ * time_mat_
        # X_train_ = np.concatenate((time_vec_[..., None], X_train_, X_t_interaction), axis=1)
        # del time_mat_
        # del X_t_interaction
        #
        # time_vec_ = np.array(list(range(1, X_test_.shape[0] + 1)))+X_train.shape[0]
        # time_mat_ = np.array([time_vec_, ] * X_test_.shape[1]).T
        # X_t_interaction = X_test_ * time_mat_
        # X_test_ = np.concatenate((time_vec_[..., None], X_test_, X_t_interaction), axis=1)
        # del time_mat_
        # del X_t_interaction

        # ols_reg = LinearRegression()

        p_ = X_test_.shape[1]
        lowbo_ = np.repeat(0, p_)
        upbo_ = np.repeat(float('inf'), p_)

        ols_reg = GeneralizedElasticNetRegressor(beta=np.ones(p_), lowbo=lowbo_, upbo=upbo_, wvec=np.zeros(p_),
                                                 sigma_ds=np.zeros(p_))
        ols_reg.fit(X_train_, y_train)
        coef = ols_reg.coef_
        coef = coef / np.sum(coef)

        # adjust for transaction cost
        X_test_[0, :] = X_test_[0, :] + transaction_cost
        X_test_[:, coef < 0] = X_test_[:, coef < 0] + short_cost
        pred_ = ols_reg.predict(X_test_)
        # print(port)
        # print(coef)

        pred = np.append(pred, pred_)
        true = np.append(true, y_test)

    return calculate_cumulative_return(pred), calculate_daily_tracking_error(pred, true)


# %%

# quarterly
# semi-annually
# annually
# buy-and-hold
zero_threshold = 1.01e-6
update_frequency = 'annually'
training_size_pool = [3, 6, 9, 12, 15, 18]
cost_adjustment_factor_pool = [30, 35, 40, 45, 50, 55, 60]
A1, B1 = np.meshgrid(training_size_pool, cost_adjustment_factor_pool)
positions = np.vstack([A1.ravel(), B1.ravel()])
Z1 = np.zeros((positions.shape[1], 1))
Z1_ = np.zeros((positions.shape[1], 1))

for t in range(positions.shape[1]):
    print('{} out of {}'.format(t, positions.shape[1]))
    training_month = positions[0, t]
    cost_adjustment = (transaction_cost + short_cost) / positions[1, t]
    target_number = 35
    Z1[t], Z1_[t] = f(X, y, update_frequency, cost_adjustment, zero_threshold, target_number, training_month)

Z1 = Z1.reshape(A1.shape)
Z1_ = Z1_.reshape(A1.shape)

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
im = ax.plot_surface(A1, B1, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                     alpha=0.8)

# ax.title.set_text('Cumulative return')
fig.colorbar(im)
ax.set_xlabel('training months')
ax.set_ylabel('cost adjustment factor')
ax.set_zlabel('cumulative return')
fig.show()
fig.savefig('3d_cumulative_return_training_months_cost_factor_{}_update.png'.format(update_frequency),
            dpi=800)

fig = plt.figure()
ax = fig.gca(projection='3d')
im = ax.plot_surface(A1, B1, Z1_, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                     alpha=0.8)

# ax.title.set_text('Cumulative return')
fig.colorbar(im)
ax.set_xlabel('training months')
ax.set_ylabel('cost adjustment factor')
ax.set_zlabel('tracking error')
fig.show()
fig.savefig('3d_tracking_error_training_months_cost_factor_{}_update.png'.format(update_frequency),
            dpi=800)

fig, ax = plt.subplots()
im = ax.matshow(Z1)
fig.colorbar(im)
ax.title.set_text('cumulative return')
ax.set_xticklabels([''] + training_size_pool)
ax.set_yticklabels([''] + cost_adjustment_factor_pool)
ax.set_xlabel('training_size_pool')
ax.set_ylabel('cost_adjustment_factor_pool')
fig.show()
fig.savefig('heat_map_cumulative_return_training_months_cost_factor_{}_update.png'.format(update_frequency),
            dpi=800)

fig, ax = plt.subplots()
im = ax.matshow(Z1_)
fig.colorbar(im)
ax.title.set_text('tracking error')
ax.set_xticklabels([''] + training_size_pool)
ax.set_yticklabels([''] + cost_adjustment_factor_pool)
ax.set_xlabel('training_size_pool')
ax.set_ylabel('cost_adjustment_factor_pool')
fig.show()
fig.savefig('heat_map_tracking_error_training_months_cost_factor_{}_update.png'.format(update_frequency),
            dpi=800)

# ----------------

target_number_pool = [15, 20, 25, 30, 35, 40, 45, 50]
cost_adjustment_factor_pool = [30, 35, 40, 45, 50, 55, 60]
A2, B2 = np.meshgrid(target_number_pool, cost_adjustment_factor_pool)
positions = np.vstack([A2.ravel(), B2.ravel()])
Z2 = np.zeros((positions.shape[1], 1))
Z2_ = np.zeros((positions.shape[1], 1))
zero_threshold = 1.01e-6

for t in range(positions.shape[1]):
    print('{} out of {}'.format(t, positions.shape[1]))
    training_month = 6
    cost_adjustment = (transaction_cost + short_cost) / positions[1, t]
    target_number = positions[0, t]
    Z2[t], Z2_[t] = f(X, y, update_frequency, cost_adjustment, zero_threshold, target_number, training_month)

Z2 = Z2.reshape(A2.shape)
Z2_ = Z2_.reshape(A2.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
im = ax.plot_surface(A2, B2, Z2, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                     alpha=0.8)

# ax.title.set_text('Cumulative return')
fig.colorbar(im)
ax.set_xlabel('target number')
ax.set_ylabel('cost adjustment factor')
ax.set_zlabel('cumulative return')
fig.show()
fig.savefig('3d_cumulative_return_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig = plt.figure()
ax = fig.gca(projection='3d')
im = ax.plot_surface(A2, B2, Z2_, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                     alpha=0.8)

# ax.title.set_text('Cumulative return')
fig.colorbar(im)
ax.set_xlabel('target number')
ax.set_ylabel('cost adjustment factor')
ax.set_zlabel('tracking error')
fig.show()
fig.savefig('3d_tracking_error_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig, ax = plt.subplots()
im = ax.matshow(Z2)
fig.colorbar(im)
ax.title.set_text('cumulative return')
ax.set_xticklabels([''] + target_number_pool)
ax.set_yticklabels([''] + cost_adjustment_factor_pool)
ax.set_xlabel('target_number_pool')
ax.set_ylabel('cost_adjustment_factor_pool')
fig.show()
fig.savefig('heat_map_cumulative_return_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig, ax = plt.subplots()
im = ax.matshow(Z2_)
fig.colorbar(im)
ax.title.set_text('tracking error')
ax.set_xticklabels([''] + target_number_pool)
ax.set_yticklabels([''] + cost_adjustment_factor_pool)
ax.set_xlabel('target_number_pool')
ax.set_ylabel('cost_adjustment_factor_pool')
fig.show()
fig.savefig('heat_map_tracking_error_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig.savefig('3d_cumulative_return_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

# ----------------

target_number_pool = [15, 20, 25, 30, 35, 40, 45, 50]
training_size_pool = [3, 6, 9, 12, 15, 18]
A3, B3 = np.meshgrid(target_number_pool, training_size_pool)
positions = np.vstack([A3.ravel(), B3.ravel()])
Z3 = np.zeros((positions.shape[1], 1))
Z3_ = np.zeros((positions.shape[1], 1))
zero_threshold = 1.01e-6

for t in range(positions.shape[1]):
    print('{} out of {}'.format(t, positions.shape[1]))
    target_number = positions[0, t]
    training_month = positions[1, t]
    cost_adjustment = (transaction_cost + short_cost) / 30
    Z3[t], Z3_[t] = f(X, y, update_frequency, cost_adjustment, zero_threshold, target_number, training_month)

Z3 = Z3.reshape(A3.shape)
Z3_ = Z3_.reshape(A3.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
im = ax.plot_surface(A3, B3, Z3, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                     alpha=0.8)

# ax.title.set_text('Cumulative return')
fig.colorbar(im)
ax.set_xlabel('target number')
ax.set_ylabel('training size pool')
ax.set_zlabel('cumulative return')
fig.show()
fig.savefig('3d_cumulative_return_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig = plt.figure()
ax = fig.gca(projection='3d')
im = ax.plot_surface(A3, B3, Z3_, rstride=1, cstride=1, cmap='viridis', edgecolor='none',
                     alpha=0.8)

# ax.title.set_text('Cumulative return')
fig.colorbar(im)
ax.set_xlabel('target number')
ax.set_ylabel('training size pool')
ax.set_zlabel('tracking error')
fig.show()
fig.savefig('3d_tracking_error_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig, ax = plt.subplots()
im = ax.matshow(Z3)
fig.colorbar(im)
ax.title.set_text('cumulative return')
ax.set_xticklabels([''] + target_number_pool)
ax.set_yticklabels([''] + training_size_pool)
ax.set_xlabel('target_number_pool')
ax.set_ylabel('training_size_pool')
fig.show()
fig.savefig('heat_map_cumulative_return_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig, ax = plt.subplots()
im = ax.matshow(Z3_)
fig.colorbar(im)
ax.title.set_text('tracking error')
ax.set_xticklabels([''] + target_number_pool)
ax.set_yticklabels([''] + training_size_pool)
ax.set_xlabel('target_number_pool')
ax.set_ylabel('training_size_pool')
fig.show()
fig.savefig('heat_map_tracking_error_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)

fig.savefig('3d_cumulative_return_target_number_training_month_{}_update.png'.format(update_frequency),
            dpi=800)


# %%
#
# quarterly
# semi-annually
# annually
# none
lam_2_pool = np.linspace(0, 1, 5)
cost_adjustment_factor = 40

cost_adjustment = (transaction_cost + short_cost) / cost_adjustment_factor
zero_threshold = 1.01e-6
target_number = 45
training_month = 6  # 3, 6, 12

dates = sp500_all['Date.1'][training_month * 21 + 1:]

# print('{num:.02%}'.format(num=-cost_adjustment))

print('c_r, '
      'a_r, '
      'a_v, '
      'd_te, '
      'd_tev',
      'm_tr')

# ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']
# [1,3,6,12]
counter = 0
fig1, ax1 = plt.subplots()
colors = ['r', 'b', 'g', 'y']
# ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']
for update_frequency in ['semi-annually']:

    training_index, testing_index = get_data_index(time_vec, update_frequency, training_month)

    nps = len(training_index)

    pred = np.array([])
    true = np.array([])
    coefs = []

    for i in range(nps):
        # print('{} out of {}'.format(i, nps))
        training_ind = training_index[i]
        testing_ind = testing_index[i]

        X_train = X[training_ind, :]
        X_test = X[testing_ind, :]
        y_train = y[training_ind]
        y_test = y[testing_ind]
        X_train = X_train - cost_adjustment
        X_test = X_test - cost_adjustment

        lam_1_low = 0
        lam_1_up = 1
        lam_1 = (lam_1_low + lam_1_up) / 2

        iter = 0

        reg2 = GeneralizedElasticNetRegressor(beta=beta, lam_1=lam_1, wvec=np.ones(p), sigma_ds=np.ones(p))
        reg2.fit(X_train, y_train)
        iter += 1
        non_zero_coef = np.sum(reg2.coef_ >= zero_threshold)
        print('iter', ',', 'lam_1_low', ',', 'lam_1', ',', 'lam_1_up', ',', 'non_zero_coef')
        print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        while iter < max_iter:

            if non_zero_coef > target_number:
                lam_1_low = lam_1
            elif non_zero_coef < target_number:
                lam_1_up = lam_1
            else:
                break

            lam_1 = (lam_1_low + lam_1_up) / 2
            reg2 = GeneralizedElasticNetRegressor(beta=beta, lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=np.ones(p),
                                                  sigma_ds=np.ones(p))
            reg2.fit(X_train, y_train)
            non_zero_coef = np.sum(reg2.coef_ > zero_threshold)
            iter += 1
            print(iter, ',', lam_1_low, ',', lam_1, ',', lam_1_up, ',', non_zero_coef)

        port = np.where(reg2.coef_ > zero_threshold)[0]
        X_train_ = X_train[:, port.tolist()]
        X_test_ = X_test[:, port.tolist()]
        coefs_ = reg2.coef_
        coefs_[coefs_ <= zero_threshold] = 0
        coefs.append(coefs_ / sum(coefs_))

        # time_vec_ = np.array(list(range(1, X_train.shape[0] + 1)))
        # time_mat_ = np.array([time_vec_, ] * X_train_.shape[1]).T
        # X_t_interaction = X_train_ * time_mat_
        # X_train_ = np.concatenate((time_vec_[..., None], X_train_, X_t_interaction), axis=1)
        # del time_mat_
        # del X_t_interaction
        #
        # time_vec_ = np.array(list(range(1, X_test_.shape[0] + 1)))+X_train.shape[0]
        # time_mat_ = np.array([time_vec_, ] * X_test_.shape[1]).T
        # X_t_interaction = X_test_ * time_mat_
        # X_test_ = np.concatenate((time_vec_[..., None], X_test_, X_t_interaction), axis=1)
        # del time_mat_
        # del X_t_interaction

        # ols_reg = LinearRegression()

        p_ = X_test_.shape[1]
        lowbo_ = np.repeat(0, p_)
        upbo_ = np.repeat(float('inf'), p_)

        ols_reg = GeneralizedElasticNetRegressor(beta=np.ones(p_), lowbo=lowbo_, upbo=upbo_, wvec=np.zeros(p_),
                                                 sigma_ds=np.zeros(p_))
        ols_reg.fit(X_train_, y_train)
        coef = ols_reg.coef_
        coef = coef / np.sum(coef)

        # adjust for transaction cost
        X_test_[0, :] = X_test_[0, :] + transaction_cost
        X_test_[:, coef < 0] = X_test_[:, coef < 0] + short_cost
        pred_ = ols_reg.predict(X_test_)
        # print(port)
        # print(coef)

        pred = np.append(pred, pred_)
        true = np.append(true, y_test)

    coefs = np.array(coefs).T
    fig, handle = plt.subplots()
    im = handle.matshow(coefs, interpolation='none', aspect='auto', cmap=plt.cm.Spectral_r)
    fig.colorbar(im)
    handle.title.set_text('{} update w/ port #: {}'.format(update_frequency, coefs.shape[1]))
    fig.show()
    # fig.savefig('training_month_{}_cost_factor_{}_stock_number_{}_update_frequency_{}.png'.format(training_month,
    #                                                                                               cost_adjustment_factor,
    #                                                                                               target_number,
    #                                                                                               update_frequency),
    #             dpi=800)

    # ols_reg_2 = LinearRegression()
    # ols_reg_2.fit(X_train, y_train)
    # pred_2 = ols_reg_2.predict(X_test)

    print('{num0:.2%},{num1:.2%},{num2:.2%},{num3:.3%},{num4:.3%},{num5:.2%}'
          .format(num0=calculate_cumulative_return(pred),
                  num1=calculate_annual_average_return(pred),
                  num2=calculated_annual_volatility(pred),
                  num3=calculate_daily_tracking_error(pred, true),
                  num4=calculate_daily_tracking_error_volatility(pred, true),
                  num5=calculate_monthly_average_turnover(coefs, update_frequency)
                  )
          )
    # print(calculate_cumulative_return(true),
    #       calculate_annual_average_return(true),
    #       calculated_annual_volatility(true),
    #       calculate_daily_tracking_error(true, true),
    #       calculate_daily_tracking_error_volatility(true, true))

    cum_pred = np.cumprod(1 + pred)
    cum_true = np.cumprod(1 + true)

    ax1.plot(dates, cum_pred, '{}--'.format(colors[counter]),
             label='{t1} update | te={t2:.2%}'.format(t1=update_frequency,
                                                      t2=calculate_daily_tracking_error(pred, true)))
    counter += 1

print('{num0:.2%},{num1:.2%},{num2:.2%},{num3:.3%},{num4:.3%}, {num5:.2%}'
      .format(num0=calculate_cumulative_return(true),
              num1=calculate_annual_average_return(true),
              num2=calculated_annual_volatility(true),
              num3=calculate_daily_tracking_error(true, true),
              num4=calculate_daily_tracking_error_volatility(true, true),
              num5=0
              )
      )
ax1.plot(dates, cum_true, 'k', label='sp500')
# ax1.legend(['buy-and-hold', 'annually', 'semi-annually', 'quarterly', 'sp500'])
ax1.legend()
ax1.title.set_text(
    'stock #: {num1}, training size: {num2}, cost adjustment factor: {num3:.02%}'.format(num1=target_number,
                                                                                         num2=training_month,
                                                                                         num3=-cost_adjustment))
fig1.show()
# fig1.savefig(
#     'cumulative_return_plot_training_month_{}_cost_factor_{}_stock_number_{}.png'.format(training_month,
#                                                                                          cost_adjustment_factor,
#                                                                                          target_number),
#     dpi=800)


#%%


lam_2_pool = np.linspace(0, 1, 5)
parameters = {'lam_2':lam_2_pool}

cost_adjustment_factor = 40

cost_adjustment = (transaction_cost + short_cost) / cost_adjustment_factor
zero_threshold = 1.01e-6
target_number = 45
training_month = 6  # 3, 6, 12

dates = sp500_all['Date.1'][training_month * 21 + 1:]

# print('{num:.02%}'.format(num=-cost_adjustment))

print('c_r, '
      'a_r, '
      'a_v, '
      'd_te, '
      'd_tev',
      'm_tr')

counter = 0
fig1, ax1 = plt.subplots()
colors = ['r', 'b', 'g', 'y']
# ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']
for update_frequency in ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']:

    training_index, testing_index = get_data_index(time_vec, update_frequency, training_month)

    nps = len(training_index)

    pred = np.array([])
    true = np.array([])
    coefs = []

    for i in range(nps):
        # print('{} out of {}'.format(i, nps))
        training_ind = training_index[i]
        testing_ind = testing_index[i]

        X_train = X[training_ind, :]
        X_test = X[testing_ind, :]
        y_train = y[training_ind]
        y_test = y[testing_ind]
        X_train = X_train - cost_adjustment
        X_test = X_test - cost_adjustment

        reg2 = GeneralizedElasticNetRegressor(beta=beta, tune_lam_1=True, target_number=target_number,wvec=np.ones(p), sigma_ds=np.ones(p))
        clf = GridSearchCV(reg2, parameters, cv=5, scoring=mse_score)
        clf.fit(X_train, y_train)


        port = np.where(clf.best_estimator_.coef_ > zero_threshold)[0]
        X_train_ = X_train[:, port.tolist()]
        X_test_ = X_test[:, port.tolist()]
        coefs_ = clf.best_estimator_.coef_
        coefs_[coefs_ <= zero_threshold] = 0
        coefs.append(coefs_ / sum(coefs_))

        p_ = X_test_.shape[1]
        lowbo_ = np.repeat(0, p_)
        upbo_ = np.repeat(float('inf'), p_)

        ols_reg = GeneralizedElasticNetRegressor(beta=np.ones(p_), lowbo=lowbo_, upbo=upbo_, wvec=np.zeros(p_),
                                                 sigma_ds=np.zeros(p_))
        ols_reg.fit(X_train_, y_train)
        coef = ols_reg.coef_
        coef = coef / np.sum(coef)

        # adjust for transaction cost
        X_test_[0, :] = X_test_[0, :] + transaction_cost
        X_test_[:, coef < 0] = X_test_[:, coef < 0] + short_cost
        pred_ = ols_reg.predict(X_test_)
        # print(port)
        # print(coef)

        pred = np.append(pred, pred_)
        true = np.append(true, y_test)

    coefs = np.array(coefs).T
    fig, handle = plt.subplots()
    im = handle.matshow(coefs, interpolation='none', aspect='auto', cmap=plt.cm.Spectral_r)
    fig.colorbar(im)
    handle.title.set_text('{} update w/ port #: {}'.format(update_frequency, coefs.shape[1]))
    fig.show()
    # fig.savefig('training_month_{}_cost_factor_{}_stock_number_{}_update_frequency_{}.png'.format(training_month,
    #                                                                                               cost_adjustment_factor,
    #                                                                                               target_number,
    #                                                                                               update_frequency),
    #             dpi=800)

    # ols_reg_2 = LinearRegression()
    # ols_reg_2.fit(X_train, y_train)
    # pred_2 = ols_reg_2.predict(X_test)

    print('{num0:.2%},{num1:.2%},{num2:.2%},{num3:.3%},{num4:.3%},{num5:.2%}'
          .format(num0=calculate_cumulative_return(pred),
                  num1=calculate_annual_average_return(pred),
                  num2=calculated_annual_volatility(pred),
                  num3=calculate_daily_tracking_error(pred, true),
                  num4=calculate_daily_tracking_error_volatility(pred, true),
                  num5=calculate_monthly_average_turnover(coefs, update_frequency)
                  )
          )
    # print(calculate_cumulative_return(true),
    #       calculate_annual_average_return(true),
    #       calculated_annual_volatility(true),
    #       calculate_daily_tracking_error(true, true),
    #       calculate_daily_tracking_error_volatility(true, true))

    cum_pred = np.cumprod(1 + pred)
    cum_true = np.cumprod(1 + true)

    ax1.plot(dates, cum_pred, '{}--'.format(colors[counter]),
             label='{t1} update | te={t2:.2%}'.format(t1=update_frequency,
                                                      t2=calculate_daily_tracking_error(pred, true)))
    counter += 1

print('{num0:.2%},{num1:.2%},{num2:.2%},{num3:.3%},{num4:.3%}, {num5:.2%}'
      .format(num0=calculate_cumulative_return(true),
              num1=calculate_annual_average_return(true),
              num2=calculated_annual_volatility(true),
              num3=calculate_daily_tracking_error(true, true),
              num4=calculate_daily_tracking_error_volatility(true, true),
              num5=0
              )
      )
ax1.plot(dates, cum_true, 'k', label='sp500')
# ax1.legend(['buy-and-hold', 'annually', 'semi-annually', 'quarterly', 'sp500'])
ax1.legend()
ax1.title.set_text(
    'stock #: {num1}, training size: {num2}, cost adjustment factor: {num3:.02%}'.format(num1=target_number,
                                                                                         num2=training_month,
                                                                                         num3=-cost_adjustment))
fig1.show()
# fig1.savefig(
#     'cumulative_return_plot_training_month_{}_cost_factor_{}_stock_number_{}.png'.format(training_month,
#                                                                                          cost_adjustment_factor,
#                                                                                          target_number),
#     dpi=800)