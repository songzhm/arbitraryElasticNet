# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 9/2/19 4:02 PM
"""

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from ARGEN import *

import numpy as np

plt.style.use('default')
plt.rcParams["axes.grid"] = False

mse_score = make_scorer(mean_squared_error, greater_is_better=False)


# utility functions

def log(file_name, msg):
    f = open(file_name, 'a')
    f.write('\n {}'.format(msg))
    f.close()


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

    train_start = int(0)
    train_end = int(training_size)
    test_start = int(train_end + 1)
    test_end = int(train_end + testing_size)

    training_indexes = []
    testing_indexes = []

    while test_start < n:
        # print(train_start, train_end, test_start, test_end)
        training_indexes.append(time_vec[train_start: train_end])
        testing_indexes.append(time_vec[test_start:test_end])

        test_start = int(test_end)
        train_end = int(test_start - 1)
        test_end = int(min(n, train_end + testing_size))
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
    excess_return = portfolio_return - index_return
    res = np.std(excess_return)
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

# Effect of OLS

# WITHOUT non-negative ols

lam_2_pool = np.linspace(0, 1, 3)
w_options = np.random.RandomState(1).randint(1, 11, size=(3, p))
w_options = w_options/w_options.sum(1)[:, np.newaxis]
parameters = {'lam_2': lam_2_pool,
              'wvec': w_options,
              'sigma_ds': np.random.RandomState(2).randint(1,11,size=(3,p))}

cost_adjustment_factor = 0

cost_adjustment = (transaction_cost + short_cost) * cost_adjustment_factor

zero_threshold = 1.01e-6
portfolio_size = 45
training_month = 6  # 3, 6, 12

dates = sp500_all['Date.1'][training_month * 21 + 1:]

print('c_r, '
      'a_r, '
      'a_v, '
      'd_te, '
      'd_tev',
      'm_tr',
      'mse')

# ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']
# [1,3,6,12]
counter = 0
fig1, ax1 = plt.subplots()
colors = ['r', 'b', 'g', 'y']

update_frequency = 'buy-and-hold'

training_index, testing_index = get_data_index(time_vec, update_frequency, training_month)

nps = len(training_index)

pred = np.array([])
true = np.array([])
coefs = []

i=0

training_ind = training_index[i]
testing_ind = testing_index[i]

X_train = X[training_ind, :]
X_test = X[testing_ind, :]
y_train = y[training_ind]
y_test = y[testing_ind]
X_train = X_train - cost_adjustment
X_test = X_test

reg = GeneralizedElasticNetRegressor(beta=beta, tune_lam_1=True, target_number=portfolio_size)
clf = RandomizedSearchCV(reg, parameters, cv=3, scoring=mse_score)
clf.fit(X_train, y_train)

port = np.where(clf.best_estimator_.coef_ > zero_threshold)[0]
X_train_ = X_train[:, port.tolist()]
X_test_ = X_test[:, port.tolist()]
coefs_ = clf.best_estimator_.coef_
coefs_[coefs_ <= zero_threshold] = 0
coefs.append(coefs_ / sum(coefs_))

coef = coefs_[coefs_ > zero_threshold]
coef = coef / np.sum(coef)

# adjust for transaction cost
X_test_[0, :] = X_test_[0, :] + transaction_cost
X_test_[:, coef < 0] = X_test_[:, coef < 0] + short_cost

pred_ = np.dot(X_test_, coef)

pred = np.append(pred, pred_)
true = np.append(true, y_test)

coefs = np.array(coefs).T
print('check stock numbers:', np.sum(coefs > 0, 0))
fig, handle = plt.subplots()
im = handle.matshow(coefs, interpolation='none', aspect='auto', cmap=plt.cm.Spectral_r)
fig.colorbar(im)
handle.title.set_text('{} update w/ port #: {}'.format(update_frequency, coefs.shape[1]))
# fig.show()
fig.savefig('no_ols_training_month_{}_cost_factor_{}_stock_number_{}_update_frequency_{}.png'.format(training_month,
                                                                                                     cost_adjustment_factor,
                                                                                                     portfolio_size,
                                                                                                     update_frequency),
            dpi=800)
# fig.close()

print(
    '{set0}, {set1}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
    '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
        .format(set0='no_ols' + update_frequency,
                set1=cost_adjustment_factor,
                set2=portfolio_size,
                set3=training_month,
                set4=clf.best_estimator_.lam_1,
                set5=clf.best_estimator_.lam_2,
                num0=calculate_cumulative_return(pred),
                num1=calculate_annual_average_return(pred),
                num2=calculated_annual_volatility(pred),
                num3=calculate_daily_tracking_error(pred, true),
                num4=calculate_daily_tracking_error_volatility(pred, true),
                num5=calculate_monthly_average_turnover(coefs, update_frequency),
                num6=mean_squared_error(true, pred)
                )
)

log('experiments.txt', '{set0}, {set1}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
                       '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
    .format(set0='no_ols' + update_frequency,
            set1=cost_adjustment_factor,
            set2=portfolio_size,
            set3=training_month,
            set4=clf.best_estimator_.lam_1,
            set5=clf.best_estimator_.lam_2,
            num0=calculate_cumulative_return(pred),
            num1=calculate_annual_average_return(pred),
            num2=calculated_annual_volatility(pred),
            num3=calculate_daily_tracking_error(pred, true),
            num4=calculate_daily_tracking_error_volatility(pred, true),
            num5=calculate_monthly_average_turnover(coefs, update_frequency),
            num6=mean_squared_error(true, pred)
            ))

cum_pred = np.cumprod(1 + pred)
cum_true = np.cumprod(1 + true)

ax1.plot(dates, cum_pred, '{}--'.format(colors[counter]),
         label='{t1} update | te={t2:.2%}'.format(t1=update_frequency,
                                                  t2=calculate_daily_tracking_error(pred, true)))
counter += 1

print('{set0}, {set1}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
      '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
      .format(set0='no_ols' + 'NA',
              set1='NA',
              set2='NA',
              set3='NA',
              set4='NA',
              set5='NA',
              num0=calculate_cumulative_return(true),
              num1=calculate_annual_average_return(true),
              num2=calculated_annual_volatility(true),
              num3=calculate_daily_tracking_error(true, true),
              num4=calculate_daily_tracking_error_volatility(true, true),
              num5=0,
              num6=0
              )
      )
log('experiments.txt', '{set0}, {set1}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
                       '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
    .format(set0='no_ols' + 'NA',
            set1='NA',
            set2='NA',
            set3='NA',
            set4='NA',
            set5='NA',
            num0=calculate_cumulative_return(true),
            num1=calculate_annual_average_return(true),
            num2=calculated_annual_volatility(true),
            num3=calculate_daily_tracking_error(true, true),
            num4=calculate_daily_tracking_error_volatility(true, true),
            num5=0,
            num6=0
            )
    )
ax1.plot(dates, cum_true, 'k', label='sp500')
# ax1.legend(['buy-and-hold', 'annually', 'semi-annually', 'quarterly', 'sp500'])
ax1.legend()
ax1.title.set_text(
    'stock #: {num1}, training size: {num2}, cost adjustment factor: {num3:.02%}'.format(num1=portfolio_size,
                                                                                         num2=training_month,
                                                                                         num3=-cost_adjustment))
# fig1.show()
fig1.savefig(
    'no_ols_cumulative_return_plot_training_month_{}_cost_factor_{}_stock_number_{}.png'.format(training_month,
                                                                                                cost_adjustment_factor,
                                                                                                portfolio_size),
    dpi=800)
# fig1.close()
