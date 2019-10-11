# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 10/6/19 10:32 AM
"""
# %%
from production.ARGEN import *
from production.utility_function import *
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import cross_val_score, train_test_split


target_number = 30
B = 0.8
A = (1 - B) / (target_number - 1)

data = get_data()

c = 0.005

transaction_cost = np.log((1 - c) / (1 + c))

X = data['X']
y = data['y']
dates = data['dates']

time_vec = np.array(list(range(1, X.shape[0] + 1))) - 1

_, p = X.shape

update_frequency = 'buy-and-hold'

training_month = 6

training_index, testing_index = get_data_index(time_vec, update_frequency, training_month)

training_ind = training_index[0]
testing_ind = testing_index[0]

X_train = X[training_ind, :]
X_test = X[testing_ind, :]
y_train = y[training_ind]
y_test = y[testing_ind]


fs_clf = FeatureSelectionRegressor(p, target_number)

fs_clf.fit(X_train, y_train)
fs_clf.score(X_test, y_test)

selected_feature_ind = np.where(fs_clf.coef_ != 0.0)[0]

X_train_ = X_train[:, selected_feature_ind]
X_test_ = X_test[:, selected_feature_ind]

X_train_, X_val_, y_train, y_val = train_test_split(X_train_, y_train, test_size=0.2, random_state=1)

_, p_ = X_train_.shape


lowbo = np.ones(p_) * A
upbo = np.ones(p_) * B

# lowbo = np.ones(p_) * 0
# upbo = np.ones(p_) * np.inf

#benchmark
nnols_clf = ARGEN(p_, 0, 0, lowbo, upbo, 0, 0)
nnols_clf.fit(X_train_, y_train)
target_score = nnols_clf.score(X_test_, y_test)
benchmark_score = nnols_clf.score(X_val_, y_val)
print("benchmark train score: %s" % nnols_clf.score(X_train_, y_train))
print("benchmark validation score: %s" % nnols_clf.score(X_val_, y_val))
print("benchmark test score: %s" % target_score)
# print(nnols_clf.coef_/nnols_clf.coef_.sum())

#%%
best_clf = ARGEN(p_, 0.0198597067072241, 1.46338759922044, lowbo, upbo, 8184, 5936)

# best_clf = ARGEN(p_, 0.0429969559618923, 9.22262235291937, lowbo, upbo, 3936, 4306)

best_clf.fit(X_train_, y_train)

best_clf.score(X_train_, y_train)

coef = best_clf.coef_

normalized_coef = coef / np.sum(coef)

import matplotlib.pyplot as plt

plt.hist(normalized_coef)
plt.show()

pred_1 = best_clf.predict(X_test_)
pred_2 = nnols_clf.predict(X_test_)
cum_pred_1 = np.cumprod(1 + pred_1) - 1
cum_pred_2 = np.cumprod(1 + pred_2) - 1
cum_true = np.cumprod(1 + y_test) - 1
true = y_test

plt.figure()
plt.plot_date(dates[testing_ind], pred_1)
plt.plot_date(dates[testing_ind], pred_2)
plt.plot_date(dates[testing_ind], true)
plt.show()

plt.figure()
plt.plot_date(dates[testing_ind], cum_pred_1, linestyle='--', markersize=1, label='NNL+ARGEN')
plt.plot_date(dates[testing_ind], cum_pred_2, linestyle='-.', markersize=1, label='NNL+NNOLS')
plt.plot_date(dates[testing_ind], cum_true, linestyle='-', markersize=1, label='sp500')
plt.legend()
plt.show()

print('type, frequency, training month, cr, avr, av, te, tev, test_mse')

print(
    'tuned, {set0}, {set3}, {num0:.2%},{num1:.2%},{num2:.2%},'
    ' {num3:.3%},{num4:.3%},{num6: .2e}'
        .format(set0=update_frequency,
                set3=training_month,
                num0=calculate_cumulative_return(pred_1),
                num1=calculate_annual_average_return(pred_1),
                num2=calculated_annual_volatility(pred_1),
                num3=calculate_daily_tracking_error(pred_1, true),
                num4=calculate_daily_tracking_error_volatility(pred_1, true),
                num6=mean_squared_error(true, pred_1)
                )
)

print(
    'benchmark, {set0}, {set3}, {num0:.2%},{num1:.2%},{num2:.2%},'
    '{num3:.3%},{num4:.3%},{num6: .2e}'
        .format(set0=update_frequency,
                set3=training_month,
                num0=calculate_cumulative_return(pred_2),
                num1=calculate_annual_average_return(pred_2),
                num2=calculated_annual_volatility(pred_2),
                num3=calculate_daily_tracking_error(pred_2, true),
                num4=calculate_daily_tracking_error_volatility(pred_2, true),
                num6=mean_squared_error(true, pred_2)
                )
)

print('best val', best_clf.score(X_val_, y_val))

print('benchmark val', nnols_clf.score(X_val_, y_val))

print(best_clf.coef_ / np.sum(best_clf.coef_))
