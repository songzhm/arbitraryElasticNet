# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 9/19/19 3:57 PM
"""
#%%
from production.ARGEN import *
from production.utility_function import *
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import cross_val_score

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

target_number = 50

fs_clf = FeatureSelectionRegressor(p, target_number)

fs_clf.fit(X_train, y_train)
fs_clf.score(X_test, y_test)

selected_feature_ind = np.where(fs_clf.coef_ != 0.0)[0]

X_train_ = X_train[:, selected_feature_ind]
X_test_ = X_test[:, selected_feature_ind]
# X_test_[0, :] = X_test_[0, :] + transaction_cost

_, p_ = X_train_.shape

lowbo = np.zeros(p_)
upbo = np.ones(p_) * np.inf

# lowbo = np.ones(p_)*1/target_number
# upbo = np.ones(p_) * 0.03
#%%

clf = ARGEN(p_,0.01, 0.00251047706973487, lowbo, upbo, 6214, 5412)

clf.fit(X_train_, y_train)
clf.coef_


#%%
#NNARLS
# import pickle
# with open('searchcv.pkl', 'rb') as fid:
#     searchcv = pickle.load(fid)
print("val. score: %s" % np.average(cross_val_score(clf, X_train_, y_train, cv=3)))
print("test score: %s" % clf.score(X_test_, y_test))
nnols_clf = ARGEN(p_, 0, 0, lowbo, upbo, 0, 0)
nnols_clf.fit(X_train_, y_train)
print("benchmark val. score: %s" % np.average(cross_val_score(nnols_clf, X_train_, y_train, cv=3)))
print("benchmark test score: %s" % nnols_clf.score(X_test_, y_test))

#%%
# print('benchmark coef')
# print(nnols_clf.coef_)
# import pickle
# with open('searchcv.pkl', 'wb') as fid:
#     pickle.dump(searchcv, fid)
# # To deserialize estimator later
# with open('searchcv.pkl', 'rb') as fid:
#     searchcv = pickle.load(fid)

#%%
import matplotlib.pyplot as plt
pred_0 = fs_clf.predict(X_test)
pred_1 = clf.predict(X_test_)
pred_2 = nnols_clf.predict(X_test_)
cum_pred_0 = np.cumprod(1+pred_0)-1
cum_pred_1 = np.cumprod(1+pred_1)-1
cum_pred_2 = np.cumprod(1+pred_2)-1
cum_true = np.cumprod(1+y_test) - 1
true = y_test

plt.figure()
plt.plot_date(dates[testing_ind], cum_pred_0, linestyle=':', markersize=1, label='NNL')
plt.plot_date(dates[testing_ind], cum_pred_1, linestyle='--', markersize=1, label='NNL+ARGEN')
plt.plot_date(dates[testing_ind], cum_pred_2, linestyle='-.', markersize=1, label='NNL+NNOLS')
plt.plot_date(dates[testing_ind], cum_true, linestyle='-', markersize=1, label='sp500')
plt.legend()
plt.show()

#%%
print('frequency, training month, cr, avr, av, te, tev, mse')

print(
        '{set0}, {set3}, {num0:.2%},{num1:.2%},{num2:.2%},'
        '{num3:.3%},{num4:.3%},{num6: .2e}'
        .format(set0=update_frequency,
                set3=training_month,
                num0=calculate_cumulative_return(pred_0),
                num1=calculate_annual_average_return(pred_0),
                num2=calculated_annual_volatility(pred_0),
                num3=calculate_daily_tracking_error(pred_0, true),
                num4=calculate_daily_tracking_error_volatility(pred_0, true),
                num6=mean_squared_error(true, pred_0)
                )
    )

print(
        '{set0}, {set3}, {num0:.2%},{num1:.2%},{num2:.2%},'
        '{num3:.3%},{num4:.3%},{num6: .2e}'
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
        '{set0}, {set3}, {num0:.2%},{num1:.2%},{num2:.2%},'
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

#%%

nnl_coefficients = fs_clf.coef_[selected_feature_ind]
nnl_coefficients = nnl_coefficients/np.sum(nnl_coefficients)

nnl_argen_lasso_coefficients = clf.coef_
nnl_argen_lasso_coefficients = nnl_argen_lasso_coefficients/np.sum(nnl_argen_lasso_coefficients)

nnl_nols_coefficients = nnols_clf.coef_
nnl_nols_coefficients = nnl_nols_coefficients/np.sum(nnl_nols_coefficients)

import pandas as pd
import seaborn as sns

coefs_data = pd.DataFrame({'stock_index':selected_feature_ind,'nnl': nnl_coefficients, 'nnl_argen': nnl_argen_lasso_coefficients, 'nnl_nols': nnl_nols_coefficients})

coefs_data = coefs_data.melt(id_vars=['stock_index'], value_vars=['nnl', 'nnl_argen', 'nnl_nols'], var_name='model', value_name='coef')

# ax = sns.swarmplot(x="model", y="coef", data=coefs_data, color=".05")
ax = sns.boxplot(x="model", y="coef", data=coefs_data)
plt.show()


sns.distplot(nnl_coefficients)
sns.distplot(nnl_nols_coefficients)
sns.distplot(nnl_argen_lasso_coefficients)
plt.show()

