# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 9/21/19 10:51 PM
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

update_frequency = 'annually'


training_month = 6

training_index, testing_index = get_data_index(
    time_vec, update_frequency, training_month)

nps = len(training_index)
#%%
for i in range(nps):
    training_ind = training_index[i]
    testing_ind = testing_index[i]

    X_train = X[training_ind, :]
    X_test = X[testing_ind, :]
    y_train = y[training_ind]
    y_test = y[testing_ind]

    target_number = 50

    fs_clf = FeatureSelectionRegressor(p, target_number)

    fs_clf.fit(X_train, y_train)
    fs_clf.score(X_test, y_test)
    pred_0 = fs_clf.predict(X_test)

    selected_feature_ind = np.where(fs_clf.coef_ != 0.0)[0]

    X_train_ = X_train[:, selected_feature_ind]
    X_test_ = X_test[:, selected_feature_ind]
    X_test_[0, :] = X_test_[0, :] + transaction_cost

    _, p_ = X_train_.shape

    lowbo = np.ones(p_) * 0.0
    upbo = np.ones(p_) * 0.2

    searchcv = BayesSearchCV(
        ARGEN(p_, 0, 0, lowbo, upbo, 0, 0),
        search_spaces={
            'lam_1': Real(1e-8, 0.5, prior='uniform'),
            'lam_2': Real(11e-8, 1e+2, prior='uniform'),
            'wvec_random_state': list(range(10000)),
            'sigma_random_state': list(range(10000))},
        n_iter=50,
        n_points=5,
        cv=3,
        random_state=42,
        n_jobs=6,
        verbose=2
    )


    searchcv.fit(X_train_, y_train)

    print("val. score: %s" % searchcv.best_score_)
    print("test score: %s" % searchcv.score(X_test_, y_test))

    import pickle
    with open('50_annualy_searchcv_port_{}.pkl'.format(i), 'wb') as fid:
        pickle.dump(searchcv, fid)
    # # To deserialize estimator later
    # with open('searchcv.pkl', 'rb') as fid:
    #     searchcv = pickle.load(fid)



#%%
pred_0 = np.array([])
pred_1 = np.array([])
pred_2 = np.array([])
true = np.array([])
for i in range(nps):

    training_ind = training_index[i]

    testing_ind = testing_index[i]

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

    lowbo = np.ones(p_) * 0.0
    upbo = np.ones(p_) * 0.2

    import pickle
    with open('50_annualy_searchcv_port_{}.pkl'.format(i), 'rb') as fid:
        searchcv = pickle.load(fid)

    nnols_clf = ARGEN(p_, 0, 0, lowbo, upbo, 0, 0)
    nnols_clf.fit(X_train_, y_train)

    pred_0_ = fs_clf.predict(X_test)
    pred_1_ = searchcv.predict(X_test_)
    pred_2_ = nnols_clf.predict(X_test_)

    pred_0 = np.append(pred_0, pred_0_)
    pred_1 = np.append(pred_1, pred_1_)
    pred_2 = np.append(pred_2, pred_2_)
    true = np.append(true, y_test)




#%%
import matplotlib.pyplot as plt
cum_pred_0 = np.cumprod(1+pred_0)-1
cum_pred_1 = np.cumprod(1+pred_1)-1
cum_pred_2 = np.cumprod(1+pred_2)-1
cum_true = np.cumprod(1+true) - 1


plt.figure()
plt.plot_date(dates[testing_index[0][0]::], cum_pred_0, linestyle=':', markersize=1, label='NNL')
plt.plot_date(dates[testing_index[0][0]::], cum_pred_1, linestyle='--', markersize=1, label='NNL+ARGEN')
plt.plot_date(dates[testing_index[0][0]::], cum_pred_2, linestyle='-.', markersize=1, label='NNL+NNOLS')
plt.plot_date(dates[testing_index[0][0]::], cum_true, linestyle='-', markersize=1, label='sp500')
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

