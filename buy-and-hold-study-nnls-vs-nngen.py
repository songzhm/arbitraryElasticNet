# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 9/22/19 12:28 PM
"""

import optuna
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

study_name = 'buy-and-hold-study-nnls-vs-nngen'  # Unique identifier of the study.

study = optuna.create_study(study_name=study_name,
                            storage='postgresql://ming:00000000@optuna.cr9g8xlyvsgj.us-west-2.rds.amazonaws.com:5432/optuna',
                            load_if_exists=True)


def objective(trial, p_, X_train_, y_train, X_test_, y_test, lowbo, upbo):
    lam_1 = trial.suggest_uniform('lam_1', 1e-8, 0.05)
    lam_2 = trial.suggest_uniform('lam_2', 1e-8, 1e+2)
    wvec_random_state = trial.suggest_int('wvec_random_state', 0, 10000)
    sigma_random_state = trial.suggest_int('sigma_random_state', 0, 10000)
    argen_clf = ARGEN(p_, lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec_random_state=wvec_random_state,
                      sigma_random_state=sigma_random_state)
    argen_clf.fit(X_train_, y_train)
    # nnols_clf = ARGEN(p_, 0, 0, lowbo, upbo, 0, 0)
    # nnols_clf.fit(X_train_, y_train)
    # print(nnols_clf.score(X_test_, y_test))
    return argen_clf.score(X_test_, y_test)


if __name__ == '__main__':
    study.optimize(lambda trial: objective(trial, p_, X_train_, y_train, X_test_, y_test, lowbo, upbo), n_trials=100000,
                   n_jobs=6)
