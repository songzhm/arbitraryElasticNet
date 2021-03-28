# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 9/24/19 10:20 PM
"""
# import packages
import optuna
from index_tracking.scripts.analysis_utils import *
from index_tracking.scripts.ARGEN import *

# configurations
UPDATE_FREQUENCY = 'buy-and-hold'

TRAINING_LENGTH = 12

TARGET_STOCK_NUMBER = 50

UPPER_BOUND = 0.6  # 0.8

LOWER_BOUND = 0.0082  # 0.0041

HP_STUDY_NAME = 'buy-and-hold-study-arls-vs-argen-up-0.6-n-50-s-100'  # Unique identifier of the study.

###

data = get_data()

X = data['X']
y = data['y']
dates = data['dates']
print(min(dates), max(dates))
time_vec = np.array(list(range(1, X.shape[0] + 1))) - 1
_, p = X.shape
training_val_index, testing_index = get_data_index(time_vec, UPDATE_FREQUENCY, TRAINING_LENGTH)
training_val_ind = training_val_index[0]
training_ind, val_ind = training_val_ind[0:-int(len(training_val_ind) * 0.2)], training_val_ind[
                                                                               -int(len(training_val_ind) * 0.2)::]
testing_ind = testing_index[0]
X_train = X[training_ind, :]
X_val = X[val_ind, :]
X_test = X[testing_ind, :]
y_train = y[training_ind]
y_val = y[val_ind]
y_test = y[testing_ind]

# feature select (portfolio stock selection)

fs_clf = FeatureSelectionRegressor(p, TARGET_STOCK_NUMBER)
fs_clf.fit(X_train, y_train)
fs_clf.score(X_test, y_test)
selected_feature_ind = np.where(fs_clf.coef_ != 0.0)[0]

# set up training/validation/testing data based on selected stocks
X_train_ = X_train[:, selected_feature_ind]
X_val_ = X_val[:, selected_feature_ind]
X_test_ = X_test[:, selected_feature_ind]

_, p_ = X_train_.shape

lowbo = np.ones(p_) * LOWER_BOUND
upbo = np.ones(p_) * UPPER_BOUND


# %%

# set up study for hyper-parameter tunning
study = optuna.create_study(study_name=HP_STUDY_NAME,
                            storage='postgresql://argen:argen@localhost:5433/argen',
                            load_if_exists=True)

def objective(trial, p_, X_train_, y_train, X_val_, y_val, lowbo, upbo):
    lam_1 = trial.suggest_uniform('lam_1', 1e-8, 0.05)
    lam_2 = trial.suggest_uniform('lam_2', 1e-8, 1e+2)
    wvec_random_state = trial.suggest_int('wvec_random_state', 0, 10000)
    sigma_random_state = trial.suggest_int('sigma_random_state', 0, 10000)
    argen_clf = ARGEN(p_, lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec_random_state=wvec_random_state,
                      sigma_random_state=sigma_random_state)
    argen_clf.fit(X_train_, y_train)
    return argen_clf.score(X_val_, y_val)


if __name__ == '__main__':

    study.optimize(lambda trial: objective(trial, p_, X_train_, y_train, X_val_, y_val, lowbo, upbo), n_trials=100,
                   n_jobs=10)

