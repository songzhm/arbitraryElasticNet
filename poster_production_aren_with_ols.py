from utility_function import *
import numpy as np
from ARGEN import *
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd


mse_score = make_scorer(mean_squared_error, greater_is_better=False)


def mydateparser(x): return pd.datetime.strptime(x, "%m/%d/%y")


sp500_all = pd.read_csv('sandp500/sp500_pct.csv', index_col='Date',
                        parse_dates=['Date.1'], date_parser=mydateparser)

sp500_all.index = sp500_all['Date.1']

constituents_names = sp500_all.columns.tolist()

constituents_names = [
    x for x in constituents_names if x not in ['Date', 'Date.1', 'SP500']]

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

# with non-negative ols

zero_threshold = 1.01e-6
portfolio_size = 45
training_month = 6

dates = sp500_all['Date.1'][training_month * 21 + 1:]

print('update_frequency, cost_factor, target_number, training_month, lam_1, \
    lam_2, c_r,a_r,a_v,d_te,d_tev,m_tr,mse')

log('poster_experiments_aren_ols.txt',
    'update_frequency, cost_factor, target_number, training_month, lam_1, \
        lam_2, c_r,a_r,a_v,d_te,d_tev,m_tr,mse')


results_return = []
results_coefs = []

true = y[training_month * 21 + 1:]

for update_frequency in \
        ['buy-and-hold', 'annually', 'semi-annually', 'quarterly']:

    training_index, testing_index = get_data_index(
        time_vec, update_frequency, training_month)

    nps = len(training_index)

    pred = np.array([])
    coefs = []

    for i in range(nps):

        training_ind = training_index[i]
        testing_ind = testing_index[i]
        X_train = X[training_ind, :]
        X_test = X[testing_ind, :]
        y_train = y[training_ind]
        y_test = y[testing_ind]
        X_train = X_train
        X_test = X_test
        reg = GeneralizedElasticNetRegressor(beta=beta, tune_lam_1=True,
                                             target_number=portfolio_size,
                                             wvec=np.ones(p),
                                             sigma_ds=np.ones(p))
        reg.fit(X_train, y_train)
        port = np.where(reg.coef_ > zero_threshold)[0]
        X_train_ = X_train[:, port.tolist()]
        X_test_ = X_test[:, port.tolist()]
        coefs_ = reg.coef_
        coefs_[coefs_ <= zero_threshold] = 0
        coefs.append(coefs_ / sum(coefs_))
        p_ = X_test_.shape[1]
        lowbo_ = np.repeat(0, p_)
        upbo_ = np.repeat(float('inf'), p_)

        ols_reg = GeneralizedElasticNetRegressor(
            beta=np.ones(p_), lowbo=lowbo_, upbo=upbo_)
        ols_reg.fit(X_train_, y_train)
        coef = ols_reg.coef_
        coef = coef / np.sum(coef)

        # adjust for transaction cost
        X_test_[0, :] = X_test_[0, :] + transaction_cost
        X_test_[:, coef < 0] = X_test_[:, coef < 0] + short_cost
        pred_ = ols_reg.predict(X_test_)

        pred = np.append(pred, pred_)

    coefs = np.array(coefs).T

    results_return.append(pred)
    results_coefs.append(coefs.T)


    print(
        '{set0}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
        '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
        .format(set0=update_frequency,
                set2=portfolio_size,
                set3=training_month,
                set4=reg.lam_1,
                set5=reg.lam_2,
                num0=calculate_cumulative_return(pred),
                num1=calculate_annual_average_return(pred),
                num2=calculated_annual_volatility(pred),
                num3=calculate_daily_tracking_error(pred, true),
                num4=calculate_daily_tracking_error_volatility(pred, true),
                num5=calculate_monthly_average_turnover(
                    coefs, update_frequency),
                num6=mean_squared_error(true, pred)
                )
    )
    log('poster_experiments_aren_ols.txt', '{set0}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
        '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
        .format(set0=update_frequency,
                set2=portfolio_size,
                set3=training_month,
                set4=reg.lam_1,
                set5=reg.lam_2,
                num0=calculate_cumulative_return(pred),
                num1=calculate_annual_average_return(pred),
                num2=calculated_annual_volatility(pred),
                num3=calculate_daily_tracking_error(pred, true),
                num4=calculate_daily_tracking_error_volatility(pred, true),
                num5=calculate_monthly_average_turnover(
                    coefs, update_frequency),
                num6=mean_squared_error(true, pred)
                ))

results_return.append(true)

print('{set0}, {set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
      '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
      .format(set0='NA',
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
log('poster_experiments_aren_ols.txt', '{set0},{set2}, {set3}, {set4}, {set5}, {num0:.2%},{num1:.2%},{num2:.2%},'
    '{num3:.3%},{num4:.3%},{num5:.2%},{num6: .2e}'
    .format(set0='NA',
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

results_return = np.array(results_return)
results_coefs = np.array(results_coefs)

np.save('AREN_OLS_returns', results_return)
np.save('AREN_OLS_coefs', results_coefs)
np.save('AREN_OLS_dates', dates)
