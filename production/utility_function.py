
# utility functions
import numpy as np
import pandas as pd


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
        train_start = train_end - training_size

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
    assert len(portfolio_return) == len(
        index_return), "two vectors need to be the same length"
    excess_return = portfolio_return - index_return
    res = np.std(excess_return)
    return res


def calculate_daily_tracking_error_volatility(portfolio_return, index_return):
    assert len(portfolio_return) == len(
        index_return), "two vectors need to be the same length"
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


def mydateparser(x): return pd.datetime.strptime(x, "%m/%d/%y")

ZERO_THRESHOLD = 1.01e-8

def get_data():
    sp500_all = pd.read_csv('./production/sp500_pct.csv', index_col='Date',
                            parse_dates=['Date.1'], date_parser=mydateparser)

    sp500_all.index = sp500_all['Date.1']

    constituents_names = sp500_all.columns.tolist()

    constituents_names = [
        x for x in constituents_names if x not in ['Date', 'Date.1', 'SP500']]

    constituents = sp500_all[constituents_names]

    sp500 = sp500_all['SP500']

    X = constituents.values

    y = sp500.values

    dates = sp500_all['Date.1']

    return {'X': X, 'y': y, 'dates': dates}

