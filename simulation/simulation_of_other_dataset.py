import os
from statistics import median

import numpy as np
from scipy.stats import sem
from skopt import dump, load, dummy_minimize
from skopt.utils import use_named_args

from .driver import Driver

# Simulate data set one example and solve the result using one model

################
# Modification

example = '8'  # choose from 1 to 8
method = 'NLS'  # choose from 'NLS', 'ARLS', 'NL', 'ARL', 'NR', 'ARR', 'NEN', 'AREN', 'NGL', 'ARGL', 'NGR', 'ARGR', 'NLEN', 'ARLEN', 'NREN', 'ARREN', 'NGEN','ARGEN'
# ##############

dr = Driver(example, method)
dr.driver()
data_dict = dr.data_dict
state = dr.state
reg = dr.reg
space = dr.space
fix_paras = dr.fix_paras
tune_paras = dr.tune_paras
f = open('eg' + example + '_' + method + '.txt', "w")
coefs = []
test_scores = []
iterations = []
for i in range(state):
    data = data_dict[i]
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    X_test = data['X_test']
    if method in ['NLS', 'ARLS']:
        coef = (reg.fit(X_train, y_train)).coef_
        score = reg.score(X_test)
        test_scores += [score]
        coefs += [coef]
    else:
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**fix_paras, **params)
            coef = (reg.fit(X_train, y_train)).coef_
            return reg.score(X_val)


        def monitor(res):
            dump(res, './tune_results_eg' + example + '_' + method + '_state' + str(i) + '.pkl')
            print('run_parameters', str(res.x_iters[-1]))
            print('run_score', res.func_vals[-1])


        HPO_PARAMS = {'n_calls': 1280,
                      "verbose": True
                      }
        # test if result file is already exist, if yes, continue from the existing results, else, start from beginning
        exists = os.path.isfile('./tune_results_eg' + example + '_' + method + '_state' + str(i) + '.pkl')
        print(exists)
        if exists:
            results_ = load('./tune_results_eg' + example + '_' + method + '_state' + str(i) + '.pkl')
            print('previous result exists')
            results = dummy_minimize(objective, space, callback=[monitor],
                                     x0=results_.x_iters,
                                     y0=results_.func_vals,
                                     **HPO_PARAMS
                                     )
        else:
            results = dummy_minimize(objective, space, callback=[monitor],
                                     **HPO_PARAMS)

        print("Best score=%.4f" % results.fun)

        for j, key in enumerate(tune_paras.keys()):
            tune_paras[key] = results.x[j]

        reg.set_params(ds=None, random_state=None,
                       err_tol=1e-8, verbose=False, text_fr=200, **tune_paras, **fix_paras)
        coef = (reg.fit(X_train, y_train)).coef_
        score = reg.score(X_test)
        print(score)
        iteration = len(results.x_iters)
        test_scores += [score]
        coefs += [coef]
        iterations += [iteration]
f.write("Median of test scores: %f\n" % median(test_scores))
f.write("Standard error of test scores: %f\n" % sem(test_scores))
f.write("Average number of zero coefficients: %f\n" % np.mean(
    [np.sum((coefs[h] >= 0) & (coefs[h] <= 1e-5)) for h in range(len(coefs))]))
f.write("Test scores:\n")
[f.write("%f " % item) for item in test_scores]
f.write("\nIterations:\n")
[f.write("%f " % item) for item in iterations]
f.write("\nCoefs:\n")
for list in coefs:
    f.writelines("%f " % item for item in list)
    f.write("\n")
f.close()
