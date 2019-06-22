import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dump, load
from skopt import forest_minimize

from generalized_elastic_net_solver_yujia import *

beta=np.array([0.85,0.85,0.85,0.85,0.85,0.85,0.85,0.85])
p=len(beta)
Ntrain=20
Ntval=20
Ntest=200
Xs=pickle.load( open( "Xs_eg2.p", "rb" ) )
ys=pickle.load( open( "ys_eg2.p", "rb" ) )



X = Xs[0]
y = ys[0]


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.8333, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

_, p = X_train.shape



HPO_PARAMS = {'n_calls':10,
              'n_random_starts': 10,
              'base_estimator': 'ET',
              'acq_func': 'EI',
              'xi': 0.02,
              'kappa': 1.96,
              'n_points': 10000,
              "n_jobs": -1,
              "verbose": True
              }

reg = GeneralizedElasticNetRegressor(beta=beta)
up = min(np.iinfo(np.int64).max, 2 ** p)-1
# define search space and parameters

space = [
    Real(0, 100,'uniform', name='lam_1'),
    Real(0, 100, 'uniform', name='lam_2'),
    Integer(0, up, name='sigma_choice'),
    Integer(0, up, name='w_choice'),
]

# define objective function, I used mse as score function for our solver,
# which can be found inn `generalized_elastic_net_solver.py`
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)
    coef=(reg.fit(X_train,y_train)).coef_
    return ((X_val.dot(beta)
                         -X_val.dot(coef))**2).mean(axis=None)

#     return np.mean(cross_val_score(reg, X_val, y_val, cv=3, n_jobs=3))


# callback function in each iteration, where I just print out the parameter that was tried in each iter
def monitor(res):
    dump(res, './hyper_optimization_results.pkl')
    print('run_parameters', str(res.x_iters[-1]))
    print('run_score', res.func_vals[-1])


import os

# test if result file is already exist, if yes, continue from the existing results, else, start from beginning
exists = os.path.isfile('./hyper_optimization_results.pkl')
print(exists)
if exists:

    results = load('./hyper_optimization_results.pkl')
    print('previous result exists')
    results = forest_minimize(objective, space, callback=[monitor],
                             x0=results.x_iters,
                             y0=results.func_vals,
                             **HPO_PARAMS
                             )

else:
    results = forest_minimize(objective, space, callback=[monitor],
                             **HPO_PARAMS)

print("Best score=%.4f" % results.fun)

# plot results
import matplotlib
# matplotlib.use('TkAgg')
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

plot_convergence(results)
plt.show()
plot_evaluations(results)
plt.show()
# plot_objective(results)
# plt.show()

reg.set_params(lam_1=results.x[0], lam_2=results.x[1], lowbo=None, upbo=None,
               ds=None,
               wvec=None, random_state=None,
                 sigma_choice=results.x[2], w_choice=results.x[3],
                 err_tol=1e-8, verbose=False, text_fr=200)
coef=(reg.fit(X_train,y_train)).coef_
print(reg.score(X_test))
