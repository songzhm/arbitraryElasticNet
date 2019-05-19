# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 2019-05-11 16:13
"""

# %%
import numpy as np

from generalized_elsatic_net_solver import GeneralizedElasticNetSover

solver = GeneralizedElasticNetSover()

# %%
lam_1 = 0.01
lam_2 = 0.01

# %%
# generate dataset
n = 1000
p = 4
np.random.seed(0)
other_variables = np.random.rand(n, p - 1)
b = np.array([1 / 6, 1 / 6, 1 / 2])
last_variable = other_variables @ b + 5 / 6 * np.random.rand(n)
last_variable = last_variable.reshape((n, 1))
x = np.concatenate((last_variable, other_variables), axis=1)

betas = np.array([0, 2, 3, 0])

y = x @ betas + np.random.rand(n)

# %%
lam_1 = 1000

lam_2 = 10

Sigma = np.diag([1] * p)

wvec = np.ones(p)

lowbo = -1 * np.ones(p)

upbo = 5 * np.ones(p)

betas_est = solver.solve(x, y, lam_1, lam_2, lowbo, upbo, wvec, Sigma)

print(betas_est)

# %%

n = 100
p = 10
np.random.seed(0)
x = np.random.rand(n, p)

betas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0, 0])

y = x @ betas + np.random.rand(n)


Sigma = np.diag([1] * p)

wvec = np.ones(p)

lowbo = -1 * np.ones(p)

upbo = 10 * np.ones(p)


lam_1s = [i for i in range(2050)]

lam_2s = [0, 1, 10, 100, 1000]

for lam_2 in lam_2s:
    betas_est_set = np.zeros((len(lam_1s), p))
    for ind, lam_1 in enumerate(lam_1s):
        betas_est = solver.solve(x, y, lam_1, lam_2, lowbo, upbo, wvec, Sigma)
        print('lam_1={}, lam_2={}'.format(lam_1, lam_2))
        betas_est_set[ind, :] = betas_est

    print(betas_est_set)

    import pandas as pd
    import seaborn as sns;sns.set()
    sns.set_style("white")
    import matplotlib.pyplot as plt

    plot_data = pd.DataFrame(betas_est_set, columns=['beta_{}'.format(i) for i in range(10)])
    plot_data['lam_1'] = lam_1s
    ax = sns.lineplot(x='lam_1', y='value', hue='variable',
                      data=pd.melt(plot_data, ['lam_1']))
    ax.set_title('lam_2={}, n={}'.format(lam_2, n))
    plt.show()
    fig = ax.get_figure()
    fig.savefig('lam_2_{}_n_{}.png'.format(lam_2, n))
