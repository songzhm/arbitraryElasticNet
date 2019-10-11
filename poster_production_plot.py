# %%

import numpy as np

results_return = np.load('AREN_OLS_returns.npy')
results_return = results_return.T
results_coef = np.load('AREN_OLS_coefs.npy')
results_dates = np.load('AREN_OLS_dates.npy')

results_return_no_ols = np.load('AREN_NO_OLS_returns.npy')
results_return_no_ols = results_return_no_ols.T
# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# sides = ('left', 'right', 'top', 'bottom')
# nolabels = {s: False for s in sides}
# nolabels.update({'label%s' % s: False for s in sides})

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4))
ax1.matshow(results_coef[0].T, cmap=cm.Blues, interpolation='none', aspect='auto')
ax2.matshow(results_coef[1].T, cmap=cm.Blues, interpolation='none', aspect='auto')
ax3.matshow(results_coef[2].T, cmap=cm.Blues, interpolation='none', aspect='auto')
img4 = ax4.matshow(results_coef[3].T, cmap=cm.Blues, interpolation='none', aspect='auto')
for ax in (ax2, ax3, ax4):
    # ax.tick_params(axis='both', which='both', **nolabels)
    ax.set(yticks=[])
    ax.xaxis.set_ticks_position('bottom')

ax1.set(xticks=[1])
ax1.set_ylabel('Individual Stock Index')
ax1.set_xlabel('Re-balance portfolios')

divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size='5%', pad=0)
plt.colorbar(img4, cax=cax, ax=[ax1, ax2, ax3, ax4], format='%.2f')
fig.show()
fig.savefig('weights_aren_ols.png', dpi=800)
# %%
from matplotlib.dates import DateFormatter

cum_returns = np.cumprod(1 + results_return, axis=0) - 1
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4))

ax1.plot_date(x=results_dates, y=cum_returns[:, -1], linestyle='-', marker='None')
ax1.plot_date(x=results_dates, y=cum_returns[:, :-1], linestyle=':', marker='None')

ax1.set_xticklabels(results_dates)
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax1.legend(['Buy-and-Hold', 'Annually', 'Semi-annually', 'Quarterly', 'S&P500'])
fig.show()
fig.savefig('cumulative_return_are_ols.png', dpi=800)

# %%
from matplotlib.dates import DateFormatter

cum_returns = np.cumprod(1 + results_return, axis=0) - 1
cum_returns_no_ols = np.cumprod(1 + results_return_no_ols, axis=0) - 1
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 5.5))
types = ['Buy-and-Hold', 'Annually', 'Semi-annually', 'Quarterly', 'S&P500']
for i, ax in enumerate((ax1, ax2, ax3, ax4)):
    ax.plot_date(x=results_dates, y=cum_returns[:, i], linestyle=':', marker='None')
    ax.plot_date(x=results_dates, y=cum_returns_no_ols[:, i], linestyle=':', marker='None')
    ax.plot_date(x=results_dates, y=cum_returns[:, -1], linestyle='-', marker='None', c='k')
    ax.set_ylabel(types[i])
    if i == 0:
        ax.legend(['AREN', 'AREN + OLS', 'S&P500'], frameon=False, ncol=3)

    if i <= 2:
        ax.set(xticks=[])
        ax.spines["bottom"].set_visible(False)
    if i <= 3:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

ax.set_xticklabels(results_dates)
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

fig.show()
fig.savefig('cumulative_return_aren_ols_vs_no_ols.png', dpi=800)

# %%
results_return = np.load('AREN_OLS_portfolio_size_returns.npy')
# results_return = results_return.T
results_coef = np.load('AREN_OLS_portfolio_size_coefs.npy')
results_dates = np.load('AREN_OLS_portfolio_size_dates.npy')
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4), dpi=800)
update_periods = ['bh', 'a', 'sa', 'q']
stock_numbers = [25, 45, 65, 85]
linestyles = [':', '--', '-.', '-']
markers = ['s', 'x', '*', 'v']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(4):
    returns = results_return[i]
    update_frequency = update_periods[i]

    for j in range(4):
        data = returns[j]
        portfolio_size = stock_numbers[j]
        ax1.plot_date(x=results_dates, y=np.cumprod(1 + data) - 1, linestyle=linestyles[i],
                      marker=markers[j], ms=0.3, c=colors[i], linewidth=0.3,
                      label='{} ({})'.format(update_frequency, portfolio_size))
        ax1.set_label('{} portfolio with {} stocks'.format(update_frequency, portfolio_size))

ax1.plot_date(x=results_dates, y=np.cumprod(1 + results_return[-1]) - 1, linestyle='-',
              marker='', c='k', linewidth=1,
              label='S&P500')
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(ncol=4,frameon=False)
fig.show()
fig.savefig('cumulative_return_with_different_portfolio_size.png', dpi=800)
