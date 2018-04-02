import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

from .paths import *


# Plotting params
marker_skip = 250
lag_skip = 25
lag_alpha = 0.3
fontsize = 13
labelpad = 3

# Choose plot colourn scheme
cmap = cm.winter
colors = cmap(np.linspace(0, 1, 10))

# Set up figures and axes
main = plt.figure(figsize=(8, 10))
curve = main.add_axes([0.1, 0.3, 0.55, 0.67])
resid = main.add_axes([0.1, 0.1, 0.55, 0.20], sharex=curve)
occ = main.add_axes([0.65, 0.1, 0.25, 0.20], sharey=resid)
lag = main.add_axes([0.65, 0.3, 0.25, 0.20])

cl = plt.figure(figsize=(8,8))
clax = cl.add_axes([0.1,0.1,0.85,0.85])


def style_curve():
    curve.set_ylabel(r'$\frac{\eta}{\eta_0}$', fontsize=fontsize,
                     labelpad=labelpad)
    curve.set_xlim([-4, 4])


def style_residuals():
    resid.set_xlabel(r'$\frac{x - ct}{L}$', fontsize=fontsize,
                     labelpad=labelpad)
    resid.set_ylabel(r'$δ$', fontsize=fontsize, labelpad=labelpad)
    resid.set_ylim([-5, 5])
    resid.set_xticks([-3, -2, -1, 0, 1, 2, 3])

    thetas = np.linspace(-4, 4, 1000)
    resid.plot(thetas, [1]*len(thetas), c='k', ls='--', zorder=2)
    resid.plot(thetas, [-1]*len(thetas), c='k', ls='--', zorder=2)


def style_lag():
    lag.set_xlabel(r'$δ_{n}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_ylabel(r'$δ_{n-1}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_xlim([-5, 5])
    lag.set_ylim([-5, 5])
    lag.set_xticks([-4, -2, 0, 2, 4])
    lag.set_yticks([-4, -2, 0, 2, 4])
    lag.tick_params(axis='x', direction='out', top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    lag.xaxis.set_label_position('top')
    lag.tick_params(axis='y', direction='out', right=True, left=False,
                    labelright=True, labelleft=False)
    lag.yaxis.set_label_position('right')

    thetas = np.linspace(-4, 4, 1000)
    lag.plot([-2]*len(thetas), np.linspace(-2, 2, len(thetas)), c='k', ls='--',
             zorder=2)
    lag.plot([2]*len(thetas), np.linspace(-2, 2, len(thetas)), c='k', ls='--',
             zorder=2)
    lag.plot(np.linspace(-2, 2, len(thetas)), [-2]*len(thetas), c='k', ls='--',
             zorder=2)
    lag.plot(np.linspace(-2, 2, len(thetas)), [2]*len(thetas), c='k', ls='--',
             zorder=2)


def style_occ(max_count):
    occ.set_xlim([max_count*1.2, 0])
    occ.tick_params(axis='y', direction='inout', left=False, right=False,
                    labelleft=False)
    occ.set_xlabel(r'$f$')
    occ.plot(np.linspace(0, 1, 100), [1]*100, ls='--', c='k')
    occ.set_xlabel(r'$f$')
    occ.plot(np.linspace(0, 1, 100), [-1]*100, ls='--', c='k')
    occ.plot(norm.pdf(np.linspace(-4, 4, 100))*max_count*(2*np.pi)**0.5,
             np.linspace(-4, 4, 100), ls='--', c='grey')


def plot_curve(psis, amplidues, n0, y_err, offset, i_color):
    color = colors[i_color]
    curve.errorbar(psis[::marker_skip], amplidues[::marker_skip] + offset,
                   fmt='x', markersize=3, yerr=y_err, capsize=1.5, color=color,
                   zorder=1)
    thetas = np.linspace(-4, 4, 1000)
    curve.plot(thetas, 1/np.cosh(thetas)**2 + offset, c='k', ls='--', zorder=2)


def plot_resid(thetas, residuals, i_color):
    color = colors[i_color]
    resid.scatter(thetas[::marker_skip], residuals[::marker_skip], s=3,
                  marker='x', color=color, zorder=1)


def plot_lag(residuals, i_color):
    color = colors[i_color]
    lag.scatter(residuals[:-1:lag_skip], residuals[1::lag_skip], marker='x',
                color=color, s=3, edgecolors='k', alpha=lag_alpha)


def plot_occ(all_residuals, bin_width):
    max_count = 0
    bins = np.arange(-5, 5, bin_width)
    bincols = cmap(np.linspace(0.2, 1, (len(bins)//2)+1))
    for i_bin, bin_start in enumerate(bins):
        greater = all_residuals > bin_start
        less = all_residuals < bin_start + bin_width
        mask = np.where(np.logical_and(greater, less))
        count = all_residuals[mask].size / all_residuals.size
        if count > max_count:
            max_count = count
        occ.barh(bin_start, count, bin_width, align='edge',
                 color=bincols[int(abs(len(bins)/2-i_bin))])
    return max_count

def plot_cl(vals, errs, i_color):
    c, l, c_t, l_t = vals
    c_err, l_err, c_t_err, l_t_err = errs
    color = colors[i_color]
    clax.errorbar(c, l, xerr=c_err, yerr=l_err, fmt='x', markersize=3,
     capsize=1.5, color=color, zorder=1)
    clax.errorbar(c_t, l_t, xerr=c_t_err, yerr=l_t_err,  fmt='x', markersize=3,
     capsize=1.5, color=color, zorder=1)

def save_main(name):
    main.savefig(name)


def show_main():
    main.show()

def save_cl(name):
    cl.savefig(name)

def show_cl():
    cl.show()
