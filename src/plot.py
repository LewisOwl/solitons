import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

from .paths import *
from .models import theory_c

# Plotting params
curve_skip = 200
resid_skip = 75
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

sec = plt.figure(figsize=(8, 10))
n0c = sec.add_axes([.1, .1, .8, .8])

resid_lims = (-5, 5)

def style_curve():
    curve.set_ylabel(r'$\frac{\eta}{\eta_0}$', fontsize=fontsize,
                     labelpad=labelpad)
    curve.set_xlim([-4, 4])
    curve.text(-3.9, 3.6, r'$(i)$')


def style_residuals():
    resid.set_xlabel(r'$\frac{x - ct}{L}$', fontsize=fontsize,
                     labelpad=labelpad)
    resid.set_ylabel(r'$δ$', fontsize=fontsize, labelpad=labelpad)
    resid.set_ylim(resid_lims)
    resid.set_xticks([-3, -2, -1, 0, 1, 2, 3])

    thetas = np.linspace(-4, 4, 1000)
    resid.plot(thetas, [1]*len(thetas), c='k', ls='--', zorder=2)
    resid.plot(thetas, [-1]*len(thetas), c='k', ls='--', zorder=2)
    resid.text(-3.9, resid_lims[1]*.87, r'$(ii)$')


def style_lag():
    lag.set_xlabel(r'$δ_{n}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_ylabel(r'$δ_{n-1}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_xlim(resid_lims)
    lag.set_ylim(resid_lims)
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
    lag.text(-4.9, 4.35, r'$(iv)$')


def style_occ(max_f):
    occ.tick_params(axis='y', direction='inout', left=False, right=False,
                    labelleft=False)
    occ.set_xlabel(r'$f$')
    occ.plot(np.linspace(0, 1, 100), [1]*100, ls='--', c='k')
    occ.set_xlabel(r'$f$')
    occ.plot(np.linspace(0, 1, 100), [-1]*100, ls='--', c='k')
    occ.text(max_f * 1.17, resid_lims[1]*.87, r'$(iii)$')
    occ.set_xlim([max_f * 1.2, 0])
    occ.set_ylim(resid_lims)


def plot_curve(psis, amplidues, y_err, offset, i_color):
    color = colors[i_color]
    curve.errorbar(psis[::curve_skip], amplidues[::curve_skip] + offset,
                   fmt='x', markersize=3, yerr=y_err[::curve_skip], capsize=1.5, color=color,
                   zorder=1)
    thetas = np.linspace(-4, 4, 1000)
    curve.plot(thetas, 1/np.cosh(thetas)**2 + offset, c='k', ls='--', zorder=2)


def plot_resid(thetas, residuals, i_color):
    color = colors[i_color]
    resid.scatter(thetas[::resid_skip], residuals[::resid_skip], s=3,
                  marker='x', color=color, zorder=1)


def plot_lag(residuals, i_color):
    color = colors[i_color]
    lag.scatter(residuals[:-1:lag_skip], residuals[1::lag_skip], marker='x',
                color=color, s=3, edgecolors='k', alpha=lag_alpha)


def calc_occ(all_residuals, bin_width):
    sx, ex = resid_lims
    occ_bins = np.arange(sx, ex, bin_width)
    bin_values = np.empty(len(occ_bins[:-1]))
    widths = np.empty(len(occ_bins[:-1]))
    for n, n_bin in enumerate(occ_bins[:-1]):
        low, upp = n_bin, occ_bins[n+1]
        widths[n] = upp - low
        bin_val = all_residuals[(low<all_residuals)*(all_residuals<upp)].size
        bin_values[n] = bin_val
    bin_values = bin_values / all_residuals.size
    norm_y = np.linspace(sx, ex, 500)
    norm_x = norm.pdf(norm_y, all_residuals.mean(), all_residuals.std())
    norm_scale = bin_width
    return occ_bins, bin_values, widths, norm_y, norm_x, norm_scale


def plot_occ(occ_bins, bin_values, widths, norm_y, norm_x, norm_scale):
    colors = cmap(np.linspace(0.2, 1, len(occ_bins)))
    for i_bin, bin_value in enumerate(bin_values):
        color = colors[int(abs(len(bin_values)/2-i_bin))]
        occ.barh(occ_bins[i_bin], bin_value, widths[i_bin], align='edge',
        facecolor=color, zorder=0)
    occ.plot(norm_scale*norm_x, norm_y, ls='--', c='grey', zorder=3)
    return np.max(bin_values)


def plot_n0c(n0s, n0_err, cs, c_errs, h, h_err, i):
    color = cm.rainbow(np.linspace(0, 1, 7))[i]
    n0c.errorbar(n0s, cs, xerr=n0_err, yerr=c_errs, color=color, fmt='x', capsize=1.5)
    xs = np.linspace(np.min(n0s), np.max(n0s), 100)
    ys, y_errs = theory_c(xs, n0_err[0], h, h_err)
    n0c.plot(xs, ys, color=color, ls='--')

def save_main(name):
    main.patch.set_alpha(0)
    main.savefig(name, bbox_inches='tight', pad_inches=0, format='png')
    main.patch.set_alpha(1)


def show_main():
    main.show()


def save_sec(name):
    sec.patch.set_alpha(0)
    sec.savefig(name, bbox_inches='tight', pad_inches=0, format='png')
    sec.patch.set_alpha(1)
