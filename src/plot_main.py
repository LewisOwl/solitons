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

resid_lims = (-5, 5)

def style_curve(ax, xlims, label):
    ax.set_ylabel(r'$\frac{\eta}{\eta_0}$', fontsize=fontsize,
                     labelpad=labelpad)
    ax.set_xlim(xlims)
    ax.text(-3.9, 3.6, label)


def style_residuals(ax, resid_lims, label):
    ax.set_xlabel(r'$\frac{x - ct}{L}$', fontsize=fontsize,
                     labelpad=labelpad)
    ax.set_ylabel(r'$δ$', fontsize=fontsize, labelpad=labelpad)
    ax.set_ylim(resid_lims)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])

    thetas = np.linspace(-4, 4, 1000)
    ax.plot(thetas, [1]*len(thetas), c='k', ls='--', zorder=2)
    ax.plot(thetas, [-1]*len(thetas), c='k', ls='--', zorder=2)
    ax.text(-3.9, resid_lims[1]*.87, label)


def style_lag(ax, resid_lims, label):
    ax.set_xlabel(r'$δ_{n}$', fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel(r'$δ_{n-1}$', fontsize=fontsize, labelpad=labelpad)
    ax.set_xlim(resid_lims)
    ax.set_ylim(resid_lims)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(axis='x', direction='out', top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='y', direction='out', right=True, left=False,
                    labelright=True, labelleft=False)
    ax.yaxis.set_label_position('right')

    thetas = np.linspace(-4, 4, 1000)
    ax.plot([-2]*len(thetas), np.linspace(-2, 2, len(thetas)), c='k', ls='--',
             zorder=2)
    ax.plot([2]*len(thetas), np.linspace(-2, 2, len(thetas)), c='k', ls='--',
             zorder=2)
    ax.plot(np.linspace(-2, 2, len(thetas)), [-2]*len(thetas), c='k', ls='--',
             zorder=2)
    ax.plot(np.linspace(-2, 2, len(thetas)), [2]*len(thetas), c='k', ls='--',
             zorder=2)
    ax.text(-4.9, 4.35, label)


def style_occ(ax, max_f, resid_lims, label):
    ax.tick_params(axis='y', direction='inout', left=False, right=False,
                    labelleft=False)
    ax.set_xlabel(r'$f$')
    ax.plot(np.linspace(0, 1, 100), [1]*100, ls='--', c='k')
    ax.set_xlabel(r'$f$')
    ax.plot(np.linspace(0, 1, 100), [-1]*100, ls='--', c='k')
    ax.text(max_f * 1.17, resid_lims[1]*.87, label)
    ax.set_xlim([max_f * 1.2, 0])
    ax.set_ylim(resid_lims)

def plot_curve(ax, psis, amplidues, y_err, offset, i_color):
    color = colors[i_color]
    ax.errorbar(psis[::curve_skip], amplidues[::curve_skip] + offset,
                   fmt='x', markersize=3, yerr=y_err[::curve_skip], capsize=1.5, color=color,
                   zorder=1)
    thetas = np.linspace(-4, 4, 1000)
    ax.plot(thetas, 1/np.cosh(thetas)**2 + offset, c='k', ls='--', zorder=2)


def plot_resid(ax, thetas, residuals, i_color):
    color = colors[i_color]
    ax.scatter(thetas[::resid_skip], residuals[::resid_skip], s=3,
                  marker='x', color=color, zorder=1)


def plot_lag(ax, residuals, i_color):
    color = colors[i_color]
    ax.scatter(residuals[:-1:lag_skip], residuals[1::lag_skip], marker='x',
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


def plot_occ(ax, occ_bins, bin_values, widths, norm_y, norm_x, norm_scale, horizontal=True):
    colors = cmap(np.linspace(0.2, 1, len(occ_bins)))
    for i_bin, bin_value in enumerate(bin_values):
        color = colors[int(abs(len(bin_values)/2-i_bin))]
        ax.barh(occ_bins[i_bin], bin_value, widths[i_bin], align='edge',
        facecolor=color, zorder=0)
    ax.plot(norm_scale*norm_x, norm_y, ls='--', c='grey', zorder=3)
    return np.max(bin_values)


def save_main(name):
    main.patch.set_alpha(0)
    main.savefig(name, bbox_inches='tight', pad_inches=0, format='png')
    main.patch.set_alpha(1)


def show_main():
    main.show()
