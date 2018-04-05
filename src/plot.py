import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

from .paths import *
from .models import theory_c


def calc_occ(all_residuals, bin_width, resid_lims):
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
