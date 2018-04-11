import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

# Choose plot colour scheme
cmap = cm.winter
colors = cmap(np.linspace(0.2, 1, 10))

fontsize = 13
labelpad = 3

def gen_main_axes():
    fig = plt.figure(figsize=(8, 10))
    curve = fig.add_axes([0.1, 0.3, 0.55, 0.67])
    resids = fig.add_axes([0.1, 0.1, 0.55, 0.20], sharex=curve)
    occ = fig.add_axes([0.65, 0.1, 0.25, 0.20], sharey=resids)
    lag = fig.add_axes([0.65, 0.3, 0.25, 0.20])
    return fig, curve, resids, occ, lag


def gen_sec_axes(xlab, ylab):
    fig = plt.figure(figsize=(10, 10))
    back = fig.add_axes([0.05, 0.05, 0.85, 0.94])
    ax1 = fig.add_axes([0.05, 0.75, 0.4, 0.22])
    ax2 = fig.add_axes([0.5, 0.75, 0.4, 0.22])
    ax3 = fig.add_axes([0.05, 0.5, 0.4, 0.22])
    ax4 = fig.add_axes([0.5, 0.5, 0.4, 0.22])
    ax5 = fig.add_axes([0.05, 0.25, 0.4, 0.22])
    ax6 = fig.add_axes([0.5, 0.25, 0.4, 0.22])
    resids = fig.add_axes([0.05, 0.0, 0.4, 0.2])
    occ = fig.add_axes([0.45, 0.0, 0.2, 0.2], sharey=resids)
    lag = fig.add_axes([0.7, 0.0, 0.2, 0.2])
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    # Edit the back to be invisible
    back.set_facecolor('none')
    back.spines['top'].set_color('none')
    back.spines['bottom'].set_color('none')
    back.spines['left'].set_color('none')
    back.spines['right'].set_color('none')
    back.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    back.xaxis.set_label_position('top')
    back.set_ylabel(ylab, labelpad=10, fontsize=fontsize)
    back.set_xlabel(xlab, labelpad=1, fontsize=fontsize)
    return fig, back, axs, resids, occ, lag


def style_main_axes(curve, resids, occ, lag, r_lims, max_occ):
    r_xlims, r_ylims = r_lims
    # Style main curve
    curve.set_ylabel(r'Normalised Amplitude $\frac{\eta}{\eta_0} + \sigma$', fontsize=fontsize,
                     labelpad=labelpad)
    curve.set_xlim(r_xlims)
    curve.text(r_xlims[0]*0.98, 3.6, r'$(a)$')
    # Style residual plot
    resids.set_xlabel(r'$\frac{x - c \cdot t}{L}$', fontsize=fontsize,
                     labelpad=labelpad)
    resids.set_ylabel(r'Norm. Residual $δ$', fontsize=fontsize, labelpad=labelpad)
    resids.set_ylim(r_ylims)
    resids.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    xs = np.linspace(*r_xlims, 100)
    resids.plot(xs, [1]*len(xs), c='k', ls='--', zorder=2)
    resids.plot(xs, [-1]*len(xs), c='k', ls='--', zorder=2)
    resids.text(r_xlims[0], r_ylims[1]*.87, r'$(b)$')
    # Style occurence plot
    occ.tick_params(axis='y', direction='inout', left=False, right=False,
                    labelleft=False)
    occ.set_xlabel(r'$Occurence$', fontsize=fontsize,
                     labelpad=labelpad)
    occ.plot(np.linspace(0, 1, 100), [1]*100, ls='--', c='k')
    occ.plot(np.linspace(0, 1, 100), [-1]*100, ls='--', c='k')
    occ.text(max_occ * 1.17, r_ylims[1]*.87, r'$(c)$')
    occ.set_xlim([max_occ * 1.2, 0])
    occ.set_ylim(r_ylims)
    # Style lag plot
    lag.set_xlabel(r'$δ_{n}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_ylabel(r'$δ_{n-1}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_xlim(r_ylims)
    lag.set_ylim(r_ylims)
    lag.set_xticks([-4, -2, 0, 2, 4])
    lag.set_yticks([-4, -2, 0, 2, 4])
    lag.tick_params(axis='x', direction='out', top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    lag.xaxis.set_label_position('top')
    lag.tick_params(axis='y', direction='out', right=True, left=False,
                    labelright=True, labelleft=False)
    lag.yaxis.set_label_position('right')

    lag.plot([-2]*len(xs), np.linspace(-2, 2, len(xs)), c='k', ls='--',
             zorder=2)
    lag.plot([2]*len(xs), np.linspace(-2, 2, len(xs)), c='k', ls='--',
             zorder=2)
    lag.plot(np.linspace(-2, 2, len(xs)), [-2]*len(xs), c='k', ls='--',
             zorder=2)
    lag.plot(np.linspace(-2, 2, len(xs)), [2]*len(xs), c='k', ls='--',
             zorder=2)
    lag.text(-4.9, 4.35, r'$(d)$')


def style_sec_axes(axs, resids, occ, lag, style, max_occ):
    # Unpack style
    xlims, ylims, labels, (r_xlims, r_ylims) = style
    # Style scatter plots
    for ax, xlim, ylim, (label, pos) in zip(axs, xlims, ylims, labels):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(*pos, label)
    # Style residuals
    resids.set_xlim(*r_xlims)
    resids.set_ylim(*r_ylims)
    resids.set_ylabel(r'Norm. Residual $δ$', fontsize=fontsize, labelpad=labelpad)
    resids.set_xlabel(r'$\eta_{0}\,/\,cm$', fontsize=fontsize, labelpad=labelpad)
    xs = np.linspace(*r_xlims, 1000)
    resids.plot(xs, [1]*len(xs), c='k', ls='--', zorder=2)
    resids.plot(xs, [-1]*len(xs), c='k', ls='--', zorder=2)
    resids.text(r_xlims[0], r_ylims[1]*.87, r'$(b)$')
    # Style Occurence
    occ.tick_params(axis='y', direction='inout', left=False, right=False,
                    labelleft=False)
    occ.set_xlabel(r'$Occurence$', fontsize=fontsize)
    occ.plot(np.linspace(0, 1, 100), [-1]*100, ls='--', c='k')
    occ.plot(np.linspace(0, 1, 100), [1]*100, ls='--', c='k')
    occ.text(max_occ * 1.19, r_ylims[1]*.87, r'$(c)$')
    occ.set_xlim([max_occ * 1.2, 0])
    occ.set_ylim(r_ylims)
    # Style lag plots
    lag.set_xlabel(r'$δ_{n}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_ylabel(r'$δ_{n-1}$', fontsize=fontsize, labelpad=labelpad)
    lag.set_xlim(r_ylims)
    lag.set_ylim(r_ylims)
    lag.tick_params(axis='x', direction='out', top=False, bottom=True,
                    labeltop=False, labelbottom=True)
    lag.xaxis.set_label_position('bottom')
    lag.tick_params(axis='y', direction='out', right=True, left=False,
                    labelright=True, labelleft=False)
    lag.yaxis.set_label_position('right')
    lag.plot([-2]*len(xs), np.linspace(-2, 2, len(xs)), c='k', ls='--',
             zorder=2)
    lag.plot([2]*len(xs), np.linspace(-2, 2, len(xs)), c='k', ls='--',
             zorder=2)
    lag.plot(np.linspace(-2, 2, len(xs)), [-2]*len(xs), c='k', ls='--',
             zorder=2)
    lag.plot(np.linspace(-2, 2, len(xs)), [2]*len(xs), c='k', ls='--',
             zorder=2)
    lag.text(r_ylims[0]*0.95, r_ylims[1]*0.87, r'$(d)$')


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


def plot_occ(ax, occ_bins, bin_values, widths, norm_y, norm_x, norm_scale, horizontal=True, norm=False):
    colors = cmap(np.linspace(0.2, 1, len(occ_bins)))
    for i_bin, bin_value in enumerate(bin_values):
        color = colors[int(abs(len(bin_values)/2-i_bin))]
        ax.barh(occ_bins[i_bin], bin_value, widths[i_bin], align='edge',
        facecolor=color, zorder=0)
    if norm:
        ax.plot(norm_scale*norm_x, norm_y, ls='--', c='grey', zorder=3)
    return np.max(bin_values)


def scatter_error(ax, xs, x_err, ys, y_errs, i, marker_step=1, markersize=6):
    color = colors[i]
    ax.errorbar(xs[::marker_step], np.absolute(ys[::marker_step]), xerr=x_err, yerr=y_errs[::marker_step], c=color, fmt='x', capsize=1.5, markersize=markersize, zorder=1)

def plot_curve(ax, xs, ys, i):
    ax.plot(xs, ys, color='k', ls='--', zorder=3)


def plot_resid(ax, xs, residuals, i_color, resid_step=1, s=20):
    color = colors[i_color]
    ax.scatter(xs[::resid_step], residuals[::resid_step], s=s,
                  marker='x', color=color, zorder=1)

def plot_lag(ax, residuals, i_color, lag_step=1, s=20, alpha=1):
    color = colors[i_color]
    ax.scatter(residuals[:-1:lag_step], residuals[1::lag_step], marker='x',
                color=color, s=s, edgecolors='k', alpha=alpha)

def save(fig, name):
    fig.patch.set_alpha(0)
    fig.savefig(name, bbox_inches='tight', pad_inches=0, format='png')
    fig.patch.set_alpha(1)
