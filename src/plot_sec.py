import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

from .paths import *
from .models import theory_c


sec = plt.figure(figsize=(10, 10))
back = sec.add_axes([0.05, 0.05, 0.85, 1.02])
n0c1 = sec.add_axes([0.05, 0.75, 0.4, 0.22])
n0c2 = sec.add_axes([0.5, 0.75, 0.4, 0.22])
n0c3 = sec.add_axes([0.05, 0.5, 0.4, 0.22])
n0c4 = sec.add_axes([0.5, 0.5, 0.4, 0.22])
n0c5 = sec.add_axes([0.05, 0.25, 0.4, 0.22])
n0c6 = sec.add_axes([0.5, 0.25, 0.4, 0.22])
secresids = sec.add_axes([0.05, 0.0, 0.4, 0.2])
secocc = sec.add_axes([0.45, 0.0, 0.2, 0.2], sharey=secresids)
seclag = sec.add_axes([0.7, 0.0, 0.2, 0.2])
n0cs = [n0c1, n0c2, n0c3, n0c4, n0c5, n0c6]

def style_back():
    back.spines['top'].set_color('none')
    back.spines['bottom'].set_color('none')
    back.spines['left'].set_color('none')
    back.spines['right'].set_color('none')
    back.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    back.xaxis.set_label_position('top')
    back.set_ylabel(r'$|c|$', labelpad=15)
    back.set_xlabel(r'$\eta_{0}$', labelpad=1)

def plot_n0c(n0s, n0_err, cs, c_errs, h, h_err, i):
    color = cmap(np.linspace(0.2, 1, 7))[i]
    n0cs[i].errorbar(n0s*100, np.absolute(cs)*100, xerr=n0_err*100, yerr=c_errs*100, c=color, fmt='x', capsize=1.5)
    xs = np.linspace(np.min(n0s)*0.9, np.max(n0s)*1.1, 100)
    ys, y_errs = theory_c(xs, n0_err[0], h, h_err)
    n0cs[i].plot(xs*100, np.absolute(ys)*100, color=color, ls='--')


def save_sec(name):
    sec.patch.set_alpha(0)
    back.set_facecolor('none')
    sec.savefig(name, bbox_inches='tight', pad_inches=0, format='png')
    sec.patch.set_alpha(1)
