import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

import proc

# Plotting params
marker_skip = 250
lag_skip = 25
lag_alpha = 0.3
fontsize=13
labelpad=3

main = plt.figure(figsize=(8,10))
curve = main.add_axes([0.1, 0.3, 0.55, 0.67])
resid = main.add_axes([0.1, 0.1, 0.55, 0.20], sharex = curve)
occ = main.add_axes([0.65, 0.1, 0.25, 0.20], sharey = resid)
lag = main.add_axes([0.65, 0.3, 0.25, 0.20])

cv = plt.figure(figsize=(8,8))
cvax = cv.add_axes([0.1,0.1,0.85,0.85])

colors = cm.winter(np.linspace(0,1,len(proc.takes)))
