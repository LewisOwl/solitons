import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
import re
import scipy.optimize

plt.rcParams["figure.figsize"] = (5,20)
fig, (curve, resid, lag) = plt.subplots(3, 1, sharex=True, gridspec_kw = {'height_ratios':[2, 1, 2]})

g = 9.81 # ms^-2

# Model function
def n(x, t, phi, c, l, *args):
    # calc wave params from n0, h
    n0, h = args
    # c0 = -(g*h)**0.5    # Our wave moves to left => negative c
    # c = c0*(1 + n0 / (2*h))
    # l = ((4*(h**3)) / (3*n0))**0.5

    # calc sech params
    arg = (x - c*t)/l + phi # phi the param offset to translate the wave
    # print(arg)
    trig = np.cosh(arg)**2
    return n0 / trig

def chi_sqr(times, xs, exp_data, exp_err, phi, c, l, *args):
    model_vals = np.empty(exp_data.shape)
    # We can loop through each frame/time and evaluate a whole line at once
    for it, time in enumerate(times):
        model_vals[:, it] = n(xs, time, phi, c, l, *args)
    residuals = model_vals - exp_data
    chi_sqr = np.sum((residuals/exp_err)**2)
    return chi_sqr/exp_data.size

THRESHS = [184, 133, 142, 170, 139, 125]
levels = ['5.85', '6.90', '8.00', '9.10', '10.05', '11.15']
colors = cm.winter(np.linspace(0,1,len(levels)))
pixel_scales_y = [0.08/280, 0.08/278, 0.08/279, 0.05/174, 0.10/346, 0.09/308]
pixel_scale_x = 0.1/331
takes = [1,2,3,4,5,6,7,8,9,10]
#takes = [5]
cols = 1
pixel_err = 3
t_ilevel = 0


path = "datas/vh/h{0}cm_{1}/"

marker_skip = 500
lag_skip = 5


for ilevel, level in enumerate(levels):
    # if ilevel != t_ilevel:
    #     continue

    print('##################  Level: {}cm  ##################'.format(level))

    pixel_scale_y = pixel_scales_y[ilevel]
    y_err = pixel_err*pixel_scale_y

    all_residuals = np.empty(1)
    all_thetas = np.empty(1)
    # Find the static water level
    stat_path = path.format(level, 'static2')
    stat_names = []
    for (_,_, filename) in os.walk(stat_path):
        stat_names += filename
    stat = cv2.imread(stat_path+stat_names[2])[::-2,5:-5,2]
    stat = stat < THRESHS[ilevel]
    stat_sum = np.sum(stat, axis=0)
    xs = np.arange(0, len(stat_sum)*pixel_scale_x, pixel_scale_x)
    # Loop over all takes for given height
    for itake, take in enumerate(takes):
        if level == '10.05' and take == 2:
            continue
        print('-------  Take: {}  -------'.format(take))
        # For specific take t
        #fig.add_subplot(math.ceil(len(takes)/cols), cols, itake + 1, projection='3d')
        img_path = path.format(level, 't{0}'.format(take))
        img_names = []
        for (_,_, filename) in os.walk(img_path):
            img_names += filename
        # For each frame
        frames = np.zeros((len(stat_sum), len(img_names)))
        times = []
        for iname, name in enumerate(img_names):
            # Find time from name
            fi, time = re.findall(r'\d+', name)

            time = int(time)/1000
            img = cv2.imread(img_path+name)[::-2,5:-5,:] # Remove aliasing
            # BGR Thresholding

            img = img[:,:,2]  # Select red channel
            img = img < THRESHS[ilevel] # Threshold the image to show wave
            img_sum = np.sum(img, axis=0)

            # # HSV Thresholding
            # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # H = 100
            # hsv_thresh = 10
            # img = cv2.inRange(hsv_img, np.array([H - hsv_thresh, 150, 150]), np.array([[H + hsv_thresh,255,255]]))
            # cv2.imshow('bin', img)
            # cv2.waitKey()
            # img = img > 0

            # Find the wave amplitude
            dif = img ^ stat
            dif_sum = np.sum(dif, axis=0)
            frames[:,iname] = dif_sum*pixel_scale_y
            # 3d plot of each frame
            ts = np.asarray([time]*len(dif_sum))
            times.append(time)

            #plt.scatter(xs, ts, dif_sum*pixel_scale, marker='o')


        # Frames is now 2d array of amplitudes for (x,t). We need to find phi offset
        # that fits the curve to the wave. We can chi_sqr min this.
        ordered_vals = frames.flatten()
        ordered_vals.sort()

        n0 = np.mean(ordered_vals[-50:])
        h = float(level)/100
        # Calc theoretical values as inital vals
        c0 = -(g*h)**0.5    # Our wave moves to left => negative c
        c = c0*(1 + n0 / (2*h))
        l = ((4*(h**3)) / (3*n0))**0.5
        t_av = times[2]
        x_av = np.mean(xs)

        c, l = c, l*0.681
        phi_av = (x_av - c*t_av)/l
        phi = -phi_av
        # Start by finding a phi that corrects the wave translation
        # phi_min_func = lambda phi: chi_sqr(times, xs, frames, y_err, phi, c, l, n0, h)
        # res = scipy.optimize.minimize(phi_min_func, -18, method='Nelder-Mead')
        # phi = res.x[0]
        # Use the corrected model to find c & l.  x=[c,l]
        min_func = lambda x: chi_sqr(times, xs, frames, y_err, x[2], x[0], x[1], n0, h)
        res = scipy.optimize.minimize(min_func,[c, l, phi], method='Nelder-Mead')
        c, l, phi = res.x
        print('n0: {:.2f}cm; C: {:.3f}ms^-1; L: {:.3f}m; Chi_sqr_red: {:.2f}'.format(n0*100, c, l, res.fun))


        # plot model lines for each time and data

        # for iname, name in enumerate(img_names):
        #     fi, time = re.findall(r'\d+', name)
        #     time = int(time)/1000
        #     # Find the wave amplitude
        #     # 3d plot of each frame
        #     xs = np.linspace(-0.05, 0.25, 1000)
        #     ts = np.asarray([time]*len(xs))
        #
        #     thetas = (xs - c*ts)/l + phi
        #
        #     heights = n(xs, time, phi, c, l, n0, h)/n0
        #     #plt.plot(xs, ts, heights)
        #     plt.plot(thetas, heights)



        for iname, name in enumerate(img_names):
            fi, time = re.findall(r'\d+', name)
            time = int(time)/1000
            ts = np.asarray([time]*len(frames[:,iname]))
            xs = np.arange(0, len(frames[:,iname])*pixel_scale_x, pixel_scale_x)
            thetas = (xs - c*ts)/l + phi
            residuals = (frames[:,iname]/n0 - n(xs, time, phi, c, l, n0, h)/n0)/(y_err/n0)
            #plt.plot(xs, ts, frames[:,iname]/n0)
            curve.errorbar(thetas[::marker_skip], frames[::marker_skip,iname]/n0 + ilevel/2, fmt='x', markersize=3, yerr=y_err/n0, capsize=4, color=colors[ilevel], zorder=1)
            resid.scatter(thetas[::marker_skip], residuals[::marker_skip], s=3, marker='x', color=colors[ilevel], zorder=1)
            # Store residuals for lag plot later
            # lag.scatter(residuals[:-1], residuals[1:], marker='x', color=colors[ilevel], s=10, edgecolors=None)
            all_residuals = np.r_[all_residuals, residuals]
            all_thetas = np.r_[all_thetas, thetas]

    all_residuals = all_residuals[np.argsort(all_thetas)]
    lag.scatter(all_residuals[:-1:lag_skip], all_residuals[1::lag_skip], marker='x', color=colors[ilevel], s=3, edgecolors='k', alpha=0.5)
# Plot dashed model and residual lines
thetas = np.linspace(-4, 4, 1000)
resid.plot(thetas, [1]*len(thetas), c='k', ls='--', zorder=2)
resid.plot(thetas, [-1]*len(thetas), c='k', ls='--', zorder=2)

for ilevel, _ in enumerate(levels):
    curve.plot(thetas, 1/np.cosh(thetas)**2 + ilevel/2, c='k', ls='--', zorder=2)

curve.set(ylabel='η/η0')
resid.set(ylabel='δ', xlabel='(x - ct)/L')
lag.set(xlabel='δn', ylabel='δn-1')
lag.set_xlim([-5, 5])
lag.set_ylim([-5, 5])


# Label and move axes

fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.show()
