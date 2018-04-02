import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
import re
import scipy.optimize
from scipy.stats import norm


plt.rcParams["figure.figsize"] = (8,10)
fig = plt.figure(figsize=(8,10))
# fig, (curve, resid) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios':[4, 1]})
curve = fig.add_axes([0.1, 0.3, 0.55, 0.67])
resid = fig.add_axes([0.1, 0.1, 0.55, 0.20], sharex = curve)
occ = fig.add_axes([0.65, 0.1, 0.25, 0.20], sharey = resid)
lag = fig.add_axes([0.65, 0.3, 0.25, 0.20])
g = 9.81 # ms^-2
g_err = 0.005
fig2 = plt.figure(figsize=(8,8))
cont = fig2.add_axes([0.1,0.1,0.85,0.85])
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
pixel_scales_y = [0.08/280, 0.08/278, 0.08/279, 0.05/174, 0.10/346, 0.09/308]
pixel_scale_x = 0.1/331
takes = [1,2,3,4,5,6,7,8,9,10]
colors = cm.winter(np.linspace(0,1,len(takes)))
#takes = [5]
cols = 1
pixel_err = 2


t_ilevel = 1
t_itake = 3
abs_all_residuals = np.zeros(1)

path = "datas/vh/h{0}cm_{1}/"

marker_skip = 250
lag_skip = 25
lag_alpha = 0.3
fontsize=13
labelpad=3

ITERS = 20
RESOL = 100





# Occurence plot params
bin_width = 1/2
max_count = 0
bins = np.arange(-5, 5, bin_width)
bincols = cm.winter(np.linspace(0.2,1,(len(bins)//2)+1))

# Data processing
for ilevel, level in enumerate(levels):
    if ilevel != t_ilevel:
        continue

    print('##################  Level: {}cm  ##################'.format(level))

    pixel_scale_y = pixel_scales_y[ilevel]
    y_err = pixel_err*pixel_scale_y


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
        if level == '11.15' and take == 6:
            continue
        if itake != t_itake:
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

            times.append(time)


        # frames = frames[:,:-2]
        # times = times[:-2]
        # Frames is now 2d array of amplitudes for (x,t). We need to find phi offset
        # that fits the curve to the wave. We can chi_sqr min this.
        ordered_vals = frames.flatten()
        ordered_vals.sort()

        n0 = np.mean(ordered_vals[-50:])
        n0_err = y_err
        h = float(level)/100

        h_err = 0.005/100
        # Calc theoretical values as inital vals
        c0 = -(g*h)**0.5    # Our wave moves to left => negative c
        c_t = c0*(1 + n0 / (2*h))
        l_t = ((4*(h**3)) / (3*n0))**0.5

        c_t_err = c_t * (0.25*((g_err/g)**2 + (h_err/h)**2) + 1/(2*h/n0 + 1) * ( (n0_err/n0)**2 + (h_err/h)**2 ))**0.5
        l_t_err = l_t/2 * ( (3*h_err/h)**2 + (n0_err/n0)**2 )**0.5


        t_av = times[2]
        x_av = np.mean(xs)

        c, l = c_t, l_t
        phi_av = (x_av - c*t_av)/l
        phi = -phi_av
        # Start by finding a phi that corrects the wave translation
        phi_min_func = lambda phi: chi_sqr(times, xs, frames, y_err, phi, c, l, n0, h)
        res = scipy.optimize.minimize(phi_min_func, phi, method='Nelder-Mead')
        phi = res.x[0]
        # Use the corrected model to find c & l.  x=[c,l]
        min_func = lambda x: chi_sqr(times, xs, frames, y_err, x[2], x[0], x[1], n0, h)
        res = scipy.optimize.minimize(min_func,[c, l, phi], method='Nelder-Mead')
        c, l, phi = res.x

        # Calc errors on params
        target_chi = res.fun + 100/frames.size
        c_dash, l_dash = c, l

        # Functions to target chi + 2.3
        t_func_c = lambda c_prime: abs(target_chi - chi_sqr(times, xs, frames, y_err, phi, c_prime, l_dash, n0, h))
        t_func_l = lambda l_prime: abs(target_chi - chi_sqr(times, xs, frames, y_err, phi, c_dash, l_prime, n0, h))

        # Functions to minimize chi
        m_func_c = lambda c_prime: chi_sqr(times, xs, frames, y_err, phi, c_prime, l_dash, n0, h)
        m_func_l = lambda l_prime: chi_sqr(times, xs, frames, y_err, phi, c_dash, l_prime, n0, h)

        # Find error in c
        c_dash, l_dash = c, l
        for it in range(ITERS):
            # Target +2.3 varing c
            c_dash = scipy.optimize.minimize(t_func_c, c_dash, method='Nelder-Mead').x[0]
            # minimize varying l
            l_dash = scipy.optimize.minimize(m_func_l, l_dash, method='Nelder-Mead').x[0]
        # End on a chi +2.3
        c_dash = scipy.optimize.minimize(t_func_c, c_dash, method='Nelder-Mead').x[0]
        c_err = abs(c - c_dash)
        # Find error in l
        c_dash, l_dash = c, l
        for it in range(ITERS):
            # Target +2.3 varing l
            l_dash = scipy.optimize.minimize(t_func_l, l_dash, method='Nelder-Mead').x[0]
            # minimize varying c
            c_dash = scipy.optimize.minimize(m_func_c, c_dash, method='Nelder-Mead').x[0]
        # End on a chi +2.3
        l_dash = scipy.optimize.minimize(t_func_l, l_dash, method='Nelder-Mead').x[0]
        l_err = abs(l_dash - l)

        # Plot contours of c & l
        # cl, cu, ll, lu = c - 5*c_err, c + 5*c_err, l - 5*l_err, l + 5*l_err
        # cs = np.linspace(cl, cu, RESOL)
        # ls = np.linspace(ll, lu, RESOL)
        # chis = np.empty((RESOL,RESOL))
        # for ic, part_c in enumerate(cs):
        #     for il, part_l in enumerate(ls):
        #         chis[ic, il] = chi_sqr(times, xs, frames, y_err, phi, part_c, part_l, n0, h)
        # cs, ls = np.meshgrid(cs, ls)
        # # cont.imshow(chis, extent=(cl, cu, ll, lu))
        # cont.contour(cs, ls, chis, levels = [chis.min(), chis.min() + 2.3/frames.size], colors=('r','g'))
        cont.errorbar(c_t,l_t, xerr=c_t_err, yerr=l_t_err, color=colors[ilevel])


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
        all_residuals = np.empty(1)
        all_thetas = np.empty(1)


        for iname, time in enumerate(times):
            # fi, time = re.findall(r'\d+', name)
            # time = int(time)/1000
            ts = np.asarray([time]*len(frames[:,iname]))
            xs = np.arange(0, len(frames[:,iname])*pixel_scale_x, pixel_scale_x)
            thetas = (xs - c*ts)/l + phi
            residuals = (frames[:,iname]/n0 - n(xs, time, phi, c, l, n0, h)/n0)/(y_err/n0)
            #plt.plot(xs, ts, frames[:,iname]/n0)
            curve.errorbar(thetas[::marker_skip], frames[::marker_skip,iname]/n0 + ilevel/2, fmt='x', markersize=3, yerr=y_err/n0, capsize=1.5, color=colors[itake], zorder=1)
            resid.scatter(thetas[::marker_skip], residuals[::marker_skip], s=3, marker='x', color=colors[itake], zorder=1)
            # Store residuals for lag plot later
            # lag.scatter(residuals[:-1], residuals[1:], marker='x', color=colors[ilevel], s=10, edgecolors=None)
            all_residuals = np.r_[all_residuals, residuals]
            all_thetas = np.r_[all_thetas, thetas]


        all_residuals = all_residuals[np.argsort(all_thetas)]
        abs_all_residuals = np.r_[abs_all_residuals, all_residuals]
        lag.scatter(all_residuals[:-1:lag_skip], all_residuals[1::lag_skip], marker='x', color=colors[itake], s=3, edgecolors='k', alpha=lag_alpha)
        derbin = np.sum((all_residuals[:-1] - all_residuals[1:])**2)/np.sum(all_residuals**2)
        print('C_t: {:.3f}±{:.3f}cm; L_t: {:.3f}±{:.3f}cm; '.format(c_t*100, c_t_err*100, l_t*100, l_t_err*100))
        print('C: {:.3f}±{:.3f}cms^-1; L: {:.3f}±{:.3f}cm; χ^2: {:.2f}; D: {:.2f}'.format(c*100, c_err*100, l*100, l_err*100, res.fun*frames.size, derbin ))



# Plot occurences of residuals

for i, bin_start in enumerate(bins):
    count = abs_all_residuals[np.where(np.logical_and(abs_all_residuals > bin_start, abs_all_residuals < bin_start + bin_width))].size / abs_all_residuals.size
    if count > max_count:
        max_count = count
    occ.barh(bin_start, count, bin_width, align='edge', color=bincols[int(abs(len(bins)/2-i))])
occ.plot(norm.pdf(np.linspace(-4,4,100))*max_count*(2*np.pi)**0.5,np.linspace(-4,4,100),ls='--', c='grey')

# Plot dashed model and residual lines and lag plot bounds
thetas = np.linspace(-4, 4, 1000)

resid.plot(thetas, [1]*len(thetas), c='k', ls='--', zorder=2)
resid.plot(thetas, [-1]*len(thetas), c='k', ls='--', zorder=2)

lag.plot([-2]*len(thetas), np.linspace(-2, 2, len(thetas)), c='k', ls='--', zorder=2)
lag.plot([2]*len(thetas), np.linspace(-2, 2, len(thetas)), c='k', ls='--', zorder=2)
lag.plot(np.linspace(-2, 2, len(thetas)), [-2]*len(thetas), c='k', ls='--', zorder=2)
lag.plot(np.linspace(-2, 2, len(thetas)), [2]*len(thetas), c='k', ls='--', zorder=2)

for ilevel, _ in enumerate(levels):
    curve.plot(thetas, 1/np.cosh(thetas)**2 + ilevel/2, c='k', ls='--', zorder=2)


# curve.set_xlabel(r'$\frac{x - ct}{L}$', fontsize=fontsize, labelpad=labelpad)
curve.set_ylabel(r'$\frac{\eta}{\eta_0}$', fontsize=fontsize, labelpad=labelpad)
# curve.tick_params(axis='x', direction='out', top = True, bottom = False,
# labeltop = True, labelbottom = False)
# curve.xaxis.set_label_position('top')
curve.set_xlim([-4, 4])

resid.set_xlabel(r'$\frac{x - ct}{L}$', fontsize=fontsize, labelpad=labelpad)
resid.set_ylabel(r'$δ$', fontsize=fontsize, labelpad=labelpad)
resid.set_ylim([-5, 5])
resid.set_xticks([-3,-2,-1,0,1,2,3])

lag.set_xlabel(r'$δ_{n}$', fontsize=fontsize, labelpad=labelpad)
lag.set_ylabel(r'$δ_{n-1}$', fontsize=fontsize, labelpad=labelpad)
lag.set_xlim([-5, 5])
lag.set_ylim([-5, 5])
lag.set_xticks([-4, -2, 0, 2, 4])
lag.set_yticks([-4, -2, 0, 2, 4])

lag.tick_params(axis='x', direction='out', top = True, bottom = False,
labeltop = True, labelbottom = False)
lag.xaxis.set_label_position('top')

lag.tick_params(axis='y', direction='out', right = True, left = False,
labelright = True, labelleft = False)
lag.yaxis.set_label_position('right')

occ.set_xlim([0.4, 0])
occ.tick_params(axis='y', direction='inout', left = False, right = False,
labelleft = False)
occ.set_xticks([0, 0.1, 0.2, 0.3])
occ.set_xlabel(r'$f$')
occ.plot(np.linspace(0,1,100), [1]*100, ls='--', c='k')
occ.set_xlabel(r'$f$')
occ.plot(np.linspace(0,1,100), [-1]*100, ls='--', c='k')
# Label and move axes

fig.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
# fig.savefig('plots/plot1.png')
# fig2.savefig('plots/plot2.png')
plt.show()
# fig2.show()
