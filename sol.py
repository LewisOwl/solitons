import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import re
import scipy.optimize

fig = plt.figure()

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
_re
def chi_sqr(times, xs, exp_data, exp_err, phi, c, l, *args):
    model_vals = np.empty(exp_data.shape)
    # We can loop through each frame/time and evaluate a whole line at once
    for it, time in enumerate(times):
        model_vals[:, it] = n(xs, time, phi, c, l, *args)
    residuals = model_vals - exp_data
    chi_sqr = np.sum((residuals/exp_err)**2)
    return chi_sqr/exp_data.size

THRESH = 150
levels = ['5.85', '6.90', '8.00', '9.10', '10.05', '11.15']
pixel_scales_y = [0.08/280, 0.08/278, 0.08/279, 0.05/174, 0.10/346, 0.09/308]
pixel_scale_x = 10/331
#takes = [1,2,3,4,5,6,7,8,9,10]
takes = [5]
cols = 1

i_set = 3
level = levels[i_set]
pixel_scale_y = pixel_scales[i_set]
path = "datas/vh/h{0}cm_{1}/"


# Find the static water level
stat_path = path.format(level, 'static2')
stat_names = []
for (_,_, filename) in os.walk(stat_path):
    stat_names += filename
stat = cv2.imread(stat_path+stat_names[0])[::-2,5:-5,2]
stat = stat < THRESH
stat_sum = np.sum(stat, axis=0)
xs = np.arange(0, len(stat_sum)*pixel_scale_x, pixel_scale_x)
# Loop over all takes for given height
for itake, take in enumerate(takes):
    # For specific take t
    fig.add_subplot(math.ceil(len(takes)/cols), cols, itake + 1, projection='3d')
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
        # Find the wave amplitude
        img = cv2.imread(img_path+name)[::-2,5:-5,2] # Remove aliasing and select red channel
        img = img < THRESH # Threshold the image to show wave
        img_sum = np.sum(img, axis=0)



        dif = img ^ stat
        dif_sum = np.sum(dif, axis=0)
        frames[:,iname] = dif_sum*pixel_scale_y
        # 3d plot of each frame
        ts = np.asarray([time]*len(dif_sum))
        times.append(time)
        #plt.plot(xs, ts, frames[:,iname])
        plt.scatter(xs, ts, frames[:,iname], marker='x')


    # Frames is now 2d array of amplitudes for (x,t). We need to find phi offset
    # that fits the curve to the wave. We can chi_sqr min this.
    n0 = np.max(frames)
    h = float(level)/100
    # Calc theoretical values as inital vals
    c0 = -(g*h)**0.5    # Our wave moves to left => negative c
    c = c0*(1 + n0 / (2*h))
    l = ((4*(h**3)) / (3*n0))**0.5

    c, l = c, l*0.68
    print(c,l)
    # Start by finding a phi that corrects the wave translation
    phi_min_func = lambda phi: chi_sqr(times, xs, frames, 3*pixel_scale_y, phi, c, l, n0, h)
    res = scipy.optimize.minimize(phi_min_func, -18, method='Nelder-Mead')
    phi = res.x[0]
    # Use the corrected model to find c & l.  x=[c,l]
    min_func = lambda x: chi_sqr(times, xs, frames, 3*pixel_scale_y, phi, x[0], x[1], n0, h)
    res = scipy.optimize.minimize(min_func,[c, l], method='Nelder-Mead')
    c, l = res.x
    print('Phi: {:.2f}; C: {:.3f}ms^-1; L: {:.3f}m; Chi_sqr_red: {:.5f}'.format(phi, c, l, res.fun))
    # plot model lines for each time

    for iname, name in enumerate(img_names):
        fi, time = re.findall(r'\d+', name)
        time = int(time)/1000
        # Find the wave amplitude
        # 3d plot of each frame
        xs = np.linspace(0, 0.2, 1000)
        ts = np.asarray([time]*len(xs))

        heights = n(xs, time, phi, c, l, n0, h)
        plt.plot(xs, ts, heights)


plt.xlabel('xs')
plt.ylabel('ts')
plt.show()
