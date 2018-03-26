import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import re


fig = plt.figure()
ax = plt.axes(projection='3d')

g = 9.81 # ms^-2

# Model function
def n(x, t, phi, *args):
    # calc wave params from n0, h
    n0, h = args
    c0 = (g*h)**0.5
    c = c0*(1 + n0 / (2*h))
    l = ((4*h**3) / (3*n0))**0.5
    # calc sech params
    arg = (x - c*t)/l - phi #a0 the param offset to translate the wave
    return n0 / (np.cosh(arg))**2


THRESH = 150
levels = ['5.85', '6.90', '8.00', '9.10', '10.05', '11.15']
takes = [1,2,3,4,5,6,7,8,9,10]
level = levels[3]
take = takes[8]
path = "datas/vh/h{0}cm_{1}/"
# Find the static water level
stat_path = path.format(level, 'static2')
stat_names = []
for (_,_, filename) in os.walk(stat_path):
    stat_names += filename
stat = cv2.imread(stat_path+stat_names[0])[::-2,5:-5,2]
stat = stat < THRESH
stat_sum = np.sum(stat, axis=0)

# For specific take t

img_path = path.format(level, 't{0}'.format(take))
img_names = []
for (_,_, filename) in os.walk(img_path):
    img_names += filename
# For each frame
frames = np.zeros((len(stat_sum), len(img_names)))
for i, name in enumerate(img_names):
    # Find time from name
    fi, time = re.findall(r'\d+', name)
    time = int(time)/1000
    # Find the wave amplitude
    img = cv2.imread(img_path+name)[::-2,5:-5,2] # Remove aliasing and select red channel
    img = img < THRESH # Threshold the image to show wave
    img_sum = np.sum(img, axis=0)

    dif = img - stat
    dif_sum = np.sum(dif, axis=0)
    # 3d plot of each frame
    ax.plot(range(len(dif_sum)), [time]*len(dif_sum), dif_sum)


    # # Plot it all!
    # plt.subplot(math.ceil(len(img_names)/4) + 1, 4, i+1)
    # # plt.title('Frame: {}, Time: {}'.format(fi, time))
    # plt.ylim((0, 200))
    # plt.xlim((0, 700))
    # plt.plot(dif_sum)


plt.show()
