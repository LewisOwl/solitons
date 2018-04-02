import os, re
import numpy as np, scipy.optimize
import cv2

from models import *

# Path to images
path = "datas/vh/h{0}cm_{1}/"

# Errors
pixel_err = 2
h_err = 0.005/100

# Set up levels, threshold values and pixel scales
takes = [1,2,3,4,5,6,7,8,9,10]
THRESHS = {'5.85':184, '6.90':133, '8.00':142,
           '9.10':170, '10.05':139, '11.15':125}
levels = ['5.85', '6.90', '8.00', '9.10', '10.05', '11.15']
pixel_scales_y = {'5.85':0.08/280, '6.90':0.08/278,
                  '8.00':0.08/279, '9.10':0.05/174,
                  '10.05':0.10/346, '11.15':0.09/308}
pixel_scale_x = 0.1/331

# Iterations in finding param errors
ITERS = 20

def chi_sqr(times, xs, exp_data, exp_err, phi, c, l, *args):
    model_vals = np.empty(exp_data.shape)
    # We can loop through each frame/time and evaluate a whole line at once
    for it, time in enumerate(times):
        model_vals[:, it] = n(time, xs, phi, c, l, *args)
    residuals = model_vals - exp_data
    chi_sqr = np.sum((residuals/exp_err)**2)
    return chi_sqr/exp_data.size


# Data processing functions
def baseline(level):
    '''Find baseline water level for a particular water level, for use in
    finding final wave calc_amplitudes
    Returns: Binary mask of static water level'''
    stat_path = path.format(level, 'static2')
    stat_names = []
    for (_,_, filename) in os.walk(stat_path):
        stat_names += filename
    # Remove aliasing, the first and last 5 columns, and take red channel
    stat = cv2.imread(stat_path+stat_names[2])[::-2,5:-5,2]
    stat = stat < THRESHS[level]
    return stat

def calc_amplitudes(level, take, static):
    '''Calculates wave amplidues for every frames for a particular wave, denoted
    by level and take
    Returns:
        Amplitudes of wave
        Times for columns of amplitude matrix
        Xs for rows of amplitude matrix'''
    # Find all frames for the level and take
    img_path = path.format(level, 't{0}'.format(take))
    img_names = []
    for (_,_, filename) in os.walk(img_path):
        img_names += filename
    # For each frame
    xs = np.arange(0, len(static[0,:])*pixel_scale_x, pixel_scale_x)
    amplitudes = np.zeros((len(xs), len(img_names)))
    times = np.empty(0)
    xs = np.arange(0, len(xs)*pixel_scale_x, pixel_scale_x)
    for iname, name in enumerate(img_names):
        # Find time from name
        fi, time = re.findall(r'\d+', name)
        time = int(time)/1000
        # Remove aliasing, the first and last 5 columns, and take red channel
        img = cv2.imread(img_path+name)[::-2,5:-5,2]

        # BGR Thresholding
        img = img < THRESHS[level] # Threshold the image to show wave
        img_sum = np.sum(img, axis=0)

        # # HSV Thresholding
        # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # H = 100
        # hsv_thresh = 10
        # img = cv2.inRange(hsv_img, np.array([H - hsv_thresh, 150, 150]), np.array([[H + hsv_thresh,255,255]]))
        # cv2.imshow('bin', img)
        # cv2.waitKey()
        # img = img > 0

        # Find the wave amplitude by xor-ing with baseline img
        dif = img ^ static
        dif_sum = np.sum(dif, axis=0)
        amplitudes[:,iname] = dif_sum*pixel_scales_y[level]

        times = np.r_[times, time]

    amplitudes = amplitudes[:,np.argsort(times)]
    times = times[np.argsort(times)]
    return amplitudes, times, xs


def model_fit(h, y_err, n0, amplitudes, times, xs):
    '''Find model parameters for particular wave defined by amplidues
    Returns:
        Wave speed and error
        Wave length and error
        Wave offset phi
        Reduced chi_sqr for model fit
        '''
    # Use theoretical values as inital vals
    c_t, c_t_err = theory_c(n0, y_err, h, h_err)
    l_t, l_t_err = theory_l(n0, y_err, h, h_err)

    # Calc average phi to center the wave on origin
    t_av = times[2]
    x_av = np.mean(xs)
    phi_av = (x_av - c_t*t_av)/l_t

    c, l, phi = c_t, l_t, -phi_av
    # Start by finding a phi that corrects the wave translation
    phi_min_func = lambda phi: chi_sqr(times, xs, amplitudes, y_err, phi, c, l, n0, h)
    res = scipy.optimize.minimize(phi_min_func, phi, method='Nelder-Mead')
    phi = res.x[0]
    # Use the corrected model to find c & l.  x=[c,l]
    min_func = lambda x: chi_sqr(times, xs, amplitudes, y_err, x[2], x[0], x[1], n0, h)
    res = scipy.optimize.minimize(min_func,[c, l, phi], method='Nelder-Mead')
    c, l, phi = res.x
    chi_sqr_red = res.fun

    # Calc errors on params
    target_chi = chi_sqr_red + 100/amplitudes.size
    c_dash, l_dash = c, l

    # Functions to target chi = target_chi
    t_func_c = lambda c_prime: abs(target_chi - chi_sqr(times, xs, amplitudes, y_err, phi, c_prime, l_dash, n0, h))
    t_func_l = lambda l_prime: abs(target_chi - chi_sqr(times, xs, amplitudes, y_err, phi, c_dash, l_prime, n0, h))

    # Functions to minimize chi
    m_func_c = lambda c_prime: chi_sqr(times, xs, amplitudes, y_err, phi, c_prime, l_dash, n0, h)
    m_func_l = lambda l_prime: chi_sqr(times, xs, amplitudes, y_err, phi, c_dash, l_prime, n0, h)

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

    return c, c_err, l, l_err, phi, chi_sqr_red

def calc_residuals(amplitudes, times, xs, c_mod, l_mod, phi_mod, n0, h, y_err):
    '''Calculate normalised residuals for the model parameters
    Returns:
        Residuals array
        Theta value for each residual'''
    residuals = np.empty(0)
    thetas = np.empty(0)
    for it, time in enumerate(times):
        ts = np.asarray([time]*len(xs))
        thetas_part = (xs - c_mod*ts)/l_mod + phi_mod
        residuals_part = (amplitudes[:, it] - n(time, xs, phi_mod, c_mod, l_mod, n0, h))/y_err
        residuals = np.r_[residuals, residuals_part]
        thetas = np.r_[thetas, thetas_part]
    residuals = residuals[np.argsort(thetas)]
    thetas = thetas[np.argsort(thetas)]
    return residuals, thetas

def derbin_watson(residuals):
    '''Calculate Derbin-Watson statistic for the residual set'''
    diffs = residuals[:-1] - residuals[1:]
    derbin = np.sum(diffs**2) / np.sum(residuals**2)
    return derbin

def process(level, take):
    h = float(level)/100
    y_err = pixel_err * pixel_scales_y[level]
    baselevel = baseline(level)
    amps, times, xs = calc_amplitudes(level, take, baselevel)

    # Calc n0
    ordered_vals = amps.flatten()
    ordered_vals.sort()
    n0 = np.mean(ordered_vals[-50:])
    # Find model params
    c, c_err, l, l_err, phi, chi_sqr_red = model_fit(h, y_err, n0, amps, times, xs)
    # Param theory values
    c_t, c_t_err = theory_c(n0, y_err, h, h_err)
    l_t, l_t_err = theory_l(n0, y_err, h, h_err)

    # Data stats
    resids, thetas = calc_residuals(amps, times, xs ,c , l, phi, n0, h, y_err)
    derbin = derbin_watson(resids)

    return (times, xs, thetas), (amps, resids), (c, c_err, l, l_err, phi,
            chi_sqr_red, derbin), (c_t, c_t_err, l_t, l_t_err)


if __name__ == '__main__':
    # Target specific wave for testing
    t_level = '6.90'
    t_take = 4

    (times, xs, thetas), (amps, resids), (c, c_err, l, l_err, phi, chi_sqr_red, derbin), (c_t, c_t_err, l_t, l_t_err) = process(t_level, t_take)


    print('C_t: {:.3f}±{:.3f}cm; L_t: {:.3f}±{:.3f}cm; '.format(c_t*100, c_t_err*100, l_t*100, l_t_err*100))
    print('C: {:.3f}±{:.3f}cms^-1; L: {:.3f}±{:.3f}cm; χ^2: {:.2f}; D: {:.2f}'.format(c*100, c_err*100, l*100, l_err*100, chi_sqr_red*amps.size, derbin))
