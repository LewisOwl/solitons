import numpy as np


g = 9.81 # ms^-2
g_err = 0.005

# Theory functions
def n(t, x, phi, c, l, *args):
    # calc wave params from n0, h
    n0, h = args
    # calc sech params
    arg = (x - c*t)/l + phi # phi the param offset to translate the wave
    trig = np.cosh(arg)**2
    return n0 / trig

def theory_c(n0, n0_err, h, h_err):
    c0 = -(g*h)**0.5    # Our wave moves to left => negative c
    c_t = c0*(1 + n0 / (2*h))
    c_t_err = c_t * (0.25*((g_err/g)**2 + (h_err/h)**2) + 1/(2*h/n0 + 1) * ( (n0_err/n0)**2 + (h_err/h)**2 ))**0.5
    return c_t, c_t_err

def theory_l(n0, n0_err, h, h_err):
    l_t = ((4*(h**3)) / (3*n0))**0.5
    l_t_err = l_t/2 * ( (3*h_err/h)**2 + (n0_err/n0)**2 )**0.5
    return l_t, l_t_err
