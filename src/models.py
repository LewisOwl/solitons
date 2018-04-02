import numpy as np


g = 9.81  # ms^-2
g_err = 0.005


# Theory functions
def n(t, x, phi, c, l, *args):
    n0, h = args
    # calc sech params
    arg = (x - c*t)/l + phi  # phi the param offset to translate the wave
    trig = np.cosh(arg)**2
    return n0 / trig


def theory_c(n0, n0_err, h, h_err):
    c0 = -(g*h)**0.5    # Our wave moves to left => negative c
    c_t = c0*(1 + n0 / (2*h))
    frac_h = h_err/h
    frac_g = g_err/g
    frac_n0 = n0_err/n0
    term1 = 0.25*(frac_g**2 + frac_h**2)
    term2 = (frac_n0)**2 + frac_h**2
    c_t_err = c_t * (term1 + 1/(2*h/n0 + 1) * term2)**0.5
    return c_t, c_t_err


def theory_l(n0, n0_err, h, h_err):
    l_t = ((4*(h**3)) / (3*n0))**0.5
    frac_h = h_err/h
    frac_n0 = n0_err/n0
    l_t_err = l_t/2 * ((3*frac_h)**2 + frac_n0**2)**0.5
    return l_t, l_t_err
