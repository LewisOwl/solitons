import numpy as np
import argparse

from paths import savepath
from proc import levels, takes

import plot
import proc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process wave data and gen plots')
    # parser.add_argument('-pa', '--processall', help='Process all data and save', action='store-true')
    # parser.add_argument('-ps')
    parser.add_argument('-pa', '--process_all', dest='proc_all', default=False, action='store_true', help='Process all data again')
    parser.add_argument('-ps', '--process_spec', dest='proc_spec', nargs=2, default=False, help='Process specific data set again. Args in the form "Level"cm "Take"')
    parser.add_argument('-gp', '--graphplot', dest='graphshow', default=False, action='store_true', help='Show a graph of data')
    parser.add_argument('-gs', '--graphsave', dest='graphsave', default=False, action='store_true', help='Save a graph of data')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help='Print out stats')
    args = parser.parse_args()



    if args.proc_all:
        proc.save_all(verbose=args.verbose)
    if args.proc_spec:
        level, take = args.proc_spec
        proc.save_spec(level, take, verbose=args.verbose)

    all_resids = np.empty(0)
    for i_level, level in enumerate(levels):
        for i_take, take in enumerate(takes):
            # Load all important values
            thetas = np.load(savepath.format('thetas', level, take)+'.npy')
            xs = np.load(savepath.format('xs', level, take)+'.npy')
            ts = np.load(savepath.format('times', level, take)+'.npy')
            amps = np.load(savepath.format('amps', level, take)+'.npy')
            resids = np.load(savepath.format('resids', level, take)+'.npy')
            all_resids = np.r_[all_resids, resids] # Save for occurence plot later

            exp_params = np.loadtxt(savepath.format('exp_params', level, take)+'.txt')
            theory_params = np.loadtxt(savepath.format('theory_params', level, take)+'.txt')

            c, c_err, l, l_err, phi, chi_sqr_red, derbin, n0, y_err, h = exp_params
            c_t, c_t_err, l_t, l_t_err = theory_params

            if args.verbose and not args.proc_all:
                print('{:#^50}'.format('  Level: {:} --- Take: {:}  '.format(level, take)))
                print('C_t: {:.3f}±{:.3f}cm; L_t: {:.3f}±{:.3f}cm; '.format(c_t*100, c_t_err*100, l_t*100, l_t_err*100))
                print('C: {:.3f}±{:.3f}cms^-1; L: {:.3f}±{:.3f}cm; χ^2: {:.2f}; D: {:.2f}'.format(c*100, c_err*100, l*100, l_err*100, chi_sqr_red*amps.size, derbin))

            if args.graphsave or args.graphshow:
                plot.plot_resid(thetas, resids, i_take)
                plot.plot_lag(resids, i_take)
                for it, t in enumerate(ts):
                    psis = (xs-c*t)/l + phi
                    plot.plot_curve(psis, amps[:,it]/n0, n0, y_err/n0, i_level/2, i_take)

    if args.graphsave or args.graphshow:
        max_occ = plot.plot_occ(all_resids, 0.5)

        plot.style_curve()
        plot.style_residuals()
        plot.style_lag()
        plot.style_occ(max_occ)
    if args.graphsave:
        plot.save_plots('plots/main_plot.png')

    if args.graphshow:
        plot.show_plot()
