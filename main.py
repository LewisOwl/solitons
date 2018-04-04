import numpy as np
from argparse import ArgumentParser as argparser

from src.paths import savepath, datapath
from src.proc import levels, takes

import src.plot as plot
import src.proc as proc

if __name__ == '__main__':
    parser = argparser(description='Process wave data and gen plots')

    parser.add_argument('-pa', '--process_all', dest='proc_all', default=False,
                        action='store_true', help='Process all data again')
    parser.add_argument('-ps', '--process_spec', dest='proc_spec', nargs=2,
                        default=False, help='Process specific data set again. \
                        Args in the form "Level"cm "Take"')
    parser.add_argument('-gpm', '--graphplot', dest='graphshow', default=False,
                        action='store_true', help='Show a graph of data')
    parser.add_argument('-gsm', '--graphsave', dest='graphsave', default=False,
                        action='store_true', help='Save a graph of data')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False,
                        action='store_true', help='Print out parameter data')
    parser.add_argument('-lt', '--latexTable', help='Output data into text \
                         file with latex table style', dest='latextab',
                         default=False, action='store_true')
    parser.add_argument('-gss', '--graphsavesec', dest='graphsavesec', default=False,
                        action='store_true', help='Save a graph of data')
    args = parser.parse_args()

    if args.proc_all:
        proc.save_all(verbose=args.verbose)
    if args.proc_spec:
        level, take = args.proc_spec
        proc.save_spec(level, take, verbose=args.verbose)


    all_resids = np.empty(0)
    datas = np.empty(13)


    for i_level, level in enumerate(levels):
        n0cs = np.empty(4)
        for i_take, take in enumerate(takes):
            # Load all important values
            thetas = np.load(savepath.format('thetas', level, take)+'.npy')
            xs = np.load(savepath.format('xs', level, take)+'.npy')
            ts = np.load(savepath.format('times', level, take)+'.npy')
            amps = np.load(savepath.format('amps', level, take)+'.npy')
            resids = np.load(savepath.format('resids', level, take)+'.npy')
            # Save for occurence plot later
            all_resids = np.r_[all_resids, resids]

            exp_params = np.loadtxt(savepath.format('exp_params',
                                                    level, take)+'.txt')
            theory_params = np.loadtxt(savepath.format('theory_params',
                                                       level, take)+'.txt')

            (c, c_err, l, l_err, phi, chi_sqr_red,
             derbin, n0, y_err, h) = exp_params
            c_t, c_t_err, l_t, l_t_err = theory_params


            if args.graphsavesec:
                n0cs = np.c_[n0cs, np.r_[n0, y_err, c, c_err]]

            if args.latextab:
                data = np.r_[i_level/2, h*100, n0*100, c*100, c_err*100, l*100,
                             l_err*100, c_t*100, c_t_err*100, l_t*100,
                             l_t_err*100, chi_sqr_red, derbin]
                datas = np.c_[datas, data]

            if args.verbose and not args.proc_all and not args.proc_spec:
                border = '{:#^50}'
                header = '  Level: {:} --- Take: {:}  '.format(level, take)
                print(border.format(header))
                theory_str = 'C_t: {:.3f}±{:.3f}cm; L_t: {:.3f}±{:.3f}cm;'
                theory_str = theory_str.format(c_t*100, c_t_err*100,
                                               l_t*100, l_t_err*100)
                print(theory_str)
                exp_str = 'C: {:.3f}±{:.3f}cms^-1; L: {:.3f}±{:.3f}cm;'
                exp_str = exp_str.format(c*100, c_err*100, l*100, l_err*100)
                print(exp_str)
                fit_str = 'χ^2: {:.2f}; D: {:.2f}'
                fit_str = fit_str.format(chi_sqr_red, derbin)
                print(fit_str)

            if args.graphsave or args.graphshow:
                plot.plot_resid(thetas, resids, i_take)
                plot.plot_lag(resids, i_take)
                for it, t in enumerate(ts):
                    psis = (xs-c*t)/l + phi
                    norm_amps = amps[:, it]/n0
                    errs = y_err/n0 * (1 + norm_amps**2)**0.5
                    plot.plot_curve(psis, norm_amps, errs, i_level/2, i_take)

        if args.graphsavesec:
            n0cs = n0cs[:,1:]
            n0s, n0_errs, cs, c_errs = n0cs
            plot.plot_n0c(n0s, n0_errs, cs, c_errs, float(level)/100, 0.005, i_level)

    if args.latextab:
        header = 'i    h      n0       c     err      l       err     c_t     err   l_t    err   chi  derbin'
        fmt = '%.1f & %.2f & %.2f & %.2f $\pm$ %.2f & %.3f $\pm$ %.3f & %.1f $\pm$ %.1f & %.1f $\pm$ %.1f & %.2f & %.2f'
        np.savetxt('latex_table.txt', datas[:,1:].T, fmt=fmt, delimiter='&',
                   newline=' \\\ \hline \n', header=header)

    if args.graphsave or args.graphshow:
        max_occ = plot.plot_occ(*plot.calc_occ(all_resids, 0.5))
        plot.style_curve()
        plot.style_residuals()
        plot.style_lag()
        plot.style_occ(max_occ)

    if args.graphsavesec:
        plot.save_sec('plots/sec.png')
    if args.graphsave:
        plot.save_main('plots/main.png')
    if args.graphshow:
        plot.show_main()
