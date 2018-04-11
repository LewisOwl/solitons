import numpy as np
from argparse import ArgumentParser as argparser

from src.paths import savepath, datapath
from src.proc import levels, takes
from src.models import theory_c, theory_l, sech2
from src.styles import s_style, l_style

import src.plot as plot
import src.proc as proc



if __name__ == '__main__':
    parser = argparser(description='Process wave data and gen plots')

    parser.add_argument('-pa', '--process_all', dest='proc_all', default=False,
                        action='store_true', help='Process all data again')
    parser.add_argument('-ps', '--process_spec', dest='proc_spec', nargs=2,
                        default=False, help='Process specific data set again. \
                        Args in the form "Level"cm "Take"')
    parser.add_argument('-gsm', '--graphsavemain', dest='gsmain', default=False,
                        action='store_true', help='Save a graph of data')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False,
                        action='store_true', help='Print out parameter data')
    parser.add_argument('-lt', '--latexTable', help='Output data into text \
                         file with latex table style', dest='latextab',
                         default=False, action='store_true')
    parser.add_argument('-gss', '--graphsavespeed', dest='gsspeed', default=False,
                        action='store_true', help='Save a graph of speeds')
    parser.add_argument('-gsl', '--graphsavelength', dest='gslength', default=False,
                        action='store_true', help='Save a graph of lengths')
    args = parser.parse_args()

    # Process data
    if args.proc_all:
        proc.save_all(verbose=args.verbose)
    if args.proc_spec:
        level, take = args.proc_spec
        proc.save_spec(level, take, verbose=args.verbose)

    # Set data stores for plotting
    if args.gsmain:
        mfig, mcurve, mresids, mocc, mlag = plot.gen_main_axes()
        all_resids = np.empty(0)
    if args.gsspeed:
        sfig, sback, saxs, sresids, socc, slag = plot.gen_sec_axes(r'Maximum Amplitude $\eta_{0}\,/\,cm$', r'Wave Speed $|c|\,/\,cm \cdot s^{-2}$')
        speed_resids = np.empty(0)
    if args.gslength:
        lfig, lback, laxs, lresids, locc, llag = plot.gen_sec_axes(r'Maximum Amplitude $\eta_{0}\,/\,cm$', r'Characteristic Length $L\,/\,cm$')
        length_resids = np.empty(0)

    if args.latextab:
        datas = np.empty(13)



    for i_level, level in enumerate(levels):
        if args.gsspeed:
            n0cs = np.empty(4)
        if args.gslength:
            n0ls = np.empty(4)
        for i_take, take in enumerate(takes):
            # Load all important values
            thetas = np.load(savepath.format('thetas', level, take)+'.npy')
            xs = np.load(savepath.format('xs', level, take)+'.npy')
            ts = np.load(savepath.format('times', level, take)+'.npy')
            amps = np.load(savepath.format('amps', level, take)+'.npy')

            # Save for occurence plot later
            if args.gsmain:
                resids = np.load(savepath.format('resids', level, take)+'.npy')
                all_resids = np.r_[all_resids, resids]

            exp_params = np.loadtxt(savepath.format('exp_params',
                                                    level, take)+'.txt')
            theory_params = np.loadtxt(savepath.format('theory_params',
                                                       level, take)+'.txt')

            (c, c_err, l, l_err, phi, chi_sqr_red,
             derbin, n0, y_err, h) = exp_params
            c_t, c_t_err, l_t, l_t_err = theory_params


            if args.gsspeed:
                n0cs = np.c_[n0cs, np.r_[n0, y_err, c, c_err]]
            if args.gslength:
                n0ls = np.c_[n0ls, np.r_[n0, y_err, l, l_err]]

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

            if args.gsmain:
                plot.plot_resid(mresids, thetas, resids, i_take, resid_step=75, s=3)
                plot.plot_lag(mlag, resids, i_take, lag_step=25, s=3, alpha=0.3)
                for it, t in enumerate(ts):
                    psis = (xs-c*t)/l + phi
                    norm_amps = amps[:, it]/n0
                    errs = y_err/n0 * (1 + norm_amps**2)**0.5
                    plot.scatter_error(mcurve, psis, 0, norm_amps + i_level/2, errs, i_take, marker_step=200, markersize=3)
                    bottoms = np.linspace(-5, 5, 1000)
                    ys = 1/np.cosh(bottoms)**2 + i_level/2
                    plot.plot_curve(mcurve, bottoms, ys, None)

        if args.gsspeed:
            n0cs = n0cs[:,1:]
            n0s, n0_errs, cs, c_errs = n0cs
            c_errs = c_errs
            height = float(level)/100
            sax = saxs[i_level]
            plot.scatter_error(sax, n0s*100, n0_errs*100, cs*100, c_errs*100, i_level)
            # Plot theory lines
            xs = np.linspace(np.min(n0s)*0.9, np.max(n0s)*1.1, 100)
            ys, y_errs = theory_c(xs, n0_errs[0], height, 0.005)
            plot.plot_curve(sax, xs*100, np.absolute(ys)*100, i_level)
            theorycs, _ = theory_c(n0s, n0_errs[0], height, 0.005)
            resids = (np.absolute(cs) - np.absolute(theorycs))/c_errs
            plot.plot_resid(sresids, n0s*100, resids, i_level)
            plot.plot_lag(slag, resids, i_level)
            speed_resids = np.r_[speed_resids, resids]
            # Fix twatty labels on that axis!
            saxs[-1].set_yticks([108, 110, 112, 114, 116, 118, 120])

        if args.gslength:
            n0ls = n0ls[:,1:]
            n0s, n0_errs, ls, l_errs = n0ls
            ls = ls * 2**0.5
            l_errs = l_errs * 4
            height = float(level)/100
            lax = laxs[i_level]
            plot.scatter_error(lax, n0s*100, n0_errs*100, ls*100, l_errs*100, i_level)
            # Plot theory lines
            xs = np.linspace(np.min(n0s)*0.9, np.max(n0s)*1.1, 100)
            ys, y_errs = theory_l(xs, n0_errs[0], height, 0.005)
            plot.plot_curve(lax, xs*100, ys*100, i_level)
            theoryls, _ = theory_l(n0s, n0_errs[0], height, 0.005)
            resids = (np.absolute(ls) - np.absolute(theoryls))/l_errs
            plot.plot_resid(lresids, n0s*100, resids, i_level)
            plot.plot_lag(llag, resids, i_level)
            length_resids = np.r_[length_resids, resids]

    if args.latextab:
        header = 'i    h      n0       c     err      l       err     c_t     err   l_t    err   chi  derbin'
        fmt = '%.1f & %.2f & %.2f & %.2f $\pm$ %.2f & %.3f $\pm$ %.3f & %.1f $\pm$ %.1f & %.1f $\pm$ %.1f & %.2f & %.2f'
        np.savetxt('latex_table.txt', datas[:,1:].T, fmt=fmt, delimiter='&',
                   newline=' \\\ \hline \n', header=header)

    if args.gsmain:
        r_ylims = (-5, 5)
        xlims = [-4.1, 4.1]
        r_lims = [xlims, r_ylims]
        max_occ = plot.plot_occ(mocc, *plot.calc_occ(all_resids, 0.5, r_ylims), norm=True)
        plot.style_main_axes(mcurve, mresids, mocc, mlag, r_lims, max_occ)
        plot.save(mfig, 'plots/main.png')

    if args.gsspeed:
        _, _, _, (_, r_ylims) = s_style
        max_occ = plot.plot_occ(socc, *plot.calc_occ(speed_resids, 2, r_ylims))
        plot.style_sec_axes(saxs, sresids, socc, slag, s_style, max_occ)
        plot.save(sfig, 'plots/speed.png')
    if args.gslength:
        _, _, _, (_, r_ylims) = l_style
        max_occ = plot.plot_occ(locc, *plot.calc_occ(length_resids, 2, r_ylims))
        plot.style_sec_axes(laxs, lresids, locc, llag, l_style, max_occ)
        plot.save(lfig, 'plots/length.png')
