import numpy as np
from argparse import ArgumentParser as argparser
import matplotlib.cm as cm

from src.paths import savepath
from src.proc import levels, takes
from src.models import theory_c, theory_l
from src.styles import s_style, l_style

import src.plot as plot
import src.proc as proc

cmap = cm.winter

if __name__ == '__main__':
    parser = argparser(description='Process wave data and gen plots')

    parser.add_argument('-pa', '--process_all', dest='proc_all', default=False,
                        action='store_true', help='Process all data again')
    parser.add_argument('-ps', '--process_spec', dest='proc_spec', nargs=2,
                        default=False, help='Process specific data set again. \
                        Args in the form "Level"cm "Take"')
    parser.add_argument('-gsm', '--graphsavemain', dest='gsmain', default=False,
                        action='store_true', help='Save a graph of data')
    parser.add_argument('-vv', '--veryverbose', dest='veryverbose', default=False,
                        action='store_true', help='Print out parameter data')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False,
                        action='store_true', help='Print out graph data')
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
        proc.save_all(verbose=args.veryverbose)
    if args.proc_spec:
        level, take = args.proc_spec
        proc.save_spec(level, take, verbose=args.veryverbose)

    # Set data stores for plotting
    if args.gsmain:
        mcolors = cmap(np.linspace(0, 1, 10))
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
        if args.verbose:
            tot_chi = 0
            tot_derb = 0
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

            if args.verbose:
                tot_chi += chi_sqr_red
                tot_derb += derbin
            if args.gsspeed:
                n0cs = np.c_[n0cs, np.r_[n0, y_err, c, c_err]]
            if args.gslength:
                n0ls = np.c_[n0ls, np.r_[n0, y_err, l, l_err]]

            if args.latextab:
                data = np.r_[i_level/2, h*100, n0*100, c*100, c_err*100, l*100,
                             l_err*100, c_t*100, c_t_err*100, l_t*100,
                             l_t_err*100, chi_sqr_red, derbin]
                datas = np.c_[datas, data]

            if args.veryverbose and not args.proc_all and not args.proc_spec:
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
                colors = cmap(np.linspace(0, 1, 10))
                plot.plot_resid(mresids, thetas, resids, i_take, resid_step=75, s=3, colors=colors)
                plot.plot_lag(mlag, resids, i_take, lag_step=25, s=3, alpha=0.3, colors=colors)
                for it, t in enumerate(ts):
                    psis = (xs-c*t)/l + phi
                    norm_amps = amps[:, it]/n0
                    errs = y_err/n0 * (1 + norm_amps**2)**0.5
                    plot.scatter_error(mcurve, psis, 0, norm_amps + i_level/2, errs, i_take, marker_step=200, markersize=3, colors=colors)
                    bottoms = np.linspace(-5, 5, 1000)
                    ys = 1/np.cosh(bottoms)**2 + i_level/2
                    plot.plot_curve(mcurve, bottoms, ys, None)

        if args.verbose:
            border = '{:#^10}'
            header = 'Level: {:}; '.format(level)
            stats = 'Average χ^2: {:.2f}; Average D: {:.3f};'.format(tot_chi/len(takes), tot_derb/len(takes))
            print(border.format(header + stats))


        if args.gsspeed:
            # Remove zero that initiated array
            n0cs = n0cs[:,1:]
            # Order all values by n0
            n0cs = n0cs[:, np.argsort(n0cs[0])]
            n0s, n0_errs, cs, c_errs = n0cs
            cs = np.absolute(cs)
            c_errs = c_errs*2
            height = float(level)/100
            sax = saxs[i_level]
            plot.scatter_error(sax, n0s*100, n0_errs*100, cs*100, c_errs*100, i_level)
            # Plot theory lines and there error bounds
            xs = np.linspace(np.min(n0s)*0.9, np.max(n0s)*1.1, 100)
            ys, y_errs = theory_c(xs, n0_errs[0], height, 0.001)
            plot.plot_curve(sax, xs*100, np.absolute(ys)*100, i_level)
            plot.plot_curve(sax, xs*100, (np.absolute(ys + y_errs))*100, i_level, color='grey')
            plot.plot_curve(sax, xs*100, (np.absolute(ys - y_errs))*100, i_level, color='grey')
            theorycs, errs = theory_c(n0s, n0_errs[0], height, 0.001)
            norm_errors = (np.absolute(errs) + np.absolute(c_errs))
            resids = (np.absolute(cs) - np.absolute(theorycs))/norm_errors
            plot.scatter_error(sresids, n0s*100, n0_errs*100, resids, c_errs/norm_errors, i_level)
            plot.plot_lag(slag, resids, i_level)
            speed_resids = np.r_[speed_resids, resids]
            # Fix twatty labels on that axis!
            saxs[-1].set_yticks([108, 110, 112, 114, 116, 118, 120])
            if args.verbose:
                chi_sqr = np.sum(resids**2)
                dw = np.sum((resids[1:] - resids[:-1])**2)/chi_sqr
                print('Speed --- h: {:.2f}; χ^2: {:.2f}; D: {:.2f}; cav: {:.3f}; n0av: {:.3f}'.format(height*100, chi_sqr/resids.size, dw, np.mean(cs)*100, np.mean(n0s)*100))

        if args.gslength:
            # Remove zero that initiated array
            n0ls = n0ls[:,1:]
            # Order all values by n0
            n0ls = n0ls[:, np.argsort(n0ls[0])]
            n0s, n0_errs, ls, l_errs = n0ls
            ls = ls * 2**0.5
            l_errs = l_errs*4
            height = float(level)/100
            lax = laxs[i_level]
            plot.scatter_error(lax, n0s*100, n0_errs*100, ls*100, l_errs*100, i_level)
            # Plot theory lines
            xs = np.linspace(np.min(n0s)*0.9, np.max(n0s)*1.1, 100)
            ys, y_errs = theory_l(xs, n0_errs[0], height, 0.001)
            plot.plot_curve(lax, xs*100, ys*100, i_level)
            plot.plot_curve(lax, xs*100, (np.absolute(ys + y_errs))*100, i_level, color='grey')
            plot.plot_curve(lax, xs*100, (np.absolute(ys - y_errs))*100, i_level, color='grey')
            theoryls, errs = theory_l(n0s, n0_errs[0], height, 0.001)
            norm_errors = (np.absolute(errs) + np.absolute(l_errs))
            resids = (np.absolute(ls) - np.absolute(theoryls))/norm_errors
            plot.scatter_error(lresids, n0s*100, n0_errs*100, resids, l_errs/norm_errors, i_level)
            plot.plot_lag(llag, resids, i_level)
            length_resids = np.r_[length_resids, resids]
            if args.verbose:
                chi_sqr = np.sum(resids**2)
                dw = np.sum((resids[1:] - resids[:-1])**2)/chi_sqr
                print('Length --- h: {:.2f}; χ^2: {:.2f}; D: {:.2f}; Lav: {:.3f}; n0av: {:.3f}'.format(height*100, chi_sqr/resids.size, dw, np.mean(ls)*100, np.mean(n0s)*100))

    if args.latextab:
        header = 'i    h      n0       c     err      l       err     c_t     err   l_t    err   chi  derbin'
        fmt = '%.1f & %.2f & %.2f & %.2f $\pm$ %.2f & %.3f $\pm$ %.3f & %.1f $\pm$ %.1f & %.1f $\pm$ %.1f & %.2f & %.2f'
        np.savetxt('latex_table.txt', datas[:,1:].T, fmt=fmt, delimiter='&',
                   newline=' \\\ \hline \n', header=header)

    if args.gsmain:
        onesigma = np.sum(np.absolute(all_resids) < 1) / all_resids.size
        twosigma = np.sum(np.absolute(all_resids) < 2) / all_resids.size
        print('All Data σ: {:.3f}; 2σ: {:.3f}; Mean: {:.3f}'.format(onesigma, twosigma, np.mean(all_resids)))
        r_ylims = (-5, 5)
        xlims = [-4.1, 4.1]
        r_lims = [xlims, r_ylims]
        max_occ = plot.plot_occ(mocc, *plot.calc_occ(all_resids, 0.5, r_ylims), norm=True)
        plot.style_main_axes(mcurve, mresids, mocc, mlag, r_lims, max_occ)
        plot.save(mfig, 'plots/main.eps')

    if args.gsspeed:
        onesigma = np.sum(np.absolute(speed_resids) < 1) / speed_resids.size
        twosigma = np.sum(np.absolute(speed_resids) < 2) / speed_resids.size
        print('Speed σ: {:.3f}; 2σ: {:.3f}; Mean: {:.3f}'.format(onesigma, twosigma, np.mean(speed_resids)))
        _, _, _, (_, r_ylims) = s_style
        max_occ = plot.plot_occ(socc, *plot.calc_occ(speed_resids, 0.5, r_ylims), norm=True)
        plot.style_sec_axes(saxs, sresids, socc, slag, s_style, max_occ)
        slag.set_xticks([-4, -2, 0, 2, 4])
        plot.save(sfig, 'plots/speed.eps')

    if args.gslength:
        onesigma = np.sum(np.absolute(length_resids) < 1) / length_resids.size
        twosigma = np.sum(np.absolute(length_resids) < 2) / length_resids.size
        print('Length σ: {:.3f}; 2σ: {:.3f}; Mean: {:.3f}'.format(onesigma, twosigma, np.mean(length_resids)))
        _, _, _, (_, r_ylims) = l_style
        max_occ = plot.plot_occ(locc, *plot.calc_occ(length_resids, 0.5, r_ylims), norm=True)
        plot.style_sec_axes(laxs, lresids, locc, llag, l_style, max_occ)
        llag.set_xticks([-4, -2, 0, 2, 4])
        plot.save(lfig, 'plots/length.eps')
