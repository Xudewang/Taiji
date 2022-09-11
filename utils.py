def set_matplotlib(style='default', usetex=False, fontsize=13, figsize=(6, 5), dpi=100):
    '''
    Default matplotlib settings, borrowed from Song Huang. I really like his plotting style.

    Parameters:
        style (str): options are "JL", "SM" (supermongo-like).
    '''

    import matplotlib.pyplot as plt
    from matplotlib.colorbar import Colorbar
    from matplotlib import rcParams
    # Use JL as a template
    if style == 'default':
        plt.style.use(os.path.join('/home/dewang/Taiji/', 'mplstyle/default.mplstyle'))
    else:
        plt.style.use(os.path.join('/home/dewang/Taiji/', 'mplstyle/JL.mplstyle'))
    rcParams.update({'font.size': fontsize,
                     'figure.figsize': "{0}, {1}".format(figsize[0], figsize[1]),
                     'text.usetex': usetex,
                     'figure.dpi': dpi})

    if style == 'DW':
        plt.style.use(['science', 'seaborn-colorblind'])

        plt.rcParams['figure.figsize'] = (10,7)
        plt.rcParams['font.size'] = 25
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams['xtick.labelsize'] = 25
        plt.rcParams['ytick.labelsize'] = 25
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['xtick.minor.pad'] = 4.8
        plt.rcParams['ytick.major.pad'] = 5
        plt.rcParams['ytick.minor.pad'] = 4.8
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['axes.labelpad'] = 8.0
        plt.rcParams['figure.constrained_layout.h_pad'] = 0
        plt.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.rcParams['legend.edgecolor'] = 'black'
        # plt.rcParams['xtick.major.width'] = 3.8
        # plt.rcParams['xtick.minor.width'] = 3.2
        # plt.rcParams['ytick.major.width'] = 3.8 
        # plt.rcParams['ytick.minor.width'] = 3.2
        # plt.rcParams['axes.linewidth'] = 5
        import matplotlib.ticker
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                       AutoMinorLocator)
        plt.close()

    if style == 'nature':
        rcParams.update({
            "font.family": "sans-serif",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.fontset": "stixsans"
        })
        