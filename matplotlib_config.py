from matplotlib import rc

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
rc('font', family = 'serif', size = 8)
rc('grid', linestyle = '--', alpha = .5)
rc('axes', axisbelow = True, grid = True)
rc('lines', linewidth = 1.33, markersize = 1.5)
rc('axes.spines', top = False, right = False, left = True, bottom = True)