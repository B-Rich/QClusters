import matplotlib.pyplot as plt
import matplotlib
from numpy import mean
from Dynamic import rsFilename, asFilename
import os

font = {'family' : 'Arial',
        'size'   :  12}
matplotlib.rc('font', **font)
matplotlib.rcParams['toolbar'] = 'None'
homeDir = os.environ['HOME']

def filePlot(filename, plotMode = "onlyValues", mk = "x"):
    xlist, ylist, xErrList, yErrList = [], [], [], []
    file = open(filename, 'r')
    for line in file:
        data = line.split(" ")
        xlist.append(float(data[0]))
        ylist.append(float(data[1]))
    file.close()
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(xlist, ylist, "^", c="#50a7f9", ls="-", lw=1.5, ms=9, markeredgecolor='none')
    if plotMode == "onlyValues":
        pass
    elif plotMode == "xError":
        plt.errorbar(xlist, ylist, xerr = xErrList, fmt = mk)
    elif plotMode == "yError":
        plt.errorbar(xlist, ylist, yerr = yErrList, fmt = mk, ls = "-")
    elif plotMode == "xyError":
        plt.errorbar(xlist, ylist, xerr = xErrList, yerr = yErrList,
                     fmt = mk)
    else:
        raise ValueError('Unknown plot mode')
    plt.show()

def glimpse(index, filename):
    file = open(filename, 'r')
    for n, line in enumerate(file):
        if n == index:
            data = line.split("; ")
            smean, series = float(data[2]), \
                            map(float, data[3].split(" "))
            smin, smax = min(series), max(series)
            break
    file.close()
    plt.plot(range(len(series)), series, "x", ls = "-")
    plt.plot(range(len(series)), [mean(series)]*len(series), ls = "-.")
    plt.show()

# Try to uncomment one of these lines
filePlot(asFilename)
#filePlot(rsFilename)