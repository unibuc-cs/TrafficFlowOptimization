import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
from itertools import cycle
import parse

sns.set(style='darkgrid', rc={'figure.figsize': (7.2, 4.45),
                            'text.usetex': True,
                            'xtick.labelsize': 16,
                            'ytick.labelsize': 16,
                            'font.size': 15,
                            'figure.autolayout': True,
                            'axes.titlesize' : 16,
                            'axes.labelsize' : 17,
                            'lines.linewidth' : 2,
                            'lines.markersize' : 6,
                            'legend.fontsize': 15})
colors = sns.color_palette("colorblind", 4)
#colors = sns.color_palette("Set1", 2)
#colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(['-', '-.', '--', ':'])
sns.set_palette(colors)
colors = cycle(colors)

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def plot_df(df, color, xaxis, yaxis, ma=1, label=''):
    # df[yaxis] = pd.to_numeric(df[yaxis], errors='coerce')  # convert NaN string to NaN value
    df = df.dropna(axis=0)

    xvalues = df[xaxis]
    yvalues = moving_average(df[yaxis], ma)

    #x = df.indices.keys()  # df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(df[xaxis], yvalues, label=label, color=color)#, linestyle=next(dashes_styles))


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Plot Traffic Signal Metrics""")
    prs.add_argument('-f', required=True, help="Files list to compare methods against\n")
    prs.add_argument('-names', required=True, help="Names of the methods in the same order as files given\n")
    prs.add_argument('-title', type=str, default="", help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default='total_wait_time', help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default='step_time', help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument('-sep', type=str, default=',', help="Values separator on file.\n")
    prs.add_argument('-xlabel', type=str, default='Second', help="X axis label.\n")
    prs.add_argument('-ylabel', type=str, default='Total waiting time (s)', help="Y axis label.\n")
    prs.add_argument('-output', type=str, default=None, help="PDF output filename.\n")

    args = prs.parse_args()

    plt.figure()

    filesList = args.f.split(",")
    methodNames = args.names.split(",")

    # For each file, plot stuff
    for file, methodName in zip(filesList, methodNames):
        dataContent = pd.read_csv(file, sep=args.sep)  # usecols=["step_time", "reward", "total_stopped", "total_wait_time"])

        # Plot DataFrame
        plot_df(dataContent,
                xaxis=args.xaxis,
                yaxis=args.yaxis,
                color=next(colors),
                ma=args.ma,
                label=methodName
                )

    plt.title(args.title)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.legend()
    plt.ylim(bottom=0)

    if args.output is not None:
        plt.savefig(args.output + '.pdf', bbox_inches="tight")

    plt.show()