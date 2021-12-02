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
    #df[yaxis] = pd.to_numeric(df[yaxis], errors='coerce')  # convert NaN string to NaN value
    df = df.dropna(axis=0)

    xaxisGroup = df.groupby(xaxis)

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = xaxisGroup.indices.keys() #df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)
    
    #plt.ylim([0,200])
    #plt.xlim([40000, 70000])


# Get run id from file, and metrics from it
def analyzeRun(dataFrame, fileName):

    # Extract ID from filename
    run_ID = fileName.find("_run")
    if run_ID is None:
        assert False, f"Can't find _runNUM in {fileName}"

    run_ID = parse.parse("_run{:d}.csv", fileName[run_ID:])
    if run_ID is None:
        assert False, f"Can't find _runNUM in {fileName}"

    run_ID = int(run_ID[0])

    max_total_wait_time = df["total_wait_time"].max()
    avg_total_wait_time = df["total_wait_time"].mean()

    max_total_stopped = df["total_stopped"].max()
    avg_total_stoppped = df["total_stopped"].mean()

    return run_ID, max_total_wait_time, avg_total_wait_time, max_total_stopped, avg_total_stoppped

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Plot Traffic Signal Metrics""")  
    prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")            
    prs.add_argument('-l', nargs='+', default=None, help="File's legends\n")
    prs.add_argument('-title', type=str, default="", help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default='total_wait_time', help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default='step_time', help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument('-sep', type=str, default=',', help="Values separator on file.\n")
    prs.add_argument('-xlabel', type=str, default='Second', help="X axis label.\n") 
    prs.add_argument('-ylabel', type=str, default='Total waiting time (s)', help="Y axis label.\n")    
    prs.add_argument('-output', type=str, default=None, help="PDF output filename.\n")
   
    args = prs.parse_args()
    labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])

    plt.figure()

    # File reading and grouping
    bestRun_maxWaitingTime_ID = None
    bestRun_avgWaitingTime_ID = None
    bestRun_maxWaitingTime_Value = None
    bestRun_avgWaitingTime_Value = None

    bestRun_maxVehiclesStopped_ID = None
    bestRun_avgVehiclesStopped_ID = None
    bestRun_maxVehiclesStopped_Value = None
    bestRun_avgVehiclesStopped_Value = None

    for file in args.f:
        main_df = pd.DataFrame()
        for f in glob.glob(file+'*'):
            try:
                df = pd.read_csv(f, sep=args.sep)# usecols=["step_time", "reward", "total_stopped", "total_wait_time"])

                # Get run id from file, and metrics from it
                run_ID, maxWaitingTime, avgWaitingTime, maxVehiclesStopped, avgVehiclesStopped = analyzeRun(df, f)

                # Augment current maximum values
                if bestRun_avgWaitingTime_ID is None or bestRun_avgWaitingTime_Value > avgWaitingTime:
                    bestRun_avgWaitingTime_ID = f
                    bestRun_avgWaitingTime_Value = avgWaitingTime

                if bestRun_maxWaitingTime_ID is None or bestRun_maxWaitingTime_Value > maxWaitingTime:
                    bestRun_maxWaitingTime_ID = f
                    bestRun_maxWaitingTime_Value = maxWaitingTime

                # Augment current maximum values
                if bestRun_avgVehiclesStopped_ID is None or bestRun_avgVehiclesStopped_Value > avgVehiclesStopped:
                    bestRun_avgVehiclesStopped_ID = f
                    bestRun_avgVehiclesStopped_Value = avgVehiclesStopped

                if bestRun_maxVehiclesStopped_ID is None or bestRun_maxVehiclesStopped_Value > maxVehiclesStopped:
                    bestRun_maxVehiclesStopped_ID = f
                    bestRun_maxVehiclesStopped_Value = maxVehiclesStopped

            except:
                continue
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        # Process data a bit...
        #main_df[args.xaxis] = df[args.xaxis].str.replace('.0.0', '0.0')

        # Plot DataFrame
        plot_df(main_df,
                xaxis=args.xaxis,
                yaxis=args.yaxis,
                label=next(labels),
                color=next(colors),
                ma=args.ma,
                )

        print(f"For method output {args.output}:")
        print(f"Best Run max WaitTime: ID {bestRun_maxWaitingTime_ID}, value {bestRun_maxWaitingTime_Value}")
        print(f"Best Run avg WaitTime: ID {bestRun_avgWaitingTime_ID}, value {bestRun_avgWaitingTime_Value}")
        print(f"Best Run max Stopped: ID {bestRun_maxVehiclesStopped_ID}, value {bestRun_maxVehiclesStopped_Value}")
        print(f"Best Run avg Stopped: ID {bestRun_avgVehiclesStopped_ID}, value {bestRun_avgVehiclesStopped_Value}")

    plt.title(args.title)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.ylim(bottom=0)

    if args.output is not None:
        plt.savefig(args.output+'.pdf', bbox_inches="tight")

    plt.show()