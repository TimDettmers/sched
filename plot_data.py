import os
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(palette="colorblind")

parser = argparse.ArgumentParser(description='Plot script for evaluation data csv.')
parser.add_argument('--csv', type=str, default='', help='Prints all argparse arguments with differences.')
parser.add_argument('--plotx', type=str, default='', help='The column name to plot on x')
parser.add_argument('--ploty', type=str, default='Mean', help='The column name to plot on y. Default: Mean.')
parser.add_argument('--filter', type=str, default='', help='Argument(s) which should be kept by value (arg=value). Multiple arguments separated with a comma.')
parser.add_argument('--out', '-o', type=str, default='', help='The output path')
parser.add_argument('--print', action='store_true', help='Prints the dataframe before it is being plotted.')
parser.add_argument('--category', type=str, default='', help='Plot all different values for a category into one plot.')
parser.add_argument('--title', type=str, default='', help='Title of the plot.')
parser.add_argument('--namey', type=str, default=None, help='Name of the y-axis')
parser.add_argument('--categoricalx', action='store_true', help='Treat x variable as categorical')
parser.add_argument('--median', action='store_true', help='Use median for plotting y-values instead of mean.')
parser.add_argument('--ci', action='store_true', help='Plot confidence intervals through the value column')
parser.add_argument('--swarm', action='store_true', help='Plot swarm instead of confidence intervals')
parser.add_argument('--tick-rotation', type=int, default=0, help='By how much to rotate the x-axis label')
parser.add_argument('--bottom', type=int, default=None, help='Filter out all the scores except the botton n entries.')
parser.add_argument('--top', type=int, default=None, help='Filter out all the scores except the top n entries.')
parser.add_argument('--scale', type=float, default=None, help='Multiply metric by this value.')
parser.add_argument('--ylim', nargs='+', type=float, default=None, help='Sets the [min, max] range of the metric value (two space separated values).')

args = parser.parse_args()

#args.ploty = 'Median' if args.median else args.ploty
#args.ploty = 'Value' if args.ci else args.ploty
#args.ploty = 'Value' if args.swarm else args.ploty
args.namey = args.ploty if args.namey is None else args.namey

if args.out == '': args.out = args.csv.replace('csv','png')
if not os.path.exists(os.path.dirname(args.out)):
    os.makedirs(os.path.dirname(args.out))

df = pd.read_csv(args.csv, sep=';')

filter_keys = [] if args.filter == '' else args.filter.split(',')
for f in filter_keys:
    k,v = f.split('=')
    k = k.strip()
    v = v.strip()
    df = df[df[k].astype(str) == v]

if args.bottom is not None or args.top is not None:
    if args.categoricalx:
        dfs = []
        for category in df[str(args.plotx)].unique():
            dfs.append(df[df[str(args.plotx)] == category].sort_values(by=args.ploty))
            if args.bottom is not None: dfs[-1] = dfs[-1].head(args.bottom)
            if args.top is not None: dfs[-1] = dfs[-1].tail(args.top)
        df = pd.concat(dfs)

    else:
        df = df.sort_values(by=args.ploty)
        if args.bottom is not None: df = df.head(args.bottom)
        if args.top is not None: df = df.tail(args.top)

if args.scale is not None:
    df[args.ploty] = df[args.ploty]*args.scale

if args.print:
    print(df)

#plt.xlim(0.0, 250)
#plt.xlim(0, 1050)

plt.title(args.title, fontsize=18)
if args.category == '':
    #sns.lineplot(df[args.plotx], df[args.ploty])
    ax = sns.regplot(x=args.plotx, y=args.ploty,data=df, scatter=True)
    plt.subplots_adjust(top=0.9)
    #ax.fig.suptitle(args.title, fontsize=18)
    #ax.set_xticklabels(rotation=args.tick_rotation)
else:
    num_values = np.unique(df[args.category]).size
    #plt.errorbar(x=df[args.plotx], y=df[args.ploty], fmt='none', xerror=df['SE'], ecolor='k', elinewidth=2)
    if args.categoricalx:
        if args.swarm:
            ax = sns.catplot(x=args.plotx, y=args.ploty,data=df, hue=args.category, palette=sns.color_palette('colorblind', num_values), legend='full', kind='swarm')
        else:
            ax = sns.catplot(x=args.plotx, y=args.ploty,data=df, hue=args.category, palette=sns.color_palette('colorblind', num_values), legend='full', kind='point', ci=95.0 if args.ci else None, err_style='bars')
        plt.subplots_adjust(top=0.9)
        ax.fig.suptitle(args.title, fontsize=18)
        ax.set_xticklabels(rotation=args.tick_rotation)
    else:
        print(df)
        #sns.lineplot(x=args.plotx, y=args.ploty,data=df, hue=args.category, palette=sns.color_palette('colorblind', num_values), legend='full', ci=95.0 if args.ci else None, err_style='bars')
        #df[args.plotx] = df[args.plotx]/df[args.plotx].max()
        #print(df[args.plotx])
        ax = sns.relplot(x=args.plotx, y=args.ploty,data=df, hue=args.category, palette=sns.color_palette('colorblind', num_values), legend='full')
        plt.subplots_adjust(top=0.9)
        ax.fig.suptitle(args.title, fontsize=18)

if args.ylim is not None:
    plt.ylim(*args.ylim)

plt.ylabel(args.namey, fontsize=13)
plt.savefig(args.out, bbox_inches='tight')
