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

args = parser.parse_args()



df = pd.read_csv(args.csv)

filter_keys = [] if args.filter == '' else args.filter.split(',')
for f in filter_keys:
    k,v = f.split('=')
    k = k.strip()
    v = v.strip()
    df = df[df[k].astype(str) == v]

if args.print:
    print(df)

#plt.ylim(0.0, 1.0)
#plt.xlim(0, 1050)

plt.title(args.title)
if args.category == '':
    sns.lineplot(df[args.plotx], df[args.ploty])
else:
    num_values = np.unique(df[args.category]).size
    #plt.errorbar(x=df[args.plotx], y=df[args.ploty], fmt='none', xerror=df['SE'], ecolor='k', elinewidth=2)
    print(df[args.plotx])
    print(df[args.ploty])
    print(df['SE'])
    sns.lineplot(x=args.plotx, y=args.ploty,data=df, hue=args.category, palette=sns.color_palette('colorblind', num_values), legend='full')

plt.savefig(args.out)
