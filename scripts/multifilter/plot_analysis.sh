#!/bin/bash


FOLDER=~/transfer/multifilter/analysis
mkdir $FOLDER -p

python plot_data.py --csv ~/num_filter2.csv --plotx num_filters --namey 'Perplexity' -o $FOLDER/num_filters.png --category filter_size --title "Performance of filter size vs #filters"
