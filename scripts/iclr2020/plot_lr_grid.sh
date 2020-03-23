#!/bin/bash

python plot_data.py --csv ~/data/csv/lr_grid_ICLR.csv --plotx lr --namey 'Test Accuracy' -o ~/data/plots/lr_grid.png --category model --title "Median Test Accuracy for different LRs" --categoricalx --median
python plot_data.py --csv ~/data/csv/lr_grid_ICLR.csv --plotx lr --namey 'Test Accuracy' -o ~/data/plots/lr_grid.pdf --category model --title "Median Test Accuracy for different LRs" --categoricalx --median

