#!/bin/bash

#PPL

python plot_data.py --csv ~/plots/csv/moe/wt/corr_data.csv --plotx "specz score 0" --ploty "ppl" "rdmppl" --namey 'Valid Perplexity' --title "PPL by Specialization Score" --swarm
#python plot_data.py --csv ~/plots/csv/moe/wt/corr_data.csv --plotx "specz score 0" --ploty "ppl" "rdmppl" --namey 'Valid Perplexity' --title "PPL by Specialization Score" --swarm --ptype pairwise
