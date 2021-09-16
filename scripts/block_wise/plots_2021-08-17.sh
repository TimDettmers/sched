#!/bin/bash

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'lr' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs lr" --swarm --out ~/plots/png/block_wise/lr.png --categoricalx  

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'adam_betas' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs betas" --swarm --out ~/plots/png/block_wise/betas.png --categoricalx  

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'adam_eps' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs eps" --swarm --out ~/plots/png/block_wise/eps.png --categoricalx  

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'stable_emb' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs stable embedding" --swarm --out ~/plots/png/block_wise/stable_emb.png --categoricalx  
