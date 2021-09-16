#!/bin/bash

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/cc_small.csv --ploty "ppl" --plotx 'lr' --category 'optim_bits' --namey 'Valid Perplexity' --title "Percentile vs lr" --swarm --out ~/plots/png/block_wise/lr.png --categoricalx  
#
#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'adam_betas' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs betas" --swarm --out ~/plots/png/block_wise/betas.png --categoricalx  
#
#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'adam_eps' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs eps" --swarm --out ~/plots/png/block_wise/eps.png --categoricalx  
#
#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/cc_small.csv --ploty "ppl" --plotx 'stable_emb' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Percentile vs stable embedding" --swarm --out ~/plots/png/block_wise/stable_emb.png --categoricalx  

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/ablations.csv --ploty "ppl" --plotx 'Learning Rate' --category 'Bits' --namey 'Valid Perplexity' --title "" --swarm --out ~/plots/png/block_wise/lr.png  --filter adam_eps=1e-07 adam_betas="0.9, 0.995" --tick-rotation 45 --rename optim_bits="Bits" lr="Learning Rate" --ylim 16 18

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/ablations.csv --ploty "ppl" --plotx 'Adam Betas' --category 'Bits' --namey 'Valid Perplexity' --title "" --swarm --out ~/plots/png/block_wise/betas.png  --filter adam_eps=1e-07 lr=0.0016329468976366 --tick-rotation 45   --rename optim_bits="Bits" adam_betas="Adam Betas" --ylim 16 18

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/block_wise/ablations.csv --ploty "ppl" --plotx 'Adam Epsilon' --category 'Bits' --namey 'Valid Perplexity' --title "" --swarm --out ~/plots/png/block_wise/eps.png  --filter adam_betas="0.9, 0.995" lr=0.0016329468976366 --tick-rotation 45   --categoricalx --rename adam_eps="Adam Epsilon" optim_bits="Bits" --ylim 16 18
