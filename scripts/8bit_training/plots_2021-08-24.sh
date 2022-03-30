#!/bin/bash

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/8bit_training/igemm.csv --ploty "ppl" --plotx 'quant_type' --category 'use_8bit_training' --namey 'Valid Perplexity' --title "Real 8-bit training results" --swarm --out ~/plots/png/8bit_training/igemm.png --categoricalx  

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/8bit_training/linear.csv --ploty "ppl" --plotx 'attention_8bit' --category 'use_8bit_training' --namey 'Valid Perplexity' --title "8-bit FFN + 8-bit Attention" --swarm --out ~/plots/png/8bit_training/linear_att.png --categoricalx  --ylim 16.8 17.8

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/8bit_training/glue.csv --ploty "accuracy" --plotx 'data' --category 'attention_8bit' --namey 'Valid Perplexity' --title "16-bit FFN + 8-bit Attention" --swarm --out ~/plots/png/8bit_training/glue.png --categoricalx 

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/meeting/primer2.csv --ploty "ppl" --plotx 'ff_block' --category 'attention' --namey 'Valid Perplexity' --title "Primer vs Transformer" --swarm --out ~/plots/png/meeting/primer2.png --categoricalx

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/meeting/primer.csv --ploty "ppl" --plotx 'ff_block' --category 'attention' --namey 'Valid Perplexity' --title "Primer vs Transformer" --swarm --out ~/plots/png/meeting/primer_limit.png --categoricalx --ylim 15.8 16.9

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/meeting/primer2.csv --ploty "ppl" --plotx 'ff_block' --category 'attention' --namey 'Valid Perplexity' --title "Primer vs Transformer" --swarm --out ~/plots/png/meeting/primer2_limit.png --categoricalx --ylim 15.8 16.9
