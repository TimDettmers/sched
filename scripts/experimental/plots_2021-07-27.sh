#!/bin/bash

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/8bit/8bit_weights.csv --ploty "ppl" --plotx 'use_8bit_training' --category 'store_8bit' --namey 'Valid Perplexity' --title "8-bit Weights" --swarm --out ~/plots/png/8bit/weights_max.png --categoricalx  

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/dendritic/initial.csv --ploty "ppl" --plotx 'num_dendritic_layers' --category 'decoder_ffn_embed_dim' --namey 'Valid Perplexity' --title "Dendritic Spike Layers" --swarm --out ~/plots/png/8bit/dendritic.png --categoricalx  
