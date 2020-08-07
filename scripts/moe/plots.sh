#!/bin/bash

python plot_data.py --csv ~/plots/moe/noise_type.csv --plotx moe_noise_type --namey 'Valid Perplexity' -o ~/plots/moe/noise_type.png --category moe_noise_init --title "Valid Perplexity WikiText-2 by Noise Type" --categoricalx

python plot_data.py --csv ~/plots/moe/experts_per_seq_wt.csv --plotx experts_per_seq --namey 'Valid Perplexity' -o ~/plots/moe/expert_per_seq_wt.png --category data --title "Valid Perplexity by Experts per Seq" --categoricalx

python plot_data.py --csv ~/plots/moe/experts_per_seq_wt_train.csv --plotx experts_per_seq --namey 'Train Perplexity' -o ~/plots/moe/expert_per_seq_wt_train.png --category data --title "Train Perplexity by Experts per Seq" --categoricalx

python plot_data.py --csv ~/plots/moe/experts_per_seq_wt_train_no_agg.csv --plotx experts_per_seq --namey 'Train Perplexity' -o ~/plots/moe/expert_per_seq_wt_train_ci.png --category data --title "Train Perplexity by Experts per Seq" --categoricalx --ci

