#!/bin/bash

#python plot_data.py --csv ~/plots/moe/noise_type.csv --plotx moe_noise_type --namey 'Valid Perplexity' -o ~/plots/moe/noise_type.png --category moe_noise_init --title "Valid Perplexity WikiText-2 by Noise Type" --categoricalx

#python plot_data.py --csv ~/plots/moe/experts_per_seq_wt.csv --plotx experts_per_seq --namey 'Valid Perplexity' -o ~/plots/moe/expert_per_seq_wt.png --category data --title "Valid Perplexity by Experts per Seq" --categoricalx

#python plot_data.py --csv ~/plots/moe/experts_per_seq_wt_train.csv --plotx experts_per_seq --namey 'Train Perplexity' -o ~/plots/moe/expert_per_seq_wt_train.png --category data --title "Train Perplexity by Experts per Seq" --categoricalx

#python plot_data.py --csv ~/plots/moe/experts_per_seq_wt_train_no_agg.csv --plotx experts_per_seq --namey 'Train Perplexity' -o ~/plots/moe/expert_per_seq_wt_train_ci.png --category data --title "Train Perplexity by Experts per Seq" --categoricalx --ci

python plot_data.py --csv ~/plots/moe/load_ppl_wt2_proj.csv --plotx bloss_weight --namey 'Valid Perplexity' -o ~/plots/moe/load_ppl_wt2_proj.png --category bloss_weight --title "Valid Perplexity by BLoss Weight" --categoricalx --ci

python plot_data.py --csv ~/plots/moe/load_reldiff_wt2_proj.csv --plotx bloss_weight --namey 'Maximum Relative Count Difference' -o ~/plots/moe/load_reldiff_wt2_proj.png --category bloss_weight --title "Rel Count Difference by BLoss Weight" --categoricalx --ci

python plot_data.py --csv ~/plots/moe/load_ppl_wt2_proj2.csv --plotx num_experts --namey 'Valid Perplexity' -o ~/plots/moe/load_ppl_wt2_proj2.png --category bloss_weight --title "Valid Perplexity by BLoss Weight" --categoricalx --ci

python plot_data.py --csv ~/plots/moe/load_reldiff_wt2_proj2.csv --plotx num_experts --namey 'Maximum Relative Count Difference' -o ~/plots/moe/load_reldiff_wt2_proj2.png --category bloss_weight --title "Rel Count Difference by BLoss Weight" --categoricalx --ci

python plot_data.py --csv ~/plots/moe/noise_proj_vs_add.csv --plotx num_experts --namey 'Valid Perplexity' -o ~/plots/moe/noise_prj_vs_add.png --category moe_noise_type --title "Valid Perplexity by Noise Type" --categoricalx --ci
