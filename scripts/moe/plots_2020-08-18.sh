#!/bin/bash

#PPL

python plot_data.py --csv ~/plots/csv/moe/wt103/ppl_e_eps.csv --plotx experts_per_seq --namey 'Valid Perplexity' -o ~/plots/png/moe/wt103/ppl_e_eps.png --category num_experts --title "Experts vs Experts per Seq" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/wt103/ppl_ff_dropout.csv --plotx moe_ff_dim --namey 'Valid Perplexity' -o ~/plots/png/moe/wt103/ppl_ff_dropout.png --category dropout --title "Dropout vs MoE Dim" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/wt103/ppl_freq_lr.csv --plotx moe_freq --namey 'Valid Perplexity' -o ~/plots/png/moe/wt103/ppl_freq_lr.png --category lr --title "LR vs MoE Frequency" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/wt103/ppl_sample_epsilon.csv --plotx sample --namey 'Valid Perplexity' -o ~/plots/png/moe/wt103/ppl_sample_epsilon.png --category epsilon --title "Sample vs Epsilon" --categoricalx --swarm

# Max count difference

python plot_data.py --csv ~/plots/csv/moe/wt103/diff_epsilon_sample_weight.csv --plotx sample --namey 'Max Count Difference' -o ~/plots/png/moe/wt103/diff_epsilon_sample.png --category epsilon --title "Sample vs Epsilon" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/wt103/diff_epsilon_iloss_weight.csv --plotx iloss_weight --namey 'Max Count Difference' -o ~/plots/png/moe/wt103/diff_epsilon_iloss_weight.png --category epsilon --title "Loss Weight vs Epsilon" --categoricalx --swarm

# Mean Max prob

python plot_data.py --csv ~/plots/csv/moe/wt103/prob_epsilon_sample_weight.csv --plotx sample --namey 'Mean Max Gate Probability' -o ~/plots/png/moe/wt103/prob_epsilon_sample.png --category epsilon --title "Sample vs Epsilon" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/wt103/prob_epsilon_iloss_weight.csv --plotx iloss_weight --namey 'Mean Max Gate Probability' -o ~/plots/png/moe/wt103/prob_epsilon_iloss_weight.png --category epsilon --title "Loss Weight vs Epsilon" --categoricalx --swarm
