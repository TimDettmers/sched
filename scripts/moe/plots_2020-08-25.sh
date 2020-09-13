#!/bin/bash

#PPL

python plot_data.py --csv ~/plots/csv/moe/scaling/full_data.csv --plotx dummy --namey 'Valid Perplexity' -o ~/plots/png/moe/scaling/ppl_dummy_arch.png --category arch --title "Valid PPL by Compute Budget" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/scaling/full_data.csv --plotx data --namey 'Valid Perplexity' -o ~/plots/png/moe/scaling/ppl_data_arch.png --category arch --title "Valid PPL by Dataset" --categoricalx --swarm --tick-rotation 90

python plot_data.py --csv ~/plots/csv/moe/scaling/full_data.csv --plotx dummy --namey 'Train Perplexity' -o ~/plots/png/moe/scaling/ppl_dummy_arch_train.png --category arch --title "Train PPL by Compute Budget" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/scaling/full_data.csv --plotx data --namey 'Train Perplexity' -o ~/plots/png/moe/scaling/ppl_data_arch_train.png --category arch --title "Train PPL by Dataset" --categoricalx --swarm --tick-rotation 90

python plot_data.py --csv ~/plots/csv/moe/scaling/params_dummy_arch.csv --plotx dummy --namey 'Parameters' -o ~/plots/png/moe/scaling/params.png --category arch --title "Model Parameters by Compute Budget" --categoricalx 

python plot_data.py --csv ~/plots/csv/moe/scaling/epoch_full.csv --plotx dummy --namey 'Convergence Epoch' -o ~/plots/png/moe/scaling/epoch_dummy.png --category arch --title "Convergence Epoch by Compute Budget" --categoricalx  --swarm

python plot_data.py --csv ~/plots/csv/moe/scaling/epoch_full.csv --plotx data --namey 'Convergence Epoch' -o ~/plots/png/moe/scaling/epoch_data.png --category arch --title "Convergence Epoch by Dataset" --categoricalx  --swarm --tick-rotation 90

python plot_data.py --csv ~/plots/csv/moe/scaling/ppl_expert_full.csv --plotx data --namey 'Valid Perplexity' -o ~/plots/png/moe/scaling/expert_data.png --category num_experts --title "Num Experts vs Dataset" --categoricalx  --swarm --tick-rotation 90

python plot_data.py --csv ~/plots/csv/moe/scaling/ppl_expert_full.csv --plotx dummy --namey 'Valid Perplexity' -o ~/plots/png/moe/scaling/expert_dummy.png --category num_experts --title "Num Experts vs Compute Budget" --categoricalx  --swarm
