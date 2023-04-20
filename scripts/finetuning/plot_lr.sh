#!/bin/bash

# CC-News baselines

python plot_data.py --csv ~/llama_lr.csv --ploty "acc" --plotx 'learning_rate' --category 'lr_scheduler_type' --categoricalx --namey 'MMLU Accuracy' --title "Learning Rate vs scheduler" --swarm --out ~/lr_llama.png
