#!/bin/bash

#PPL

python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_10k.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 2% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_20k.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 4% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_30k.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 6% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_40k.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 8% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_50k.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 10% Data" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_train_10k.csv --plotx dummy --namey 'Train Perplexity' --category arch --title "PPL by Compute Budget 2% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_train_20k.csv --plotx dummy --namey 'Train Perplexity' --category arch --title "PPL by Compute Budget 4% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_train_30k.csv --plotx dummy --namey 'Train Perplexity' --category arch --title "PPL by Compute Budget 6% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_train_40k.csv --plotx dummy --namey 'Train Perplexity' --category arch --title "PPL by Compute Budget 8% Data" --categoricalx --swarm
python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news_train_50k.csv --plotx dummy --namey 'Train Perplexity' --category arch --title "PPL by Compute Budget 10% Data" --categoricalx --swarm

python plot_data.py --csv ~/plots/csv/moe/scaling/params_dummy_arch_cc_news.csv --plotx dummy --namey 'Parameters' -o ~/plots/png/moe/scaling/params_cc_news.png --category arch --title "Model Parameters by Compute Budget" --categoricalx --swarm

