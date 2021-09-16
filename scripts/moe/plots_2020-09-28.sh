#!/bin/bash

#PPL

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_0.5_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 50% Data" --categoricalx --swarm
#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_0.5_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 50% Data" --categoricalx --swarm --filter decoder_layers=9,lr=[0.003] --out ~/plots/png/moe/scaling/cld/full_grid_cld_1.0_good_specialization_filtered.png


#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_1.0_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm
#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_1.0_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --bottom 20 --out ~/plots/png/moe/scaling/cld/full_grid_cld_1.0_good_specialization_filtered.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/params_small.csv --plotx dummy --namey 'Parameters Millions' --category arch --title "Parameters by Compute Budget (dummy) " --categoricalx --swarm --scale 1e-6

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/params_large.csv --plotx dummy --namey 'Parameters Millions' --category arch --title "Parameters by Compute Budget (dummy) " --categoricalx --swarm --scale 1e-6

# small

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_small.csv --plotx decoder_embed_dim --namey 'Specialization score' --category decoder_layers --title "Specialization score by Network Size" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_small_dec_emb.png


#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_small.csv --plotx iloss_weight --namey 'Specialization score' --category lr --title "Specialization score by Loss Weight/LR" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_small_lr_iloss_weight.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_small.csv --plotx decoder_embed_dim --namey 'Specialization score' --category decoder_layers --title "Specialization score by Network Size" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_small_dec_emb_filtered2.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_small.csv --plotx moe_start_layer --namey 'Specialization score' --category decoder_layers --title "Specialization score by Layer Order" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_small_dec_start_filtered.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_small.csv --plotx dropout --namey 'Specialization score' --category attention_dropout --title "Specialization score by Dropouts" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_small_dropout.png

# large

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_large.csv --plotx decoder_embed_dim --namey 'Specialization score' --category decoder_layers --title "Specialization score by Network Size" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_large_dec_emb.png
#
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_large.csv --plotx iloss_weight --namey 'Specialization score' --category lr --title "Specialization score by Loss Weight/LR" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_large_lr_iloss_weight.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_large.csv --plotx decoder_embed_dim --namey 'Specialization score' --category decoder_layers --title "Specialization score by Network Size" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_large_dec_emb_filtered2.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/scores_large.csv --plotx moe_start_layer --namey 'Specialization score' --category decoder_layers --title "Specialization score by Layer Order" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/scores_large_dec_start_filtered.png


#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_0.5_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/full_filter_small.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_1.0_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --bottom 20 --out ~/plots/png/moe/scaling/cld/full_filter_large.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/large_ppl_filtered.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/large_filtered_with_score.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cld/full_grid_cld_1.0_good_specialization.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --out ~/plots/png/moe/scaling/cld/large_filtered_with_lr_params.png --filter lr=[0.001],decoder_layers=15

#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/wt10.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --out ~/plots/png/moe/scaling/wt/wt10_ppl.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/wt10.csv --plotx dummy --namey 'Valid Perplexity' --category arch --title "PPL by Compute Budget 100% Data" --categoricalx --swarm --out ~/plots/png/moe/scaling/wt/wt10_ppl_filtered.png --bottom 20


python plot_data.py --csv ~/plots/csv/test/decoder.csv --plotx lr --namey 'PPL' --category decoder_layers --title "LR by decoder layers" --categoricalx --swarm --out ~/plots/png/moe/scaling/wt/score_wt10.png
