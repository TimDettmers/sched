#!/bin/bash

# CC-News baselines

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adaptive/adaptive.csv --ploty "ppl" --plotx 'runtime' --category 'adaptive_span' --namey 'Valid Perplexity' --title "PPL vs Runtime Adaptive" --swarm --out ~/plots/png/adaptive/adaptive.png

#python plot_data.py --csv ~/plots/csv/moe/scaling/cc_news/baselines_word_level.csv --ploty "ppl" --plotx 'arch' --category 'decoder_embed_dim' --categoricalx --namey 'Valid Perplexity' --title "PPL Baselines vs Word-level" --swarm --out ~/plots/png/moe/scaling/wt/cc_news_baselines_word_level.png


#python get_results_from_logs.py -f ~/logs/moe/scaling/wt/moe4/ --start "ppl" --end "\| wps" --contains "| valid |" --smaller-is-better --agg min --metric-file scripts/moe/metric_file_2_rdm.csv  --groupby data iloss_weight decoder_embed_dim sample_type experts_per_seq --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv
#
##PPL vs rdm ppl
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-103,decoder_embed_dim=1024 --out ~/plots/png/moe/scaling/wt/eval_ppl_103_1024.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-103,decoder_embed_dim=512 --out ~/plots/png/moe/scaling/wt/eval_ppl_103_512.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-103,decoder_embed_dim=256 --out ~/plots/png/moe/scaling/wt/eval_ppl_103_256.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-50,decoder_embed_dim=512 --out ~/plots/png/moe/scaling/wt/eval_ppl_50_512.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-50,decoder_embed_dim=1024 --out ~/plots/png/moe/scaling/wt/eval_ppl_50_1024.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-50,decoder_embed_dim=256 --out ~/plots/png/moe/scaling/wt/eval_ppl_50_256.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-25,decoder_embed_dim=512 --out ~/plots/png/moe/scaling/wt/eval_ppl_25_512.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-25,decoder_embed_dim=1024 --out ~/plots/png/moe/scaling/wt/eval_ppl_25_1024.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-25,decoder_embed_dim=256 --out ~/plots/png/moe/scaling/wt/eval_ppl_25_256.png
#
##PPL vs specialization score
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-103,decoder_embed_dim=1024 --out ~/plots/png/moe/scaling/wt/score_ppl_103_1024.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-103,decoder_embed_dim=512 --out ~/plots/png/moe/scaling/wt/score_ppl_103_512.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-103,decoder_embed_dim=256 --out ~/plots/png/moe/scaling/wt/score_ppl_103_256.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-50,decoder_embed_dim=512 --out ~/plots/png/moe/scaling/wt/score_ppl_50_512.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-50,decoder_embed_dim=1024 --out ~/plots/png/moe/scaling/wt/score_ppl_50_1024.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-50,decoder_embed_dim=256 --out ~/plots/png/moe/scaling/wt/score_ppl_50_256.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-25,decoder_embed_dim=512 --out ~/plots/png/moe/scaling/wt/score_ppl_25_512.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-25,decoder_embed_dim=1024 --out ~/plots/png/moe/scaling/wt/score_ppl_25_1024.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "PPL vs Random Expert PPL" --filter data=data/wikitext-25,decoder_embed_dim=256 --out ~/plots/png/moe/scaling/wt/score_ppl_25_256.png

# Sample-type


#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'arch' --category "decoder_embed_dim" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Baseline - WT25" --swarm --filter data=data/wikitext-25 --out ~/plots/png/moe/scaling/wt/baseline_wt25.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'arch' --category "decoder_embed_dim" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Baseline - WT50" --swarm --filter data=data/wikitext-50 --out ~/plots/png/moe/scaling/wt/baseline_wt50.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/eval_ppl.csv --ploty "ppl" --plotx 'arch' --category "decoder_embed_dim" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Baseline - WT103" --swarm --filter data=data/wikitext-103 --out ~/plots/png/moe/scaling/wt/baseline_wt103.png
#
#python get_results_from_logs.py -f ~/logs/moe/scaling/wt/moe6/ --start "ppl" --end "\| wps" --contains "| valid |" --smaller-is-better --agg min --metric-file scripts/moe/metric_file_2_rdm.csv  --groupby data iloss_weight decoder_embed_dim sample_type experts_per_seq agg_type --csv ~/plots/csv/moe/scaling/wt/agg_type.csv
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/agg_type.csv --ploty "ppl" --plotx 'agg_type' --category "decoder_embed_dim" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Aggregation Type - WT103" --swarm --filter data=data/wikitext-103 --out ~/plots/png/moe/scaling/wt/agg_type_wt103.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/agg_type.csv --ploty "ppl" --plotx 'agg_type' --category "decoder_embed_dim" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Aggregation Type - WT50" --swarm --filter data=data/wikitext-50 --out ~/plots/png/moe/scaling/wt/agg_type_wt50.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/agg_type.csv --ploty "ppl" --plotx 'agg_type' --category "decoder_embed_dim" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Aggregation Type - WT25" --swarm --filter data=data/wikitext-25 --out ~/plots/png/moe/scaling/wt/agg_type_wt25.png
#
#python get_results_from_logs.py -f ~/logs/moe/scaling/wt/moe5/ --start "ppl" --end "\| wps" --contains "| valid |" --smaller-is-better --agg min --metric-file scripts/moe/metric_file_2_rdm.csv  --groupby data iloss_weight decoder_embed_dim sample_type experts_per_seq gate_type num_experts --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv --ploty "ppl" --plotx 'gate_type' --category "experts_per_seq" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Experts per Seq - WT103" --swarm --filter data=data/wikitext-103,num_experts=8 --out ~/plots/png/moe/scaling/wt/experts_103_8.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv --ploty "ppl" --plotx 'gate_type' --category "experts_per_seq" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Experts per Seq - WT103" --swarm --filter data=data/wikitext-103,num_experts=16 --out ~/plots/png/moe/scaling/wt/experts_103_16.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv --ploty "ppl" --plotx 'gate_type' --category "experts_per_seq" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Experts per Seq - WT103" --swarm --filter data=data/wikitext-103,num_experts=4 --out ~/plots/png/moe/scaling/wt/experts_103_4.png
#
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv --ploty "ppl" --plotx 'gate_type' --category "experts_per_seq" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Experts per Seq - WT50" --swarm --filter data=data/wikitext-50,num_experts=8 --out ~/plots/png/moe/scaling/wt/experts_50_8.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv --ploty "ppl" --plotx 'gate_type' --category "experts_per_seq" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Experts per Seq - WT50" --swarm --filter data=data/wikitext-50,num_experts=16 --out ~/plots/png/moe/scaling/wt/experts_50_16.png
#python plot_data.py --csv ~/plots/csv/moe/scaling/wt/experts_per_seq.csv --ploty "ppl" --plotx 'gate_type' --category "experts_per_seq" --categoricalx --namey 'Valid Perplexity' --title "PPL vs Experts per Seq - WT50" --swarm --filter data=data/wikitext-50,num_experts=4 --out ~/plots/png/moe/scaling/wt/experts_50_4.png
#
