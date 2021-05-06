#!/bin/bash

# CC-News baselines

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/dynamic_tree.csv --ploty "ppl" --plotx 'use_emb_norm' --category 'adam8bits_method' --namey 'Valid Perplexity' --title "PPL vs Quant method" --swarm --out ~/plots/png/adam/method_instability.png --categoricalx

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/dynamic_tree_clean.csv --ploty "ppl" --plotx 'use_emb_norm' --category 'adam8bits_method' --namey 'Valid Perplexity' --title "PPL vs Quant method" --swarm --out ~/plots/png/adam/method_emb.png --categoricalx

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/dynamic_tree_clean.csv --ploty "ppl" --plotx 'percentile_clipping' --category 'adam8bits_method' --namey 'Valid Perplexity' --title "PPL vs Quant method" --swarm --out ~/plots/png/adam/method_clip.png --categoricalx

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/dynamic_tree_clean.csv --ploty "ppl" --plotx 'percentile_clipping' --category 'adam8bits_method' --namey 'Valid Perplexity' --title "PPL vs Quant method with norm" --swarm --out ~/plots/png/adam/method_clip_norm.png --categoricalx --filter use_emb_norm=True

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/dynamic_tree_clean.csv --ploty "ppl" --plotx 'percentile_clipping' --category 'adam8bits_method' --namey 'Valid Perplexity' --title "PPL vs Quant method without Norm" --swarm --out ~/plots/png/adam/method_clip_nonorm.png --categoricalx --filter use_emb_norm=False


#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/glue.csv --ploty "accuracy" --plotx 'data' --category 'memory_efficient_fp16' --namey 'Valid Accuracy' --title "Acc vs master weights" --swarm --out ~/plots/png/adam/glue_32_mw.png --categoricalx --filter adam_bits=32 --tick-rotation 45

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/glue.csv --ploty "accuracy" --plotx 'data' --category 'adam_bits' --namey 'Valid Accuracy' --title "32 vs 8 bits without MW" --swarm --out ~/plots/png/adam/glue_32_vs_8_no_mw.png --categoricalx --filter memory_efficient_fp16=True --tick-rotation 45

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/glue.csv --ploty "accuracy" --plotx 'data' --category 'adam_bits' --namey 'Valid Accuracy' --title "32 vs 8 bits with MW" --swarm --out ~/plots/png/adam/glue_32_vs_8mw.png --categoricalx --filter memory_efficient_fp16=False --tick-rotation 45

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/glue.csv --ploty "accuracy" --plotx 'data' --category 'adam8bits_method' --namey 'Valid Accuracy' --title "8-bit Methods" --swarm --out ~/plots/png/adam/glue_8.png --categoricalx --filter adam_bits=8 --tick-rotation 45

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/cifar.csv --ploty "default" --plotx 'model' --category 'adam_bits' --namey 'Test Accuracy' --title "32-bit vs 8-bit" --swarm --out ~/plots/png/adam/cifar_32_vs_8.png --categoricalx --filter apex_level=O2 --tick-rotation 45 --namey "accuracy"
#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/cifar.csv --ploty "default" --plotx 'model' --category 'adam8bits_method' --namey 'Test Accuracy' --title "8-bit Methods" --swarm --out ~/plots/png/adam/cifar_8.png --categoricalx --filter apex_level=O2 --tick-rotation 45 --namey "accuracy"

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/lr.csv --ploty "ppl" --plotx 'lr' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Clipping and LR 8-bit" --swarm --out ~/plots/png/adam/lr_clip_8bit.png --filter adam_bits=8,use_emb_norm=True

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/lr.csv --ploty "ppl" --plotx 'lr' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Clipping and LR 32-bit" --swarm --out ~/plots/png/adam/lr_clip_32bit.png --filter adam_bits=32,use_emb_norm=True
#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/lr.csv --ploty "ppl" --plotx 'lr' --category 'percentile_clipping' --namey 'Valid Perplexity' --title "Clipping and LR 32-bit no norm" --swarm --out ~/plots/png/adam/lr_clip_32bit_nonorm.png --filter adam_bits=32,use_emb_norm=False

#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/lr.csv --ploty "ppl" --plotx 'lr' --category 'adam_bits' --namey 'Valid Perplexity' --title "Bits and LR" --swarm --out ~/plots/png/adam/32_vs_8.png --filter use_emb_norm=True
#python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/lr.csv --ploty "ppl" --plotx 'lr' --category 'adam_bits' --namey 'Valid Perplexity' --title "Bits and LR w/ Perc Clipping=5" --swarm --out ~/plots/png/adam/32_vs_8_filtered.png --filter use_emb_norm=True,percentile_clipping=5

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/mt.csv --ploty "ppl" --plotx 'lr' --category 'memory_efficient_fp16' --namey 'Valid Perplexity' --title "PPL vs master weights" --swarm --out ~/plots/png/adam/mt_32_mw.png --categoricalx --filter adam_bits=32 
￼
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/mt.csv --ploty "ppl" --plotx 'lr' --category 'adam_bits' --namey 'Valid Perplexity' --title "32 vs 8 bits without MW" --swarm --out ~/plots/png/adam/mt_32_vs_8_no_mw.png --categoricalx --filter memory_efficient_fp16=True 

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/mt.csv --ploty "ppl" --plotx 'lr' --category 'adam_bits' --namey 'Valid Perplexity' --title "32 vs 8 bits with MW" --swarm --out ~/plots/png/adam/mt_32_vs_8mw.png --categoricalx --filter memory_efficient_fp16=False 

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/mt.csv --ploty "ppl" --plotx 'lr' --category 'adam8bits_method' --namey 'Valid Perplexity' --title "8-bit Methods" --swarm --out ~/plots/png/adam/mt_8.png --categoricalx --filter adam_bits=8 
