#!/bin/bash

# CC-News baselines

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/1bn_bench.csv --ploty "ppl" --plotx 'decoder_ffn_embed_dim' --category 'adam_bits' --namey 'Valid Perplexity' --title "PPL vs Adam Bits" --swarm --out ~/plots/png/adam/1bn_full.png --categoricalx --filter emb_max_norm=0.0
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/1bn_bench.csv --ploty "ppl" --plotx 'decoder_ffn_embed_dim' --category 'adam8bits_offset' --namey 'Valid Perplexity' --title "PPL vs Offset" --swarm --out ~/plots/png/adam/1bn_offset.png --categoricalx --filter adam_bits=8,emb_max_norm=0.0
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/1bn_bench.csv --ploty "ppl" --plotx 'prob_quant' --category 'decoder_ffn_embed_dim' --namey 'Valid Perplexity' --title "PPL vs Prob Quant" --swarm --out ~/plots/png/adam/1bn_prob_quant.png --categoricalx --filter adam_bits=8,emb_max_norm=0.0

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/1bn_bench.csv --ploty "ppl" --plotx 'memory_efficient_fp16' --category 'adam_bits' --namey 'Valid Perplexity' --title "PPL vs 64k hidden size" --swarm --out ~/plots/png/adam/1bn_master_weights.png --categoricalx --filter decoder_ffn_embed_dim=65536,emb_max_norm=0.0
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/1bn_bench_clean.csv --ploty "ppl" --plotx 'decoder_ffn_embed_dim' --category 'adam_bits' --namey 'Valid Perplexity' --title "PPL vs 64k hidden size" --swarm --out ~/plots/png/adam/1bn_filtered.png --categoricalx --filter memory_efficient_fp16=True,emb_max_norm=0.0


python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/moe/1bn_sparse.csv --ploty "ppl" --plotx 'sparsity1' --category 'decoder_ffn_embed_dim' --namey 'Valid Perplexity' --title "PPL vs Sparsity layer 1" --swarm --out ~/plots/png/adam/1bn_sparse_1.png 
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/moe/1bn_sparse.csv --ploty "ppl" --plotx 'sparsity2' --category 'decoder_ffn_embed_dim' --namey 'Valid Perplexity' --title "PPL vs Sparsity layer 2" --swarm --out ~/plots/png/adam/1bn_sparse_2.png 
