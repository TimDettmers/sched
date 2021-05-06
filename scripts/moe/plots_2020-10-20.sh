#!/bin/bash

#PPL

#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --plotx "experts_per_seq" --ploty "ppl" --category 'gate_type' --namey 'Valid Perplexity' --title "4K: PPL by Expert-per-seq" --swarm --categoricalx  --filter decoder_ffn_embed_dim=4096,use_ff_norm=False --out ~/plots/png/moe/1bn/4k.png
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --plotx "experts_per_seq" --ploty "ppl" --category 'gate_type' --namey 'Valid Perplexity' --title "8k: PPL by Expert-per-seq" --swarm --categoricalx  --filter decoder_ffn_embed_dim=8192,use_ff_norm=False --out ~/plots/png/moe/1bn/8k.png
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --plotx "experts_per_seq" --ploty "ppl" --category 'gate_type' --namey 'Valid Perplexity' --title "64k: PPL by Expert-per-seq" --swarm --categoricalx  --filter decoder_ffn_embed_dim=65536,use_ff_norm=False --out ~/plots/png/moe/1bn/64k.png
#
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --plotx "decoder_ffn_embed_dim" --ploty "ppl" --category 'use_ff_norm' --namey 'Valid Perplexity' --title "PPL by Layer Norm" --swarm --categoricalx  --filter gate_type=segments --out ~/plots/png/moe/1bn/ff_norm.png
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --category "decoder_ffn_embed_dim" --ploty "ppl" --plotx 'iloss_weight' --namey 'Valid Perplexity' --title "PPL by Loss Weight" --swarm --categoricalx  --filter gate_type=segments,use_ff_norm=False --out ~/plots/png/moe/1bn/loss_weight.png
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --category "decoder_ffn_embed_dim" --ploty "ppl" --plotx 'iloss_weight' --namey 'Valid Perplexity' --title "PPL by Loss Weight /w Norm" --swarm --categoricalx  --filter gate_type=segments,use_ff_norm=True --out ~/plots/png/moe/1bn/loss_weight_norm.png
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Rdm PPL " --swarm --filter gate_type=segments,use_ff_norm=True --out ~/plots/png/moe/1bn/rdm_ppl_norm.png
#python plot2.py --csv ~/plots/csv/moe/1bn/data1.csv --ploty "ppl" --plotx 'rdmppl' --namey 'Valid Perplexity' --title "PPL vs Rdm PPL" --swarm --filter gate_type=segments,use_ff_norm=False --out ~/plots/png/moe/1bn/rdm_ppl.png
python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "4k, Layer 1: PPL vs Specialization Score " --swarm --filter decoder_ffn_embed_dim=4096 --out ~/plots/png/moe/1bn/special_4k.png
python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'score0' --namey 'Valid Perplexity' --title "8k, Layer 1: PPL vs Specialization Score " --swarm --filter decoder_ffn_embed_dim=8192 --out ~/plots/png/moe/1bn/special_8k.png

python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'score1' --namey 'Valid Perplexity' --title "4k, Layer 2: PPL vs Specialization Score " --swarm --filter decoder_ffn_embed_dim=4096 --out ~/plots/png/moe/1bn/special_4k_2.png
python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'score1' --namey 'Valid Perplexity' --title "8k, Layer 2: PPL vs Specialization Score " --swarm --filter decoder_ffn_embed_dim=8192 --out ~/plots/png/moe/1bn/special_8k_2.png

python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "rdmppl" --plotx 'score0' --title "4k, Layer 1: Rdm PPL vs Specialization Score " --swarm --filter decoder_ffn_embed_dim=4096 --out ~/plots/png/moe/1bn/special_4k_rdm.png
python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "rdmppl" --plotx 'score0' --title "8k, Layer 1: Rdm PPL vs Specialization Score " --swarm --filter decoder_ffn_embed_dim=8192 --out ~/plots/png/moe/1bn/special_8k_rdm.png

python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'p0' --namey 'Valid Perplexity' --title "4k, Layer 1: PPL vs Median Gate Prob" --swarm --filter decoder_ffn_embed_dim=4096 --out ~/plots/png/moe/1bn/gate_4k.png
python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'p0' --namey 'Valid Perplexity' --title "8k, Layer 1: PPL vs Median Gate Prob " --swarm --filter decoder_ffn_embed_dim=8192 --out ~/plots/png/moe/1bn/gate_8k.png

python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'p1' --namey 'Valid Perplexity' --title "4k, Layer 2: PPL vs Median Gate Prob " --swarm --filter decoder_ffn_embed_dim=4096 --out ~/plots/png/moe/1bn/gate_4k_2.png
python plot2.py --csv ~/plots/csv/moe/1bn/special1.csv --ploty "ppl" --plotx 'p1' --namey 'Valid Perplexity' --title "8k, Layer 2: PPL vs Median Gate Prob " --swarm --filter decoder_ffn_embed_dim=8192 --out ~/plots/png/moe/1bn/gate_8k_2.png
