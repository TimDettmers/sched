#!/bin/bash

python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/66e2846d21d56d4338484df1447cb06e/analysis/ -s ~/plots/png/block_wise/blockwise_dynamic_abs.png --mode abs --vlimits 0 0.02
python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/9a78d2bb11162eb2b9084b7387a887b2/analysis/ -s ~/plots/png/block_wise/dynamic_abs.png --mode abs --vlimits 0 0.02
python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/ec27529163131a7b81d28b2ae31d1b5f/analysis/ -s ~/plots/png/block_wise/linear_abs.png --mode abs  --vlimits 0 0.02

python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/66e2846d21d56d4338484df1447cb06e/analysis/ -s ~/plots/png/block_wise/blockwise_dynamic_rel.png --mode rel  --vlimits 0 0.1
python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/9a78d2bb11162eb2b9084b7387a887b2/analysis/ -s ~/plots/png/block_wise/dynamic_rel.png --mode rel --vlimits 0 0.1
python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/ec27529163131a7b81d28b2ae31d1b5f/analysis/ -s ~/plots/png/block_wise/linear_rel.png --mode rel  --vlimits 0 0.1

python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/66e2846d21d56d4338484df1447cb06e/analysis/ -s ~/plots/png/block_wise/blockwise_dynamic_counts.png --mode counts --vlimits 0 3.5e-4
python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/9a78d2bb11162eb2b9084b7387a887b2/analysis/ -s ~/plots/png/block_wise/dynamic_counts.png --mode counts --vlimits 0 3.5e-4
python ~/git/fairseq_private/heatmap.py -f /checkpoint/timdettmers/block_wise/cc_small/analysis3/ec27529163131a7b81d28b2ae31d1b5f/analysis/ -s ~/plots/png/block_wise/linear_counts.png --mode counts --vlimits 0 3.5e-4
