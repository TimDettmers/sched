#!/bin/bash

mkdir ~/transfer/pics3 -p

python plot_data.py --csv ~/baseline.csv --plotx topk --filter task=pos,direct=False -o ~/transfer/pics3/pos_faclinear.png --category layer --title "POS Factorized Linear Rank 1000"
python plot_data.py --csv ~/baseline.csv --plotx topk --filter task=pos,direct=True -o ~/transfer/pics3/pos_linear.png --category layer --title "POS Linear Rank 1000"
python plot_data.py --csv ~/baseline.csv --plotx topk --filter task=corrupted-pos,direct=False -o ~/transfer/pics3/corrupted-pos_faclinear.png --category layer --title "Control Task POS Factorized Linear Rank 1000"
python plot_data.py --csv ~/baseline.csv --plotx topk --filter task=corrupted-pos,direct=True -o ~/transfer/pics3/corrupted-pos_linear.png --category layer --title "Control Task POS Linear Rank 1000"

python plot_data.py --csv ~/baseline.csv --plotx topk --filter task=edge -o ~/transfer/pics3/edge_bilinear.png --category layer --title "Edge Label Bilinear Rank 1000"
python plot_data.py --csv ~/baseline.csv --plotx topk --filter task=corrupted-edge -o ~/transfer/pics3/corrupted_edge_bilinear.png --category layer --title "Corrupted Edge Label Bilinear Rank 1000"
