#!/bin/bash

rm -rf ~/plots/*



#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_14b.csv --ploty "p" --plotx 'limit' --namey 'Issues Resolved' --namex 'Test samples used' --title "Resolved for nTestSamples" --swarm --out ~/plots/limit_14b.png --categoricalx   --ylim 0.04 0.42
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench.csv --ploty "p" --plotx 'limit' --namey 'Issues Resolved' --namex 'Test samples used' --title "Resolved for nTestSamples" --swarm --out ~/plots/limit_32b.png --categoricalx   --ylim 0.04 0.42
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_72b.csv --ploty "p" --plotx 'limit' --namey 'Issues Resolved' --namex 'Test samples used' --title "Resolved for nTestSamples" --swarm --out ~/plots/limit_72b.png --categoricalx   --ylim 0.04 0.42
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_32b_new.csv --ploty "p" --plotx 'limit' --namey 'Issues Resolved' --namex 'Test samples used' --title "Resolved for nTestSamples" --swarm --out ~/plots/limit_32b_new.png --categoricalx   --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/temp_swebench.csv --ploty "p" --plotx 'limit' --category 'temperature' --namey 'Issues Resolved' --namex 'Test samples used' --title "Resolved for nTestSamples" --swarm --out ~/plots/limit_32b_temp.png --categoricalx   --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/context_window_swebench.csv --ploty "p" --plotx 'context_window' --namey 'Issues Resolved' --namex 'Function context window size' --title "Resolved for context_window" --swarm --out ~/plots/context_window.png --categoricalx   --ylim 0.04 0.42


#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_32b_new.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_new_256.png --categoricalx   --filter limit=256 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_32b_new.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_new_128.png --categoricalx   --filter limit=128 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_32b_new.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_new_64.png --categoricalx   --filter limit=64 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_32b_new.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_new_32.png --categoricalx   --filter limit=32 --ylim 0.04 0.42



#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_256.png --categoricalx   --filter limit=256 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_128.png --categoricalx   --filter limit=128 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_64.png --categoricalx   --filter limit=64 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_32b_swebench_32.png --categoricalx   --filter limit=32 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_72b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_72b_swebench_256.png --categoricalx   --filter limit=256 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_72b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_72b_swebench_128.png --categoricalx   --filter limit=128 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_72b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_72b_swebench_64.png --categoricalx   --filter limit=64 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_72b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_72b_swebench_32.png --categoricalx   --filter limit=32 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_14b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_14b_swebench_256.png --categoricalx   --filter limit=256 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_14b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_14b_swebench_128.png --categoricalx   --filter limit=128 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_14b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_14b_swebench_64.png --categoricalx   --filter limit=64 --ylim 0.04 0.42
#
#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/seeds_swebench_14b.csv --ploty "p" --plotx 'top_n' --category 'max_samples' --namey 'Issues Resolved' --title "Resolved vs top_n vs max_samples" --swarm --out ~/plots/seeds_14b_swebench_32.png --categoricalx   --filter limit=32 --ylim 0.04 0.42



#python /data/input/timd/git/sched/plot_data.py --csv ~/csv/swe_grid_data.csv --ploty "p2" --plotx 'folder' --category 'folder' --namey 'Issues Resolved' --title "Qwen 2.5 32B development" --swarm --out ~/plots/32b_improvements.png --categoricalx   --ylim 0.04 0.30 --tick-rotation 45 --namex "Method"

python /data/input/timd/git/sched/plot_data.py --csv ~/csv/swe_models.csv --ploty "p2" --plotx 'model_name' --category 'model_name' --namey 'Issues Resolved' --title "" --swarm --out ~/plots/models.png --categoricalx   --ylim 0.02 0.28 --tick-rotation 90 --namex "Method"

python /data/input/timd/git/sched/plot_data.py --csv ~/csv/swe_models.csv --ploty "p" --plotx 'model_name' --category 'model_name' --namey 'Issues Resolved (empty patches excluded)' --title "" --swarm --out ~/plots/models2.png --categoricalx   --ylim 0.02 0.28 --tick-rotation 90 --namex "Method"
