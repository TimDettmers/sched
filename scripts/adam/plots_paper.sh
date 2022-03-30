
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/sensitivity_betas.csv --ploty "ppl" --plotx 'adam_betas' --category 'method' --namey 'Valid Perplexity' --title "" --swarm --out ~/plots/png/adam/sensi_betas_all.png --categoricalx  --tick-rotation 45
python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/sensitivity_eps.csv --ploty "ppl" --plotx 'adam_eps' --category 'method' --namey 'Valid Perplexity' --title "" --swarm --out ~/plots/png/adam/sensi_eps.png --categoricalx  --tick-rotation 45

python /private/home/timdettmers/git/sched/plot_data.py --csv ~/plots/csv/adam/lr_cleaned.csv --ploty "ppl" --plotx 'lr' --category 'method' --namey 'Valid Perplexity' --title "" --swarm --out ~/plots/png/adam/sensi_lr.png 
