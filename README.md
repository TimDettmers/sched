# sched
Schedule GPU jobs over ssh and slurm.



## How to use for slurm

### Install & Config

0. `pip install -r requirements.txt`
1. `python setup.py install`
2. Edits the config file at `config/slurm_config.cfg`.
  - `GIT_HOME` folder to your github directory. All sub-folders should be repos.
  - `LOG_HOME` folder where you want the log files to be stored.
  - `ANACONDA_HOME` the path to the anaconda folder
  - `SCRIPT_HISTORY` The path where the executed commands will be stored. This is used to restart jobs that failed.

3. Edit your software so that is prints argparse at the beginning of what you do:
```=python
  args = parser.parse_args()
  print(args)
```
4. Run some models with a template script. Fairseq example can be found in script/experimental/cc_small.py
5. You can now filter log files for results. You can try the example files to see how this works like so:
   `python get_results_from_logs -f example_logs/cc_small --metric-file fairseq.csv --groupby lr clip_norm`

You can use `--diff` to see all possible hyperparameters to group by.

The general structure of the call is:
   `python get_results_from_logs -f PATH_TO_FOLDER_WITH_LOGS --metric-file METRIC_FILE_WITH_REGEX_VALUES.csv --groupby lr clip_norm my_args_parse_variable`

6. To plot first save a csv value:
   `python get_results_from_logs -f example_logs/cc_small --metric-file fairseq.csv --groupby lr clip_norm --csv examples_output.csv`
7. Now you can plot values like so:
`python sched/plot_data.py --csv example_output.csv --ploty "ppl" --plotx 'lr' --category 'clip_norm' --namey 'Valid Perplexity' --title "PPL by lr and clip_norm" --swarm --out plot.png --categoricalx`  

For presentations I usually have multiple commands in a bash file like the one that you can find under `scripts/experimental/plots_2021-07-27.sh`.

I then copy the images from the slurm cluster to my desktop and if required adjust the plot variables in the bash file and replot.

