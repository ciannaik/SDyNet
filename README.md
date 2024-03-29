# SDyNet

This repository contains the code used the run the experiments in Bayesian Nonparametrics for Sparse Dynamic Neworks

In order to run the reddit and reuters experiments, download the **soc-redditHyperlinks-title.tsv** file from https://snap.stanford.edu/data/soc-RedditHyperlinks.html and the **Days.net** file from http://vlado.fmf.uni-lj.si/pub/networks/data/CRA/terror.htm respectively, and place them in the **data** folder.

Then, run `load_reddit.py` or `load_reuters.py` to preprocess the data.

Finally, run the experiments using `reddit_experiment.py` or `reuters_experiment.py`.

After running the desired number of chains, combine them using `combine_chains_reddit.py` or `combine_chains_reuters.py`. Then, the results can be plotted using the combined results using `plot_results_reddit.py` or `plot_results_reuters.py`. 

The simulated data experiment can be run directly by running `simulated_experiment.py`, which will also plot the results.

