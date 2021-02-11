# Learning-Based Cutoff Median-Sketch

## Repoducing experiments

We describe how to reproduce the experiments performed for our paper [TODO: insert link]. 

### Datasets 
First, you need to obtain the datasets we use in our evaluation. 

#### Internet traffic dataset

Dataset website: http://www.caida.org/data/passive/passive_dataset.xml. 
You may request data access on the CAIDA website (https://www.caida.org/data/passive/passive_dataset_request.xml). 
We can share the preprocessed data used in our evaluation once you recevie approval from CAIDA (usually takes 2~3 days).

#### Search query dataset

Dataset website: https://jeffhuang.com/search_query_logs.html

The preprocessed data can be downloaded here:
https://drive.google.com/open?id=1jWot63TJY8WHk5_bQPgIz-EPV0a6BDVX (~37GB after unpacking)


### Running the experiments

#### Model Training
Example code available  in ```neuralnet/run.sh```.
See https://github.com/chenyuhsu/learnedsketch for additional details on how to train the models. 

#### Evaluation

To reproduce the paper plots using our trained models (found in ```paper_models/``` directory) run:
1. Run ```eval.sh``` (uncomment the desired dataset to evaluate)
2. Run ```plot_loss.py``` as follows:

* To plot results for the search query dataset: 
```python plot_loss.py --results ./experiments/aol_learned_sketch_experiment_results.npz --algo "Learned Count Sketch" --aol```

* To plot results for the internet traffic dataset: 
```python plot_loss.py --results ./experiments/ip_learned_sketch_experiment_results.npz --algo "Learned Count Sketch" --ip```

* To plot results for synthetic datasets (Pareto and Zipfian): 
```python plot_synth.py --results ./experiments/sythn_learned_sketch_experiment_results_synth_pareto.npz --synth_pareto```
```python plot_synth.py --results ./experiments/sythn_learned_sketch_experiment_results_synth_zipfian.npz --synth_zipfian```

All plot results will be saved to ```./experiments```. 

#### Acknowledgements 
This code is partially based on the implementation of https://github.com/chenyuhsu/learnedsketch/. 