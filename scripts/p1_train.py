#!/usr/bin/env python3

"""
Script to train multievolve neural network models.

Example usage:

conda activate multievolve

p1_train.py \
--experiment-name multievolve_example \
--protein-name example_protein \
--wt-files apex.fasta \
--training-dataset-fname example_dataset.csv \
--wandb-key <key> \
--mode test
"""

import wandb
import argparse
import sys
import matplotlib
matplotlib.use('Agg')

from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multievolveneural network models')

    parser.add_argument(
        '-e',
        '--experiment-name',
        required=True,
        help='Name of the experiment'   
    )
    parser.add_argument(
        '-p',
        '--protein-name',
        required=True,
        help='Name of the protein'
    )
    parser.add_argument(
        '-wt',
        '--wt-files',
        required=True,
        help='Comma separated list of paths to the wildtype FASTA files'
    )
    parser.add_argument(
        '-t',
        '--training-dataset-fname',
        required=True,
        help='Path to the training dataset CSV file'
    )
    parser.add_argument(
        '-k',
        '--wandb-key',
        required=True,
        help='WandB API key for authentication'
    )
    parser.add_argument(
        '-m',
        '--mode',
        required=True,
        help='Training method of the experiment, options include: test or standard'
    )
    args = parser.parse_args()
    args.wt_files = [f.strip() for f in args.wt_files.split(',')]
    return args

def main():
    
    """Main function."""

    # Parse command line arguments
    args = parse_args()

    try:
        # Login to WandB
        wandb.login(key=args.wandb_key)
    except Exception as e:
        print(f"Error logging into WandB: {e}")
        sys.exit(1)

    # Define variables
    experiment_name = args.experiment_name
    protein_name = args.protein_name
    wt_files = args.wt_files
    training_dataset_fname = args.training_dataset_fname

    try:
        # Define splits
        fold_splitter = KFoldProteinSplitter(protein_name, training_dataset_fname, wt_files, csv_has_header=True, use_cache=True, y_scaling=False, val_split=0.15)
        splits = fold_splitter.generate_splits(n_splits=5)
    except Exception as e:
        print(f"Error generating splits: {e}")
        sys.exit(1)

    try:
        # Define features
        onehot = OneHotFeaturizer(protein=protein_name, use_cache=True) 
        features = [onehot]
    except Exception as e:
        print(f"Error generating features: {e}")
        sys.exit(1)

    # Define models
    models = [Fcn]

    if args.mode == 'test':
        print("Running in test mode")
        sweep_depth = 'test'
        search_method = 'test'
    elif args.mode == 'standard':
        print("Running in standard mode")
        sweep_depth = 'standard'
        search_method = 'grid'

    try:
        # Run experiments
        print(f"Running experiments for {experiment_name} with {protein_name}...")        
        run_nn_model_experiments(splits, 
                                features, 
                                models, 
                                experiment_name=experiment_name,
                                use_cache=False,
                                sweep_depth=sweep_depth,
                                search_method=search_method,
                                show_plots=True, # prevents issue when running script in terminal
                                )
    except Exception as e:
        print(f"Error running experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()