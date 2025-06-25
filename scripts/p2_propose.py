#!/usr/bin/env python3

"""
Script to propose mutations using trained multievolve models.

Example usage:

conda activate multievolve

p2_propose.py \
--experiment-name multievolve_example \
--protein-name example_protein \
--wt-files apex.fasta \
--training-dataset example_dataset.csv \
--mutation-pool combo_muts.csv \
--top-muts-per-load 3 \
--export-name multievolve_proposals
"""

import wandb
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
import matplotlib
matplotlib.use('Agg')

from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Propose mutations using trained models')
    parser.add_argument(
        '--experiment-name',
        required=True,
        help='Name of experiment'
    )
    parser.add_argument(
        '--protein-name',
        required=True, 
        help='Name of protein'
    )
    parser.add_argument(
        '--wt-files',
        required=True,
        help='Comma separated list of paths to the wildtype FASTA files'
    )
    parser.add_argument(
        '--training-dataset',
        required=True,
        help='Path to training dataset CSV'
    )
    parser.add_argument(
        '--mutation-pool',
        required=True,
        help='Path to mutation pool CSV'
    )
    parser.add_argument(
        '--top-muts-per-load',
        type=int,
        default=3,
        help='Number of top mutations to select per load (default: 3)'
    )
    parser.add_argument(
        '--export-name',
        required=True,
        help='Name for export files'
    )

    args = parser.parse_args()
    args.wt_files = [f.strip() for f in args.wt_files.split(',')]
    return args


def main():

    """Main function."""

    # Parse command line arguments
    args = parse_args()

    # Define variables from args
    experiment_name = args.experiment_name
    protein_name = args.protein_name
    wt_files = args.wt_files
    training_dataset_fname = args.training_dataset
    mutation_pool_fname = args.mutation_pool
    top_muts_per_load = args.top_muts_per_load
    export_name = args.export_name

    # Processed variables
    mutation_pool = pd.read_csv(mutation_pool_fname, header=None).values.flatten().tolist()
    wt_seq = "".join([str(SeqIO.read(wt_file, "fasta").seq.upper()) for wt_file in wt_files])

    # Retrieve wandb runs
    api = wandb.Api()
    runs = api.runs(experiment_name) # Project is specified by <entity/project-name>

    # Create dataframe of all runs

    ## get column names
    test_run = runs[0]
    summary_keys = []
    for key in test_run.summary._json_dict:
        if not key.startswith("_"):
            if key != "Plot":
                summary_keys.append(key)
    config_keys = [key for key in test_run.config.keys()]
    combined_keys = summary_keys + config_keys +['name']

    ## get data from runs
    values_list = []
    for run in runs: 
        values = []
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        for key in summary_keys:
            if run.summary._json_dict.get(key) is not None:
                values.append(run.summary._json_dict[key])
            else:
                values.append(np.inf)
        for key in config_keys:
            values.append(run.config[key])
        values.append(run.name)
        values_list.append(values)

    ## create starting dataframe
    df = pd.DataFrame(values_list, columns=combined_keys)

    ## create condition column to average runs with the same nn architecture across folds later on
    df['condition'] = (
                        df['batch_size'].astype(str) + '|' 
                    + df['learning_rate'].astype(str) + '|' 
                    + df['layer_size'].astype(str) + '|' 
                    + df['num_layers'].astype(str) + '|'  
                    + df['Feature'].astype(str)
                    )

    ## get splits, split up runs by split method and get rank of test loss within each split
    splits = list(set(df['Split Method']))
    df_ls = []
    for x in splits:
        current_df = (df[df['Split Method'] == x].copy())
        current_df['rank test loss'] = current_df['Test Loss'].rank()
        df_ls.append(current_df)

    ## concat all splits
    df_mod = pd.concat(df_ls)

    ## get mean of test loss, pearson, spearman for each condition
    data_p1 = df_mod.groupby('condition')[['rank test loss', 'Test Loss', 'Pearson', 'Spearman']].mean().sort_values(by='rank test loss', ascending=True)

    ## get list of rank test losses for each condition
    data_p2 = df_mod.groupby('condition')[['rank test loss']].agg(list).sort_values(by='rank test loss', key=lambda x: x.map(len), ascending=True)
    data_p2.rename(columns={'rank test loss': 'rank test losses - random'}, inplace=True)

    ## merge data_p1 and data_p2
    data_collated = pd.merge(data_p1, data_p2, on='condition', how='inner')
    data_collated.reset_index(inplace=True)

    ## get bs, lr, hidden, layers, ft from condition column
    data_collated[['bs', 'lr', 'hidden', 'layers', 'ft']] = data_collated['condition'].str.split('|', expand=True)

    # return the bs, lr, hidden, layers, ft for the top rank architecture
    top_arch = data_collated.sort_values(by='rank test loss', ascending=True).head(1)
    bs = int(top_arch['bs'].values[0])
    lr = float(top_arch['lr'].values[0])
    hidden = int(top_arch['hidden'].values[0])
    layers = int(top_arch['layers'].values[0])
    print(bs, lr, hidden, layers)

    # Train fully connected neural network model with best architecture

    ## config of best architecture
    config = {
            'layer_size': hidden,
            'num_layers' : layers,
            'learning_rate': lr,
            'batch_size': bs,
            'optimizer': 'adam',
            'epochs': 300
    }

    ## initialize splits
    split = KFoldProteinSplitter(protein_name, training_dataset_fname, wt_files, csv_has_header=True, use_cache=True, y_scaling=False, val_split=0.15)
    splits = split.generate_splits(n_splits=10)

    ## initialize feature
    feature = OneHotFeaturizer(protein=protein_name, use_cache=True)

    ## initialize and train models
    models = []
    for split in splits:
        model = Fcn(split, feature, config=config, use_cache=True)
        model.run_model()
        models.append(model)

    print("Proposing mutations...")

    ## initialize proposer and evaluate proposals
    proposer = CombinatorialProposer(
        start_seq=wt_seq, 
        models=models, 
        trust_radius=11, 
        num_seeds=-1, # evaluate all seeds
        mutation_pool=mutation_pool)
    proposer.propose(output_df=False)
    proposer.evaluate_proposals()
    proposer.save_proposals(f'{experiment_name}_proposals_all.csv')

    # get top n variants per mutational load
    df = proposer.proposals
    df_ls = []
    for num_mut in range(3, 11, 1):
        subset = df[df['num_muts'] == num_mut].copy()
        subset.sort_values(by='average', ascending=False, inplace=True)
        top_subset = subset.head(top_muts_per_load).copy()
        df_ls.append(top_subset)
    top_df = pd.concat(df_ls, ignore_index=True)

    # Export results
    print('Saving all proposals...')
    top_df.to_csv(os.path.join(splits[0].file_attrs['dataset_dir'], 'proposers/results', f'{experiment_name}_proposals_top_{top_muts_per_load}.csv'), index=False)
    top_df[['Mut_string']].to_csv(os.path.join(splits[0].file_attrs['dataset_dir'], f'{export_name}.csv'), index=False,header=None)
    
    # functions for multichain proteins

    def reverse_multichain_mutations(mut_strings, chain_lengths):
        """
        Reverse the position adjustments for mutations in a multi-chain protein.
        
        Args:
            mut_strings (list): List of mutation strings (e.g. ["A50G/L120M", "R30K"])
            chain_lengths (list): List of lengths for each chain (e.g. [100, 150] for two chains)
        
        Returns:
            dict: Dictionary mapping original mutation string to dict of chain-specific mutation lists
                e.g. {"A50G/L120M": {0: ["A50G"], 1: ["L20M"]}}
        """
        # Calculate cumulative lengths for each chain
        cumulative_lengths = [sum(chain_lengths[:i]) for i in range(len(chain_lengths))]
        
        mutation_map = {}
        
        for mut_string in mut_strings:
            # Split into individual mutations
            mutations = mut_string.split('/')
            
            # Initialize dictionary with empty lists for each chain
            chain_mutations = {i: [] for i in range(len(chain_lengths))}
            
            # iterate over each mutation and add to correct chain in chain_mutations
            for mut in mutations:
                position = int(mut[1:-1])  # Extract position number
                wt_aa = mut[0]  # Wild type amino acid
                mut_aa = mut[-1]  # Mutant amino acid
                
                # Find which chain this mutation belongs to
                for chain_idx, start_pos in enumerate(cumulative_lengths):
                    if position <= cumulative_lengths[chain_idx + 1] if chain_idx + 1 < len(cumulative_lengths) else float('inf'):
                        # Adjust position back to chain-specific numbering
                        chain_pos = position - start_pos
                        # Add mutation to the appropriate chain's list
                        chain_mutations[chain_idx].append(f"{wt_aa}{chain_pos}{mut_aa}")
                        break
            
            mutation_map[mut_string] = chain_mutations
        
        return mutation_map

    def mutation_map_to_df(mutation_map):
        """
        Convert mutation map to DataFrame with columns for mut_string and chain-specific mutations
        
        Args:
            mutation_map (dict): Dictionary mapping mutation strings to chain mutations
                            e.g. {"A50G/L120M": {0: ["A50G"], 1: ["L20M"]}}
        
        Returns:
            pd.DataFrame: DataFrame with columns ['mut_string', 'chain_1', 'chain_2', ...]
        """
        # Create list of dictionaries for DataFrame
        rows = []
        for mut_string, chain_muts in mutation_map.items():
            row = {'Mut_string': mut_string}
            # Add chain mutations as comma-separated strings if multiple mutations exist
            for chain_idx, mutations in chain_muts.items():
                row[f'chain_{chain_idx + 1}'] = '/'.join(mutations) if mutations else ''
            rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure consistent column ordering
        chain_cols = [col for col in df.columns if col.startswith('chain_')]
        df = df[['Mut_string'] + sorted(chain_cols)]
        
        return df

    if len(wt_files) > 1:

        mutations = top_df['Mut_string'].values.tolist()
        chain_lens = splits[0].wt_seq_lens
        dict_mutations = reverse_multichain_mutations(mutations, chain_lens)
        df_mutations = mutation_map_to_df(dict_mutations)

        top_df = pd.merge(top_df, df_mutations, on='Mut_string', how='left')
        top_df.to_csv(os.path.join(splits[0].file_attrs['dataset_dir'], 'proposers/results', f'{experiment_name}_proposals_top_{top_muts_per_load}.csv'), index=False)
        

        for col in df_mutations.columns[1:]:
            mutations = set(df_mutations[col].tolist())
            if '' in mutations:
                mutations.remove('')
            #convert mutations to a dataframe for csv export
            df_mutations_col = pd.DataFrame(mutations, columns=[col])
            df_mutations_col.to_csv(os.path.join(splits[0].file_attrs['dataset_dir'], f'{export_name}_{col}_mutants.csv'), index=False, header=None)


if __name__ == "__main__":
    main() 