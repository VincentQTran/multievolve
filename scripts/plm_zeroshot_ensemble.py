#!/usr/bin/env python3

"""
Script to generate zeroshot mutations for ESM and ESM-IF.

Example usage:

conda activate multievolve

plm_zeroshot_ensemble.py \
--wt-file apex.fasta \
--pdb-files apex.cif, apex_2.cif \
--chain-id A \
--variants 24 \
--normalizing-method aa_substitution_type \
--excluded-positions 1,14,41,112
"""

import argparse
from Bio import SeqIO
import pandas as pd
import os

from multievolve import zero_shot_esm_dms, zero_shot_esm_if_dms

def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='Nominate mutations using a PLM zeroshot ensemble')
    
    parser.add_argument(
        '--wt-file',
        required=True,
        help='Path to the wildtype FASTA file'
    )
    parser.add_argument(
        '--pdb-files',
        required=True,
        help='Comma-separated list of PDB/CIF structure files'
    )
    parser.add_argument(
        '--chain-id',
        required=True,
        help='Chain ID to include in the zeroshot predictions'
    )
    parser.add_argument(
        '--variants',
        type=int,
        required=True,
        help='Number of variants to nominate per method'
    )
    parser.add_argument(
        '--normalizing-method',
        required=True,
        help='Method for normalizing fold-change scores to generate z-scores'
    )
    parser.add_argument(
        '--excluded-positions', 
        required=False,
        help='Comma-separated list of positions to exclude from mutation'
    )

    args = parser.parse_args()
    
    # Process arguments
    args.pdb_files = [f.strip() for f in args.pdb_files.split(',')]
    args.excluded_positions = [int(p) for p in args.excluded_positions.split(',')] if args.excluded_positions else []
    
    return args

def main():

    args = parse_args()
    wt_file = args.wt_file
    pdb_files = args.pdb_files
    variants = args.variants
    excluded_positions = args.excluded_positions
    normalizing_method = args.normalizing_method
    chain_id = args.chain_id


    wt_seq = str(SeqIO.read(wt_file, "fasta").seq)

    print('Running ESM zeroshot...')
    esm_zeroshot = zero_shot_esm_dms(wt_seq)

    print('Running ESM-IF zeroshot...')
    esm_if_zeroshot_ls = []
    for pdb_file in pdb_files:
        esm_if_zeroshot_ls.append(zero_shot_esm_if_dms(wt_seq, pdb_file, chain_id = chain_id, scoring_strategy='wt-marginals'))

    def sample_mutations(df, total_muts, excluded_positions):

        muts = []
        pos = excluded_positions.copy()
        
        # iterate over each row
        for index, row in df.iterrows():
            if row['pos'] not in pos:
                muts.append(row.to_frame().T)
                pos.append(row['pos'])
            if len(muts) == total_muts:
                break

        result = pd.concat(muts, ignore_index=True)
        return result

    def calculate_z_scores(df, col_name, activity_col):
        """
        Calculate z-scores for activity values grouped by a column and filter groups with sufficient samples.
        
        Args:
            df (pd.DataFrame): Input dataframe
            col_name (str): Column name to group by
            activity_col (str): Column name containing activity values to calculate z-scores for
            
        Returns:
            pd.DataFrame: Dataframe with z-scores calculated and sorted, filtered to groups with >= 5 samples
        """
        dfs = []
        col_values = df[col_name].unique()
        for value in col_values:
            subset = df[df[col_name] == value].copy()
            subset['z_logratio'] = (subset[activity_col] - subset[activity_col].mean()) / subset[activity_col].std()
            subset['n'] = len(subset)

            if len(subset) >= 5:
                dfs.append(subset)
        df = pd.concat(dfs, ignore_index=True)
        df.sort_values(by='z_logratio', ascending=False, inplace=True)
        
        return df

    # Function to merge dataframes
    def merge_mutation_dfs(df_dict):
        # Start with the first dataframe
        first_key = list(df_dict.keys())[0]
        result = df_dict[first_key][0][['mutations', df_dict[first_key][1]]].copy()
        
        # Merge all remaining dataframes
        for key in list(df_dict.keys())[1:]:
            df = df_dict[key][0]
            col = df_dict[key][1]
            result = pd.merge(result, df[['mutations', col]], on='mutations', how='outer')
        
        return result.fillna(0)

    # average results for esm if zeroshot across multiple structure models
    subset_ls = []
    for j in range(len(esm_if_zeroshot_ls)):
        subset_ls.append(esm_if_zeroshot_ls[j][['mutations','logratio']].copy())
        subset_ls[j].rename(columns={'logratio': f'logratio_model{j}'}, inplace=True)

    # Start with first dataframe
    esm_if_zeroshot = subset_ls[0].copy()

    # Merge remaining dataframes iteratively
    for j in range(1, len(subset_ls)):
        esm_if_zeroshot = pd.merge(esm_if_zeroshot, subset_ls[j], on='mutations', how='outer')
        
    # Calculate average across all model logratios
    logratio_cols = [f'logratio_model{j}' for j in range(len(subset_ls))]
    esm_if_zeroshot['average_model_logratio'] = esm_if_zeroshot[logratio_cols].mean(axis=1)

    # sort esm_zeroshot by number of total models with FC > 1 and then by FC value

    esm_zeroshot_ls = []

    total_model_pass_list = list(set(esm_zeroshot['total_model_pass'].values))
    total_model_pass_list = total_model_pass_list[::-1]

    for model_pass_value in total_model_pass_list:
        subset = esm_zeroshot[esm_zeroshot['total_model_pass'] == model_pass_value].copy()
        subset.sort_values(by='average_model_logratio', ascending=False, inplace=True)
        esm_zeroshot_ls.append(subset)

    esm_zeroshot_sorted = pd.concat(esm_zeroshot_ls)

    # modify dataframes with columns for amino acid substitution type and remove wt

    esm_zeroshot_sorted['aa_mutation'] = esm_zeroshot_sorted['mutations'].apply(lambda x: x[-1])
    esm_zeroshot_sorted['aa_substitution_type'] = esm_zeroshot_sorted['mutations'].apply(lambda x: f'{x[0]}-{x[-1]}')
    esm_zeroshot_sorted['pos'] = esm_zeroshot_sorted['mutations'].apply(lambda x: int(x[1:-1]))

    esm_if_zeroshot['aa_mutation'] = esm_if_zeroshot['mutations'].apply(lambda x: x[-1])
    esm_if_zeroshot['aa_substitution_type'] = esm_if_zeroshot['mutations'].apply(lambda x: f'{x[0]}-{x[-1]}')
    esm_if_zeroshot['pos'] = esm_if_zeroshot['mutations'].apply(lambda x: int(x[1:-1]))

    # ESM FC

    muts_esm = sample_mutations(esm_zeroshot_sorted, variants, excluded_positions)
    muts_esm['esm_sampled'] = 1

    # # ESM-IF FC

    esm_if_zeroshot.sort_values(by='average_model_logratio', ascending=False, inplace=True)
    muts_esm_if = sample_mutations(esm_if_zeroshot, variants, excluded_positions)
    muts_esm_if['esm_if_sampled'] = 1

    # ESM Z
    df = esm_zeroshot_sorted.copy()
    activity_col = 'average_model_logratio'
    df = calculate_z_scores(df, normalizing_method, activity_col)
    muts_esm_z = sample_mutations(df, variants, excluded_positions)
    muts_esm_z['esm_z_sampled'] = 1

    # ESM-IF Z
    df = esm_if_zeroshot.copy()
    activity_col = 'average_model_logratio'
    df = calculate_z_scores(df, normalizing_method, activity_col)
    muts_esm_if_z = sample_mutations(df, variants, excluded_positions)
    muts_esm_if_z['esm_if_z_sampled'] = 1

    # Define dataframes to combine
    dfs = {
        'esm': [muts_esm, 'esm_sampled'],
        'esm_if': [muts_esm_if, 'esm_if_sampled'],
        'esm_z': [muts_esm_z, 'esm_z_sampled'],
        'esm_if_z': [muts_esm_if_z, 'esm_if_z_sampled']
    }

    muts_combined = merge_mutation_dfs(dfs)

    muts_combined.to_csv(os.path.join(os.path.dirname(wt_file), 'plm_zeroshot_ensemble_nominated_mutations.csv'))

if __name__ == '__main__':
    main()