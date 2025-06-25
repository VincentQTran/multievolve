import pandas as pd
import numpy as np

from pathlib import Path
import re

from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *


def prepare_proteingym_dataset_multiround(row, use_cache=True):
    """
    Prepares a ProteinGym dataset for benchmarking by processing and filtering mutation data.
    
    Args:
        row (pd.Series): Row containing dataset info including target sequence and DMS ID
        
    Returns:
        tuple: (protein_name, wt_file_path, filtered_dataset) or None if dataset doesn't meet criteria
            - protein_name (str): Path to protein benchmark data 
            - wt_file_path (str): Path to wild-type sequence FASTA file
            - filtered_dataset (pd.DataFrame): Filtered dataset with mutant, activity and sampling columns
    """
    sequence = row['target_seq']
    dataset_name = row['DMS_id']

    # Setup paths
    output_dir = Path('../../data/benchmark_data') / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    wt_file = output_dir / 'sequence.fasta'
    filtered_file = output_dir / 'dataset_filtered_multiround.csv'

    # Return existing data if available
    if wt_file.exists() and filtered_file.exists() and use_cache:
        fil_dataset = pd.read_csv(filtered_file)
        if len(fil_dataset[fil_dataset["mutations"] > 5]) > 100:
            return f"benchmark_data/{dataset_name}", str(wt_file), fil_dataset
        print(f'{dataset_name} has only {len(fil_dataset[fil_dataset["mutations"] > 5])} mutants with over 5 mutations')
        return None

    # Write sequence to FASTA
    with open(wt_file, 'w') as f:
        f.write(f'>{dataset_name}\n{sequence}\n')
        
    # Load and preprocess dataset
    working_df = pd.read_csv('../../data/benchmark_data/datasets/' + row['DMS_filename'])
    working_df['mutant'] = working_df['mutant'].str.replace(':', '/')
    
    # Extract relevant columns and compute mutations
    dataset = working_df[['mutant', 'DMS_score', 'mutated_sequence']].copy()
    dataset['mutations'] = dataset['mutant'].apply(lambda x: len(x.split('/')))
    singles = dataset[dataset['mutations'] == 1]['mutant'].tolist()

    # Filter for variants with existing single mutants
    def check_existing_mutants(mutant):
        return all(m in singles for m in mutant.split('/'))

    dataset['singles_exist'] = dataset['mutant'].apply(check_existing_mutants)
    fil_dataset = dataset[dataset['singles_exist'] == True].sort_values(by='mutations', ascending=False)

    if len(fil_dataset[fil_dataset["mutations"] > 5]) <= 100:
        print(f'{dataset_name} has only {len(fil_dataset[fil_dataset["mutations"] > 5])} mutants with over 5 mutations')
        return None
    
    # For iteration embedding optimization, grab all variants with 5-10 mutations
    dataset_5_10 = dataset[dataset['mutations'].isin(range(5,11))]
    dataset_5_10 = dataset_5_10[~dataset_5_10['singles_exist']]
    fil_dataset = pd.concat([fil_dataset, dataset_5_10], ignore_index=True)

    # Add sampling column and save
    fil_dataset['sampled'] = -1

    # make a copy of the DMS_score column
    fil_dataset['DMS_score_unscaled'] = fil_dataset['DMS_score']

    # modify dms_score to be between 0 and 1
    fil_dataset['DMS_score'] = (fil_dataset['DMS_score'] - fil_dataset['DMS_score'].min()) / (fil_dataset['DMS_score'].max() - fil_dataset['DMS_score'].min()) 
    
    fil_dataset.to_csv(filtered_file, index=False)

    return f"benchmark_data/{dataset_name}", str(wt_file), fil_dataset

def perform_mlde_round_multievolve(working_df,
                                    train_df, 
                                    round_num, 
                                    protein_name,
                                    wt_file,
                                    encoding):  # outputs an activity_df
    
    print(f"Generating splits for round {round_num}")
    # get splits
    fold_split = KFoldProteinSplitter(train_df, wt_file, is_csv=False, y_scaling=False, val_split=0.15)
    splits = fold_split.generate_splits(n_splits=5)
    
    # get feature
    if encoding == 'esm2_15b':
        feature = ESM2_15b_EmbedFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'esm2_3b':
        feature = ESM2EmbedFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'esmc_6b':
        feature = Forge_ESMC_6B_EmbedFeaturizer(protein=protein_name, use_cache=True, token="8JzRfpnjN8cDPNgkcZj28")
    elif encoding == 'onehot':
        feature = OneHotFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'georgiev':
        feature = GeorgievFeaturizer(protein=protein_name, use_cache=True)

    # get new df for model activity results 
    activity_df = working_df[['mutant', 'mutated_sequence']].copy()

    # train models
    batchsize = round(splits[0].splits['X_train'].shape[0] / 10)
    
    for index, split in enumerate(splits):

        # config
        config = {
            'layer_size': 100,
            'num_layers' : 2,
            'learning_rate': 0.001,
            'batch_size': batchsize,
            'optimizer': 'adam',
            'epochs': 300
        }

        print(f"Training model {index} for round {round_num}")

        # evaluate model on mutational load >= 5
        model = Fcn(split, feature, config=config, use_cache=False)
        stat = model.run_model()

        activity_df[f'model{index}'] = model.predict(activity_df['mutated_sequence'].values)

    activity_df[f'activity_r{round_num}'] = activity_df[['model0', 'model1', 'model2', 'model3', 'model4']].mean(axis=1)
    activity_df.drop(columns=['model0', 'model1', 'model2', 'model3', 'model4', 'mutated_sequence'], inplace=True)
      
    return activity_df # output df only has columns "mutant" and "activity_r#"

def perform_mlde_round_EVOLVEpro(working_df,
                       train_df, 
                       round_num, 
                       protein_name,
                       wt_file,
                       encoding):  # outputs an activity_df
    
    split = RandomProteinSplitter(train_df, wt_file, is_csv=False, y_scaling=False)
    split.split_data(test_size=0)

    feature = ESM2_15b_EmbedFeaturizer(protein=protein_name, use_cache=True)

    # get new df for model activity results 
    activity_df = working_df[['mutant', 'mutated_sequence']].copy()

    model = RandomForestRegressor(split, feature)
    model.run_model(eval=False)
    
    activity_df[f'model0'] = model.predict(activity_df['mutated_sequence'].values)

    activity_df[f'activity_r{round_num}'] = activity_df[['model0']].mean(axis=1)
    activity_df.drop(columns=['model0', 'mutated_sequence'], inplace=True)

    return activity_df # output df only has columns "mutant" and "activity_r#"
    
def sample_muts_multievolve(working_df, num_muts, sampling_round, round_type):

    working_df_copy = working_df.copy()

    if round_type == 'initial':
        mask = working_df_copy['mutations'].isin([1, 2])
        working_df_copy.loc[mask, 'sampled'] = sampling_round

    elif round_type == 'next':
        for mut_num in range(5,11,1):
            total_sampled = 0
            subset = working_df_copy[working_df_copy['mutations'] == mut_num]
            for index, row in subset.iterrows():
                if total_sampled < num_muts: # check if total sampled is less than num_muts
                    if row['sampled'] == -1: # check if sampled is 0
                        working_df_copy.loc[index, 'sampled'] = sampling_round # mark as sampled
                        total_sampled += 1 # increment total sampled
                    else:
                        continue
                    
    return working_df_copy
            

########################

# MLDE STEP 1: perform mlde round and output df with new activity values (EVOLVEpro benchmarking)
def perform_zero_round(zeroshot_esm_df, zeroshot_esmif_df, rd0_type): # outputs an activity_df
    rd0types = ['esm', 'esm_if', 'random', 'esm_norm_by_aa', 'esm_norm_by_aa_switch', 'esm_z_by_aa', 'esm_z_by_aa_switch', 'esm_if_norm_by_aa', 'esm_if_norm_by_aa_switch', 'esm_if_z_by_aa', 'esm_if_z_by_aa_switch']
    # check if rd0_type is in rd0types
    if rd0_type not in rd0types:
        raise ValueError(f'Invalid rd0_type: {rd0_type}. Please choose from {rd0types}.')

    def process_df(df):
        df = df[df['mutations'].apply(lambda x: x[0] != x[-1])].copy()
        df['new_aa'] = df['mutations'].apply(lambda x: x[0])
        df['aa_switch'] = df['mutations'].apply(lambda x: x[0] + x[-1])
        df.rename(columns={'mutations': 'mutant'}, inplace=True)
        return df

    def normalize_or_z_score(df, col_name, activity_col, method='normalize'):
        dfs = []
        for value in df[col_name].unique():
            subset = df[df[col_name] == value].copy()
            if method == 'normalize':
                subset['activity'] = (subset[activity_col] - subset[activity_col].min()) / (subset[activity_col].max() - subset[activity_col].min())
            else:  # z-score
                subset['activity'] = (subset[activity_col] - subset[activity_col].mean()) / subset[activity_col].std()
            if col_name == 'aa_switch' and len(subset) < 5:
                continue
            dfs.append(subset)
        
        return pd.concat(dfs, ignore_index=True)

    if rd0_type == 'esm':
        activity_df = process_df(zeroshot_esm_df)
        activity_df_p1 = activity_df[activity_df['total_model_pass'] >= 1]
        activity_df_p2 = activity_df[activity_df['total_model_pass'] == 0].sort_values(by='average_model_logratio', ascending=False)
        activity_df = pd.concat([activity_df_p1, activity_df_p2], axis=0)
        activity_df['activity'] = range(len(activity_df), 0, -1)
    elif rd0_type == 'esm_if':
        activity_df = process_df(zeroshot_esmif_df)
        activity_df.rename(columns={'logratio': 'activity'}, inplace=True)
    elif rd0_type == 'random':
        activity_df = process_df(zeroshot_esm_df)
        activity_df['activity'] = np.random.rand(len(activity_df))
    elif rd0_type in ['esm_norm_by_aa', 'esm_z_by_aa']:
        activity_df = normalize_or_z_score(process_df(zeroshot_esm_df), 'new_aa', 'average_model_logratio', 'normalize' if 'norm' in rd0_type else 'z')
    elif rd0_type in ['esm_norm_by_aa_switch', 'esm_z_by_aa_switch']:
        activity_df = normalize_or_z_score(process_df(zeroshot_esm_df), 'aa_switch', 'average_model_logratio', 'normalize' if 'norm' in rd0_type else 'z')
    elif rd0_type in ['esm_if_norm_by_aa', 'esm_if_z_by_aa']:
        activity_df = normalize_or_z_score(process_df(zeroshot_esmif_df), 'new_aa', 'logratio', 'normalize' if 'norm' in rd0_type else 'z')
    elif rd0_type in ['esm_if_norm_by_aa_switch', 'esm_if_z_by_aa_switch']:
        activity_df = normalize_or_z_score(process_df(zeroshot_esmif_df), 'aa_switch', 'logratio', 'normalize' if 'norm' in rd0_type else 'z')
    
    return activity_df[['mutant', 'activity']]
        
def perform_mlde_round(working_df,
                       train_df, 
                       round, 
                       protein_name,
                       wt_file,
                       mlde_type):  # outputs an activity_df
    
    if mlde_type == 'random':
        activity_df = working_df[['mutant', 'mutated_sequence']].copy()
        activity_df['random_float'] = np.random.rand(len(working_df))
        activity_df.rename(columns={'random_float':f'activity_r{round}'}, inplace=True)
        activity_df.drop(columns=['mutated_sequence'], inplace=True)

        return activity_df
    
    elif mlde_type.split('/')[0] == 'esm15b' and mlde_type.split('/')[1] == 'randomforest':
        splits = []
        
        fold_split = RandomProteinSplitter(train_df, wt_file, is_csv=False, y_scaling=False)
        fold_split.split_data(test_size=0)
        splits.append(fold_split)

        feature = ESM2_15b_EmbedFeaturizer(protein=f'benchmark_data/{protein_name}', use_cache=True)

        models = []

        # get new df for model activity results 
        activity_df = working_df[['mutant', 'mutated_sequence']].copy()

        for model_iter, split in enumerate(splits):
            model = RandomForestRegressor(split, feature)
            # stat, fig = model.run_model(eval=False)
            model.run_model(eval=False)
            # plt.close(fig)
            models.append(model)
            activity_df[f'model{model_iter}'] = model.predict(activity_df['mutated_sequence'].values)

        # activity_df[f'activity_r{round}'] = activity_df[['model0', 'model1', 'model2', 'model3', 'model4']].mean(axis=1)
        activity_df[f'activity_r{round}'] = activity_df[['model0']].mean(axis=1)
        # drop model columns
        # activity_df.drop(columns=['model0', 'model1', 'model2', 'model3', 'model4', 'mutated_sequence'], inplace=True)
        activity_df.drop(columns=['model0', 'mutated_sequence'], inplace=True)

        return activity_df # output df only has columns "mutant" and "activity_r#"

# MLDE STEP 2: merge working dataframe with new model results 
def process_model_results(working_df, activity_df, current_activity_col): # outputs a working_df

    # merge working dataframe with activity dataframe
    # ignore all mutants that do not have experimental activity
    working_df = pd.merge(working_df, activity_df, on='mutant', how='inner')
    
    # sort by activity
    working_df.sort_values(by=current_activity_col, ascending=False, inplace=True)
    
    return working_df

# MLDE STEP 3: sample mutants from working dataframe based on new model results (EVOLVEpro benchmarking)
def sample_muts(working_df, num_muts, sampling_round, diverse_pos=False): # outputs a working_df

    working_df_copy = working_df.copy()

    # check if diverse_pos is False
    if diverse_pos == False:
        # iterate over each row and mark top "num_muts" mutants as sampled
        total_sampled = 0
        for index, row in working_df_copy.iterrows():
            if total_sampled < num_muts: # check if total sampled is less than num_muts
                if row['sampled'] == -1: # check if sampled is 0
                    working_df_copy.loc[index, 'sampled'] = sampling_round # mark as sampled
                    total_sampled += 1 # increment total sampled
                else:
                    continue
    
    # check if diverse_pos is True
    elif diverse_pos == True:

        # # identify positions sampled
        # sampled_positions = list(working_df_copy[working_df_copy['sampled'] == round]['pos'].unique())
        sampled_positions = [] # create empty list to store sampled positions, every round is refreshed

         # iterate over each row and mark top "num_muts" mutants as sampled
        total_sampled = 0
        for index, row in working_df_copy.iterrows():
            if total_sampled < num_muts: # check if total sampled is less than num_muts
                if row['sampled'] == -1 and row['pos'] not in sampled_positions: # check if sampled is 0 and not in sampled positions
                    working_df_copy.loc[index, 'sampled'] = sampling_round # mark as sampled
                    total_sampled += 1 # increment total sampled
                    sampled_positions.append(row['pos'])
                else:
                    continue


     # Check if adjusting diversity by percentage of unique positions
    elif isinstance(diverse_pos, str) and re.match(r'^\d+% Unique Positions$', diverse_pos):
        # Extract percentage value from string
        percent = int(diverse_pos.split('%')[0])

        # Calculate the number of unique mutations
        num_muts_unique = round(num_muts * percent / 100, 0)
        num_muts_mod = num_muts - num_muts_unique

        total_sampled_unique = 0
        total_sampled_mod = 0

        # Sample unique positions first
        sampled_positions = list(working_df_copy[working_df_copy['sampled'] == sampling_round]['pos'].unique())
        # Iterate over each row and mark top "num_muts" mutants as sampled
        for index, row in working_df_copy.iterrows():
            if total_sampled_unique < num_muts_unique:  # Check if total unique sampled is less than num_muts_unique
                if row['sampled'] == -1 and row['pos'] not in sampled_positions:  # Check if not sampled and unique position
                    working_df_copy.loc[index, 'sampled'] = sampling_round  # Mark as sampled
                    total_sampled_unique += 1  # Increment total unique sampled
                    sampled_positions.append(row['pos'])
            else:
                break
        
        # Sample to fill up to total mutants
        for index, row in working_df_copy.iterrows():
            if total_sampled_mod < num_muts_mod:  # Check if total sampled is less than num_muts_mod
                if row['sampled'] == -1:  # Check if not sampled
                    working_df_copy.loc[index, 'sampled'] = sampling_round  # Mark as sampled
                    total_sampled_mod += 1  # Increment total sampled
            else:
                break

    elif diverse_pos == 'Unique AA Mutations':
        
        unique_aas = working_df_copy['new_aa'].unique()
        sampled_aas = {aa: 0 for aa in unique_aas}
        total_sampled = 0

        # determine how many times each aa can be sampled evenly
        num_muts_per_aa = num_muts // len(unique_aas) + 2
        for num_mut_per_aa in range(1, num_muts_per_aa, 1):
            for index, row in working_df_copy.iterrows():
                # check if every aa is sampled at least once
                if not all(sampled_aas[aa] == num_mut_per_aa for aa in sampled_aas):
                    if total_sampled < num_muts: 
                        if row['sampled'] == -1 and sampled_aas[row['new_aa']] < num_mut_per_aa:
                            sampled_aas[row['new_aa']] += 1
                            working_df_copy.loc[index, 'sampled'] = sampling_round
                            total_sampled += 1
                        else:
                            continue
                else:
                    break

    elif diverse_pos == 'Unique AA Switches':

        unique_switches = working_df_copy['aa_switch'].unique()
        sampled_switches = {switch: 0 for switch in unique_switches}
        total_sampled = 0

        for index, row in working_df_copy.iterrows():
            if not all(sampled_switches[switch] == 1 for switch in sampled_switches):
                if total_sampled < num_muts:
                    if row['sampled'] == -1 and sampled_switches[row['aa_switch']] < 1:
                        sampled_switches[row['aa_switch']] += 1
                        working_df_copy.loc[index, 'sampled'] = sampling_round
                        total_sampled += 1

    
    return working_df_copy

# MLDE STEP 4: set up training data for training models
def export_rd0(working_df): # outputs a train_df

    train_df = working_df[['mutated_sequence', 'DMS_score', 'sampled']]
    train_df_final = train_df[train_df['sampled'] >= 0][['mutated_sequence', 'DMS_score']]
    # deduplicate based on 'mutated_sequence'
    train_df_final = train_df_final.drop_duplicates(subset=['mutated_sequence'])
    # reset index
    train_df_final.reset_index(drop=True, inplace=True)

    return train_df_final

def export_rdmlde(working_df): # outputs a train_df

    train_df = working_df[['mutated_sequence', 'DMS_score', 'sampled']]
    train_df_final = train_df[train_df['sampled'] >= 0][['mutated_sequence', 'DMS_score']]
    train_df_final.reset_index(drop=True, inplace=True)

    return train_df_final