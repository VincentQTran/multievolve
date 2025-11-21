import pandas as pd
import numpy as np

from pathlib import Path
import json, hashlib
import re

from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *

class TrainingCache:
    """
    A cache class for storing training results with an index-based lookup system.
    
    Attributes:
        dir (Path): Directory path for cache storage
        index_path (Path): Path to the index CSV file
        index (pd.DataFrame): DataFrame containing cache metadata and lookup information
    """
    
    def __init__(self, cache_dir: str | Path):
        """
        Initialize the TrainingCache.
        
        Args:
            cache_dir (str | Path): Directory path where cache files will be stored
        """
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.csv"

        if self.index_path.exists():
            # Just load it back; columns will include whatever keys you added
            self.index = pd.read_csv(self.index_path)
        else:
            # Start empty; no need to predefine columns, they’ll be added by set()
            self.index = pd.DataFrame()

    def _key_id(self, keys: dict) -> str:
        """
        Generate a unique hash ID for the given keys dictionary.
        
        Args:
            keys (dict): Dictionary of keys to hash
            
        Returns:
            str: MD5 hash of the serialized keys
        """
        blob = json.dumps(keys, sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()

    def _path(self, key_id: str) -> Path:
        """
        Generate the file path for a given key ID.
        
        Args:
            key_id (str): Unique identifier for the cache entry
            
        Returns:
            Path: Path object pointing to the pickle file
        """
        return self.dir / f"{key_id}.pkl"

    def _check_index(self, row: pd.DataFrame) -> None:

        # check if index exists
        if self.index_path.exists():
            self.index = pd.read_csv(self.index_path)

        # overwrite existing entry in case of updating path
        if not self.index.empty and "key_id" in self.index.columns:
            self.index = self.index[self.index["key_id"] != row['key_id']]

        self.index = pd.concat([self.index, pd.DataFrame([row])], ignore_index=True)
        self.index.to_csv(self.index_path, index=False)


    def get(self, keys: dict) -> pd.DataFrame | None:
        """
        Check if a pkl file for run exists, if so retrieve dataframe and check index if run is in index.
        
        Args:
            keys (dict): Dictionary of keys to look up
            
        Returns:
            pd.DataFrame | None: Cached DataFrame if found, None otherwise
        """

        # Check if pkl file for run exists else return None
        path = self._path(self._key_id(keys))

        if not path.exists():
            return None

        # retrieve dataframe
        df = pd.read_pickle(path)

        # update index
        row = {
            "key_id": self._key_id(keys),
            "path": str(path),
            "variants": df.shape[0],
            **keys,  # expand keys directly into columns
        }
        self._check_index(row)

        return df
        
    def set(self, keys: dict, df: pd.DataFrame) -> None:
        """
        Store a DataFrame in the cache with the given keys and check index to update with new entry.
        
        Args:
            keys (dict): Dictionary of keys to associate with the DataFrame
            df (pd.DataFrame): DataFrame to cache
        """

        key_id = self._key_id(keys)
        path = self._path(key_id)

        # save dataframe
        df.to_pickle(path)

        # build row with expanded keys
        row = {
            "key_id": key_id,
            "path": str(path),
            "variants": df.shape[0],
            **keys,  # expand keys directly into columns
        }

        # update index
        self._check_index(row)
        

def summary_df_check_dms_completion(summary_df, threshold=0.8):
    
    def check_dms_completion(row):
        fraction_dms = row['DMS_number_single_mutants'] / (row['seq_len'] * 19)
        if fraction_dms >= threshold:
            return fraction_dms, True
        else:
            return fraction_dms, False

    summary_df[['fraction_dms', 'dms_threshold_met']] = summary_df.apply(check_dms_completion, axis=1, result_type='expand')
    return summary_df

# function to receive dataset name, dataset filename, and sequence from dataframe row
def receive_dataset_vars(row):
    # generate fasta file of sequence
    dataset_name = row['DMS_id']
    dataset_fname = row['DMS_filename']
    sequence = row['target_seq']

    return dataset_name, dataset_fname, sequence

# function to generate fasta file of sequence
def retrieve_wt_file(dataset_name, seq_dir, sequence):
    
    output_dir = Path(seq_dir)
    wt_file = output_dir / f'{dataset_name}.fasta'
    if wt_file.exists():
        return str(wt_file)
    else:
        # Prepare folder to save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(wt_file, 'w') as file:
            file.write(f'>{dataset_name}\n')
            file.write(sequence + '\n')
        
        return str(wt_file)

# function to preprocess dataset (mark valid multimutants), add relevant columns
def preprocess_dataset(dataset_fname, data_dir, stringency='singles'):
    # options for stringency: 'singles', 'singles_or_doubles', 'singles_positions'

    # check for existing processed datasets
    processed_datasets_dir = os.path.join(data_dir, 'processed')
    processed_datasets_all_dir = os.path.join(data_dir, 'processed', 'all')
    processed_datasets_stringency_dir = os.path.join(data_dir, 'processed', stringency)
    processed_all_filename = os.path.join(processed_datasets_all_dir, f'{dataset_fname}.csv')
    processed_stringency_filename = os.path.join(processed_datasets_stringency_dir, f'{dataset_fname}.csv')
    if os.path.exists(processed_all_filename) and os.path.exists(processed_stringency_filename):
        return pd.read_csv(processed_all_filename), pd.read_csv(processed_stringency_filename)

    # read csv file
    working_df_head = pd.read_csv(os.path.join(data_dir, dataset_fname))

    # replace colon with slash in mutant column
    working_df_head['mutant'] = working_df_head['mutant'].apply(lambda x: x.replace(':', '/'))

    # retrieve number of mutations
    working_df_head['num_mutations'] = working_df_head['mutant'].apply(lambda x: len(x.split('/')))

    # retrieve single mutants
    singles = working_df_head[working_df_head['num_mutations'] == 1]['mutant'].tolist()
    
    # get positions from single mutants
    singles_positions = []
    pattern = r'[A-Z]\d+[A-Z]'
    for mutant in singles:
        if len(re.findall(pattern, mutant)) != 0:
            singles_positions.append(int(mutant[1:-1]))
    singles_positions = list(set(singles_positions))
    
    # retrieve mutations doubles
    doubles = [single for double in working_df_head[working_df_head['num_mutations'] == 2]['mutant'].tolist() for single in double.split('/')]

    # function to check for multimutants where are single mutants are present
    def check_existing_mutants_in_singles(mutant):
        mutant_list = mutant.split('/')
        for m in mutant_list:
            if m in singles:
                pass
            else:
                return False
        return True

    def check_existing_mutants_in_singles_or_doubles(mutant):
        mutant_list = mutant.split('/')
        for m in mutant_list:
            if m in singles or m in doubles:
                pass
            else:
                return False
        return True
    
    def check_existing_mutants_in_singles_positions(mutant):
        mutant_list = mutant.split('/')
        if mutant_list[0] == 'WT':
            return True
        else:
            for m in mutant_list:
                if int(m[1:-1]) in singles_positions:
                    pass
                else:
                    return False
            return True

    # filter dataset keeping only combo variants with all single mutants existing in the dataset
    working_df_head['singles_exist'] = working_df_head['mutant'].apply(check_existing_mutants_in_singles)
    working_df_head['singles_or_doubles_exist'] = working_df_head['mutant'].apply(check_existing_mutants_in_singles_or_doubles)
    working_df_head['singles_positions_exist'] = working_df_head['mutant'].apply(check_existing_mutants_in_singles_positions)

    if stringency == 'singles':
        working_df_head_valid = working_df_head[working_df_head['singles_exist'] == True].sort_values(by='num_mutations', ascending=False)
    elif stringency == 'singles_or_doubles':
        working_df_head_valid = working_df_head[working_df_head['singles_or_doubles_exist'] == True].sort_values(by='num_mutations', ascending=False)
    elif stringency == 'singles_positions':
        working_df_head_valid = working_df_head[working_df_head['singles_positions_exist'] == True].sort_values(by='num_mutations', ascending=False)
    else:
        raise ValueError(f'Invalid stringency: {stringency}. Please choose from ["singles", "singles_or_doubles"].')

    # save results
    os.makedirs(os.path.join(processed_datasets_all_dir), exist_ok=True)
    os.makedirs(os.path.join(processed_datasets_stringency_dir), exist_ok=True)
    working_df_head.to_csv(processed_all_filename, index=False)
    working_df_head_valid.to_csv(processed_stringency_filename, index=False)

    return working_df_head, working_df_head_valid

def select_feature(encoding, protein_name, batch_size=1000):

    # check if encoding_name is in the list of available encodings
    if encoding not in ['esm2_15b', 'esm2_3b', 'onehot', 'georgiev', 'aaidx', 'onehot_and_esm2_15b', 'ankh_base', 'ankh_large', 'ProtT5_XL_U50_Embed']:
        raise ValueError(f'Invalid encoding_name: {encoding}. Please choose from {["esm2_15b", "esm2_3b", "onehot", "georgiev", "aaidx", "ankh_base", "ankh_large", "ProtT5_XL_U50_Embed"]}.')
    
    # get feature
    if encoding == 'esm2_15b':
        feature = ESM2_15b_EmbedFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'esm2_3b':
        feature = ESM2EmbedFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'onehot':
        feature = OneHotFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'georgiev':
        feature = GeorgievFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'aaidx':
        feature = AAIdxFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'onehot_and_esm2_15b':
        feature = OnehotAndESM2_15bEmbedFeaturizer(protein=protein_name, use_cache=True)
    elif encoding == 'ankh_base':
        feature = AnkhBaseFeaturizer(protein=protein_name, use_cache=True, batch_size=batch_size)
    elif encoding == 'ankh_large':
        feature = AnkhLargeFeaturizer(protein=protein_name, use_cache=True, batch_size=batch_size)
    elif encoding == 'ProtT5_XL_U50_Embed':
        feature = ProtT5_XL_U50_EmbedFeaturizer(protein=protein_name, use_cache=True, batch_size=batch_size)
    return feature