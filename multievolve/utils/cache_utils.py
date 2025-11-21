# This module contains utility functions for caching and loading feature model data

import numpy as np
import os
import pickle
import sys

def cache_namespace(fmodel_type, protein):
    """
    Creates a namespace directory for caching feature models of a specific protein.
    
    Args:
    - fmodel_type (str): Type of feature model.
    - protein (str): Name of the protein.
    
    Returns:
    - str: Path to the namespace directory.
    """
    fmodel_type = fmodel_type.replace('/', '-')
    root_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    namespace = f'{root_folder}/proteins/{protein}/feature_cache/{fmodel_type}'
    if not os.path.exists(namespace):
        os.makedirs(namespace)
    return namespace

def load_cache(fmodel_type, protein, verbose=1):
    """
    Loads cached feature model data for a specific protein.
    
    Args:
    - fmodel_type (str): Type of feature model.
    - protein (str): Name of the protein.
    - verbose (int): Whether to print the number of sequences loaded from cache.
    
    Returns:
    - dict: Cached data where keys are sequences and values are feature arrays.
    """
    dirname = cache_namespace(fmodel_type, protein)
    
    if not os.path.exists(f'{dirname}/seqs.pkl') or \
       not os.path.exists(f'{dirname}/X.npy'):
        sys.stderr.write(f'Warning: Could not load cache in {dirname}\n')
        return {}

    with open(f'{dirname}/seqs.pkl', 'rb') as f:
        seqs = pickle.load(f)

    X = np.load(f'{dirname}/X.npy')

    cache = {
        seq: X[idx] for idx, seq in enumerate(seqs)
    }
    
    if verbose > 0:
        print(f'Loaded {len(cache)} sequences from cache.')

    return cache

def update_cache(fmodel_type, protein, updating_cache_values):
    """
    Update the existing cache with new values.
    
    Args:
    - fmodel_type (str): Type of feature model.
    - protein (str): Name of the protein.
    - updating_cache_values (dict): New values to update the cache with, where keys are sequences and values are feature arrays.
    """
    dirname = cache_namespace(fmodel_type, protein)

    existing_cache = load_cache(fmodel_type, protein, verbose=0)
    new_cache_values = { seq: val for seq, val in updating_cache_values.items() if seq not in existing_cache.keys() }
    # print(f'Existing cache: {len(existing_cache)}')
    print(f'Updating cache with {len(new_cache_values)} new values for {fmodel_type}')
    if len(new_cache_values) > 0:
        updated_cache = existing_cache | new_cache_values
        print(f'Updated cache: {len(updated_cache)}')

        seqs = list(updated_cache.keys())
        X = np.array([ updated_cache[seq] for seq in seqs ])
        
        with open(f'{dirname}/seqs.pkl', 'wb') as f:
            pickle.dump(seqs, f)

        np.save(f'{dirname}/X.npy', X)
