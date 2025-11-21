# This utils page is for functions for manipulation and handling datasets 

from concurrent.futures import ProcessPoolExecutor
import re
import math

import pandas as pd
import numpy as np
import Levenshtein
from torch.utils.data import DataLoader, Dataset
import torch

# Given a set of mutations separated by "/" (e.g "G19S/R420G"), convert it into a list; if given 'WT', return ['WT']
def convert_mutation_list(string):
    """
    Convert a mutation string into a list of mutations.
    
    Args:
    - string (str): Mutation string separated by "/" (e.g. "G19S/R420G") or 'WT'.
    
    Returns:
    - list: List of mutations or ['WT'] if input is 'WT'.
    """
    if isinstance(string, float) and math.isnan(string):
        return ['WT']
    else:
        mutation_list = string.split('/')
        filtered_mutation_list = [mutation for mutation in mutation_list if re.search(r'[a-zA-Z]\d+[a-zA-Z]', mutation) or mutation == 'WT']
    return filtered_mutation_list

# Given a wild-type sequence and a list of mutations, generate the mutant sequence
def make_mutations(seq, mutations):
    """
    Given a wild-type sequence and a list of mutations, generate the mutant sequence.
    
    Args:
    - seq (str): Wild-type sequence.
    - mutations (list): List of mutations (e.g. ["G19S", "R420G"]).
    
    Returns:
    - str: Mutant sequence.
    """
    mut_seq = [char for char in seq]
    
    for mutation in mutations:
        if mutation == 'WT':
            break
        else:
            wt, pos, mt = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]
            assert seq[pos] == wt, f"{wt}{pos+1}{mt} is not a true mutation from {seq[pos]}{pos+1}"
            mut_seq[pos] = mt
    mut_seq = ''.join(mut_seq).replace('-', '')
    return mut_seq

def mutation_format_check(mutation):
    """
    Check the format of the mutation.
    
    Args:
    - mutation (str or list): Mutation in string or list format.
    
    Returns:
    - str: Format of the mutation ('Mutation String', 'Mutation List', or 'Full Sequence').
    """
    if type(mutation) == str:
        if re.search(r'[a-zA-Z]\d+[a-zA-Z]', mutation) or mutation == 'WT':
            return 'Mutation String'
        else:
            return 'Full Sequence'
        
    if type(mutation) == list or type(mutation) == tuple:
        assert re.search(r'[a-zA-Z]\d+[a-zA-Z]', mutation[0]), f"{mutation[0]} is not a true mutation"
        return 'Mutation List'

    raise ValueError('mutation not in Mutation String, Mutation List, or Full Sequence format')

def find_mutation_positions(seq1, seq2):
    """
    Find the positions of mutations between two sequences.
    
    Args:
    - seq1 (str): First sequence (wild-type).
    - seq2 (str): Second sequence (mutant).
    
    Returns:
    - list: List of mutation positions.
    """
    mutation_set = []
    pos1 = 0

    for wt, mt in zip(seq1, seq2):
        pos1 += 1
        if wt != mt:
            mut_str = pos1
            mutation_set.append(mut_str)
    
    if len(mutation_set) == 0:
        mutation_set = [0]

    return mutation_set

def find_mutation_positions_helper(args):
    """
    Helper function to find mutation positions.
    
    Args:
    - args (tuple): Tuple containing wild-type sequence and mutant sequence.
    
    Returns:
    - list: List of mutation positions.
    """
    wt_seq, seq = args
    seq = seq.replace('X', '')
    mutation_set = find_mutation_positions(wt_seq, seq)
    return mutation_set

def find_mutation_positions_multithreaded(wt_seq, seqs):
    """
    Find mutation positions using multithreading.
    
    Args:
    - wt_seq (str): Wild-type sequence.
    - seqs (list): List of mutant sequences.
    
    Returns:
    - list: List of mutation positions for each mutant sequence.
    """
    args = [(wt_seq, seq) for seq in seqs]
    with ProcessPoolExecutor() as executor:
        mutation_sets = executor.map(find_mutation_positions_helper, args)
    return list(mutation_sets)

def find_mutations(seq1, seq2):
    """
    Find mutations between two sequences.
    
    Args:
    - seq1 (str): First sequence (wild-type).
    - seq2 (str): Second sequence (mutant).
    
    Returns:
    - list: List of mutations in the format 'wt_pos_mt'.
    """
    mutation_set = []
    pos1 = 0
    for wt, mt in zip(seq1, seq2):
        pos1 += 1
        if wt != mt:
            mut_str = f'{wt}{pos1}{mt}'
            mutation_set.append(mut_str)

    return mutation_set

def find_mutations_helper(args):
    """
    Helper function to find mutations.
    
    Args:
    - args (tuple): Tuple containing wild-type sequence and mutant sequence.
    
    Returns:
    - list: List of mutations in the format 'wt_pos_mt'.
    """
    wt_seq, seq = args
    seq = seq.replace('X', '')
    mutation_set = find_mutations(wt_seq, seq)
    return mutation_set

def find_mutations_multithreaded(wt_seq, seqs):
    """
    Find mutations using multithreading.
    
    Args:
    - wt_seq (str): Wild-type sequence.
    - seqs (list): List of mutant sequences.
    
    Returns:
    - list: List of mutations for each mutant sequence.
    """
    args = [(wt_seq, seq) for seq in seqs]
    with ProcessPoolExecutor() as executor:
        mutations = list(executor.map(find_mutations_helper, args))
    return mutations

class MutationFormat:
    """
    Class to handle different mutation formats.

    Attributes:
        mutation (str or list): The mutation in its original format.
        wt_seq (str): The wild-type sequence.
        format (str): The determined format of the mutation.
        formats (dict): Dictionary storing the mutation in different formats.
    """
    def __init__(self, mutation, wt_seq):
        """
        Initialize MutationFormat.
        
        Args:
        - mutation (str or list): Mutation in string or list format.
        - wt_seq (str): Wild-type sequence.
        """
        self.mutation = mutation
        self.wt_seq = wt_seq
        self._determine_type()
        self.formats = {}
        self.formats[self.format] = mutation

    def _determine_type(self):
        """
        Determine the format of the mutation.
        """
        self.format = mutation_format_check(self.mutation)

    def to_full_sequence(self):
        """
        Convert mutation to full sequence format.
        
        Returns:
        - str: Full sequence.
        """
        if 'Full Sequence' in self.formats.keys():
            return self.formats['Full Sequence']
        
        if 'Mutation List' in self.formats.keys():
            full_sequence = make_mutations(self.wt_seq, self.formats['Mutation List'])
            self.formats['Full Sequence'] = full_sequence
            return full_sequence

        if 'Mutation String' in self.formats.keys():
            mutation_list = self.formats['Mutation String'].split('/')
            full_sequence = make_mutations(self.wt_seq, mutation_list)
            self.formats['Mutation List'] = mutation_list
            self.formats['Full Sequence'] = full_sequence
            return full_sequence

    def to_mutation_list(self):
        """
        Convert mutation to mutation list format.
        
        Returns:
        - list: List of mutations.
        """
        if 'Mutation List' in self.formats.keys():
            return self.formats['Mutation List']
        
        if 'Mutation String' in self.formats.keys():
            mutation_list = self.formats['Mutation String'].split('/')
            self.formats['Mutation List'] = mutation_list
            return mutation_list
        
        if 'Full Sequence' in self.formats.keys():
            mutation_list = find_mutations(self.wt_seq, self.formats['Full Sequence'])
            self.formats['Mutation List'] = mutation_list
            return mutation_list

    def to_mutation_string(self):
        """
        Convert mutation to mutation string format.
        
        Returns:
        - str: Mutation string.
        """
        if 'Mutation String' in self.formats.keys():
            return self.formats['Mutation String']
        
        if 'Mutation List' in self.formats.keys():
            mutation_string = "/".join(self.formats['Mutation List'])
            self.formats['Mutation String'] = mutation_string
            return mutation_string

        if 'Full Sequence' in self.formats.keys():
            mutation_list = find_mutations(self.wt_seq, self.formats['Full Sequence'])
            mutation_string = "/".join(mutation_list)
            self.formats['Mutation List'] = mutation_list
            self.formats['Mutation String'] = mutation_string
            return mutation_string

class MutationListFormats:
    """
    Class to handle different formats of mutation lists.

    Attributes:
        mutation_list (list): List of mutations.
        wt_seq (str): The wild-type sequence.
        format (str): The determined format of the mutations.
        formats (dict): Dictionary storing the mutations in different formats.

    Example Usage:
    
    muts = pd.read_csv('muts.csv', header=None) # load csv file with sequences in first column
    muts_ls = muts[0].tolist()
    mut_seqs = MutationListFormats(muts_ls, wt_seq)
    
    # get mutation strings
    muts['mut_strings'] = mut_seqs.to_mutation_strings()
    
    # get mutation lists
    muts['mut_lists'] = mut_seqs.to_mutation_lists()
    
    # get full sequences
    muts['full_seqs'] = mut_seqs.to_full_sequences()
    
    """
    def __init__(self, mutation_list, wt_seq):
        """
        Initialize MutationListFormats.
        
        Args:
        - mutation_list (list or pd.Series or pd.DataFrame): List of mutations.
        - wt_seq (str): Wild-type sequence.
        """
        if isinstance(mutation_list, pd.Series):
            mutation_list = mutation_list.tolist()
        elif isinstance(mutation_list, pd.DataFrame):
            cols = mutation_list.columns
            mutation_list = mutation_list[cols[0]].tolist()
        assert isinstance(mutation_list, list), 'mutation_list must be a list'
        self.mutation_list = mutation_list
        self.wt_seq = wt_seq
        self._determine_type(mutation_list[0])
        self.formats = {}
        self.formats[self.format] = self.mutation_list
    
    def _determine_type(self, mutation):
        """
        Determine the format of the mutation.
        
        Args:
        - mutation (str): Mutation in string format.
        """
        self.format = mutation_format_check(mutation)

    def to_full_sequences(self):
        """
        Convert mutation list to full sequences format.
        
        Returns:
        - list: List of full sequences.
        """
        if 'Full Sequence' in self.formats.keys():
            return self.formats['Full Sequence']
        
        if 'Mutation List' in self.formats.keys():
            full_sequences = [make_mutations(self.wt_seq, mutation_list) for mutation_list in self.formats['Mutation List']]
            self.formats['Full Sequences'] = full_sequences
            return full_sequences
        
        if 'Mutation String' in self.formats.keys():
            mutation_lists = [mutation_string.split('/') for mutation_string in self.formats['Mutation String']]
            full_sequences = [make_mutations(self.wt_seq, mutation_list) for mutation_list in mutation_lists]
            self.formats['Mutation Lists'] = mutation_lists
            self.formats['Full Sequences'] = full_sequences
            return full_sequences
        
    def to_mutation_lists(self):
        """
        Convert mutation list to mutation lists format.
        
        Returns:
        - list: List of mutation lists.
        """
        if 'Mutation List' in self.formats.keys():
            return self.formats['Mutation List']
        
        if 'Mutation String' in self.formats.keys():
            mutation_lists = [mutation_string.split('/') for mutation_string in self.formats['Mutation String']]
            self.formats['Mutation Lists'] = mutation_lists
            return mutation_lists
        
        if 'Full Sequence' in self.formats.keys():
            mutation_lists = find_mutations_multithreaded(self.wt_seq, self.formats['Full Sequence'])
            self.formats['Mutation Lists'] = mutation_lists
            return mutation_lists
        
    def to_mutation_strings(self):
        """
        Convert mutation list to mutation strings format.
        
        Returns:
        - list: List of mutation strings.
        """
        if 'Mutation String' in self.formats.keys():
            return self.formats['Mutation String']
        
        if 'Mutation List' in self.formats.keys():
            mutation_strings = ["/".join(mutation_list) for mutation_list in self.formats['Mutation List']]
            self.formats['Mutation Strings'] = mutation_strings
            return mutation_strings
        
        if 'Full Sequence' in self.formats.keys():
            mutation_lists = find_mutations_multithreaded(self.wt_seq, self.formats['Full Sequence'])
            mutation_strings = ["/".join(mutation_list) for mutation_list in mutation_lists]
            self.formats['Mutation Lists'] = mutation_lists
            self.formats['Mutation Strings'] = mutation_strings
            return mutation_strings

    def get_mutation_pool(self):
        """
        Get all the pool of single mutations in the mutation list.
        
        Returns:
        - list: List of unique single mutations.
        """
        mutation_lists = self.to_mutation_lists()
        mutation_pool = set()
        for mutation_list in mutation_lists:
            mutation_pool.update(mutation_list)
        return list(mutation_pool)

# This code snippet was taken from https://github.com/VincentQTran/low-N-protein-engineering/blob/master/analysis/common/utils.py
def levenshtein_distance_matrix(a_list, b_list=None, verbose=False):
    """
    Computes a len(a_list) x len(b_list) Levenshtein distance matrix.
    
    Args:
    - a_list (list): List of sequences.
    - b_list (list, optional): List of sequences. If None, computes the distance matrix for a_list against itself.
    - verbose (bool, optional): If True, prints progress.
    
    Returns:
    - np.ndarray: Levenshtein distance matrix.
    """
    if b_list is None:
        single_list = True
        b_list = a_list
    else:
        single_list = False
    
    H = np.zeros(shape=(len(a_list), len(b_list)))
    for i in range(len(a_list)):
        if verbose:
            print(i)
        
        if single_list:  
            # only compute upper triangle.
            for j in range(i+1, len(b_list)):
                H[i, j] = Levenshtein.distance(a_list[i], b_list[j])
                H[j, i] = H[i, j]
        else:
            for j in range(len(b_list)):
                H[i, j] = Levenshtein.distance(a_list[i], b_list[j])

    return H

# Classes to handle data
class TorchCustomDataset(Dataset):
    """
    Class to create a PyTorch dataset from a list of sequences and labels.

    Attributes:
        encodings (list): List of encoded sequences.
        labels (list): List of labels corresponding to the sequences.
        original_sequences (list): List of original sequences before encoding.
    """
    def __init__(self, encodings, labels, original_sequences):
        """
        Initialize TorchCustomDataset.
        
        Args:
        - encodings (list): List of encoded sequences.
        - labels (list): List of labels.
        - original_sequences (list): List of original sequences.
        """
        self.encodings = encodings
        self.labels = labels
        self.original_sequences = original_sequences

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
        - int: Number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - tuple: Encoded sequence, label, and original sequence.
        """
        return self.encodings[idx], self.labels[idx], self.original_sequences[idx]

class TorchDataProcessor:
    """
    Processes data for neural network models.

    Attributes:
        featurizer (object): Object to featurize sequences.
        X_train, X_val, X_test (list): Lists of sequences for training, validation, and testing.
        y_train, y_val, y_test (list): Lists of labels for training, validation, and testing.
        split_name (str): Name of the data split.
        bs (int): Batch size for data loading.
        X_train_feat, X_val_feat, X_test_feat (np.array): Featurized sequences.
        train_dataset, val_dataset, test_dataset (TorchCustomDataset): PyTorch datasets.
        train_loader, val_loader, test_loader (DataLoader): PyTorch DataLoaders.
    """
    def __init__(self, split, featurizer, batch_size):
        """
        Initialize TorchDataProcessor.
        
        Args:
        - split (object): Object containing data splits.
        - featurizer (object): Object to featurize sequences.
        - batch_size (int): Batch size for data loading.
        """
        self.featurizer = featurizer
        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
            self.split_name,
        ) = (
            split.splits['X_train'],
            split.splits['X_val'],
            split.splits['X_test'],
            split.splits['y_train'],
            split.splits['y_val'],
            split.splits['y_test'],
            split.splits['split_name'],
        )

        self.bs = batch_size

    def featurize(self, X):
        """
        Featurizes a list of sequences X.
        
        Args:
        - X (list): List of sequences.
        
        Returns:
        - list: List of featurized sequences.
        """
        X_featurized = self.featurizer.featurize(X)
        return X_featurized

    def setup_train_loader(self):
        """
        Setup the train loader if not already created.
        """
        if hasattr(self, 'train_loader'):
            return self.train_loader

        print("Featurizing training data...")
        self.X_train_feat = self.featurizer.featurize(self.X_train)
        
        self.train_dataset = TorchCustomDataset(
            torch.from_numpy(self.X_train_feat.astype(np.float32)),
            torch.from_numpy(self.y_train.astype(np.float32)),
            self.X_train
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        return self.train_loader
    
    def setup_val_loader(self):
        """
        Setup the validation loader if not already created.
        """
        if hasattr(self, 'val_loader'):
            return self.val_loader

        print("Featurizing validation data...")
        self.X_val_feat = self.featurizer.featurize(self.X_val)
        
        self.val_dataset = TorchCustomDataset(
            torch.from_numpy(self.X_val_feat.astype(np.float32)),
            torch.from_numpy(self.y_val.astype(np.float32)),
            self.X_val
        )
        
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=True)
        return self.val_loader
    
    def setup_test_loader(self):
        """
        Setup the test loader if not already created.
        """
        if hasattr(self, 'test_loader'):
            return self.test_loader

        print("Featurizing testing data...")
        self.X_test_feat = self.featurizer.featurize(self.X_test)
        
        self.test_dataset = TorchCustomDataset(
            torch.from_numpy(self.X_test_feat.astype(np.float32)),
            torch.from_numpy(self.y_test.astype(np.float32)),
            self.X_test
        )
        
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, shuffle=True)
        return self.test_loader

    def preprocess_data(self):
        """
        Set up all data loaders.
        """
        self.setup_train_loader()
        self.setup_val_loader() 
        self.setup_test_loader()