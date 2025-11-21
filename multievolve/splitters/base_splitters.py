import random
import pandas as pd
from abc import ABC, abstractmethod
from Bio import SeqIO, PDB
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import copy
import shutil

import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

from multievolve.utils.other_utils import aa_dict_3to1
from multievolve.utils.data_utils import find_mutation_positions_multithreaded, MutationFormat

class BaseSplitter(ABC):
    """Abstract base class for splitters."""

    """
    Attributes:
        wt_seq (str): Wild-type sequence of the protein.
        use_cache (bool): Flag to determine if caching should be used.
        random_state (int): Random state for reproducibility.
        data (pd.DataFrame): DataFrame containing the protein data.
        file_attrs (dict): Dictionary containing file attributes and paths.

    Example Usage:

    splitter = BaseSplitter(training_dataset_fname, 
                            wt_file, 
                            csv_has_header=True,  # Whether input CSV has a header row (default: True)
                            use_cache=True        # Whether to cache split results (default: True)
                            )
    splitter.split_data()
    """

    def __init__(self, name, data, wt_file, csv_has_header=False, use_cache=False,
                 random_state=42, type='biomolecules',
                 **kwargs):
        """
        Args:
        - name (str): Name of the biomolecule.
        - data (str or pd.DataFrame): Accepts a table of data, containing two columns: one for sequences (for multi-chain proteins, colon-separated sequences) and the second one for the labels.
        - wt_file (str or list): File path(s) to wild-type sequence of protein of interest in FASTA format.
        - csv_has_header (bool): Flag to modify if your data has headers (True) or does not (False).
        - use_cache (bool): Flag to use cache.
        - random_state (int): Random state for reproducibility.
        - type (str): Type of biomolecule.
        - **kwargs: Additional keyword arguments.
        """

        self.wt_seq_lens = []
        self.wt_seqs = []
        if isinstance(wt_file, str):
            self.wt_seq_lens.append(len(str(SeqIO.read(wt_file, "fasta").seq)))
            self.wt_seqs.append(str(SeqIO.read(wt_file, "fasta").seq))
        elif isinstance(wt_file, list):
            for file in wt_file:
                self.wt_seq_lens.append(len(str(SeqIO.read(file, "fasta").seq)))
                self.wt_seqs.append(str(SeqIO.read(file, "fasta").seq))
        self.wt_seq = ''.join(self.wt_seqs)
        self.use_cache = use_cache
        self.random_state = random_state

         # If the data is a CSV file
        if isinstance(data, str) and data.endswith('.csv'):
            self.data = pd.read_csv(data, header=0 if csv_has_header else None)
            # Rename columns for consistency
            self.data.rename(columns={self.data.columns[0]: 0, self.data.columns[1]: 1}, inplace=True)
            dataset_name = os.path.splitext(os.path.basename(data))[0]
            dataset_file = os.path.join(root_folder, type, name, dataset_name + '.csv')
        elif isinstance(data, pd.DataFrame):  # If the data is already a DataFrame
            self.data = data.copy()
            # Ensure column names are standardized
            self.data.rename(columns={self.data.columns[0]: 0, self.data.columns[1]: 1}, inplace=True)
            dataset_name = 'dataframe_input'
            dataset_file = os.path.join(root_folder, type, name, dataset_name + '.csv')
        else:
            raise ValueError("Invalid data format: data must be a file path to a CSV or a DataFrame.")

        # Define file attributes and split directory
        self.file_attrs = {
            'dataset_file': dataset_file,
            'dataset_name': dataset_name,
            'dataset_dir': os.path.join(root_folder, type, name),
            'split_dir': os.path.join(root_folder, type, name, 'split_cache', dataset_name)
        }

        # Create cache directory if needed
        if self.use_cache:
            os.makedirs(self.file_attrs['split_dir'], exist_ok=True)

        # copy dataset file to new location
        if not os.path.exists(self.file_attrs['dataset_file']):
            os.makedirs(os.path.dirname(self.file_attrs['dataset_file']), exist_ok=True)
            
            if isinstance(data, str):  # data is a file path
                shutil.copy(data, self.file_attrs['dataset_file'])
            else:  # data is a DataFrame
                data.to_csv(self.file_attrs['dataset_file'], index=False)
        
# Note: For new classes on top of ProteinSplitter, all unique args for split_data() should be used for generate split_type attribute for proper cache storage

class ProteinSplitter(BaseSplitter):
    """Class for splitting protein datasets."""    

    """
    Attributes:
        y_scaling (str): String indicating whether y values are scaled or not.
        val_split (float): Fraction of data for validation set.
        file_attrs (dict): Dictionary containing file attributes and paths.
            base_splitter_path (str): Path to the base splitter pickle file.
        splits (dict): Dictionary to store the data splits.

    Example Usage:

    splitter = ProteinSplitter(training_dataset_fname, 
                                wt_file, 
                                csv_has_header=True,  # Whether input CSV has a header row
                                use_cache=True,       # Whether to cache results to disk
                                y_scaling=True,       # Whether to scale y values to [0,1]
                                val_split=None        # Fraction of data for validation (None=no validation split)
                                )
    splitter.split_data()
    """

    def __init__(self, protein_name, data, wt_file, csv_has_header=False, use_cache=False, 
                 random_state=42,
                 y_scaling=False, 
                 val_split=None,
                 **kwargs):
        """
        Args:
        - protein_name (str): Name of the protein.
        - data (str or pd.DataFrame): Accepts a table of data, containing two columns: one for sequences and the second one for the labels.
        - wt_file (str or list): File path(s) to wild-type sequence of protein of interest in FASTA format.
        - csv_has_header (bool): Flag to modify if your data has headers (True) or does not (False).
        - use_cache (bool): Flag to use cache.
        - y_scaling (bool): Flag to scale y values.
        - val_split (float): Fraction of data to partition into the validation set.
        - random_state (int): Random state for reproducibility.
        - **kwargs: Additional keyword arguments.
        """

        super().__init__(protein_name, data, wt_file, csv_has_header=csv_has_header, use_cache=use_cache, 
                         random_state=random_state, type='proteins',
                         **kwargs)
        
        # Define base protein splitter path
        if y_scaling == False:
            self.y_scaling = "y_unscaled"
        else:
            self.y_scaling = "y_scaled"

        self.val_split = val_split
        self.kfold_splits = False

        self.file_attrs['base_splitter_path'] = os.path.join(self.file_attrs['split_dir'], "base_splitter" + '_' + self.y_scaling + ".pkl")

        # Load existing base protein splitter or create a new one
        if self.use_cache and os.path.exists(self.file_attrs['base_splitter_path']):
            self.data = pd.read_pickle(self.file_attrs['base_splitter_path'])
            
        # Create base protein splitter if not found and save is use_cache is True
        else:
            
            if len(self.wt_seq_lens) > 1:
                # check MutationFormat of column 0
                if MutationFormat(self.data[0].iloc[0].split(':')[0], self.wt_seq).format == 'Mutation String':
                    self.data[0] = self.data[0].apply(lambda x: self._shift_mutation_position(x, self.wt_seq_lens, 'Mutation String'))
                elif MutationFormat(self.data[0].iloc[0].split(':')[0], self.wt_seq).format == 'Mutation List':
                    self.data[0] = self.data[0].apply(lambda x: self._shift_mutation_position(x, self.wt_seq_lens, 'Mutation List'))
                elif MutationFormat(self.data[0].iloc[0].split(':')[0], self.wt_seq).format == 'Full Sequence':
                    self.data[0] = self.data[0].apply(lambda x: ''.join(x.split(':')))
                else:
                    raise ValueError('Mutation format not recognized')

            self.data[0] = self.data[0].apply(lambda x: MutationFormat(x, self.wt_seq).to_full_sequence())
            mut_positions = find_mutation_positions_multithreaded(self.wt_seq, self.data[0].tolist())
            self.data['mut_positions'] = mut_positions
            self.data['muts'] = self.data[0].apply(lambda x: MutationFormat(x, self.wt_seq).to_mutation_string())
            self.data['mut_load'] = self.data['mut_positions'].apply(lambda x: len(x))

            if y_scaling == True:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(np.array(self.data[1]).reshape(-1, 1))
                self.data[1] = scaled_data

            # save as pickle file
            if self.use_cache:
                self.data.to_pickle(self.file_attrs['base_splitter_path'])

    def _shift_mutation_position(self, inputs, lengths, type):
        # inputs is a list of mutation strings or mutation lists
        # goal is to return a combined mutation string/mutation list
        inputs = inputs.split(':')
        lengths = [sum(lengths[:i+1]) for i in range(len(lengths))]
        new_inputs  = []
        # Process first input differently based on type
        if type == 'Mutation String':
            new_inputs.extend(inputs[0].split('/'))
        
        # Process remaining inputs
        for i in range(1, len(inputs)):
            mutations = inputs[i].split('/') if type == 'Mutation String' else inputs[i]
            for mut in mutations:
                if mut == 'WT':
                    break
                else:
                    mut_pos = mut[1:-1]
                    mut_pos = str(int(mut_pos) + lengths[i-1]) 
                    mut = mut[0] + mut_pos + mut[-1]
                    new_inputs.append(mut)
        
        if len(new_inputs) > 1 and "WT" in new_inputs:
            new_inputs.remove("WT")

        return '/'.join(new_inputs)

    def split_data(self, iter=None):
        """
        Splits data into training and test sets. 
        Data split into the test set is given a group label of 1,
        while data split into the training set is given a group label of 0.
        
        Args:
        - iter (int, optional): Iteration number for naming the split file.
        """
                   
        self.split_type = 'base'

        raise NotImplementedError("This method should be implemented by the subclass.")

    def _save_splits(self, iter=None):
        """
        Save splits for specific split type.
        
        Args:
        - iter (int, optional): Iteration number for naming the split file.
        """

        test_size = self.data['group'].sum() / len(self.data)

        # 0 for train, 1 for test, 2 for val

        if self.val_split is not None:

            if self.kfold_splits == True:
                pass
            else:
                # Separate train set into train and val sets
                rows_with_marker_0 = self.data[self.data['group'] == 0]
                num_to_sample = int(len(rows_with_marker_0) * self.val_split)
                sampled_indices = rows_with_marker_0.sample(n=num_to_sample).index
                self.data.loc[sampled_indices, 'group'] = 2

            X_train = self.data[self.data['group'] == 0][0].values
            X_val = self.data[self.data['group'] == 2][0].values
            X_test = self.data[self.data['group'] == 1][0].values
            y_train = self.data[self.data['group'] == 0][1].values
            y_val = self.data[self.data['group'] == 2][1].values
            y_test = self.data[self.data['group'] == 1][1].values

            train_size = len(X_train) / len(self.data)
            val_size = len(X_val) / len(self.data)

            # return splits
            # initialize dictionary to store splits
            self.splits = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test}


            # check if iter is none:
            if iter is None:
                split_name = f'split_by_{self.split_type}_{int(train_size*100)}-{int(val_size*100)}-{int(test_size*100)}-{self.y_scaling}'
            else:
                split_name = f'split_by_{self.split_type}_{int(train_size*100)}-{int(val_size*100)}-{int(test_size*100)}-{self.y_scaling}_iter{iter}'
            
        else:
            X_train = self.data[self.data['group'] == 0][0].values
            X_test = self.data[self.data['group'] == 1][0].values
            y_train = self.data[self.data['group'] == 0][1].values
            y_test = self.data[self.data['group'] == 1][1].values

            # initialize dictionary to store splits
            self.splits = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
            
            # check if iter is none:
            if iter is None:
                split_name = f'split_by_{self.split_type}_{int((1-test_size)*100)}-{int(test_size*100)}-{self.y_scaling}'
            else:
                split_name = f'split_by_{self.split_type}_{int((1-test_size)*100)}-{int(test_size*100)}-{self.y_scaling}_iter{iter}'

        # Save split or load split if it already exists to prevent overwriting
        self.splits['split_name'] = split_name

        file_name = os.path.join(self.file_attrs['split_dir'], split_name + ".pkl")

        if self.use_cache:
            if not os.path.exists(file_name):
                with open(file_name, 'wb') as file:
                    print("Split saved.")
                    pickle.dump(self.splits, file)
            
            else:
                with open(file_name, 'rb') as file:
                    print("Split already exists. Generated splits not saved. Loading pre-existing split.")
                    self.splits = pickle.load(file)

    def _assign_folds(self, k_folds):
        """
        Assigns fold labels to the data for K-Fold cross-validation if there is a validation split.
        
        Args:
        - k_folds (int): Number of folds.
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # initialize fold columns
        self.data['fold'] = None

        # get indices of rows designated for training/validation
        train_indices = self.data[self.data['group'] == 0].index

        # Shuffle the indices
        shuffled_indices = np.random.permutation(train_indices)

        # Determine the number of points per fold
        fold_sizes = [len(shuffled_indices) // k_folds] * k_folds
        for i in range(len(shuffled_indices) % k_folds):
            fold_sizes[i] += 1

        # Assign fold labels
        groups = np.zeros(len(self.data), dtype=int)
        current_idx = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            for i in range(current_idx, current_idx + fold_size):
                self.data.loc[shuffled_indices[i], 'fold'] = fold_idx
            current_idx += fold_size

    def _save_folds(self,k_folds):

        # assert at least 2 folds or code breaks
        assert k_folds >= 2, "At least 2 folds are required"

        # mark object as having kfold splits
        self.kfold_splits = True
        self._assign_folds(k_folds=k_folds)

        folds = []
        for fold_num in range(k_folds):

            # Create a deep copy of self and add it to folds
            fold_copy = copy.deepcopy(self)
            
            # Assign fold num to validation set if match fold num, otherwise keep in train or test
            fold_copy.data['group'] = np.where(
                fold_copy.data['fold'] == fold_num, 2, fold_copy.data['group']
            )

            fold_copy._save_splits(iter=fold_num)
            folds.append(fold_copy)
            
        self.folds = folds
    
    def load_splits(self, file_path):
        """
        Loads the split from a pickle file.
        
        Args:
        - file_path (str): Path to the pickle file containing the splits.
        """
        with open(file_path, 'rb') as file:
            self.splits = pickle.load(file)

class KFoldProteinSplitter(ProteinSplitter):
    """
    Class for K-Fold splitting of protein datasets.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.

    Example Usage:

    splitter = KFoldProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    splitter.split_data(5) # Performs a 5-Fold split for 5-Fold cross-validation
    splits = splitter.generate_splits() # Returns a list of 5 splitter objects, one for each fold
    """

    def _assign_folds(self, n_splits):
        """
        Assigns fold labels to the data for K-Fold cross-validation.
        
        Args:
        - n_splits (int): Number of folds.
        """
    
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Shuffle the DataFrame indices
        shuffled_indices = np.random.permutation(self.data.index)
        
        # Determine the number of points per fold
        fold_sizes = [len(shuffled_indices) // n_splits] * n_splits
        for i in range(len(shuffled_indices) % n_splits):
            fold_sizes[i] += 1
        
        # Assign fold labels
        groups = np.zeros(len(self.data), dtype=int)
        current_idx = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            for i in range(current_idx, current_idx + fold_size):
                groups[shuffled_indices[i]] = fold_idx
            current_idx += fold_size
        
        # Add 'group' column to the DataFrame
        self.data['fold'] = groups

    def _split_data(self, fold_number):
        """
        Splits data into training and test sets based on the specified fold number.
        
        Args:
        - fold_number (int): Fold number to include in the test set.
        """

        self.data['group'] = np.where(
            self.data['fold'] == fold_number, 1, 0
        )

        # Save splits
        self.split_type = f'kfold-{fold_number}'
        self._save_splits()

    def generate_splits(self, n_splits):
      
      self._assign_folds(n_splits=n_splits)

      splits = []
      for fold_num in range(n_splits):
          self._split_data(fold_number=fold_num)
          # Create a deep copy of self and add it to splits
          split_copy = copy.deepcopy(self)
          splits.append(split_copy)
      
      return splits

class RoundProteinSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets based on round number.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.

    Example Usage:

    splitter = RoundProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    splitter.split_data(1, 3) # Splits data such that all data from round 1 and 2 are in the training set, and all data from round 3 and above are in the test set
    """

    def split_data(self, max_train_round, min_test_round, iter=None, k_folds=None):
        """
        Splits data into training and test sets based on round number.
        
        Args:
        - max_train_round (int): Maximum round number to include in the training set.
        - min_test_round (int): Minimum round number to include in the test set.
        - iter (int, optional): Iteration number for naming the split file.
        - k_folds (int, optional): Number of folds to generate.
        """
        assert 'round' in self.data.columns, "DataFrame must contain a 'round' column"
        assert max_train_round < min_test_round, "Maximum training round must be less than minimum test round"

        self.data['group'] = np.where(
            self.data['round'] <= max_train_round, 0,
            np.where(
                self.data['round'] >= min_test_round, 1, np.nan
            )
        )

        # Save splits
        self.split_type = f'round-{max_train_round}-{min_test_round}'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return

        else:
            
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)

class RandomProteinSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets randomly.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.

    Example Usage:

    splitter = RandomProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    splitter.split_data(test_size=0.2) # Splits data into 80% training and 20% test sets
    """

    def split_data(self, test_size=0.2, iter=None, k_folds=None):
        """
        Splits data into training and test sets randomly.
        
        Args:
        - test_size (float): Fraction of data to partition into the test set.
        - iter (int, optional): Iteration number for naming the split file.
        - k_folds (int, optional): Number of folds to generate.
        """      
        random.seed(self.random_state)  # Add this line
        test_len = int(test_size * len(self.data))
        test_indices = random.sample(range(len(self.data)), test_len)
        self.data['group'] = 0
        self.data.loc[test_indices, 'group'] = 1
        
        # Save splits
        self.split_type = 'random'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return

        else:
            
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)

class PositionProteinSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets based on mutation positions.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.
    
    Example Usage:

    splitter = PositionProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    # Splits data into training and test sets based on mutation positions:
    # 1. Randomly samples test_size_sample fraction of variants to get mutation positions to exclude
    # 2. Any variant containing those positions goes into test set
    # 3. Repeats sampling up to iter times until test set size is between test_size_min and test_size_max
    # 4. If test set size requirements not met after iter attempts, uses best attempt

    splitter.split_data(test_size_sample=0.2, iter=3, test_size_min=0.2, test_size_max=0.3)
    """

    def split_data(self, test_size_sample, sample_iter=3, test_size_min=0.2, test_size_max=0.3, iter=None, k_folds=None):
        """
        Splits the dataset into training and test sets based on mutation positions occupied in each variant.
        
        Args:
        - test_size_sample (float): Fraction to sample to retrieve mutation positions to exclude out of the training set.
        - sample_iter (int): Number of sampling iterations.
        - test_size_min (float): Minimum test size set desired.
        - test_size_max (float): Maximum test size set allowed.
        - iter (int, optional): Iteration number for naming the split file.
        - k_folds (int, optional): Number of folds to generate.
        """

        def are_any_elements_present(row):
            return 1 if any(element in test_positions for element in row['mut_positions']) else 0

        i = 0
        test_size = test_size_sample
        random.seed(self.random_state)
        while not (test_size_min < test_size < test_size_max) and i < sample_iter:
            i += 1
            positions = [sublist for sublist in self.data['mut_positions'].values]
            test_len = int(test_size_sample * len(positions))
            test_positions_ls = random.sample(positions, test_len)
            test_positions = [item for sublist in test_positions_ls for item in sublist]
            self.data['group'] = self.data.apply(are_any_elements_present, axis=1)
            test_size = self.data['group'].sum() / len(self.data)

        if (test_size_min < test_size < test_size_max):
            print(f'Test set size ({round(test_size,2)}) passes the recommended requirements (i.e. between {test_size_min} and {test_size_max}).')
        elif test_size < test_size_min: 
            print(f'Test set size ({round(test_size,2)}) is lower than the recommended minimum size ({test_size_min}). If necessary, rerun with the same or higher test set sample size.')
        elif test_size > test_size_max:
            print(f'Test set size ({round(test_size,2)}) is higher than the recommended maximum size ({test_size_max}). If necessary, rerun with the same or lower test set sample size.')

        # Save splits
        self.split_type = 'position'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return

        else:
            
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)
    
    
class RegionProteinSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets based on mutation positions.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.

    Example Usage:

    splitter = RegionProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    splitter.split_data(region=[1, 60]) # Splits data such that all variants containing mutations in the first 60 positions are in the test set
    """

    def split_data(self, region, iter=None, k_folds=None):
        """
        Exclude a region or domain of a protein into the test set, the remaining regions are placed into the test set.
        
        Args:
        - region (list): Provided as a 2-number list defining the boundaries of the region to exclude (e.g. [1, 60]).
        - iter (int, optional): Iteration number for naming the split file.
        - k_folds (int, optional): Number of folds to generate.
        """

        region_i = region[0]
        region_f = region[1]

        def are_any_mutations_present(row):
            return 1 if any(element in range(region_i, region_f + 1, 1) for element in row['mut_positions']) else 0

        self.data['group'] = self.data.apply(are_any_mutations_present, axis=1)
        
        # Save splits
        self.split_type = f'region_{region[0]}-{region[1]}'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return

        else:
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)
    
class PropertyProteinSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets by value.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.

    Example Usage:

    splitter = PropertyProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    splitter.split_data(
        property=0.5,      # Value to split on (e.g. 0.5 for median split)
        above_or_below='above'  # 'above': variants with y > property in test set
                               # 'below': variants with y < property in test set
    )
    """
    
    def split_data(self, property, above_or_below, iter=None, k_folds=None):
        """
        Splits data by the property represented by the y values.
        
        Args:
        - property (float): Value of property to split on.
        - above_or_below (str): 'above' or 'below', values to leave out into the test set based on the given property value.
        - iter (int, optional): Iteration number for naming the split file.
        - k_folds (int, optional): Number of folds to generate.
        """

        if above_or_below == 'above':
            self.data['group'] = np.where(self.data[1] > property, 1, 0)
        
        elif above_or_below == 'below':
            self.data['group'] = np.where(self.data[1] < property, 1, 0)

        # Save splits
        self.split_type = f'y_{above_or_below}_{property}'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return
    
        else:
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)

class MutLoadProteinSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets by mutational load.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.

    Example Usage:

    splitter = MutLoadProteinSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None        # Fraction of data for validation (None=no validation split)
                                    )
    splitter.split_data(
        max_train_muts=2,      # Maximum number of mutations to include in training set
        min_test_muts=5        # Minimum number of mutations to include in test set
    )
    """
    
    def split_data(self, max_train_muts, min_test_muts, iter=None, k_folds=None):
        """
        Splits data into training and test sets based on mutational load.
        
        Args:
        - max_train_muts (int): Maximum mutational load to include in the training set.
        - min_test_muts (int): Minimum mutational load to include in the test set.
        - iter (int, optional): Iteration number for naming the split file.
        - k_folds (int, optional): Number of folds to generate.
        """
        assert 'mut_load' in self.data.columns, "DataFrame must contain a 'mut_load' column"
        assert max_train_muts < min_test_muts, "Maximum training mutational load must be less than minimum test mutational load"

        self.data['group'] = np.where(
            self.data['mut_load'] <= max_train_muts, 0,
            np.where(
                self.data['mut_load'] >= min_test_muts, 1, np.nan
            )
        )

        # Save splits
        self.split_type = f'muts-{max_train_muts}-{min_test_muts}'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return

        else:
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)

class ResidueDistanceSplitter(ProteinSplitter):
    """
    Class for splitting protein datasets based on residue distances in 3D structure.

    Attributes:
        data (pd.DataFrame): The protein dataset.
        random_state (int): Random seed for reproducibility.
        splits (dict): Dictionary to store the data splits.
        split_type (str): Type of split being performed.
        file_attrs (dict): Dictionary containing file attributes.
        use_cache (bool): Whether to use cached splits.
        y_scaling (str): Method for scaling y values.
        val_split (float): Fraction of data to use for validation.
        pdb_file (str): Path to PDB/CIF structure file.
        chain_ids (list): List of chain IDs to analyze.
        dist_dict (dict): Dictionary mapping mutation pairs to distances.

    Example Usage:

    splitter = ResidueDistanceSplitter(training_dataset_fname, 
                                    wt_file, 
                                    csv_has_header=True,  # Whether input CSV has a header row
                                    use_cache=True,       # Whether to cache results to disk
                                    y_scaling=True,       # Whether to scale y values to [0,1]
                                    val_split=None,       # Fraction of data for validation (None=no validation split)
                                    pdb_file='1abc.pdb', # Path to structure file
                                    chain_ids=['A','B']  # Chain IDs to analyze
                                    )
    splitter.split_data(
        percentile_threshold=50,  # Distance percentile threshold for training set
        min_test_muts=5,         # Minimum mutations for test set
        max_train_muts=2         # Maximum mutations for training set
    )
    """

    def __init__(self, protein_name, data, wt_file, csv_has_header=False, use_cache=False, 
                 y_scaling=False, 
                 val_split=None,
                 random_state=42,
                 pdb_file=None,
                 chain_ids=None,
                 **kwargs):
        """
        Args:
            data (str or pd.DataFrame): Input data containing sequences and labels.
            wt_file (str or list): Path(s) to wild-type sequence file(s).
            csv_has_header (bool): Whether input CSV has header.
            use_cache (bool): Whether to cache results.
            y_scaling (bool): Whether to scale y values.
            val_split (float): Fraction of data for validation.
            random_state (int): Random seed.
            pdb_file (str): Path to PDB/CIF structure file.
            chain_ids (list): List of chain IDs to analyze.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(protein_name, data, wt_file, csv_has_header=csv_has_header, use_cache=use_cache, 
                         random_state=random_state,
                         y_scaling=y_scaling, 
                         val_split=val_split,
                         **kwargs)
        
        self.pdb_file = pdb_file
        self.chain_ids = chain_ids

    def _calculate_ca_distances(self):
        """
        Calculate pairwise distances between all alpha carbons in a protein structure.
        
        Calculates distances between CA atoms and stores in self.dist_dict mapping
        mutation pairs to their 3D distance in Angstroms.
        """
        # Initialize PDB parser
        if self.pdb_file.endswith(".pdb"):
            parser = PDB.PDBParser(QUIET=True)
        elif self.pdb_file.endswith(".cif"):
            parser = PDB.MMCIFParser(QUIET=True)
        else:
            raise ValueError("Invalid file type. Please provide a PDB or CIF file.")
        
        structure = parser.get_structure('protein', self.pdb_file)
        
        # Get all alpha carbons
        ca_atoms = []
        residue_info = []
        
        for model in structure:
            for chain in model:
                if chain.id in self.chain_ids:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_atoms.append(residue['CA'])
                            residue_info.append((
                                chain.id,
                                residue.get_id()[1],  # residue number
                                residue.get_resname()  # residue name
                            ))
        
        # Calculate distance matrix
        n_residues = len(ca_atoms)
        distance_matrix = np.zeros((n_residues, n_residues))
        
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                distance = ca_atoms[i] - ca_atoms[j]  # Returns distance in Angstroms
                distance_matrix[i,j] = distance
                distance_matrix[j,i] = distance
        
        dist_dict = {}
        
        for i in range(len(residue_info)):
            for j in range(i+1, len(residue_info)):  # Only upper triangle to avoid duplicates
                chain_i, resnum_i, resname_i = residue_info[i]
                chain_j, resnum_j, resname_j = residue_info[j]
                resname_i = aa_dict_3to1[resname_i]
                resname_j = aa_dict_3to1[resname_j]
                
                # Calculate adjusted residue numbers based on chain index
                resnum_i_adj = resnum_i + self.wt_seq_lens[self.chain_ids.index(chain_i)-1] if self.chain_ids.index(chain_i) != 0 else resnum_i
                resnum_j_adj = resnum_j + self.wt_seq_lens[self.chain_ids.index(chain_j)-1] if self.chain_ids.index(chain_j) != 0 else resnum_j
                
                # Create key string with adjusted residue numbers
                key = f'{resname_i}{resnum_i_adj}_{resname_j}{resnum_j_adj}'
                
                # Store distance in dictionary
                dist_dict[key] = distance_matrix[i,j]

        if self.randomized_control:
            # modify dist_dict to be randomized
            # Set random seed before shuffling
            random.seed(self.random_state)
            # Get all values and shuffle them
            values = list(dist_dict.values())
            random.shuffle(values)
            # Reassign shuffled values to the same keys
            dist_dict = dict(zip(dist_dict.keys(), values))
            
        self.dist_dict = dist_dict

        self.data['dist'] = self.data['muts'].apply(lambda x: self._get_dist(x.split('/')))

        self._get_dist_percentile()

    def _get_dist(self, muts):
        """
        Get sum of pairwise distances between mutations.
        
        Args:
            muts (list): List of mutation strings in format 'A123B'.
            
        Returns:
            float: Sum of pairwise distances between mutations.
        """
        distances = []
        for i in range(len(muts)):
            for j in range(i+1, len(muts)):
                mut_pair = f"{muts[i][:-1]}_{muts[j][:-1]}"
                if mut_pair in self.dist_dict:
                    distances.append(self.dist_dict[mut_pair])
        return sum(distances)
    
    def _get_dist_percentile(self):
        """
        Calculate distance percentile for each variant within its mutational load group.
        Updates self.data with 'dist_percentile' column.
        """
        for mut_load in self.data['mut_load'].unique():
            subset = self.data[self.data['mut_load'] == mut_load].copy()
            if mut_load == 0 or mut_load == 1:
                subset['dist_percentile'] = 0
            else:
                subset['dist_percentile'] = subset['dist'].rank(pct=True) *100
            self.data.loc[self.data['mut_load'] == mut_load, 'dist_percentile'] = subset['dist_percentile']

    def split_data(self, percentile_threshold, min_test_muts, max_train_muts, randomized_control=False, iter=None, k_folds=None):
        """
        Split data based on mutation distances and counts.
        
        Args:
            percentile_threshold (float): Maximum distance percentile for training set.
            min_test_muts (int): Minimum mutations for test set.
            max_train_muts (int): Maximum mutations for training set.
            randomized_control (bool): Whether to randomize the distance dictionary.
            iter (int, optional): Iteration number for naming the split file.
            k_folds (int, optional): Number of folds to generate.
        """
        assert 'mut_load' in self.data.columns, "DataFrame must contain a 'mut_load' column"
        assert max_train_muts < min_test_muts, "Maximum training mutational load must be less than minimum test mutational load"

        self.randomized_control = randomized_control
        self._calculate_ca_distances()

        self.data['group'] = np.where(
            (self.data['mut_load'] <= max_train_muts) & (self.data['dist_percentile'] <= percentile_threshold), 0,
            np.where(
                self.data['mut_load'] >= min_test_muts, 1, np.nan
            )
        )

        # Save splits
        self.split_type = f'dist-p{percentile_threshold}-{max_train_muts}-{min_test_muts}{"-randomized-" + str(self.random_state) if self.randomized_control else ""}'

        if k_folds is not None:
            if self.val_split is not None:
                self._save_folds(k_folds=k_folds)
            elif self.val_split is None:
                print("No validation split, cannot generate kfolds")
                return
        else:
            if iter is None:
                self._save_splits()
            else:
                self._save_splits(iter=iter)

