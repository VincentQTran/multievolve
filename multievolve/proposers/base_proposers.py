import copy
import os
from joblib import Parallel, delayed
import random
from typing import List, Tuple

from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from itertools import combinations, product

from multievolve.predictors import BaseRegressor, GPRegressor
from multievolve.utils.data_utils import MutationFormat, MutationListFormats, levenshtein_distance_matrix
from multievolve.utils.other_utils import deep_mutational_scan, wt_only_mutational_pool_to_dict, mutational_pool_to_dict, mut_pool_searcher

# Definitions:
# Mutant = single substitution, deletion, or insertion
# Variant = single sequence with multiple mutations

#########################################
# Unsupervised proposers: These propose new sequences without using previous data.


class BaseProposer:
    """
    Base class for proposing mutations on protein sequences.

    Attributes:
        start_seq (str): The starting protein sequence.
        models (list): List of models for evaluating proposed mutants.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
        mutation_pool (list): List of possible mutations to propose from.
        experiment_name (str): Name of the experiment run.
        proposals (pd.DataFrame): DataFrame to store proposed mutations.
    """

    def __init__(
        self,
        start_seq,
        models=None,
        trust_radius=None,
        num_seeds=None,
        mutation_pool=None,
        experiment_name="base_proposer_run"
    ):
        """
        Initialize the BaseProposer.

        Args:
            start_seq (str): Starting protein sequence.
            experiment_name (str): Name of the experiment run.
            models (list): List of models for evaluating proposed mutants.
            trust_radius (int): Maximum number of mutations allowed in a variant.
            num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
            mutation_pool (list): List of possible mutations to propose from. If None, generates a full deep mutational scan.
        """
        self.start_seq = start_seq
        self.experiment_name = experiment_name
        self.models = models
        self.trust_radius = trust_radius
        self.num_seeds = num_seeds
        self.proposals = None

        if mutation_pool is None:
            # Generate default mutation pool if not provided
            mutation_pool = list(deep_mutational_scan(self.start_seq))
            mutation_pool = [mut for mut in mutation_pool if mut[2] != "*"]
            mutation_pool = [f"{mut[1]}{mut[0] + 1}{mut[2]}" for mut in mutation_pool]
            self.mutation_pool = mutation_pool
        elif mutation_pool is not None:
            # Convert provided mutation pool to standard format
            mutation_lists = MutationListFormats(mutation_pool, self.start_seq)
            self.mutation_pool = mutation_lists.get_mutation_pool()

    def get_proposals(self):
        """
        Retrieve the current proposals.

        Returns:
            pd.DataFrame: Current proposals.
        """
        return self.proposals

    def save_proposals(self, filename):
        """
        Save the current proposals to a CSV file.

        Args:
            filename (str): Name of the file to save proposals.

        Raises:
            ValueError: If no proposals have been made.
        """
        if self.proposals is None:
            raise ValueError("No proposals have been made.")
        else:
            dir_path = os.path.join(self.models[0].file_attrs["dataset_dir"], "proposers/results/")
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            self.proposals.to_csv(
                os.path.join(dir_path, f"{filename}.csv"), index=False
            )

    def proposal_list_to_dataframe(self, muts) -> pd.DataFrame:
        """
        Convert a list of mutations to a DataFrame.

        Args:
            muts (list): List of mutation lists.

        Returns:
            pd.DataFrame: DataFrame containing mutations, full sequences, and mutation strings.
        """
        proposals = pd.DataFrame({'Mutations': muts})
        proposals["Full_Sequence"] = proposals.apply(lambda row: MutationFormat(row['Mutations'], self.start_seq).to_full_sequence(), axis=1)
        proposals['Mut_string'] = proposals.apply(lambda row: MutationFormat(row['Full_Sequence'], self.start_seq).to_mutation_string(), axis=1)
        
        return proposals

    def propose(self) -> pd.DataFrame:
        """
        Generate and return proposals.

        Returns:
            pd.DataFrame: DataFrame of proposal mutations and full sequences.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluate_proposals(self) -> pd.DataFrame:
        """
        Evaluate the generated proposals with the supplied models.

        Returns:
            pd.DataFrame: DataFrame with evaluation results added.

        Raises:
            ValueError: If no model is available or no proposals have been made.
        """
        if self.models is None:
            raise ValueError("No model to evaluate.")
        if self.proposals is None:
            raise ValueError("No proposals have been made.")

        for model in tqdm(self.models):
            print(f"Evaluating proposals with model: {model.file_attrs['model_name']}")
            self.proposals[model.file_attrs['model_name']] = model.predict(self.proposals["Full_Sequence"])
        
        # Calculate average across all models
        self.proposals['average'] = self.proposals.iloc[:,-len(self.models):].mean(axis=1)


    def get_variables(self) -> dict:
        """
        Retrieve all instance variables.

        Returns:
            dict: Dictionary of all instance variables.
        """
        return self.__dict__


class AlanineScanningProposer(BaseProposer):
    """
    Proposer that replaces every position with an alanine.

    Attributes:
        start_seq (str): The starting protein sequence.
        models (list): List of models for evaluating proposed mutants.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
        mutation_pool (list): List of possible mutations to propose from.
        experiment_name (str): Name of the experiment run.
        proposals (pd.DataFrame): DataFrame to store proposed mutations.

    Example Usage:

        # Ignore: num_seeds, experiment_name, trust_radius, mutation_pool
        proposer = AlanineScanningProposer(
            start_seq="MKTSTGNFKIVILMGVNRRMKTSTGNFKI",
            models=[model1, model2],  # List of trained models
        )

        # Generate and get proposals
        proposer.propose()

        # Predict activity of proposed mutants
        proposer.evaluate_proposals()

        # Save proposals to file
        proposer.save_proposals("alanine_scanning_proposals")
    """

    def propose(self) -> pd.DataFrame:
        """
        Generate proposals by replacing each position with alanine.

        Returns:
            pd.DataFrame: DataFrame of alanine scanning proposals.
        """
        muts = []
        for i, wt in enumerate(self.start_seq):
            muts.append(f"{wt}{i+1}A")

        self.proposals = self.proposal_list_to_dataframe(muts)

        return self.proposals


class DeepMutationalScanningProposer(BaseProposer):
    """
    Proposer that generates every possible single amino acid substitution.

    Attributes:
        start_seq (str): The starting protein sequence.
        models (list): List of models for evaluating proposed mutants.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
        mutation_pool (list): List of possible mutations to propose from.
        experiment_name (str): Name of the experiment run.
        proposals (pd.DataFrame): DataFrame to store proposed mutations.

    Example Usage:

        # Ignore: num_seeds, experiment_name, trust_radius, mutation_pool
        proposer = DeepMutationalScanningProposer(
            start_seq="MKTSTGNFKIVILMGVNRRMKTSTGNFKI",
            models=[model1, model2],  # List of trained models
        )

        # Generate and get proposals
        proposer.propose()

        # Predict activity of proposed mutants
        proposer.evaluate_proposals()

        # Save proposals to file
        proposer.save_proposals("deep_mutational_scanning_proposals")
    """

    def propose(self) -> pd.DataFrame:
        """
        Generate proposals for every possible substitution.

        Returns:
            pd.DataFrame: DataFrame containing all possible single amino acid substitutions.
        """
        muts = list(deep_mutational_scan(self.start_seq))
        # Filter out stop codons, which are represented by *
        muts = [mut for mut in muts if mut[2] != "*"]
        muts = [f"{mut[1]}{mut[0] + 1}{mut[2]}" for mut in muts]

        self.proposals = self.proposal_list_to_dataframe(muts)

        return self.proposals


class RandomMutagenesisProposer(BaseProposer):
    """
    Proposer that generates random variants with a specified number of mutations.

    Attributes:
        start_seq (str): The starting protein sequence.
        models (list): List of models for evaluating proposed mutants.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed. -1 for all combinations.
        mutation_pool (list): List of possible mutations to propose from.
        experiment_name (str): Name of the experiment run.
        proposals (pd.DataFrame): DataFrame to store proposed mutations.

    Example Usage:
    
        # Initialize proposer with sequence and parameters
        # Ignore: experiment_name
        proposer = RandomMutagenesisProposer(
            start_seq="MKTSTGNFKIVILMGVNRRMKTSTGNFKI",
            models=[model1, model2],  # List of trained models
            trust_radius=2,  # Maximum 2 mutations per variant
            num_seeds=-1,  # -1 for all combinations.
            mutation_pool=["A1G", "D2E", "K3R"],  # Allowed mutations
        )

        # Generate and get proposals
        proposer.propose()

        # Predict activity of proposed mutants
        proposer.evaluate_proposals()

        # Save proposals to file
        proposer.save_proposals("random_mutagenesis_proposals")
    """

    def propose(self) -> pd.DataFrame:
        """
        Generate random proposals with the specified number of mutations.

        Returns:
            pd.DataFrame: DataFrame of random mutagenesis proposals.
        """
        mutation_pool = self.mutation_pool

        # Generate variants based on trust_radius
        if self.num_seeds == -1:
            muts = combinations(mutation_pool, self.trust_radius)
        else:
            muts = [
                random.sample(mutation_pool, self.trust_radius)
                for _ in range(self.num_seeds)
            ]

        self.proposals = self.proposal_list_to_dataframe(muts)

        return self.proposals


class CombinatorialProposer(BaseProposer):
    """
    Proposer that generates combinatorial proposals by combining mutations.

    Attributes:
        start_seq (str): The starting protein sequence.
        models (list): List of models for evaluating proposed mutants.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed, -1 for all combinations.
        mutation_pool (list): List of possible mutations to propose from.
        experiment_name (str): Name of the experiment run.
        proposals (pd.DataFrame): DataFrame to store proposed mutations.

    Example Usage:

        # Initialize proposer with sequence and parameters
        # Ignore: experiment_name
        proposer = CombinatorialProposer(
            start_seq="MKTSTGNFKIVILMGVNRRMKTSTGNFKI",
            models=[model1, model2],  # List of trained models
            trust_radius=2,  # Maximum 2 mutations per variant
            num_seeds=-1,  # -1 for all combinations.
            mutation_pool=["A1G", "D2E", "K3R"],  # Allowed mutations
        )

        # Generate and get proposals
        proposer.propose()

        # Predict activity of proposed mutants
        proposer.evaluate_proposals()

        # Save proposals to file
        proposer.save_proposals("combinatorial_proposals")
    """

    def propose(self, output_df=True) -> pd.DataFrame:
        """
        Generate combinatorial proposals by combining mutations.

        Returns:
            pd.DataFrame: DataFrame of combinatorial proposals, including the number of mutations for each proposal.
        """

        # Function to generate all possible combinations of mutations
        def generate_permutations(mutations, num_positions):
            positions = list(mutations.keys())
            all_combinations_ls = []
            
            # Get all combinations of the given number of positions
            for combo in combinations(positions, num_positions):
                # Generate all permutations for the selected combination of positions
                perms_ls = [permutation for permutation in product(*(mutations[pos] for pos in combo))]
                all_combinations_ls.extend(perms_ls)

            return all_combinations_ls

        # Initialize an empty dictionary to store the mutations
        mutations_dict = {}

        # Iterate over the list and populate the dictionary
        for mutation in self.mutation_pool:
            # Extract the position and mutation from the string
            position = int(''.join(filter(str.isdigit, mutation)))
            
            # If the position is not in the dictionary, add it with an empty list
            if position not in mutations_dict:
                mutations_dict[position] = []
            
            # Append the mutation to the list at the current position
            mutations_dict[position].append(mutation)

        muts = []
        if self.num_seeds == -1:
            # Create combinations for all sizes from 2 up to trust_radius
            for r in range(2, self.trust_radius + 1):
                muts.extend(generate_permutations(mutations_dict, r))
        else:
            # Randomly sample num_seeds combinations for each size from 2 up to trust_radius
            for r in range(2, self.trust_radius + 1):
                muts.extend(random.sample(generate_permutations(mutations_dict, r), self.num_seeds))

        self.proposals = self.proposal_list_to_dataframe(muts)
        self.proposals['num_muts'] = [len(mut) for mut in muts]

        if output_df:
            return self.proposals


#########################################
# Model-guided proposers: These proposer mechanisms iteratively propose mutations with guidance from a model.
# Examples include MCMC, simulated annealing, ICE, etc.


class ModelGuidedProposer(BaseProposer):
    """
    A model-guided proposer that iteratively proposes mutations with guidance from a model.
    Evaluations by the model are made throughout the proposal process.
    Models can be data-driven or pre-trained.

    Attributes:
        start_seq (str): The starting protein sequence.
        models (list): List of models for final evaluation of proposals.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
        mutation_pool (list): List of possible mutations to propose from.
        proposals (pd.DataFrame): DataFrame of proposed mutations and their evaluations.
        guiding_model: The model used to evaluate variants during the proposal process.
        experiment_name (str): Name of the experiment run.
    """

    def __init__(
        self,
        guiding_model,
        start_seq,
        experiment_name="model_guided_run",
        models=None,
        trust_radius=1,
        num_seeds=10,
        mutation_pool=None,
    ):
        """
        Initialize the ModelGuidedProposer.

        Args:
            guiding_model: Model of class Regressor used to evaluate variants during the proposal process.
            start_seq (str): Starting protein sequence.
            experiment_name (str): Name of the experiment run.
            models (list): List of models for final evaluation of proposals.
            trust_radius (int): Maximum number of mutations allowed in a variant.
            num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
            mutation_pool (list): List of possible mutations to propose from. If None, generates a full deep mutational scan including WT amino acids.
        """
        if mutation_pool is None:
            # Generate full mutation pool including WT amino acids for iterative proposers
            mutation_pool = list(deep_mutational_scan(start_seq, exclude_noop=False))
            mutation_pool = [mut for mut in mutation_pool if mut[2] != "*"]
            mutation_pool = [f"{mut[1]}{mut[0] + 1}{mut[2]}" for mut in mutation_pool]
            self.mutation_pool = mutation_pool
        elif mutation_pool is not None:
            mutation_lists = MutationListFormats(mutation_pool, start_seq)
            self.mutation_pool = mutation_lists.get_mutation_pool()

        super().__init__(
            start_seq,
            experiment_name,
            [guiding_model],
            trust_radius,
            num_seeds,
            self.mutation_pool,
        )
        self.guiding_model = guiding_model


class SimulatedAnnealingProposer(ModelGuidedProposer):
    """
    Propose new sequences using Simulated Annealing, a flavor of MCMC.

    Attributes:
        start_seq (str): The starting protein sequence.
        experiment_name (str): Name of the experiment run.
        models (list): List of models for final evaluation of proposals.
        trust_radius (int): Maximum number of mutations allowed in a variant.
        num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
        mutation_pool (list): List of possible mutations to propose from.
        proposals (pd.DataFrame): DataFrame of proposed mutations and their evaluations.
        guiding_model: The model used to evaluate variants during the proposal process.
        variant_seeds (list): List of sequences to seed the proposal process.
        avg_muts_per_seq (int): Average number of mutations per sequence.
        trajectories (int): Number of trajectories to run.
        n_iter (int): Number of iterations for the simulated annealing process.
        T_max (float): Maximum temperature for simulated annealing.
        decay_rate (float): Temperature decay rate.
        k (float): Annealing schedule parameter.
        min_mut_pos (int): Minimum position to mutate.
        max_mut_pos (int): Maximum position to mutate.
        use_cache (bool): Whether to use cached fitness values.
        n_jobs (int): Number of parallel jobs to run.
        verbose (int): Verbosity level.
        start_seq_ls (list): Starting sequence as a list.
        wt_mutational_pool_dict (dict): Dictionary of wild-type mutations.
        mutational_pool_dict (dict): Dictionary of all possible mutations.
        acceptance_rate_history (list): History of acceptance rates.
        best_fitness_values (list): History of best fitness values.
        avg_fitness_values (list): History of average fitness values.
        serial_corr_best (list): History of serial correlations for best fitness.
        serial_corr_avg (list): History of serial correlations for average fitness.
    """

    def __init__(
        self,
        guiding_model: BaseRegressor,
        start_seq,
        trust_radius,
        mutation_pool,
        avg_muts_per_seq,
        models=None,
        num_seeds=10,
        experiment_name="simulated_annealing_run",
        variant_seeds=None,
        n_iter=1000,
        T_max=0.01,
        decay_rate=0.99,
        k=1,
        min_mut_pos=0,
        max_mut_pos=1e6,
        use_cache=True,
        n_jobs=1,
        verbose=1,

    ):
        """
        Initialize the SimulatedAnnealingProposer.

        Args:
            guiding_model (BaseRegressor): The model used to evaluate variants during the proposal process.
            start_seq (str): Starting sequence.
            trust_radius (int): Maximum number of mutations allowed per variant.
            mutation_pool (list): Pool of possible mutations.
            avg_muts_per_seq (int): Average number of mutations per sequence.
            models (list): List of models for final evaluation.
            num_seeds (int): Maximum number of sequences or evolutionary trajectories allowed.
            experiment_name (str): Name of the experiment.
            variant_seeds (list): List of sequences to seed the proposal process.
            n_iter (int): Number of iterations for the simulated annealing process.
            T_max (float): Maximum temperature for simulated annealing.
            decay_rate (float): Temperature decay rate.
            k (float): Annealing schedule parameter.
            min_mut_pos (int): Minimum position to mutate.
            max_mut_pos (int): Maximum position to mutate.
            use_cache (bool): Whether to use cached fitness values.
            n_jobs (int): Number of parallel jobs to run.
            verbose (int): Verbosity level.
        """
        # If variant_seeds are provided, then update num_seeds to match
        if variant_seeds:
            num_seeds = len(variant_seeds)
        super().__init__(
            guiding_model,
            start_seq,
            experiment_name,
            [guiding_model],
            trust_radius,
            num_seeds,
            mutation_pool,
        )
        self.variant_seeds = variant_seeds
        self.avg_muts_per_seq = avg_muts_per_seq
        self.trajectories = len(self.variant_seeds)
        self.n_iter = n_iter
        self.T_max = T_max
        self.decay_rate = decay_rate
        self.k = k
        self.min_mut_pos = min_mut_pos
        self.max_mut_pos = max_mut_pos
        self.use_cache = use_cache
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.start_seq_ls = list(self.start_seq)

        # Double check to make sure mutations are within the min + max range
        mutation_pool = self.mutation_pool
       
        filtered_mutation_pool = []
        for mut in mutation_pool:
            idx = int(mut[1:-1]) - 1
            if min_mut_pos <= idx <= max_mut_pos:
                filtered_mutation_pool.append(mut)
        self.mutation_pool = filtered_mutation_pool # mutations should be 1-indexed

        # Create dictionaries from the mutational pool
        self.wt_mutational_pool_dict = wt_only_mutational_pool_to_dict(self.mutation_pool, self.start_seq)
        self.mutational_pool_dict = mutational_pool_to_dict(self.mutation_pool)

    def save_proposals(self):
        """Save the proposals to a CSV file."""
        if self.proposals is None:
            raise ValueError("No proposals have been made.")
        else:
            dir_path = os.path.join(self.models[0].file_attrs["dataset_dir"], "proposers/results/")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            self.proposals.to_csv(
                os.path.join(dir_path, f"{self.trust_radius}-{self.trajectories}-{self.n_iter}-{self.T_max}-{self.decay_rate}-{self.k}-{self.experiment_name}-proposals.csv"), index=False
            )

    def graph_annealing_history(self, seeds_to_plot=30):
        """
        Generate graphs showing the annealing history.

        Args:
            seeds_to_plot (int): Number of seeds to include in the fitness plot.
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # Create a figure and a 4x1 grid of subplots

        # Plot histories
        metrics = [
            (self.acceptance_rate_history, 'Acceptance Rate'),
            (self.best_fitness_values, 'Best Fitness'),
            (self.avg_fitness_values, 'Average Fitness')
        ]
        for i, (data, title) in enumerate(metrics):
            axs[i].plot(data)
            axs[i].set(title=f'{title} History', xlabel='Iteration', ylabel=title)

        # Plot fitness scores by seed
        sns.set_theme(style="darkgrid")
        for seed in self.proposals['Seed'].unique()[:seeds_to_plot]:
            subset = self.proposals[self.proposals['Seed'] == seed]
            axs[3].plot(subset['Iteration'], subset['Fitness'], label=f'Seed {seed}', marker='o')
        axs[3].set(xlabel='Iteration', ylabel='Fitness', title='Fitness Scores by Seed Over Time')

        plt.tight_layout()  # Adjust layout to not overlap
        
        dir_path = os.path.join(self.models[0].file_attrs["dataset_dir"], "proposers/graphs/")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        plt.savefig(os.path.join(dir_path, f"{self.trust_radius}-{self.trajectories}-{self.n_iter}-{self.T_max}-{self.decay_rate}-{self.k}-{self.experiment_name}-graphs.jpg"), dpi=600)

    def __acceptance_prob(self, f_proposal, f_current, i):
        """
        Calculate the acceptance probability for the simulated annealing process.

        Args:
            f_proposal (float): Fitness of the proposed sequence.
            f_current (float): Fitness of the current sequence.
            i (int): Current iteration.

        Returns:
            float: Acceptance probability.
        """
        current_temperature = self.T_max * self.decay_rate**i
        ap = np.exp((f_proposal - f_current) / (self.k * current_temperature))
        ap[ap > 1] = 1
        return ap

    def __make_n_mutations(self, seq, n_edits, seq_mut_positions, backward=False):
        """
        Make n mutations to a given sequence.

        Args:
            seq (str): The sequence to mutate.
            n_edits (int): Number of mutations to make.
            seq_mut_positions (list): Current mutated positions in the sequence.
            backward (bool): Whether to revert mutations or make new ones.

        Returns:
            tuple: Mutated sequence and updated list of mutated positions.
        """
        # Enforce trust radius part 2
        if not backward:
            if len(seq_mut_positions) + n_edits > self.trust_radius:
                n_edits = self.trust_radius - len(seq_mut_positions)
            random_mutations = random.sample(self.mutation_pool, n_edits)
        else:
            random_wt_mutations = []
            random_pos_mutations = []
            # adjust n_edits to be less than the total number of mutations
            if len(seq_mut_positions) - n_edits < 0:
                n_edits = len(seq_mut_positions)
            # randomly determine how many of the total edits will be to revert to wildtype
            n_edits_wt = random.choice(range(0, n_edits+1))
            random_wt_mutations = random.sample(mut_pool_searcher(seq_mut_positions, self.wt_mutational_pool_dict), n_edits_wt)
            if n_edits - n_edits_wt > 0:
                random_pos_mutations = random.sample(mut_pool_searcher(seq_mut_positions, self.mutational_pool_dict), n_edits - n_edits_wt)
            # combine the two lists
            random_mutations = random_wt_mutations + random_pos_mutations

        lseq = list(seq)
        pos_ls = seq_mut_positions.copy()

        for mut in random_mutations: 
            idx = int(mut[1:-1]) - 1 # adjust for 1-index based residue positioning
            lseq[idx] = mut[-1]
            # Remove mutation from position list if mutating back to wildtype
            if mut[-1] == self.start_seq_ls[idx]:
                if (idx+1) in pos_ls:
                    pos_ls.remove(idx+1) # account for 1-index based residue positioning
            else:
                if (idx+1) not in pos_ls:
                    pos_ls.append(idx+1) # account for 1-index based residue positioning

        mutated_seq = ''.join(lseq)

        return mutated_seq, pos_ls

    def __propose_seqs(self, seqs, seqs_mut_positions):
        """
        Propose new sequences based on the current sequences.

        Args:
            seqs (list): Current sequences.
            seqs_mut_positions (list): Current mutated positions for each sequence.

        Returns:
            tuple: Proposed sequences and their mutated positions.
        """
        mu_muts_per_seq = np.array([self.avg_muts_per_seq] * len(seqs))
        n_edits = np.random.poisson(mu_muts_per_seq - 1) + 1

        # Enforce trust radius part 1 (by limiting the mutational pool to sample from)
        nmut = np.array([len(pos_ls) for pos_ls in seqs_mut_positions])
        mut_mask = nmut >= self.trust_radius # if number of mutations is greater than or equal to maximum allowed, then edit walk direction

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.__make_n_mutations)(
                seq,
                n_edits[i],
                seqs_mut_positions[i],
                backward=mut_mask[i]
            )
            for i, seq in enumerate(seqs)
        )

        mseqs, mpositions = zip(*results)

        return mseqs, mpositions
    
    def __get_fitness_fn(self, seqs, seqs_mut_positions):
        """
        Calculate fitness for the given sequences.

        Args:
            seqs (list): Sequences to evaluate.
            seqs_mut_positions (list): Mutated positions for each sequence.

        Returns:
            tuple: Predicted fitness, uncertainties, and original predictions.
        """
        y_pred = self.guiding_model.predict(seqs)
        y_pred_original = y_pred.copy()

        if isinstance(self.guiding_model, GPRegressor):
            uncertainties = self.guiding_model.uncertainties_
        else:
            uncertainties = np.zeros(y_pred.shape)
    
        # Enforce trust radius part 3
        nmut = np.array([len(pos_ls) for pos_ls in seqs_mut_positions])
        mut_mask = nmut > self.trust_radius # if number of mutations is greater than maximum allowed, then prevent accepting the change
        y_pred[mut_mask] = -np.inf

        return y_pred, uncertainties, y_pred_original
    
    def save_best_proposals(self, num_vars_per_mut_dist=6, min_mut_distance=2):
        """
        Save the best proposals, filtered by mutation distance.

        Args:
            num_vars_per_mut_dist (int): Number of variants to save per mutation distance.
            min_mut_distance (int): Minimum mutation distance to consider.
        """
        if self.proposals is None:
            raise ValueError("No proposals have been made.")
        else:
            dir_path = os.path.join(self.models[0].file_attrs["dataset_dir"], "proposers/designs/")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            df_dedup = self.proposals.drop_duplicates('Full_Sequence').copy()
            df_dedup.sort_values(by='Fitness',ascending=False, inplace=True)
            sequences = df_dedup['Full_Sequence'].tolist()

            num_mutations_seed = list(levenshtein_distance_matrix([self.start_seq], sequences).reshape(-1).astype(int))
            df_dedup['dist_from_start_seq'] = num_mutations_seed

            df_ls = []
            for num in list(set(num_mutations_seed)):
                if num >= min_mut_distance:
                    df_current = df_dedup[df_dedup['dist_from_start_seq'] == num].copy()
                    df_current.sort_values(by='Fitness',ascending=False, inplace=True)
                    df_current = df_current[:num_vars_per_mut_dist]
                    df_ls.append(df_current)

            df_top = pd.concat(df_ls)

            df_top.to_csv(
                os.path.join(dir_path, f"{self.trust_radius}-{self.trajectories}-{self.n_iter}-{self.T_max}-{self.decay_rate}-{self.k}-{self.experiment_name}-top_proposals.csv"), index=False
            )


    def propose(self) -> pd.DataFrame:
        # [TODO] implement maximum mutation distance from WT
        """
        1. Start with a set of sequence *seeds*, which can either be user-input or
            generated from the WT sequence and mutation pool.
        2. For each seed, iteratively run simulated annealing.
        2a. Propose a mutation
        2b. Evaluate fitness
        2c. Accept or reject, and update the acceptance probability given the current temperature.
        3. Repeat 2a-2c until the stopping criteria is met.
        4. Return the seeds, the proposed mutations, and the fitness of the proposed mutations in a dataframe.

        Returns:
            pd.DataFrame: DataFrame containing the proposed sequences and their properties.
        """
        # [TODO] implement maximum mutation distance from WT
        if self.verbose:
            print("Initializing")

        # 1. Get seeds. If none are provided, generate single variants from the WT sequence and mutation pool.
        if self.variant_seeds is None:
            if self.verbose:
                print("Generating seeds since none were provided.")
            random_proposer = RandomMutagenesisProposer(
                start_seq=self.start_seq,
                trust_radius=1,
                num_seeds=self.num_seeds,
                mutation_pool=self.mutation_pool,
            )
            random_proposals = random_proposer.propose()
            self.variant_seeds = random_proposals["Full_Sequence"].values

        # 1b. Evaluate fitness of seeds
        state_seqs = copy.deepcopy(self.variant_seeds)
        state_seqs_mut_positions = [[] for _ in range(len(self.variant_seeds))]
        state_fitness, state_fitness_std, __ = self.__get_fitness_fn(self.variant_seeds, state_seqs_mut_positions)
        seq_history = [copy.deepcopy(state_seqs)]
        fitness_history = [copy.deepcopy(state_fitness)]
        fitness_std_history = [copy.deepcopy(state_fitness_std)]

        # Initialize monitoring variables
        self.acceptance_rate_history = []
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.serial_corr_best = []
        self.serial_corr_avg = []

        # 2. Iteratively run simulated annealing
        for i in range(self.n_iter):
            if self.verbose:
                print(f"Iteration: {i}")

            if self.verbose:
                print("\tProposing sequences.")
            proposal_seqs, proposal_seqs_mut_pos = self.__propose_seqs(state_seqs, state_seqs_mut_positions)

            if self.verbose:
                print("\tCalculating predicted fitness.")
            proposal_fitness, proposal_fitness_std, proposal_fitness_original = self.__get_fitness_fn(
                proposal_seqs, proposal_seqs_mut_pos
            )

            if self.verbose:
                print("\tMaking acceptance/rejection decisions.")
            aprob = self.__acceptance_prob(proposal_fitness, state_fitness, i)

            # Monitor simulated annealing
            self.accepted_proposals = 0

            if self.verbose:
                print("\tUpdating state.")
            for j, ap in enumerate(aprob):
                if np.random.rand() < ap:
                    state_seqs[j] = copy.deepcopy(proposal_seqs[j])
                    state_seqs_mut_positions[j] = copy.deepcopy(proposal_seqs_mut_pos[j])
                    state_fitness[j] = copy.deepcopy(proposal_fitness[j])
                    state_fitness_std[j] = copy.deepcopy(proposal_fitness_std[j])
                    self.accepted_proposals += 1
            
            # Update monitoring variables
            acceptance_rate = round(self.accepted_proposals/len(state_seqs)*100, 2)
            self.acceptance_rate_history.append(acceptance_rate)

            self.best_fitness_values.append(proposal_fitness_original.max())
            self.avg_fitness_values.append(proposal_fitness_original.mean())
            if i > 0:
                best_fitness_array = np.array(self.best_fitness_values)
                avg_fitness_array = np.array(self.avg_fitness_values)
                serial_correlation_best = np.corrcoef(best_fitness_array[:-1], best_fitness_array[1:])[0, 1]
                serial_correlation_avg = np.corrcoef(avg_fitness_array[:-1], avg_fitness_array[1:])[0, 1]
                self.serial_corr_best.append(serial_correlation_best)
                self.serial_corr_avg.append(serial_correlation_avg)


            seq_history.append(copy.deepcopy(state_seqs))
            fitness_history.append(copy.deepcopy(state_fitness))
            fitness_std_history.append(copy.deepcopy(state_fitness_std))

            if self.verbose:
                print(f'\tCurrent best fitness: {state_fitness.max()}')
                print(f"Acceptance Rate: {acceptance_rate}")            
                

        # 3. Save results to pd.DataFrame and then csv

        # Label each row with the sequence, seed number, iteration number, fitness score, fitness std, mutation string, and number of mutations
        data = [
                [seq, j, i, fitness_history[i][j][0], fitness_std_history[i][j][0]]
                for i, seqs in enumerate(seq_history)
                for j, seq in enumerate(seqs)
                ]

        df = pd.DataFrame(
            data,
            columns=["Full_Sequence", "Seed", "Iteration", "Fitness", "Fitness_Std"],
        )

        self.proposals = df