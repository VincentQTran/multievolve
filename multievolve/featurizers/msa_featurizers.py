from Bio import SeqIO
import numpy as np
import torch

from multievolve.featurizers.base_featurizers import BaseFeaturizer
from multievolve.featurizers.model_choices import FEATURE_MODELS
from multievolve.utils.other_utils import read_msa, greedy_select, msa_splicer


class MSABaseFeaturizer(BaseFeaturizer):
    """Base class for MSA-based featurizers.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to model files.
        msa_file (str): Path to MSA file.

    Example Usage:

    featurizer = MSABaseFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        msa_file='msa.fasta',    # Path to MSA file
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        msa_file=None,
        batch_size=968,
        model_type="msa",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            msa_file (str): Path to MSA file.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model_type, protein, use_cache, **kwargs)
        self.batch_size = batch_size
        self.model_locations = model_locations
        self.msa_file = msa_file
        self.device = torch.device("cpu")  # MSAs might be too big for GPU

        

    def featurize_msa(self, seqs, msa_file, output_type, **kwargs):
        """
        Featurizes sequences using MSA Transformer model.

        Args:
            seqs (list): List of sequences to featurize.
            msa_file (str): Path to MSA file.
            output_type (str): Type of output features to extract.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of sequence features.
        """
        from esm import pretrained
        
        torch.set_grad_enabled(False)
        # Check to see if there is an MSA file in **kwargs.
        if msa_file is None:
            assert self.msa_file is not None, "No MSA file provided."
            msa_file = self.msa_file
        assert output_type in [
            "sequence_representations",
            "msa_representations",
            "log_probabilities",
        ]
        msa = read_msa(msa_file)

        # Instantiate the model
        (
            msa_transformer,
            msa_transformer_alphabet,
        ) = pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer = msa_transformer.eval().to(self.device)
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

        # Prep the MSA, making the appropriate mutations
        inputs = greedy_select(
            msa, num_seqs=128
        )  # can change this to pass more/fewer sequences
        # This splices the MSA to exclude gaps in the first sequence, due to MSATransformer context window
        # size limit of 1024. If your MSA width is less than 1024, then you don't need to do this
        inputs = msa_splicer(inputs)
        name, wt_seq = inputs[0][0], inputs[0][1]
        reps = []

        # Batch processing
        batch_size = self.batch_size
        num_batches = len(seqs) // batch_size + (len(seqs) % batch_size != 0)
        for batch_i in range(num_batches):  # TODO: change this to num_batches
            start_idx = batch_i * batch_size
            end_idx = start_idx + batch_size
            batch_seqs = seqs[start_idx:end_idx]

            input_msas = []
            for seq in batch_seqs:
                # Replace the first sequence in the MSA with the mutant sequence
                assert len(wt_seq) == len(seq)
                inputs[0] = (name, seq)
                input_msas.append(inputs)
            # print(len(input_msas), len(input_msas[0]), len(input_msas[0][0][1]), len(input_msas[0][1][1]))
            # Run the MSA Transformer
            (
                msa_transformer_batch_labels,
                msa_transformer_batch_strs,
                msa_transformer_batch_tokens,
            ) = msa_transformer_batch_converter(input_msas)
            num_msas = len(msa_transformer_batch_tokens)
            msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(
                next(msa_transformer.parameters()).device
            )
            msa_transformer_predictions = msa_transformer.forward(
                msa_transformer_batch_tokens, repr_layers=[12]
            )

            # Extract features
            msa_reps = (
                msa_transformer_predictions["representations"][12]
                .detach()
                .cpu()
                .numpy()
            )
            for i in range(num_msas):
                logits = (
                    msa_transformer_predictions["logits"][i][0].detach().cpu().numpy()
                )
                avg_msa_rep = msa_reps[i].mean((0, 1))
                sequence_rep = msa_reps[i][0].mean(0)

                if output_type == "msa_representations":
                    reps.append(avg_msa_rep)

                elif output_type == "sequence_representations":
                    reps.append(sequence_rep)

                elif output_type == "log_probabilities":
                    reps.append(logits)

        X = np.array(reps)
    
        return X


class MSAEmbedFeaturizer(MSABaseFeaturizer):
    """Class for generating MSA embedding features.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to model files.
        msa_file (str): Path to MSA file.

    Example Usage:

    featurizer = MSAEmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        msa_file='msa.fasta',    # Path to MSA file
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        msa_file=None,
        model_locations=FEATURE_MODELS["msa_embed"],
        batch_size=968,
        model_type="msa_embed",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            msa_file (str): Path to MSA file.
            model_locations (list): Paths to model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            msa_file,
            batch_size,
            model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, msa_file=None, **kwargs):
        """
        Featurizes sequences using MSA embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            msa_file (str): Path to MSA file.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of MSA embedding features.
        """
        X = self.featurize_msa(
            seqs, msa_file, output_type="msa_representations", **kwargs
        )
        return X


class MSASequenceEmbedFeaturizer(MSABaseFeaturizer):
    """Class for generating MSA sequence embedding features.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to model files.
        msa_file (str): Path to MSA file.

    Example Usage:

    featurizer = MSASequenceEmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        msa_file='msa.fasta',    # Path to MSA file
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        msa_file=None,
        model_locations=FEATURE_MODELS["msa_sequence_embed"],
        batch_size=968,
        model_type="msa_sequence_embed",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            msa_file (str): Path to MSA file.
            model_locations (list): Paths to model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            msa_file,
            batch_size,
            model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, msa_file=None, **kwargs):
        """
        Featurizes sequences using MSA sequence embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            msa_file (str): Path to MSA file.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of MSA sequence embedding features.
        """
        X = self.featurize_msa(
            seqs, msa_file, output_type="sequence_representations", **kwargs
        )
        return X


class MSALogitsFeaturizer(MSABaseFeaturizer):
    """Class for generating MSA logits features.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to model files.
        msa_file (str): Path to MSA file.

    Example Usage:

    featurizer = MSALogitsFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        msa_file='msa.fasta',    # Path to MSA file
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        msa_file=None,
        model_locations=FEATURE_MODELS["msa_logits"],
        batch_size=968,
        model_type="msa_logits",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            msa_file (str): Path to MSA file.
            model_locations (list): Paths to model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            msa_file,
            batch_size,
            model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, msa_file=None, **kwargs):
        """
        Featurizes sequences using MSA logits.

        Args:
            seqs (list): List of sequences to featurize.
            msa_file (str): Path to MSA file.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of MSA logits features.
        """
        X = self.featurize_msa(
            seqs, msa_file, output_type="log_probabilities", **kwargs
        )
        return X
