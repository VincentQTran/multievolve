from multievolve.featurizers.base_featurizers import *
from multievolve.featurizers.esm_featurizers import *
from multievolve.featurizers.msa_featurizers import *
from multievolve.featurizers.zeroshot_featurizers import *

from multievolve.featurizers.model_choices import FEATURIZE_CHOICES

FEATURIZE_CLASSES = {
    # Dictionary of model names to model classes.
    # Base Featurizers
    "onehot": OneHotFeaturizer,
    "georgiev": GeorgievFeaturizer,
    "aa_idx": AAIdxFeaturizer,
    # MSA Featurizers
    "msa_embed": MSAEmbedFeaturizer,
    "msa_sequence_embed": MSASequenceEmbedFeaturizer,
    "msa_logits": MSALogitsFeaturizer,
    # ESM Featurizers
    "esm_logits": ESMLogitsFeaturizer,
    "esm_embed_1v": ESM1vEmbedFeaturizer,
    "esm_embed_2_3b": ESM2EmbedFeaturizer,
    "esm_embed_2_15b": ESM2_15b_EmbedFeaturizer,
    # Zeroshot Featurizers
    "zeroshot_msa": ZeroshotMSAFeaturizer,
    "zeroshot_esm": ZeroshotESMFeaturizer,
    "zeroshot_prose": ZeroshotProseFeaturizer,
    "zeroshot_cscs": ZeroshotCSCSFeaturizer,
    "zeroshot_cscs_gram": ZeroshotCSCSGramFeaturizer,
    "zeroshot_cscs_sem": ZeroshotCSCSSemFeaturizer,
}


class CombinatorialFeaturizer():
    """Base class for combining multiple featurizers.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = CombinatorialFeaturizer(
        featurize_methods=['onehot', 'georgiev'],  # List of featurizers to combine
        protein='protein1',                        # Name of protein for caching
        use_cache=True                            # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods, **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        for featurize_method in featurize_methods:
            assert (
                featurize_method in FEATURIZE_CHOICES
            ), f"{featurize_method} not in {FEATURIZE_CHOICES}"

        model_type = "-".join(featurize_methods)
        self.name = str(model_type)

        self.featurizers = {
            featurize_method: FEATURIZE_CLASSES[featurize_method](**kwargs)
            for featurize_method in featurize_methods
        }

    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using all component featurizers.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined features from all featurizers.
        """
        X = []
        for featurizer in self.featurizers.values():
            X.append(featurizer.featurize(seqs, **kwargs))

        X = np.concatenate(X, axis=-1)

        return X
    
class ESMAugmentedFeaturizer(CombinatorialFeaturizer):
    """Class for combining ESM features with one-hot encoding.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = ESMAugmentedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["zeroshot_esm", "onehot"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)
    
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM and one-hot encoding.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined ESM and one-hot features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        X.append(featurizer_0.featurize(seqs, **kwargs))
        
        featurizer_1 = list(self.featurizers.values())[1]
        onehot = featurizer_1.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        X = np.concatenate(X, axis=1)

        return X

class MSAAugmentedFeaturizer(CombinatorialFeaturizer):
    """Class for combining MSA features with one-hot encoding.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = MSAAugmentedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["zeroshot_msa", "onehot"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using MSA and one-hot encoding.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined MSA and one-hot features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        X.append(featurizer_0.featurize(seqs, **kwargs))
        
        featurizer_1 = list(self.featurizers.values())[1]
        onehot = featurizer_1.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        X = np.concatenate(X, axis=1)

        return X

class OnehotAndGeorgievFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot and Georgiev encodings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndGeorgievFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["onehot", "georgiev"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

class OnehotAndAAIdxFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot and amino acid index encodings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndAAIdxFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["onehot", "aa_idx"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)      

class OnehotAndESMLogitsFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM logits.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESMLogitsFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["onehot", "esm_logits"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)
    
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM logits.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM logits features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        x = featurizer_0.featurize(seqs, **kwargs)
        zero_vectors = np.zeros((x.shape[0], 1, x.shape[2]))
        X.append(np.concatenate((zero_vectors, x, zero_vectors), axis=1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=-1)

        return X

class OnehotAndESMMSALogitsFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM-MSA logits.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESMMSALogitsFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["onehot", "msa_logits"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)
    
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM-MSA logits.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM-MSA logits features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        x = featurizer_0.featurize(seqs, **kwargs)
        zero_vectors = np.zeros((x.shape[0], 1, x.shape[2]))
        X.append(np.concatenate((zero_vectors, x), axis=1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=-1)

        return X
    

class OnehotAndESM2EmbedFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM2 embeddings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESM2EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["onehot", "esm_embed_2_3b"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM2 embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM2 embedding features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        onehot = featurizer_0.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=1)

        return X

class OnehotAndESM2_15bEmbedFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM2 embeddings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESM2EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(self, featurize_methods=["onehot", "esm_embed_2_15b"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM2 embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM2 embedding features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        onehot = featurizer_0.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=1)

        return X