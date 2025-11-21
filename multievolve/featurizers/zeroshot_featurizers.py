from Bio import SeqIO

from multievolve.featurizers.base_featurizers import BaseFeaturizer
from multievolve.featurizers.model_choices import FEATURE_MODELS
from multievolve.utils.data_utils import find_mutations_multithreaded

class ZeroshotBaseFeaturizer(BaseFeaturizer):
    """Base class for zero-shot featurizers.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotBaseFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta',      # Path to wild-type sequence
        model_locations=[]        # Paths to model files
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model_type, protein, use_cache, **kwargs)
        self.model_locations = model_locations
        self.wt_file = wt_file
        self.wt_seq = str(SeqIO.read(self.wt_file, "fasta").seq)

    def featurize_zeroshot(
        self, seqs, model_locations, wt_file, zeroshot_model, **kwargs
    ):
        """
        Featurizes sequences using zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            zeroshot_model (callable): Zero-shot prediction function.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Zero-shot prediction scores.
        """
        assert (self.wt_file is not None) or (
            wt_file is not None
        ), "No wt sequence provided."
        assert (self.model_locations is not None) or (
            model_locations is not None
        ), "No model locations provided."

        wt_file = wt_file or self.wt_file
        model_locations = model_locations or self.model_locations

        wt_seq = str(SeqIO.read(self.wt_file, "fasta").seq)
        model_locations = self.model_locations

        mutations = find_mutations_multithreaded(wt_seq, seqs)

        # make sure to remove model_locations and sequence from kwargs
        kwargs.pop("model_locations", None)
        kwargs.pop("sequence", None)
        kwargs['device'] = self.device
        X = zeroshot_model(
            mutations, model_locations=model_locations, sequence=wt_seq, **kwargs
        )

        X = X.reshape(-1, 1)

        return X


class ZeroshotESMFeaturizer(ZeroshotBaseFeaturizer):
    """Class for ESM zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotESMFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_esm"],
        wt_file=None,
        model_type="zeroshot_esm",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: ESM zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_esm as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


class ZeroshotMSAFeaturizer(ZeroshotBaseFeaturizer):
    """Class for MSA zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.
        msa_file (str): Path to MSA file.

    Example Usage:

    featurizer = ZeroshotMSAFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta',      # Path to wild-type sequence
        msa_file='msa.fasta'     # Path to MSA file
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_msa"],
        wt_file=None,
        msa_file=None,
        model_type="zeroshot_msa",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            msa_file (str): Path to MSA file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )
        self.msa_file = msa_file

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using MSA zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: MSA zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_msa as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            msa_file=self.msa_file,
            **kwargs
        )
        return X


class ZeroshotCSCSFeaturizer(ZeroshotBaseFeaturizer):
    """Class for CSCS zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotCSCSFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot_cscs",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using CSCS zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: CSCS zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_cscs as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


class ZeroshotCSCSGramFeaturizer(ZeroshotBaseFeaturizer):
    """Class for CSCS-Gram zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotCSCSGramFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot_cscs_gram",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using CSCS-Gram zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: CSCS-Gram zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_cscs_gram as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


class ZeroshotCSCSSemFeaturizer(ZeroshotBaseFeaturizer):
    """Class for CSCS-Sem zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotCSCSSemFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot_cscs_sem",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using CSCS-Sem zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: CSCS-Sem zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_cscs_sem as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


class ZeroshotProseFeaturizer(ZeroshotBaseFeaturizer):
    """Class for ProSE zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotProseFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_prose"],
        wt_file=None,
        model_type="zeroshot_prose",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ProSE zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: ProSE zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_prose as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


class ZeroshotESMIFFeaturizer(ZeroshotBaseFeaturizer):
    """Class for ESM-IF zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.
        pdb_file (str): Path to PDB structure file.
        chain_id (str): Chain identifier in PDB file.

    Example Usage:

    featurizer = ZeroshotESMIFFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta',      # Path to wild-type sequence
        pdb_file='struct.pdb',   # Path to structure file
        chain_id='A'             # Chain identifier
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_esmif"],
        wt_file=None,
        model_type="zeroshot_esmif",
        pdb_file=None,
        chain_id='A',
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            pdb_file (str): Path to PDB structure file.
            chain_id (str): Chain identifier in PDB file.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            pdb_file=pdb_file,
            chain_id=chain_id,
            **kwargs
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM-IF zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: ESM-IF zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_esm_if as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            pdb_file=self.pdb_file,
            chain_id=self.chain_id,
            **kwargs
        )
        return X