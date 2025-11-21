import numpy as np
import torch

from multievolve.featurizers.model_choices import FEATURE_MODELS
from multievolve.featurizers.base_featurizers import BaseFeaturizer
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

class ForgeESMFeaturizer(BaseFeaturizer):
    """Class for generating ESM Forge-based protein embeddings or log probabilities.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model (str): ESM model name.
        url (str): Forge API URL.
        token (str): API token.
        output_type (str): Output format type.

    Example Usage:

    featurizer = ForgeESMFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        flatten_features=False,   # Whether to flatten output features
        model='esm2_t33_650M',   # ESM model to use
        token='api_token',       # Forge API token
        output_type='sequence_representations'  # Output type
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        flatten_features=False,
        model=None,
        url="https://forge.evolutionaryscale.ai",
        token=None,
        output_type=None,
        model_type="esm",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            flatten_features (bool): Whether to flatten output features.
            model (str): ESM model name.
            url (str): Forge API URL.
            token (str): API token.
            output_type (str): Output format type.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model_type=model_type, protein=protein, use_cache=use_cache, flatten_features=flatten_features, **kwargs)
        self.model = model
        self.url = url
        if token is None:
            raise ValueError("ESM Forge API token must be provided")
        self.token = token
        self.output_type = output_type

    def process_single_protein(self, sequence, model, url, token, output_type):
        """Process a single protein sequence using ESM Forge.
        
        Args:
            sequence (str): Protein sequence to process.
            model (str): ESM model name.
            url (str): Forge API URL.
            token (str): Forge API access token.
            output_type (str): Either "log_probabilities" or "sequence_representations".
            
        Returns:
            numpy.ndarray: Protein embeddings or log probabilities.
            
        Raises:
            ValueError: If output_type is invalid.
            RuntimeError: If API call fails.
        """
        if output_type not in ["log_probabilities", "sequence_representations"]:
            raise ValueError("output_type must be 'log_probabilities' or 'sequence_representations'")
         
        try:
            
            from esm.sdk.forge import ESM3ForgeInferenceClient
            from esm.sdk.api import ESMProtein, LogitsConfig

            forge_client = ESM3ForgeInferenceClient(model=model, url=url, token=token)
            protein = ESMProtein(sequence=sequence)
            protein_tensor = forge_client.encode(protein)
            logits_output = forge_client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
        except Exception as e:
            raise RuntimeError(f"ESM Forge API call failed: {str(e)}")

        if output_type == "log_probabilities":
            return np.array(logits_output.logits.sequence.numpy())
        elif output_type == "sequence_representations":
            embeddings_float32 = logits_output.embeddings.squeeze().float()
            return (np.array(embeddings_float32.numpy()).mean(axis=0))
        

    def process_proteins_parallel(self, seqs, model, url, token, output_type):
        """Process a list of protein sequences in parallel using ESM Forge.
        
        Args:
            seqs (list): List of protein sequences to process.
            model (str): ESM model name to use.
            url (str): Forge API URL.
            token (str): Forge API access token.
            output_type (str): Type of output features.
            
        Returns:
            list: List of protein embeddings as numpy arrays.
        """
        with ProcessPoolExecutor(max_workers=16) as executor:
            future_to_index = {
                executor.submit(self.process_single_protein, seq, model, url, token, output_type): i 
                for i, seq in enumerate(seqs)
            }
            
            results = [None] * len(seqs)
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                
        return results
        
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM Forge.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of featurized sequences.
        """
        X = self.process_proteins_parallel(seqs, model=self.model, url=self.url, token=self.token, output_type=self.output_type)
        return X


class Forge_ESMC_6B_EmbedFeaturizer(ForgeESMFeaturizer):
    """Class for generating ESM-C 6B embeddings using Forge.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model (str): ESM model name.
        url (str): Forge API URL.
        token (str): API token.
        output_type (str): Output format type.

    Example Usage:

    featurizer = Forge_ESMC_6B_EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        flatten_features=False,   # Whether to flatten output features
        token='api_token'        # Forge API token
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        flatten_features=False,
        model="esmc-6b-2024-12",
        url="https://forge.evolutionaryscale.ai",
        token=None,
        output_type="sequence_representations",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            flatten_features (bool): Whether to flatten output features.
            model (str): ESM model name.
            url (str): Forge API URL.
            token (str): API token.
            output_type (str): Output format type.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model_type="esmc_6b", protein=protein, use_cache=use_cache, flatten_features=flatten_features, output_type=output_type, model=model, url=url, token=token, **kwargs)


class ESMBaseFeaturizer(BaseFeaturizer):
    """Base class for ESM model featurizers.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to ESM model files.

    Example Usage:

    featurizer = ESMBaseFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        batch_size=968,          # Processing batch size
        model_locations=['path/to/model']  # Model file paths
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        batch_size=968,
        model_type="esm",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to ESM model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model_type, protein, use_cache, **kwargs)
        self.batch_size = batch_size
        self.model_locations = model_locations

    def eval_esm(self, model, batch_tokens, sequence_data, output_type):
        """
        Evaluates sequences using ESM model.

        Args:
            model (torch.nn.Module): ESM model.
            batch_tokens (torch.Tensor): Tokenized sequences.
            sequence_data (list): Original sequence data.
            output_type (str): Type of output features.

        Returns:
            np.ndarray: Model outputs.
        """
        if output_type == "log_probabilities":
            # Featurize as sequence log probabilities.
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    model(batch_tokens.to(self.device))["logits"], dim=-1
                )
            output = token_probs.cpu().numpy()

        elif output_type == "sequence_representations":
            # Featurize as sequence embeddings (last hidden layer).
            last_layer = len(model.layers)
            with torch.no_grad():
                results = model(
                    batch_tokens.to(self.device), repr_layers=[last_layer]
                )
            token_representations = results["representations"][last_layer]

            sequence_representations = []
            for i, (_, seq) in enumerate(sequence_data):
                seq_embed = token_representations[i, 1 : len(seq) + 1].mean(0)
                sequence_representations.append(seq_embed.cpu().numpy().ravel())

            output = np.array(sequence_representations)

        return output

    def featurize_esm(self, seqs, output_type):
        """
        Featurizes sequences using ESM model.

        Args:
            seqs (list): List of sequences to featurize.
            output_type (str): Type of output features.

        Returns:
            np.ndarray: Array of featurized sequences.
        """
        from esm import pretrained
        model_loc_to_model = {}
        for model_location in self.model_locations:
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.backends.mps.is_available() or torch.cuda.is_available():
                model = model.to(self.device)
            else:
                print("GPU device not available")
                return
            model_loc_to_model[model_location] = model

        output = []

        batch_size = self.batch_size
        n_batches = ((len(seqs[0]) - 1) // batch_size) + 1
        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size

            model_features = []
            for model_location in self.model_locations:
                model = model_loc_to_model[model_location]

                seq_batch = []
                sbatch_size = 3
                n_sbatches = ((len(seqs) - 1) // sbatch_size) + 1
                for batchj in range(n_sbatches):
                    sb_start = batchj * sbatch_size
                    sb_end = (batchj + 1) * sbatch_size

                    sequence_data = [
                        (f"protein{sbidx}", seq[start:end])
                        if output_type == "log_probabilities"
                        else (f"protein{sbidx}", seq[start:end].replace("X", ""))
                        for sbidx, seq in enumerate(seqs[sb_start:sb_end])
                    ]

                    batch_converter = alphabet.get_batch_converter()
                    batch_labels, batch_strs, batch_tokens = batch_converter(
                        sequence_data
                    )

                    if n_batches > 1:
                        if batchi == 0:
                            batch_tokens = batch_tokens[:, :-1]
                        elif batchi == n_batches - 1:
                            batch_tokens = batch_tokens[:, 1:]
                        else:
                            batch_tokens = batch_tokens[:, 1:-1]

                    seq_batch.append(
                        self.eval_esm(model, batch_tokens, sequence_data, output_type)
                    )

                model_features.append(np.concatenate(seq_batch))

            output.append(np.mean(model_features, axis=0))

        X = np.hstack(output)

        return X


class ESMLogitsFeaturizer(ESMBaseFeaturizer):
    """Class for generating ESM model log probabilities.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to ESM model files.

    Example Usage:

    featurizer = ESMLogitsFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["esm_logits"],
        batch_size=968,
        model_type="esm_logits",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to ESM model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            batch_size,
            model_type,
            **kwargs,
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM model log probabilities.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of log probabilities.
        """
        X = self.featurize_esm(seqs, output_type="log_probabilities")
        return X


class ESM1vEmbedFeaturizer(ESMBaseFeaturizer):
    """Class for generating ESM-1v model embeddings.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to ESM model files.

    Example Usage:

    featurizer = ESM1vEmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["esm_embed_1v"],
        batch_size=968,
        model_type="esm_embed_1v",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to ESM model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            batch_size,
            model_type,
            **kwargs,
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM-1v model embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of sequence embeddings.
        """
        X = self.featurize_esm(seqs, output_type="sequence_representations")
        return X


class ESM2EmbedFeaturizer(ESMBaseFeaturizer):
    """Class for generating ESM-2 model embeddings.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to ESM model files.

    Example Usage:

    featurizer = ESM2EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["esm_embed_2_3b"],
        batch_size=968,
        model_type="esm_embed_2_3b",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to ESM model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            batch_size,
            model_type,
            **kwargs,
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM-2 model embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of sequence embeddings.
        """
        X = self.featurize_esm(seqs, output_type="sequence_representations")
        return X


class ESM2_15b_EmbedFeaturizer(ESMBaseFeaturizer):
    """Class for generating ESM-2 15B model embeddings.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.
        model_locations (list): Paths to ESM model files.

    Example Usage:

    featurizer = ESM2_15b_EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        batch_size=968           # Processing batch size
    )
    features = featurizer.featurize(sequences)
    """

    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["esm_embed_2_15b"],
        batch_size=968,
        model_type="esm_embed_2_15b",
        **kwargs,
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to ESM model files.
            batch_size (int): Batch size for processing.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein,
            use_cache,
            model_locations,
            batch_size,
            model_type,
            **kwargs,
        )

    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM-2 15B model embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of sequence embeddings.
        """
        X = self.featurize_esm(seqs, output_type="sequence_representations")
        return X