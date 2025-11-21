# This module contains utility functions for zero-shot predictions using various protein language models

import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
from tqdm import tqdm

from multievolve.utils.other_utils import read_msa, greedy_select, msa_splicer, AAs

def zero_shot_esm_dms(wt_seq, 
                    model_locations = ['esm1v_t33_650M_UR90S_1',
                    'esm1v_t33_650M_UR90S_2',
                    'esm1v_t33_650M_UR90S_3',
                    'esm1v_t33_650M_UR90S_4',
                    'esm1v_t33_650M_UR90S_5',
                    'esm2_t36_3B_UR50D'],
                    scoring_strategy='wt-marginals',
                    num_msa_seqs=400,
                    **kwargs):
    """
    Perform deep mutational scanning using ESM model.
    
    Args:
        wt_seq (str): Wild-type protein sequence
        scoring_strategy (str): 'wt-marginals' or 'masked-marginals'
        **kwargs: Additional arguments
    
    Returns:
        pandas.DataFrame: DataFrame containing mutation scores and statistics
    """
    from esm import pretrained, MSATransformer
    
    # create list of all possible single point mutations in the wildtype sequence
    amino_acids = AAs[:-1]
    mutations = []
    for i, residue in enumerate(wt_seq):
        for aa in amino_acids:
            if wt_seq[i] == aa:
                continue
            mutations.append(wt_seq[i] + str(i + 1) + aa)

    # Compute token probs for each model.
    model_probs = []

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    for model_location in model_locations:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        model = model.to(device)

        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):

            assert kwargs['msa_file'] is not None, 'No MSA file provided.'
            msa = read_msa(kwargs['msa_file'])

            # Prep the MSA, making the appropriate mutations
            inputs = greedy_select(msa, num_seqs=num_msa_seqs) # can change this to pass more/fewer sequences
            #This splices the MSA to exclude gaps in the first sequence, due to MSATransformer context window 
            #size limit of 1024. If your MSA width is less than 1024, then you don't need to do this
            data = [msa_splicer(inputs)]

            # Run the model, retrieve logits
            _, __, batch_tokens = batch_converter(data)
            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked.to(device))["logits"], dim=-1
                    )
                all_token_probs.append(token_probs[:, 0, i])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

        else: 
            data = [
                ('protein1', wt_seq),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if scoring_strategy == 'wt-marginals':
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens.to(device))['logits'], dim=-1)

            elif scoring_strategy == 'masked-marginals':
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.to(device))['logits'], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

            else:
                raise ValueError(f'Invalid scoring strategy {scoring_strategy}')
            
        model_probs.append(token_probs.cpu().numpy()[0])
    
    X = []
    for model_prob in model_probs:

        X_sub = []
        for mutation in mutations:
            wt, idx, mt = mutation[0], int(mutation[1:-1])-1, mutation[-1]
            assert wt_seq[idx] == wt, 'Wild-type residue does not match input sequence.'
            
            wt_encoded, mt_encoded = alphabet.tok_to_idx[wt], alphabet.tok_to_idx[mt]

            score = model_prob[idx + 1, mt_encoded] - model_prob[idx + 1, wt_encoded]
            if not np.isfinite(score):
                score = 0.
            
            X_sub.append(score)

        X.append(X_sub)
    
    # set up dataframe
    data = {'mutations': mutations}
    for i in range(len(X)):
        data[f'model_{i+1}_logratio'] = X[i]
    df = pd.DataFrame(data)

    # Calculate average log ratio across all models
    logratio_cols = [f'model_{i+1}_logratio' for i in range(len(X))]
    df['average_model_logratio'] = df[logratio_cols].mean(axis=1)

    # Calculate pass/fail for each model
    for i in range(len(X)):
        df[f'model_{i+1}_pass'] = df[f'model_{i+1}_logratio'].apply(lambda x: 1 if x > 0 else 0)

    # Sum up total passes
    pass_cols = [f'model_{i+1}_pass' for i in range(len(X))]
    df['total_model_pass'] = df[pass_cols].sum(axis=1)
    df.sort_values(by='average_model_logratio', ascending=False, inplace=True)
    df.sort_values(by='total_model_pass', ascending=False, inplace=True)

    df_ls = []

    # sort dataframe by total_model_pass and then by average_model_logratio

    total_model_pass_list = list(set(df['total_model_pass'].values))
    total_model_pass_list = total_model_pass_list[::-1]

    for model_pass_value in total_model_pass_list:
        subset = df[df['total_model_pass'] == model_pass_value].copy()
        subset.sort_values(by='average_model_logratio', ascending=False, inplace=True)
        df_ls.append(subset)

    df_sorted = pd.concat(df_ls)

    return df_sorted

def zero_shot_esm_if_dms(wt_seq, pdb_file, chain_id = 'A', scoring_strategy='wt-marginals', **kwargs):
    """
    Perform deep mutational scanning using ESM-IF (Inverse Folding) model.
    
    Args:
        wt_seq (str): Wild-type protein sequence
        pdb_file (str): Path to PDB file
        chain_id (str): Chain ID in the PDB file
        scoring_strategy (str): Currently not used, kept for consistency
        **kwargs: Additional arguments
    
    Returns:
        pandas.DataFrame: DataFrame containing mutation scores
    """
    import torch_geometric
    import torch_sparse
    from torch_geometric.nn import MessagePassing
    import esm
    from esm import pretrained
    from esm.inverse_folding.util import CoordBatchConverter

    amino_acids = AAs[:-1]
    mutations = []
    for i, residue in enumerate(wt_seq):
        for aa in amino_acids:
            if wt_seq[i] == aa:
                continue
            mutations.append(wt_seq[i] + str(i + 1) + aa)

    model_locations = ['esm_if1_gvp4_t16_142M_UR50']

    model, alphabet = pretrained.load_model_and_alphabet(model_locations[0])
    model = model.eval()

    structure = esm.inverse_folding.util.load_structure(pdb_file, chain_id)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    
    if native_seq == wt_seq:
        print(f"Native sequence from structure matches input sequence ({len(native_seq)} residues)")
    else:
        print(f"Warning: Native sequence from structure ({len(native_seq)} residues) does not match input sequence ({len(wt_seq)} residues)")

    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, wt_seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    # Average model scores and find scores for the mutations-of-interest.

    scores = logits.detach().numpy()[0]
    mutation_score = {}
    for pos in range(len(wt_seq)):
        wt = wt_seq[pos]
        for mt in alphabet.all_toks:
            mutation = f'{wt}{pos + 1}{mt}'
            mutation_score[mutation] = scores[alphabet.tok_to_idx[mt], pos]

    X = []
    for mutation in mutations:
        wt = mutation[0]+mutation[1:-1]+mutation[0]
        score = mutation_score[mutation] - mutation_score[wt]
        if not np.isfinite(score):
            score = 0.
        
        X.append(score)

    df = pd.DataFrame({'mutations': mutations, 'logratio': X})

    return df

def zero_shot_esm(
        mutations,
        model_locations,
        sequence,
        scoring_strategy='wt-marginals',
        **kwargs
):
    """
    Perform zero-shot prediction using ESM (Evolutionary Scale Modeling) model.
    
    Args:
        mutations (list): List of mutation sets
        model_locations (list): List of ESM model file paths
        sequence (str): Original protein sequence
        scoring_strategy (str): 'wt-marginals' or 'masked-marginals'
        **kwargs: Additional arguments (e.g., device)
    
    Returns:
        numpy.ndarray: Array of mutation scores
    """
    from esm import pretrained

    # Compute token probs for each model.

    model_probs = []
    
    for model_location in model_locations:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        model = model.to(kwargs['device'])

        batch_converter = alphabet.get_batch_converter()

        data = [
            ('protein1', sequence),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if scoring_strategy == 'wt-marginals':
            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens.to(kwargs['device']))['logits'], dim=-1)

        elif scoring_strategy == 'masked-marginals':
            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(1))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, i] = alphabet.mask_idx
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked.to(kwargs['device']))['logits'], dim=-1
                    )
                all_token_probs.append(token_probs[:, i])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

        else:
            raise ValueError(f'Invalid scoring strategy {scoring_strategy}')
        
        model_probs.append(token_probs.cpu().numpy()[0])

    
    # Sum model scores and find scores for the mutations-of-interest.
    scores = np.sum(model_probs, axis=0)
    mutation_score = {}
    for pos in range(len(sequence)):
        wt = sequence[pos]
        for mt in alphabet.all_toks:
            mutation = f'{wt}{pos + 1}{mt}'
            mutation_score[mutation] = scores[pos + 1, alphabet.tok_to_idx[mt]]

    X = []
    for mutation_set in mutations:
        score = np.mean([
            mutation_score[mutation] for mutation in mutation_set
        ])
        if not np.isfinite(score):
            score = 0.
        X.append(score)

    return np.array(X)

def zero_shot_msa(
        mutations,
        sequence,   
        **kwargs,
):
    """
    Perform zero-shot prediction using MSA Transformer model.
    
    Args:
        mutations (list): List of mutation sets
        sequence (str): Original protein sequence
        **kwargs: Additional arguments (must include 'msa_file')
    
    Returns:
        numpy.ndarray: Array of mutation scores
    """
    import esm
    import torch
    torch.set_grad_enabled(False)
    # Check to see if there is an MSA file in **kwargs.
    assert kwargs['msa_file'] is not None, 'No MSA file provided.'
    msa = read_msa(kwargs['msa_file'])

    # Instantiate the model
    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval()
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

    # Prep the MSA, making the appropriate mutations
    inputs = greedy_select(msa, num_seqs=128) # can change this to pass more/fewer sequences
    #This splices the MSA to exclude gaps in the first sequence, due to MSATransformer context window 
    #size limit of 1024. If your MSA width is less than 1024, then you don't need to do this
    inputs = [msa_splicer(inputs)]

    # Run the model, retrieve logits
    _, __, msa_transformer_batch_tokens = msa_transformer_batch_converter(inputs)
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
    predictions = msa_transformer.forward(msa_transformer_batch_tokens, repr_layers=[12])
    logits = predictions['logits'][0][0]
    token_probs = torch.softmax(logits, dim=-1)
    print(token_probs.shape)

    print('pulling out mutations')
    # Pull specific logits out for the mutations-of-interest.
    mutation_score = {}
    for pos in range(len(sequence)):
        wt = sequence[pos]
        for mt in msa_transformer_alphabet.all_toks:
            mutation = f'{wt}{pos + 1}{mt}'
            mutation_score[mutation] = token_probs[pos + 1, msa_transformer_alphabet.tok_to_idx[mt]]

    X = []
    for mutation_set in mutations:
        score = np.mean([
            mutation_score[mutation] for mutation in mutation_set
        ])
        if not np.isfinite(score):
            score = 0.
        X.append(score)
    
    return np.array(X)

def zero_shot_esm_if(       
        mutations,
        model_locations,
        sequence,
        pdb_file, 
        chain_id,
        **kwargs
):
    """
    Perform zero-shot prediction using ESM-IF (Inverse Folding) model.
    
    Args:
        mutations (list): List of mutation sets
        model_locations (list): List of ESM-IF model file paths
        sequence (str): Original protein sequence
        pdb_file (str): Path to PDB file
        chain_id (str): Chain ID in the PDB file
        **kwargs: Additional arguments
    
    Returns:
        numpy.ndarray: Array of mutation scores
    """
    # Check that imports are correctly installed
    import torch_geometric
    import torch_sparse
    from torch_geometric.nn import MessagePassing
    import esm
    from esm import pretrained
    from esm.inverse_folding.util import CoordBatchConverter

    # If one of the above fails, run the following in your conda environment
    # import torch

    # def format_pytorch_version(version):
    #   return version.split('+')[0]

    # TORCH_version = torch.__version__
    # TORCH = format_pytorch_version(TORCH_version)

    # def format_cuda_version(version):
    #   return 'cu' + version.replace('.', '')

    # CUDA_version = torch.version.cuda
    # CUDA = format_cuda_version(CUDA_version)

    # !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    # !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    # !pip install -q torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    # !pip install -q torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    # !pip install -q torch-geometric

    # # Install esm
    # !pip install -q git+https://github.com/facebookresearch/esm.git

    # # Install biotite
    # !pip install -q biotite

    # Compute token probs for each model.

    model, alphabet = pretrained.load_model_and_alphabet(model_locations[0])
    model = model.eval()

    structure = esm.inverse_folding.util.load_structure(pdb_file, chain_id)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    # Average model scores and find scores for the mutations-of-interest.

    scores = logits.detach().numpy()[0]
    mutation_score = {}
    for pos in range(len(sequence)):
        wt = sequence[pos]
        for mt in alphabet.all_toks:
            mutation = f'{wt}{pos + 1}{mt}'
            mutation_score[mutation] = scores[alphabet.tok_to_idx[mt], pos] # logits are vocab x length (no padding)

    X = []
    for mutation_set in mutations:
        score = np.mean([
            mutation_score[mutation] for mutation in mutation_set
        ])
        if not np.isfinite(score):
            score = 0.
        X.append(score)

    return np.array(X)
