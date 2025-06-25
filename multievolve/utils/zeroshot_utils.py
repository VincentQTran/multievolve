# This module contains utility functions for zero-shot predictions using various protein language models

import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
import tqdm

from multievolve.utils.other_utils import read_msa, greedy_select, msa_splicer, AAs

def create_parser():
    """Create and return an argument parser for zero-shot prediction."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sequence-fname',
        type=str,
        help='Sequence FASTA to do deep mutational scan.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='prose',
        help='Zero shot model',
    )
    return parser


def compute_score_prose(
        mutation,
        sequence,
        token_probs,
        alphabet,
        raw_score=False
):
    """
    Compute the score for a single mutation using ProSE model.
    
    Args:
        mutation (str): Mutation in the format 'WT{position}MT'
        sequence (str): Original protein sequence
        token_probs (torch.Tensor): Token probabilities from the model
        alphabet (Alphabet): ProSE alphabet
        raw_score (bool): If True, return raw score; if False, return delta score
    
    Returns:
        float: Mutation score
    """
    wt, idx, mt = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]
    assert(sequence[idx] == wt)
    
    wt_encoded, mt_encoded = (
        alphabet.encode(wt.encode())[0], alphabet.encode(mt.encode())[0]
    )

    score = token_probs[idx, mt_encoded]
    if not raw_score:
        score -= token_probs[idx, wt_encoded]

    return score.item()


def zero_shot_prose(
        mutations,
        model_locations,
        sequence,
        **kwargs
):
    """
    Perform zero-shot prediction using ProSE model.
    
    Args:
        mutations (list): List of mutation sets
        model_locations (list): List of ProSE model file paths
        sequence (str): Original protein sequence
        **kwargs: Additional arguments (e.g., device)
    
    Returns:
        numpy.ndarray: Array of mutation scores
    """
    from prose.alphabets import Uniprot21
    from prose.utils import pack_sequences
    
    alphabet = Uniprot21()
    x = [ torch.from_numpy(alphabet.encode(sequence.encode())).long() ]
    x, _ = pack_sequences(x)

    X = []

    for model_location in model_locations:
        
        model = torch.load(model_location).to(kwargs['device'])
        logits = model(x.to(kwargs['device'])).data

        with torch.no_grad():
            token_probs = torch.log_softmax(
                logits, dim=1
            )

        X_model = []
        for mutation_set in mutations:
            score = np.mean([
                compute_score_prose(
                    mutation, sequence, token_probs, alphabet,
                    raw_score=False,
                )
                for mutation in mutation_set
            ])

            if not np.isfinite(score):
                score = 0.

            X_model.append(score)

        X.append(X_model)

    return np.array(X).mean(0)


def zero_shot_cscs(
        mutations,
        model_locations,
        sequence,
        exclude_gram=False,
        exclude_sem=False,
        **kwargs
):
    """
    Perform zero-shot prediction using CSCS (Computational Saturation Cost Score) model.
    
    Args:
        mutations (list): List of mutation sets
        model_locations (list): List of CSCS model file paths
        sequence (str): Original protein sequence
        exclude_gram (bool): If True, exclude grammaticality scores
        exclude_sem (bool): If True, exclude semantic change scores
        **kwargs: Additional arguments
    
    Returns:
        numpy.ndarray: Array of mutation scores
    """
    if exclude_gram and exclude_sem:
        raise ValueError(
            'Must have grammaticality, semantic change, or both.'
        )
    
    data = []
    with open(model_locations[0]) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split('\t')
            pos, wt, mt, prob, sem = fields[:5]
            if mt in { 'B', 'J', 'U', 'X', 'Z' }:
                continue
            mut_str = f'{wt}{int(pos) + 1}{mt}'
            data.append([ mut_str, float(prob), float(sem) ])
    df = pd.DataFrame(data, columns=[ 'mutation', 'prob', 'sem' ])

    df['cscs'] = (
        ss.rankdata(np.array(df['prob'])) +
        ss.rankdata(np.array(df['sem']))
    )

    mut_to_gram = {
        df['mutation'][i]: df['prob'][i] for i in range(len(df))
    }
    mut_to_sem = {
        df['mutation'][i]: df['sem'][i] for i in range(len(df))
    }
    mut_to_cscs = {
        df['mutation'][i]: df['cscs'][i] for i in range(len(df))
    }

    X = []
    for mutation_set in mutations:
        if not exclude_gram and exclude_sem:
            score = np.mean([
                mut_to_gram[mut] for mut in mutation_set
            ])
        elif exclude_gram and not exclude_sem:
            score = np.mean([
                mut_to_sem[mut] for mut in mutation_set
            ])
        elif not exclude_gram and not exclude_sem:
            score = np.mean([
                mut_to_cscs[mut] for mut in mutation_set
            ])
        if not np.isfinite(score):
            score = 0.
        X.append(score)

    return np.array(X)

def zero_shot_cscs_gram(
        mutations,
        model_locations,
        sequence,
        **kwargs
):
    """
    Perform zero-shot prediction using CSCS model with only grammaticality scores.
    
    This is a wrapper function for zero_shot_cscs with exclude_sem=True.
    """
    return zero_shot_cscs(
        mutations,
        model_locations,
        sequence,
        exclude_sem=True,
    )

def zero_shot_cscs_sem(
        mutations,
        model_locations,
        sequence,
):
    """
    Perform zero-shot prediction using CSCS model with only semantic change scores.
    
    This is a wrapper function for zero_shot_cscs with exclude_gram=True.
    """
    return zero_shot_cscs(
        mutations,
        model_locations,
        sequence,
        exclude_gram=True,
    )


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

def zero_shot_esm_dms(wt_seq, scoring_strategy='wt-marginals', **kwargs):
    """
    Perform deep mutational scanning using ESM model.
    
    Args:
        wt_seq (str): Wild-type protein sequence
        scoring_strategy (str): 'wt-marginals' or 'masked-marginals'
        **kwargs: Additional arguments
    
    Returns:
        pandas.DataFrame: DataFrame containing mutation scores and statistics
    """
    from esm import pretrained
    # create list of all possible single point mutations in the wildtype sequence
    
    amino_acids = AAs[:-1]
    mutations = []
    for i, residue in enumerate(wt_seq):
        for aa in amino_acids:
            mutations.append(wt_seq[i] + str(i + 1) + aa)

    model_locations = [
    'esm1v_t33_650M_UR90S_1',
    'esm1v_t33_650M_UR90S_2',
    'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4',
    'esm1v_t33_650M_UR90S_5',
    'esm2_t36_3B_UR50D'
    ] 

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

        scores = model_prob
        mutation_score = {}
        for pos in range(len(wt_seq)):
            wt = wt_seq[pos]
            for mt in alphabet.all_toks:
                mutation = f'{wt}{pos + 1}{mt}'
                mutation_score[mutation] = scores[pos + 1, alphabet.tok_to_idx[mt]] # logits are arranged seq length x vocab (add one to pos due to additional CLS token)

        X_sub = []
        for mutation in mutations:
            wt = mutation[0]+mutation[1:-1]+mutation[0]
            score = mutation_score[mutation] - mutation_score[wt]
            if not np.isfinite(score):
                score = 0.
            
            X_sub.append(score)

        X.append(X_sub)
    
    df = pd.DataFrame({'mutations': mutations, 
                       'model_1_logratio': X[0], 
                       'model_2_logratio': X[1], 
                       'model_3_logratio': X[2], 
                       'model_4_logratio': X[3], 
                       'model_5_logratio': X[4], 
                       'model_6_logratio': X[5]})

    df['average_model_logratio'] = (df['model_1_logratio'] + df['model_2_logratio'] + df['model_3_logratio'] + df['model_4_logratio'] + df['model_5_logratio'] + df['model_6_logratio']) / 6
    df['model_1_pass'] = df['model_1_logratio'].apply(lambda x: 1 if x > 0 else 0)
    df['model_2_pass'] = df['model_2_logratio'].apply(lambda x: 1 if x > 0 else 0)
    df['model_3_pass'] = df['model_3_logratio'].apply(lambda x: 1 if x > 0 else 0)
    df['model_4_pass'] = df['model_4_logratio'].apply(lambda x: 1 if x > 0 else 0)
    df['model_5_pass'] = df['model_5_logratio'].apply(lambda x: 1 if x > 0 else 0)
    df['model_6_pass'] = df['model_6_logratio'].apply(lambda x: 1 if x > 0 else 0)
    df['total_model_pass'] = df['model_1_pass'] + df['model_2_pass'] + df['model_3_pass'] + df['model_4_pass'] + df['model_5_pass'] + df['model_6_pass']
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
