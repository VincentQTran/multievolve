# updated 3/25/2025
from concurrent.futures import ProcessPoolExecutor
import copy
import re

import pandas as pd
import numpy as np
import os

from Bio import Align, SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
import matplotlib.pyplot as plt
from Bio.SeqRecord import SeqRecord
from typing import Optional, Tuple

codon_dicts = {
    'human': {
    'F': 'TTT', 'L': 'CTG', 'Y': 'TAT', 'H': 'CAT', 'Q': 'CAG',
    'I': 'ATT', 'M': 'ATG', 'N': 'AAT', 'K': 'AAG', 'V': 'GTG',
    'D': 'GAT', 'E': 'GAG', 'S': 'TCT', 'C': 'TGT', 'W': 'TGG',
    'P': 'CCT', 'R': 'CGG', 'T': 'ACT', 'A': 'GCT', 'G': 'GGG',
    },
    'ecoli': {
        'F': 'TTT', 'L': 'CTG', 'Y': 'TAT', 'H': 'CAT', 'Q': 'CAG',
        'I': 'ATT', 'M': 'ATG', 'N': 'AAC', 'K': 'AAA', 'V': 'GTG',
        'D': 'GAT', 'E': 'GAA', 'S': 'TCT', 'C': 'TGC', 'W': 'TGG',
        'P': 'CCG', 'R': 'CGT', 'T': 'ACC', 'A': 'GCG', 'G': 'GGC',
    },
    'yeast': {
        'F': 'TTT', 'L': 'CTA', 'Y': 'TAT', 'H': 'CAT', 'Q': 'CAA',
        'I': 'ATT', 'M': 'ATG', 'N': 'AAT', 'K': 'AAA', 'V': 'GTT',
        'D': 'GAT', 'E': 'GAA', 'S': 'TCT', 'C': 'TGT', 'W': 'TGG',
        'P': 'CCA', 'R': 'AGA', 'T': 'ACT', 'A': 'GCT', 'G': 'GGT',
    }
}

class MultiAssemblyDesigner:

    """
    Designs oligos for protein mutations.

    Args:
        data (pd.DataFrame): DataFrame containing mutation data.
        start_seq_fasta (str): Path to FASTA file with starting sequence.
        overhang (int): Overhang length.
        species (str): Species, 'human', 'ecoli', or 'yeast'.
        oligo_direction (str): Direction of oligo, 'bottom' or 'top'.
        tm (float): Target melting temperature.
        output (str): Type of output, 'design' or 'update'.
    """

    def __init__(self, data, start_seq_fasta, overhang, species='human', oligo_direction='bottom', tm=80, output='design'):
        
        print("Initializing MultiAssemblyDesigner...")
        self.data = data.rename(columns={data.columns[0]:'aa_mut'})
        self.data['aa_mut'] = self.data['aa_mut'].apply(lambda x: self._sort_mutations(x))
        self.fasta_dir = os.path.dirname(start_seq_fasta)

        print(f'The melting temperature is {tm}')
        self.tm = tm
        self.start_seq = SeqIO.read(start_seq_fasta, "fasta").seq.upper()
        self.overhang = overhang
        self.oligo_direction = oligo_direction
        self.codon_dict = codon_dicts[species]
        # print('Processing mutations...')
        self._process_mutations()

        # print('Designing oligos...')
        self._design_oligos()
        self._find_unique_mutant_oligos()

        if output == 'design':
            print('Exporting design...')
            self._export_design()

        elif output == 'update':
            print('Updating oligo IDs...')
            self._modify_oligo_id()

    def _sort_mutations(self, mutation_string):
        """
        Sort mutations within a string based on their position numbers.
        
        Args:
            mutation_string (str): String containing mutations (e.g., 'A167R/T192V')
        
        Returns:
            str: Sorted mutation string
        """
        mutations = mutation_string.split('/')
        sorted_mutations = sorted(mutations, key=lambda x: int(''.join(filter(str.isdigit, x))))
        return sorted_mutations

    def _process_mutations(self):
        """Processes mutations to extract positions and bases."""
        self.data[['Positions','Reference_bases','Alternative_bases']] = self.data.apply(
            lambda x: pd.Series(self._get_codon_mutation_list(x['aa_mut'], self.codon_dict, self.overhang, str(self.start_seq))),
            axis=1
        )
        
        self.data['mut_seq'] = self.data.apply(
            lambda x: self._get_mut_seq(x['Positions'], x['Alternative_bases'], x['aa_mut']),
            axis=1
        )

    def _design_oligos(self):
        """Designs oligos for each mutation in the dataset."""
        self.data[['oligos','oligo_mut']] = self.data.apply(
            lambda x: pd.Series(self._design_oligo_pipeline(x)),
            axis=1
        )

    def _get_codon_mutation_list(self, mut_ls, codon_dict, overhang, start_seq):
        """
        Retrieves list of codon mutations.

        Args:
            mut_ls (list): List of mutations.
            codon_dict (dict): Codon dictionary.
            overhang (int): Overhang length.
            start_seq (str): Starting sequence.

        Returns:
            tuple: Lists of positions, old codons, and new codons.
        """
        pos_ls, old_codon_ls, new_codon_ls = [], [], []
        for mut in mut_ls:
            pos, new_codon = self._get_codon_mutation(mut, codon_dict)
            pos_ls.append(int(pos)+overhang)
            old_codon_ls.append(start_seq[(int(pos)+overhang)-1:(int(pos)+overhang+2)])
            new_codon_ls.append(new_codon)
        return pos_ls, old_codon_ls, new_codon_ls
    
    def _get_codon_mutation(self, mut, codon_dict):
        """
        Retrieves codon mutation details.

        Args:
            mut (str): Mutation string.
            codon_dict (dict): Codon dictionary.

        Returns:
            tuple: Position and new codon.
        """
        new_codon = codon_dict[mut[-1]]
        pos = str(int(mut[1:-1])*3 - 2)
        return pos, new_codon
    
    def _design_oligo_pipeline(self, row):
        """
        Designs oligos for a row of mutations.

        Args:
            row (pd.Series): Row of mutation data.

        Returns:
            tuple: Lists of oligos and oligo mutations.
        """
        pos_start_ls, pos_end_ls = [], []
        for i, pos in enumerate(row['Positions']):
            pos_start, pos_end = self._design_mutant_oligo(self.start_seq, pos, row['Alternative_bases'][i], row['Reference_bases'][i], result='positions')
            pos_start_ls.append(pos_start)
            pos_end_ls.append(pos_end)

        oligos, oligo_mt_mapping = [], []
        i = 0
        while i < len(row['Positions']):
            mut = [row['aa_mut'][i]]
            start_index = pos_start_ls[i]
            index_i = i

            if i < len(row['Positions'])-1:
                n = 0
                while pos_end_ls[i+n] >= pos_start_ls[i+n+1]:
                    n += 1
                    mut.append(row['aa_mut'][i+n])
                    if i+n+1 == len(row['Positions']):
                        break
                index_f = i + n
                i = i + n
                end_index = pos_end_ls[index_f]
            else:
                index_f = i
                end_index = pos_end_ls[i]
            
            oligos.append(str(self._get_mutant_oligo_by_pos(self.start_seq, row['Positions'], row['Alternative_bases'], row['Reference_bases'], start_index, end_index, index_i, index_f)))
            i += 1
            oligo_mt_mapping.append("-".join(mut))

        return oligos, oligo_mt_mapping
    
    def _get_mut_seq(self, pos_ls, new_codon_ls, mut_ls):
        """
        Generates mutated sequence.

        Args:
            pos_ls (list): List of positions.
            new_codon_ls (list): List of new codons.
            mut_ls (list): List of mutations.

        Returns:
            str: Mutated sequence.
        """
        mut_seq = copy.deepcopy(self.start_seq)
        for i, pos in enumerate(pos_ls):
            mod_pos = int(pos) - 1
            wt_aa = mut_ls[i][0]
            wt_aa_retrieved = Seq(self.start_seq[mod_pos:mod_pos+3]).translate()
            assert wt_aa == wt_aa_retrieved, f"{mut_ls[i]} is not a true mutation from {wt_aa_retrieved}{mut_ls[i][1:-1]}"
            mut_seq = mut_seq[:mod_pos] + new_codon_ls[i].lower() + mut_seq[mod_pos+3:]
        return str(mut_seq)
    
    def _design_mutant_oligo(self, seq, pos, new_codon, old_codon, result='oligo'):
        """
        Designs mutant oligo.

        Args:
            seq (str): Sequence.
            pos (int): Position.
            new_codon (str): New codon.
            old_codon (str): Old codon.
            result (str): Type of result to return.

        Returns:
            tuple: Oligo sequence and wild-type oligo sequence, or start and end positions.
        """
        mod_pos = int(pos) - 1
        mut_seq = seq[:mod_pos] + new_codon.lower() + seq[mod_pos+3:]
        wt_seq = seq[:mod_pos] + old_codon.lower() + seq[mod_pos+3:]
        
        start_index = mod_pos - 11
        end_index = mod_pos + 14

        if self.oligo_direction == 'bottom':
            oligo = mut_seq[start_index:end_index].reverse_complement()
            wt_oligo = wt_seq[start_index:end_index].reverse_complement()
        else:
            oligo = mut_seq[start_index:end_index]
            wt_oligo = wt_seq[start_index:end_index]
        
        while mt.Tm_NN(oligo, Na=50, K=25, Tris=35, Mg=10) <= self.tm:
            if len(oligo) % 2 == 0:
                start_index -= 1
            else: 
                end_index += 1

            if self.oligo_direction == 'bottom': 
                oligo = mut_seq[start_index:end_index].reverse_complement()
                wt_oligo = wt_seq[start_index:end_index].reverse_complement()
            else:
                oligo = mut_seq[start_index:end_index]
                wt_oligo = wt_seq[start_index:end_index]

        if result == 'oligo':
            return str(oligo), str(wt_oligo), round(mt.Tm_NN(oligo, Na=50, K=25, Tris=35, Mg=10), 2)
        else:
            return start_index+1, end_index+1
    
    def _get_mutant_oligo_by_pos(self, seq, pos_ls, new_codon_ls, old_codon_ls, start, end, index_i, index_f):
        """
        Retrieves mutant oligo by position.

        Args:
            seq (str): Sequence.
            pos_ls (list): List of positions.
            new_codon_ls (list): List of new codons.
            old_codon_ls (list): List of old codons.
            start (int): Start position.
            end (int): End position.
            index_i (int): Start index.
            index_f (int): End index.

        Returns:
            str: Mutant oligo sequence.
        """
        mod_pos_ls = pos_ls[index_i:index_f+1]
        mod_new_codon_ls = new_codon_ls[index_i:index_f+1]
        mod_old_codon_ls = old_codon_ls[index_i:index_f+1]

        for i, mod_pos in enumerate(mod_pos_ls):
            mod_pos = int(mod_pos) - 1
            old_codon = seq[mod_pos:mod_pos+3]
            assert old_codon.upper() == mod_old_codon_ls[i].upper()
            seq = seq[:mod_pos] + mod_new_codon_ls[i].lower() + seq[mod_pos+3:]

        mod_start, mod_end = int(start) - 1, int(end) - 1
        return seq[mod_start:mod_end].reverse_complement() if self.oligo_direction == 'bottom' else seq[mod_start:mod_end]
    
    def _find_unique_mutant_oligos(self):
        """Identifies unique mutant oligos in the dataset."""
        oligos = [item for sublist in self.data['oligos'].tolist() for item in sublist]
        oligo_mutation = [item for sublist in self.data['oligo_mut'].tolist() for item in sublist]

        df = pd.DataFrame({'oligos': oligos, 'mutation': oligo_mutation}).drop_duplicates(subset=['mutation'], keep='first')
        df['oligo_id'] = range(len(df))

        oligo_dict = {oligo: i for i, oligo in enumerate(df['oligos'])}
        self.data['oligo_id'] = self.data['oligos'].apply(lambda x: [oligo_dict[oligo] for oligo in x])

        self.oligos = df

        # Apply the sorting function to each row
        self.data[['oligo_id', 'oligo_mut']] = self.data.apply(self._sort_oligos, axis=1)

    def _sort_oligos(self, row):
            """Sort oligo_id and corresponding oligo_mut values in sync."""

            # Convert oligo_id string to list of integers
            oligo_ids = row['oligo_id']
            # Convert oligo_mut string to list
            oligo_muts = row['oligo_mut']
            
            # Zip together for sorting
            paired_data = list(zip(oligo_ids, oligo_muts))
            # Sort by oligo_id
            paired_data.sort(key=lambda x: x[0])
            
            # Unzip the sorted data
            sorted_ids, sorted_muts = map(list, zip(*paired_data))
            
            # Return new row with sorted, comma-joined values
            return pd.Series({
                'oligo_id': sorted_ids,
                'oligo_mut': sorted_muts
            })
    
    def _export_df_with_lists(self, df, filepath, delimiter=','):
        """
        Export DataFrame with list columns to CSV, converting lists to delimiter-separated strings
        without brackets for better readability.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing list columns
        filepath (str): Path where CSV will be saved
        delimiter (str): Delimiter to separate list items (default ';')
        """
        # Create a copy to avoid modifying the original
        df_to_save = df.copy()
        
        # Convert list columns to delimited strings
        for column in df_to_save.columns:
            if df_to_save[column].apply(lambda x: isinstance(x, list)).any():
                df_to_save[column] = df_to_save[column].apply(
                    lambda x: delimiter.join(str(item) for item in x) if isinstance(x, list) else x
                )
        
        # Save to CSV
        df_to_save.to_csv(filepath, index=False)

    def _import_df_with_lists(self, filepath, delimiter=','):
        """
        Import CSV file and convert delimiter-separated strings back to lists.
        
        Parameters:
        filepath (str): Path to the CSV file
        delimiter (str): Delimiter used to separate list items (default ';')
        
        Returns:
        pandas.DataFrame: DataFrame with list columns properly restored
        """
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Try to convert delimited strings back to lists
        for column in df.columns:
            try:
                # Check if the column contains delimiter-separated values
                if df[column].dtype == 'object':
                    first_value = str(df[column].iloc[0])
                    if delimiter in first_value:
                        # Convert to list and handle type conversion
                        def convert_to_list(value):
                            if pd.isna(value):
                                return []
                            items = str(value).split(delimiter)
                            # Try to convert to numbers if possible
                            try:
                                return [float(item) if '.' in item else int(item) 
                                    for item in items]
                            except ValueError:
                                return items
                        
                        df[column] = df[column].apply(convert_to_list)
            except:
                # If conversion fails, keep the column as is
                continue
        
        return df

    def _export_design(self):
        """Exports the cloning sheet and oligos."""

        self._export_df_with_lists(self.data[['oligo_id', 'oligo_mut']].copy(), os.path.join(self.fasta_dir, 'cloning_sheet.csv'))
        self.oligos.to_csv(os.path.join(self.fasta_dir, 'oligos.csv'), index=False)

    def _modify_oligo_id(self):
        """Modifies the oligo_id in the cloning sheet to match the updated oligo_id in the oligos file."""
        self.oligos = self._import_df_with_lists(os.path.join(self.fasta_dir, 'oligos.csv'))
        oligo_dict = dict(zip(self.oligos['mutation'], self.oligos['oligo_id']))
        self.data['oligo_id'] = self.data['oligo_mut'].apply(lambda x: [oligo_dict[mutation] for mutation in x])
        self.data[['oligo_id', 'oligo_mut']] = self.data.apply(self._sort_oligos, axis=1)
                
        self._export_df_with_lists(self.data[['oligo_id', 'oligo_mut']].copy(), os.path.join(self.fasta_dir, 'cloning_sheet.csv'))

class SequenceTrimmer:
    """
    Trims adapter sequences from DNA sequences, handling both forward and reverse orientations.
    
    Args:
        five_prime (str): 5' adapter sequence to find and trim before
        three_prime (str): 3' adapter sequence to find and trim after
        max_error_rate (float): Maximum mismatch rate allowed when matching adapters (default: 0.1)
        min_length (int): Minimum sequence length after trimming (default: 15)
        
    Attributes:
        five_prime (str): Uppercase 5' adapter sequence
        three_prime (str): Uppercase 3' adapter sequence
        max_error_rate (float): Maximum allowed mismatch rate
        min_length (int): Minimum allowed sequence length
    """

    def __init__(self, 
                 five_prime: str,
                 three_prime: str,
                 min_length: int,
                 max_error_rate: float = 0
                 ):
        self.five_prime = five_prime.upper() 
        self.three_prime = three_prime.upper()
        self.max_error_rate = max_error_rate
        self.min_length = min_length
    
    def _count_mismatches(self, seq1: str, seq2: str) -> int:
        """
        Count mismatches between two sequences of equal length.
        
        Args:
            seq1 (str): First sequence
            seq2 (str): Second sequence
            
        Returns:
            int: Number of mismatched positions
        """
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    
    def _reverse_complement(self, seq: str) -> str:
        """
        Generate reverse complement of a DNA sequence.
        
        Args:
            seq (str): Input DNA sequence
            
        Returns:
            str: Reverse complement sequence
        """
        seq = seq.upper()
        complement = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
        return ''.join(complement.get(base, base) for base in reversed(seq))
    
    def _find_with_mismatches(self, sequence: str, pattern: str) -> Optional[Tuple[Tuple[int, int], str]]:
        """
        Find pattern in sequence and its reverse complement, allowing mismatches.
        
        Args:
            sequence (str): Input sequence to search
            pattern (str): Pattern to find
            
        Returns:
            Optional[Tuple[Tuple[int, int], str]]: Tuple of ((start, end), strand) if found, None if not found
        """
        sequence = sequence.upper()
        pattern_len = len(pattern)
        
        if len(sequence) < pattern_len:
            return None
            
        scores = {}
        rev_comp = self._reverse_complement(sequence)
        
        for i, seq in enumerate([sequence, rev_comp]):
            for start in range(len(seq) - pattern_len + 1):
                window = seq[start:start + pattern_len]
                score = self._count_mismatches(window, pattern)
                scores[(start, start + pattern_len), "fwd" if i == 0 else "rev"] = score
                
        if not scores:
            return None
            
        best_pos = min(scores.items(), key=lambda x: x[1])
        return best_pos[0] if best_pos[1] <= (1 - self.max_error_rate) * pattern_len else None

    def _trim_record(self, seq: str) -> Optional[str]:
        """
        Trim adapters from a single sequence.
        
        Args:
            seq (str): Input DNA sequence
            
        Returns:
            Optional[str]: Trimmed sequence if successful, None if discarded
        """
        if len(seq) < self.min_length:
            return None
        
        sequence = seq
        sequence_rev_comp = self._reverse_complement(sequence)
        start = 0
        end = len(sequence)
        strand = "fwd"
        
        five_prime_pos = self._find_with_mismatches(sequence, self.five_prime)
        if five_prime_pos:
            start = five_prime_pos[0][0]
            strand = five_prime_pos[1]
                
        three_prime_pos = self._find_with_mismatches(sequence, self.three_prime)
        if three_prime_pos:
            end = three_prime_pos[0][1]


        # check if start position is less than end position
        if start < end:

            if end - start < self.min_length:
                return None
                
            return sequence[start:end] if strand == "fwd" else sequence_rev_comp[start:end]
        
        # if start is greater than end, then the region of interest is wrapping around (given the sequence is circular)
        else:

            if strand == "fwd":
                trim = sequence[start:] + sequence[:end]
            else:
                trim = sequence_rev_comp[start:] + sequence_rev_comp[:end]

            return trim

    def trim_file(self, input, input_type: str = 'fasta') -> Optional[list]:
        """
        Process FASTQ file and output trimmed sequences.
        
        Args:
            input: Path to input FASTQ file or FASTA file or list of either (fasta, fastq, fasta list, fastq list)
            input_type (str): Type of input, either 'fastq' or 'fasta'
            
        Returns:
            Optional[list]: List of trimmed sequences if output='list', None otherwise
        """
        records_stored = []
        if input_type == 'fastq':
            records_stored = [record for record in SeqIO.parse(input, "fastq")]
            seqs = [str(record.seq) for record in SeqIO.parse(input, "fastq")]
        elif input_type == 'fasta':
            records_stored = [record for record in SeqIO.parse(input, "fasta")]
            seqs = [str(record.seq) for record in SeqIO.parse(input, "fasta")]
        elif input_type == 'fasta list':
            records_stored = [record for file in input for record in SeqIO.parse(file, "fasta")]
            seqs = [str(record.seq) for file in input for record in SeqIO.parse(file, "fasta")]
        elif input_type == 'fastq list':
            records_stored = [record for file in input for record in SeqIO.parse(file, "fastq")]
            seqs = [str(record.seq) for file in input for record in SeqIO.parse(file, "fastq")]

        with ProcessPoolExecutor(max_workers=10) as executor:
            trimmed_seqs = list(executor.map(self._trim_record, seqs))

        records = [] 
        for seq, record in zip(trimmed_seqs, records_stored):
            if seq is not None and len(seq) >= self.min_length:
                records.append(SeqRecord(seq=Seq(seq), id=record.id, 
                                      name=record.name, description=record.description))
                
        if input_type == 'fasta list' or input_type == 'fastq list':
            SeqIO.write(records, f"seqs_trimmed.fasta", "fasta")
        else:
            SeqIO.write(records, f"{input.split('.')[0]}_trimmed.fasta", "fasta")

class BaseProteinCDSAnalyzer:
    """
    Analyzes coding sequences (CDS) of proteins.

    Args:
        seqs (str or list): Path to FASTA file or list of sequences.
        ref_seqs (str or list): Path to reference FASTA file or list of reference sequences.
        input_type (str): Type of input, either 'fasta' or 'list'.
    """
    
    def __init__(self, seqs, ref_seqs, input_type='fasta'):
        self._load_sequences(seqs, ref_seqs, input_type)
        self._run_pipeline()
       
    def _load_sequences(self, seqs, ref_seqs, input_type):
        """
        Loads sequences from input file or list.

        Args:
            seqs (str or list): Path to FASTA file or list of sequences.
            ref_seqs (str or list): Path to reference FASTA file or list of reference sequences.
            input_type (str): Type of input, either 'fasta' or 'list'.
        """
        if input_type == 'fasta':
            self.data = pd.DataFrame([str(record.seq).upper() for record in SeqIO.parse(seqs, "fasta")], columns=['seqs'])
            self.ref_seq = str(next(SeqIO.parse(ref_seqs, "fasta")).seq).upper()
        elif input_type == 'list':
            self.data = pd.DataFrame(seqs, columns=['seqs'])
            self.ref_seq = ref_seqs[0]

    def _align_sequences(self, query_sequence):
        """
        Aligns a query sequence to the reference sequence.

        Args:
            query_sequence (str): The sequence to align.

        Returns:
            list: Aligned sequence and its length.
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = 0
        aligner.open_gap_score = -4
        aligner.extend_gap_score = -2
        alignment = next(aligner.align(self.ref_seq, query_sequence))
        return [alignment[1], len(alignment[1])]

    def _align_sequences_multithreaded(self):
        """Aligns sequences using multiple threads for improved performance."""
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._align_sequences, self.data['seqs'])
        self.data[['aligned_seqs', 'aligned_seqs_length']] = pd.DataFrame(list(results))

    def _generate_mutation_name(self, input_list):
        """
        Generates a mutation name from a list of mutations.

        Args:
            input_list (list): List of mutations.

        Returns:
            str: Generated mutation name.
        """
        if not input_list:
            return 'WT'
        if input_list[0] in ['indel', 'deletion', 'contains_N']:
            return input_list[0]
        return '/'.join(sorted(input_list, key=lambda s: int(''.join(filter(str.isdigit, s)))))

    def _compare_codon_to_ref(self, sequence):
        """
        Compares codons in a sequence to the reference sequence.

        Args:
            sequence (str): The sequence to compare.

        Returns:
            tuple: Dictionary of mutation counts and dictionary of mutation details.
        """
        ref_codon_seq = [self.ref_seq[i:i+3] for i in range(0, len(self.ref_seq), 3)]
        codon_seq = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        
        if 'N' in sequence:
            return [0, 0, 0, 0, [], ['contains_N'], [], [], 'contains_N']
        
        if "-" in sequence:
            return [0, 0, 0, 0, [], ['deletion'], [], [], 'deletion']
        
        if len(sequence) > len(self.ref_seq):
            return [0, 0, 0, 0, [], ['indel'], [], [], 'indel']
        
        if len(sequence) == len(self.ref_seq):
            muts = [0, 0, 0, 0]
            seq_mutations = [[], [], [], [], '']
            for pos, (codon, ref_codon) in enumerate(zip(codon_seq, ref_codon_seq), 1):
                mismatches = sum(c1 != c2 for c1, c2 in zip(codon, ref_codon))
                if mismatches:
                    seq_mutations[mismatches].append(ref_codon + str(pos) + codon)
                muts[mismatches] += 1
            return muts + seq_mutations

    def _compare_codon_to_ref_multithreaded(self):
        """Compares codons to reference using multiple threads for improved efficiency."""
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._compare_codon_to_ref, self.data['aligned_seqs'])
        self.data[['Num_Changes_0', 'Num_Changes_1', 'Num_Changes_2', 'Num_Changes_3',
                   'nt_0_mut', 'nt_1_mut', 'nt_2_mut', 'nt_3_mut', 'error']] = pd.DataFrame(list(results))

    def _convert_codon_mut_to_aa_mut(self, codon_mut_ls):
        """
        Converts codon mutations to amino acid mutations.

        Args:
            codon_mut_ls (list): List of codon mutations.

        Returns:
            list: List of amino acid mutations.
        """
        aa_mut_ls = []
        for mut in codon_mut_ls:
            if mut in ['indel', 'deletion']:
                aa_mut_ls.append(mut)
                continue
            match = re.match(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', mut)
            if match:
                part1, part2, part3 = match.groups()
                aa_i = str(Seq(part1).translate())
                aa_f = str(Seq(part3).translate())
                aa_mut_ls.append(aa_i + part2 + aa_f)
        return [aa_mut_ls]
    
    def _convert_codon_mut_to_aa_mut_multithreaded(self):
        """Converts codon mutations to amino acid mutations using multiple threads for better performance."""
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._convert_codon_mut_to_aa_mut, self.data['codon_mut_ls'])
        self.data['aa_mut_ls'] = pd.DataFrame(list(results))
        self.data['aa_mutation'] = self.data['aa_mut_ls'].apply(self._generate_mutation_name)
    
    def _generate_mutation_names_all(self):
        """Generates mutation names for all sequences in the dataset."""
        self.data['codon_mut_ls'] = self.data['nt_1_mut'] + self.data['nt_2_mut'] + self.data['nt_3_mut']
        self.data['codon_mutation'] = self.data['codon_mut_ls'].apply(self._generate_mutation_name)

    def _run_pipeline(self):
        """Executes the full analysis pipeline."""
        self._align_sequences_multithreaded()
        self._compare_codon_to_ref_multithreaded()
        self._generate_mutation_names_all()
        self._convert_codon_mut_to_aa_mut_multithreaded()
        self.mutants = self.data[['aa_mut_ls','aa_mutation']]

class RawNanoporeProteinCDSAnalyzer(BaseProteinCDSAnalyzer):
    """
    Manages raw nanopore sequencing data with high error rate.

    Inherits from BaseProteinCDSAnalyzer.
    """

    def _remove_insertions(self, reference_aligned, query_aligned):
        """
        Removes insertions from aligned query sequence.

        Args:
            reference_aligned (str): Aligned reference sequence.
            query_aligned (str): Aligned query sequence.

        Returns:
            str: Query sequence with insertions removed.
        """
        return ''.join(char for i, char in enumerate(query_aligned) if reference_aligned[i] != '-')

    def _align_sequences(self, query_sequence):
        """
        Aligns a query sequence to the reference sequence, removing insertions.

        Args:
            query_sequence (str): The sequence to align.

        Returns:
            list: Aligned sequence without insertions and its length.
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = 0
        aligner.open_gap_score = aligner.extend_gap_score = -2
        alignment = next(aligner.align(self.ref_seq, query_sequence))
        query_aligned_no_ins = self._remove_insertions(*alignment)
        return [query_aligned_no_ins, len(query_aligned_no_ins)]

    # def _generate_mutation_names_all(self):
    #     """Generates mutation names for all sequences, considering only 2 and 3 nucleotide changes."""
    #     self.data['codon_mut_ls'] = self.data['nt_2_mut'] + self.data['nt_3_mut']
    #     self.data['codon_mutation'] = self.data['codon_mut_ls'].apply(self._generate_mutation_name)
    
    def _compare_codon_to_ref(self, sequence):
        """
        Compares codons in a sequence to the reference sequence, ignoring deletions within codons.

        Args:
            sequence (str): The sequence to compare.

        Returns:
            tuple: Dictionary of mutation counts and dictionary of mutation details.
        """
        ref_codon_seq = [self.ref_seq[i:i+3] for i in range(0, len(self.ref_seq), 3)]
        codon_seq = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        
        muts = [0, 0, 0, 0]
        seq_mutations = [[], [], [], [], '']
        for pos, (codon, ref_codon) in enumerate(zip(codon_seq, ref_codon_seq), 1):
            mismatches = sum(c1 != c2 for c1, c2 in zip(codon, ref_codon))
            if mismatches:
                seq_mutations[mismatches].append(ref_codon + str(pos) + codon)
            muts[mismatches] += 1
        return muts + seq_mutations