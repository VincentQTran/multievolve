"""
Streamlit app for MULTI-evolve: A framework for engineering hyperactive multi-mutants.

This app provides an interactive web app to:
1. Train neural network models on protein mutation data
2. Propose optimized multi-mutant combinations
3. Generate MULTI-assembly mutagenic oligos for gene synthesis
4. Perform zero-shot predictions with protein language models
"""

import streamlit as st
import pandas as pd
from Bio import SeqIO
import os
import wandb
from pathlib import Path
import subprocess

from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *

def setup_page():
    """Configure basic Streamlit page settings"""
    st.set_page_config(
        page_title="MULTI-evolve",
        page_icon="🧬",
        layout="wide"
    )
    st.title("MULTI-evolve")

def create_protein_directory(protein_name):
    """
    Create directory structure for a protein project
    
    Args:
        protein_name (str): Name of the protein project
        
    Returns:
        Path: Path object pointing to protein directory
    """
    protein_dir = Path("proteins") / protein_name
    protein_dir.mkdir(parents=True, exist_ok=True)
    
    return protein_dir

def save_uploaded_file(uploaded_file, protein_dir):
    """
    Save an uploaded file to the protein directory
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file
        protein_dir (Path): Path to protein directory
        
    Returns:
        Path: Path to saved file
    """

    if uploaded_file is None:
        return None
        
    save_path = protein_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def validate_files(protein_name, wt_files_aa=None, wt_file_aa=None,wt_file_dna=None, dataset_file=None, mutations_file=None, pdb_files=None):
    """Validate uploaded files with protein-specific directory handling"""
    try:
        protein_dir = create_protein_directory(protein_name)
        
        # Validate and save FASTA
        if wt_files_aa:
            for wt_file_aa in wt_files_aa:
                fasta_path = save_uploaded_file(wt_file_aa, protein_dir)
                wt_seq_aa = str(SeqIO.read(fasta_path, "fasta").seq.upper())
        elif wt_file_aa:
            fasta_path = save_uploaded_file(wt_file_aa, protein_dir)
            wt_seq_aa = str(SeqIO.read(fasta_path, "fasta").seq.upper())

        if wt_file_dna:
            fasta_path = save_uploaded_file(wt_file_dna, protein_dir)
            wt_seq_dna = str(SeqIO.read(fasta_path, "fasta").seq.upper())

        # Validate and save dataset CSV
        if dataset_file:
            dataset_path = save_uploaded_file(dataset_file, protein_dir)
            df = pd.read_csv(dataset_path)
            required_cols = ['mutation', 'property_value']
            if not all(col in df.columns for col in required_cols):
                st.error("Training dataset must contain 'mutation' and 'property_value' columns")
                return False
                
        # Validate and save mutation pool
        if mutations_file:
            pool_path = save_uploaded_file(mutations_file, protein_dir)
            df = pd.read_csv(pool_path, header=None)
            if df.empty:
                st.error("Mutation pool file is empty")
                return False
            
        if pdb_files:
            for pdb_file in pdb_files:
                pdb_path = save_uploaded_file(pdb_file, protein_dir)
                if not str(pdb_path).endswith('.pdb') and not str(pdb_path).endswith('.cif'):
                    st.error("PDB/CIF files must be in PDB or CIF format")
                    return False
                
        return True
        
    except Exception as e:
        st.error(f"File validation error: {str(e)}")
        return False

def train_models():
    """Train neural network models section"""
    # st.header("1. Train Neural Network Models")
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        protein_name = st.text_input("Protein Name")
        wt_files_aa = st.file_uploader("Upload Wildtype Amino Acid Sequence FASTA", accept_multiple_files=True, type=['fasta', 'fa'])
        dataset_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'], accept_multiple_files=False)
        experiment_name = st.text_input("Experiment Name", value="test")
        wandb_key = st.text_input("WandB API Key", type="password")
        mode = st.selectbox("Training Mode", ["test", "standard"])
        
    with col2:
        st.markdown("""
        ### Step 1: Train Neural Network Models
        
        This tool performs a grid search over many neural network architectures to find the best performing model for a given protein and dataset.
        
        #### Input Files and Parameters
        
        - **Training Dataset (CSV)**: CSV file with columns 'mutation' and 'property_value'. Variants should be formatted as ```A40P/E61Y```, or for protein complexes as ```A40P/E61Y:WT```, where ```:``` separates the individual chains (e.g. ```chain 1 mutations:chain 2 mutations```), ```/``` separates the individual mutations, and ```WT``` indicates the wildtype sequence. A sample dataset for APEX peroxidase can be found in ```data/example_protein/example_dataset.csv```. For a protein complex example, use ```data/example_multichain_protein/example_dataset.csv```.
        - **Wildtype Amino Acid Sequence FASTA**: Protein sequence in FASTA format. Upload multiple sequence files if working with a protein complex in the same order as you formatted the variants in the training dataset. A sample sequence of APEX peroxidase can be found in ```data/example_protein/apex.fasta```. For a protein complex example, upload in the following order: ```data/example_multichain_protein/vh_chain1.fasta```, ```data/example_multichain_protein/vl_chain2.fasta```.
        - **Experiment Name**: Name of the model training experiment (e.g. APEX_gridsearch). This should be used for the subsequent step 2 for proposing mutations.
        - **WandB API Key**: API key for logging to WandB. Create an account and get an API key from [WandB](https://wandb.ai/authorize).
        - **Training Mode**: 
            - `test`: Test the training process for a single architecture.
            - `standard`: Performs a grid search over many architectures. Will take a longer time to run.
        """)
        
    if st.button("Train Models"):
        if not all([experiment_name, protein_name, wandb_key, wt_files_aa, dataset_file]):
            st.error("Please fill in all required fields")
            return
            
        if not validate_files(protein_name, wt_files_aa=wt_files_aa, dataset_file=dataset_file):
            return
            
        try:
            protein_dir = Path("proteins") / protein_name
            wt_paths = [protein_dir / wt_file_aa.name for wt_file_aa in wt_files_aa]
            dataset_path = protein_dir / dataset_file.name
            
            # Force wandb relogin
            subprocess.run(["wandb", "login", "--relogin", wandb_key], capture_output=True)
            
            # Show the command that will be executed
            command = [
                "python", "scripts/p1_train.py",
                "--experiment-name", experiment_name,
                "--protein-name", protein_name,
                "--wt-files", ",".join(str(wt_path) for wt_path in wt_paths),
                "--training-dataset-fname", str(dataset_path),
                "--wandb-key", wandb_key,
                "--mode", mode
            ]
            
            st.subheader("Terminal Output:")
            # Add CSS styling for scrollable container
            st.markdown("""
                <style>
                    .terminal-container {
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        background-color: #f0f0f0;
                        font-family: monospace;
                        white-space: pre-wrap;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Show the command in a separate code block
            st.code(f"$ {' '.join(command)}", language="bash")
            
            # Create container for scrollable output
            with st.container():
                terminal_output = st.empty()
            
            with st.spinner("Training models..."):
                # Run the command and capture all output
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                output_lines = []
                
                # Stream output in real-time
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip())
                    # Update the terminal display with all output so far
                    terminal_text = '\n'.join(output_lines)
                    terminal_output.markdown(f'<div class="terminal-container"><pre><code>{terminal_text}</code></pre></div>', 
                                          unsafe_allow_html=True)
                
                process.wait()
                
                # Keep the final scrollable output instead of replacing with code block
                final_output = '\n'.join(output_lines)
                terminal_output.markdown(f'<div class="terminal-container"><pre><code>{final_output}</code></pre></div>', 
                                      unsafe_allow_html=True)
                
                if process.returncode == 0:
                    st.success("✅ Model training completed successfully!")
                else:
                    st.error(f"❌ Training failed with exit code: {process.returncode}")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.exception(e)

def propose_mutations():
    """Propose mutations section"""
    # st.header("2. Propose MULTI-evolve Variants")
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        protein_name = st.text_input("Protein Name", key="propose_protein")
        wt_files_aa = st.file_uploader("Upload Wildtype Amino Acid Sequence FASTA", accept_multiple_files=True, type=['fasta', 'fa'], key="propose_wt")
        dataset_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'], key="propose_dataset")
        mutation_pool = st.file_uploader("Upload Mutation Pool (CSV)", type=['csv'])
        experiment_name = st.text_input("Experiment Name", key="propose_exp")
        top_muts = st.number_input("Top Mutations per Load", min_value=1, value=3)
        export_name = st.text_input("Export Name", value="multievolve_proposals")
        
    with col2:
        st.markdown("""
        ### Step 2: Propose MULTI-evolve Variants
        
        This tool proposes MULTI-evolve variants using a trained neural network model, whose ideal architecture is selected from a grid search in Step 1.

        #### Input Files and Parameters
        
        - **Wildtype Amino Acid Sequence FASTA**: Protein sequence in FASTA format. Upload multiple sequence files if working with a protein complex. Same file(s) as Step 1. A sample sequence of APEX peroxidase can be found in ```data/example_protein/apex.fasta```. For a protein complex example, upload in the following order: ```data/example_multichain_protein/vh_chain1.fasta```, ```data/example_multichain_protein/vl_chain2.fasta```.
        - **Training Dataset (CSV)**: CSV file with columns 'mutation' and 'property_value'. Same file as Step 1. A sample dataset for APEX peroxidase can be found in ```data/example_protein/example_dataset.csv```. For a protein complex example, use ```data/example_multichain_protein/example_dataset.csv```. 
        - **Mutation Pool (CSV)**: Path to the mutation pool CSV file, which is a list of mutations to be used to generate the proposed combinatorial variants. It is a one column no header CSV file. Example is provided in ```data/example_protein/combo_muts.csv```. For a protein complex example, use ```data/example_multichain_protein/combo_muts.csv```.
        - **Experiment Name**: Name of the model training experiment (e.g. APEX_gridsearch). Same experiment name as Step 1.
        - **Top Mutations per Load**: Number of top mutations to propose per mutational load.
        - **Export Name**: Name of the exported csv file containing the list of the proposed variants. This csv file can be used to generate MULTI-assembly mutagenic oligos for cloning the proposed variants in the ```Design MULTI-assembly Oligos``` tab.
       
        #### Outputs:
        
        A CSV file will be generated:
        - `<Export Name>.csv`: List of proposed variants. If it is a protein complex, it will export files for each chain (e.g. ```<Export Name>_chain_1_mutants.csv```)

                    
        """)
        
    if st.button("Propose Mutations"):
        if not all([experiment_name, protein_name, wt_files_aa, dataset_file, mutation_pool, export_name]):
            st.error("Please fill in all required fields")
            return
            
        if not validate_files(protein_name, wt_files_aa=wt_files_aa, dataset_file=dataset_file, mutations_file=mutation_pool):
            return
            
        try:
            protein_dir = Path("proteins") / protein_name
            wt_paths = [protein_dir / wt_file_aa.name for wt_file_aa in wt_files_aa]
            dataset_path = protein_dir / dataset_file.name
            mutation_pool_path = protein_dir / mutation_pool.name

            # Show the command that will be executed
            command = [
                "python", "scripts/p2_propose.py",
                "--experiment-name", experiment_name,
                "--protein-name", protein_name,
                "--wt-files", ",".join(str(wt_path) for wt_path in wt_paths),
                "--training-dataset", str(dataset_path),
                "--mutation-pool", str(mutation_pool_path),
                "--top-muts-per-load", str(top_muts),
                "--export-name", export_name
            ]

            st.subheader("Terminal Output:")
            # Add CSS styling for scrollable container
            st.markdown("""
                <style>
                    .terminal-container {
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        background-color: #f0f0f0;
                        font-family: monospace;
                        white-space: pre-wrap;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Show the command in a separate code block
            st.code(f"$ {' '.join(command)}", language="bash")
            
            # Create container for scrollable output
            with st.container():
                terminal_output = st.empty()

            with st.spinner("Proposing mutations..."):
                # Run the command and capture all output
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                output_lines = []
                
                # Stream output in real-time
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip())
                    # Update the terminal display with all output so far
                    terminal_text = '\n'.join(output_lines)
                    terminal_output.markdown(f'<div class="terminal-container"><pre><code>{terminal_text}</code></pre></div>', 
                                          unsafe_allow_html=True)
                    
                process.wait()
                
                # Keep the final scrollable output instead of replacing with code block
                final_output = '\n'.join(output_lines)
                terminal_output.markdown(f'<div class="terminal-container"><pre><code>{final_output}</code></pre></div>', 
                                      unsafe_allow_html=True)
                
                if process.returncode == 0:
                    st.success("✅ MULTI-evolve variants proposed successfully!")
                else:
                    st.error(f"❌ MULTI-evolve variants proposal failed with exit code: {process.returncode}")

        except Exception as e:
            st.error(f"Error during mutation proposal: {str(e)}")
            st.exception(e)

def design_oligos():
    """Design MULTI-assembly oligos section"""
    # st.header("Design MULTI-assembly Oligos")
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        protein_name = st.text_input("Protein Name", key="MULTI-assembly_protein")
        wt_file_dna = st.file_uploader("Upload Wildtype DNA Sequence FASTA", type=['fasta', 'fa'], key="oligo_wt")
        mutations_file = st.file_uploader("Upload Mutations File (CSV)", type=['csv'])
        species = st.selectbox("Species", ["human", "ecoli", "yeast"])
        tm = st.number_input("Melting Temperature (°C)", value=80.0)
        overhang = st.number_input("Overhang Length", value=33)
        oligo_direction = st.selectbox("Oligo Direction", ["top", "bottom"])
        output_type = st.selectbox("Output Type", ["design", "update"])
        
    with col2:
        # information from readme
        st.markdown("""
        ### Step 3: Generate MULTI-assembly Mutagenic Oligos
        
        This tool generates mutagenic oligos for MULTI-assembly cloning of multi-mutant variants.
        
        #### Input Files and Parameters:
                    
        - **Protein Name**: Name of the protein to generate oligos for.
        - **Wildtype DNA Sequence FASTA**: DNA sequence of the wildtype protein with overhangs from the protein's MULTI-assembly vector. The sequence should include overhangs for the MULTI-assembly oligos, wherein the overhangs are the same length on both ends of the DNA sequence. Recommended overhang length is 33 bp or longer. An example is found in ```data/example_protein/APEX_33overhang.fasta```.
        - **Mutations File (CSV)**: List of proposed variants to generate oligos for. It is a one column no-header csv file with the variants. See ```data/example_protein/MULTI-assembly_input.csv``` for an example of the csv format.
        - **Species**: Codon usage table selection (human/ecoli/yeast).
        - **Melting Temperature**: Target Tm for oligos (recommended: 80°C).
        - **Overhang Length**: Length of overhangs on both ends of sequence.
        - **Oligo Direction**: 
            - `top`: Oligos bind 5' to 3' in top strand orientation.
            - `bottom`: Oligos bind 3' to 5' in bottom strand orientation.
        - **Output Type**:
            - `design`: Generate new oligo designs.
            - `update`: Update existing oligo IDs.
        
        #### Outputs:
        
        Two CSV files will be generated:
        1. `cloning_sheet.csv`: Assembly instructions describing which oligos to pool for each variant.
        2. `oligos.csv`: Oligo sequences and IDs.
        
        The oligo IDs in `oligos.csv` can be customized with user-defined IDs. After editing the `oligos.csv` file, re-running with `update` will sync IDs between files.
        """)
        
    if st.button("Design Oligos"):
        if not all([protein_name, mutations_file, wt_file_dna]):
            st.error("Please fill in all required fields")
            return
            
        if not validate_files(protein_name, wt_file_dna=wt_file_dna, mutations_file=mutations_file):
            return
            
        try:
            protein_dir = Path("proteins") / protein_name
            mutations_path = protein_dir / mutations_file.name
            wt_path = protein_dir / wt_file_dna.name

            # Show the command that will be executed
            command = [
                "python", "scripts/p3_assembly_design.py",
                "--mutations-file", str(mutations_path),
                "--wt-fasta", str(wt_path), 
                "--overhang", str(overhang),
                "--species", species,
                "--oligo-direction", oligo_direction,
                "--tm", str(tm),
                "--output", output_type
            ]

            st.subheader("Terminal Output:")
            # Add CSS styling for scrollable container
            st.markdown("""
                <style>
                    .terminal-container {
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        background-color: #f0f0f0;
                        font-family: monospace;
                        white-space: pre-wrap;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Show the command in a separate code block
            st.code(f"$ {' '.join(command)}", language="bash")

            # Create container for scrollable output
            with st.container():
                terminal_output = st.empty()

            with st.spinner("Designing oligos..."):
                # Run the command and capture all output
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                output_lines = []
                
                # Stream output in real-time
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip())
                    # Update the terminal display with all output so far
                    terminal_text = '\n'.join(output_lines)
                    terminal_output.markdown(f'<div class="terminal-container"><pre><code>{terminal_text}</code></pre></div>', 
                                          unsafe_allow_html=True)
                
                process.wait()
                
                # Keep the final scrollable output instead of replacing with code block
                final_output = '\n'.join(output_lines)
                terminal_output.markdown(f'<div class="terminal-container"><pre><code>{final_output}</code></pre></div>', 
                                      unsafe_allow_html=True)
                
                if process.returncode == 0:
                    st.success("✅ Oligo design completed successfully!")
                else:
                    st.error(f"❌ Oligo design failed with exit code: {process.returncode}")

        except Exception as e:
            st.error(f"Error during oligo design: {str(e)}")
            st.exception(e)

def zeroshot_predictions():
    """Perform zero-shot predictions section"""
    # st.header("Perform Protein Language Model Zero-shot Ensemble")
    
    col1, col2 = st.columns([1,1])

    with col1:
        protein_name = st.text_input("Protein Name", key="zeroshot_protein")
        wt_file_aa = st.file_uploader("Upload Wildtype Amino Acid Sequence FASTA", type=['fasta', 'fa'], key="zeroshot_wt")
        pdb_files = st.file_uploader("Upload PDB/CIF Files", type=['pdb', 'cif'], accept_multiple_files=True, key="zeroshot_pdb")
        chain_id = st.text_input("Chain ID", value="A", key="zeroshot_chain")
        variants = st.number_input("Number of Variants", min_value=1, value=24)
        excluded_pos = st.text_input("Excluded Positions (comma-separated, optional)", value="1,10,30", key="zeroshot_excluded")
        norm_method = st.selectbox("Normalizing Method", ["aa_substitution_type", "aa_mutation"], key="zeroshot_norm")
    
    with col2:
        st.markdown("""
        ### MULTI-evolve: Protein Language Model Zero-shot Ensemble
                    
        This tool performs zero-shot predictions with a protein language model ensemble to nominate mutations.
        
        #### Input Files and Parameters
        
        - **Wildtype Amino Acid Sequence FASTA**: Protein sequence in FASTA format.
        - **PDB/CIF Files**: One or more structure files in PDB or CIF format. Provide multiple structure files if there are different models (e.g. top 5 predicted structures from AlphaFold).
        - **Chain ID**: Chain ID of the targeted protein in the structure files.
        - **Number of Variants**: Number of variants to nominate per method (default: 24)
        - **Excluded Positions**: Comma-separated list of positions to exclude from mutation (e.g. 1,5,20). Leave empty if no positions should be excluded.
        - **Normalizing Method**: Method for normalizing fold-change scores:
            - `aa_substitution_type`: Group by specific amino acid substitution type (e.g. all alanine to proline mutations, A→P mutations).
            - `aa_mutation`: Group by amino acid mutation (e.g. all mutations to proline, →P).
        
        #### Outputs:
        
        A CSV file will be generated:
        - `plm_zeroshot_ensemble_nominated_mutations.csv`: List of proposed variants and nominating methods.
        """)
        
    if st.button("Run Zero-shot Predictions"):
        if not all([protein_name, wt_file_aa, pdb_files, chain_id]):
            st.error("Please fill in all required fields")
            return
            
        if not validate_files(protein_name, wt_file_aa=wt_file_aa, pdb_files=pdb_files):
            return
            
        try:
            protein_dir = Path("proteins") / protein_name
            wt_path = protein_dir / wt_file_aa.name
            pdb_paths = [protein_dir / pdb_file.name for pdb_file in pdb_files]
            
            
            # Show the command that will be executed
            command = [
                "python", "scripts/plm_zeroshot_ensemble.py",
                "--wt-file", str(wt_path),
                "--pdb-files", ",".join(str(path) for path in pdb_paths),
                "--chain-id", chain_id,
                "--variants", str(variants),
                "--normalizing-method", norm_method
            ]
            
            # Only add excluded-positions flag if it's provided and not empty
            if excluded_pos.strip():
                command.extend(["--excluded-positions", excluded_pos])

            st.subheader("Terminal Output:")
            # Add CSS styling for scrollable container
            st.markdown("""
                <style>
                    .terminal-container {
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        background-color: #f0f0f0;
                        font-family: monospace;
                        white-space: pre-wrap;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Show the command in a separate code block
            st.code(f"$ {' '.join(command)}", language="bash")

            # Create container for scrollable output
            with st.container():
                terminal_output = st.empty()

            with st.spinner("Running Zero-shot Predictions..."):
                # Run the command and capture all output
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                output_lines = []
                
                # Stream output in real-time
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip())
                    # Update the terminal display with all output so far
                    terminal_text = '\n'.join(output_lines)
                    terminal_output.markdown(f'<div class="terminal-container"><pre><code>{terminal_text}</code></pre></div>', 
                                          unsafe_allow_html=True)
                
                process.wait()
                
                # Keep the final scrollable output instead of replacing with code block
                final_output = '\n'.join(output_lines)
                terminal_output.markdown(f'<div class="terminal-container"><pre><code>{final_output}</code></pre></div>', 
                                      unsafe_allow_html=True)
                
                if process.returncode == 0:
                    st.success("✅ Zero-shot predictions completed successfully!")
                else:
                    st.error(f"❌ Zero-shot predictions failed with exit code: {process.returncode}")

        except Exception as e:
            st.error(f"Error during zero-shot predictions: {str(e)}")
            st.exception(e)

def about():
    st.markdown("""
    
    This tool provides an interactive web app to perform the computational steps of MULTI-evolve (model-guided, universal, targeted installation of multi-mutants), an end-to-end framework for efficiently engineering hyperactive multi-mutants.
    
    The interactive web app has the following uses:
    1. Implement the MULTI-evolve framework to propose multi-mutants and generate the associated MULTI-assembly mutagenic oligos for gene synthesis:
                
        (a) Train fully connected neural networks to predict the fitness of a given sequence.
                
        (b) Choose the best performing neural network and use it to predict combinatorial variants.
                
        (c) For the chosen multi-mutants, generate the MULTI-assembly mutagenic oligos for gene synthesis.
                
    2. Perform the Protein Language Model Zero-shot Ensemble Approach used in the MULTI-evolve framework.
    """)

def main():
    """Main function to run the Streamlit app"""
    setup_page()
    
    # Create tabs for different functionalities
    tab5, tab1, tab2, tab3, tab4 = st.tabs([
        "About:",
        "Train Models", 
        "Propose Multi-mutants", 
        "Generate MULTI-assembly Oligos",
        "Perform PLM Zero-shot Ensemble",
    ])
    
    with tab5:
        about()

    with tab1:
        train_models()
        
    with tab2:
        propose_mutations()
        
    with tab3:
        design_oligos()
        
    with tab4:
        zeroshot_predictions()
        


if __name__ == "__main__":
    main()