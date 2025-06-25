## Usage

The workflow for the MULTI-evolve framework is as follows:
1. Train fully connected neural networks to predict the fitness of a given sequence.
2. Choose the best performing neural network and use it to predict combinatorial variants.
3. For the chosen multi-mutants, generate the MULTI-assembly mutagenic oligos for gene synthesis.

In certain iterations, the MULTI-evolve framework involves using a protein language model zeroshot ensemble approach to nominate single mutants to evaluate.

### Command-line

To run MULTI-evolve on the command line refer to the instructions below.

Run all the following steps in the folder containing the protein of interest.
```bash
conda activate multievolve
cd data/example_protein
```

#### Step 0: Set Up WandB:

Create a WandB account and get an API key, which is found in 'API keys' under 'User settings'.

#### Step 1: Train Neural Network Models:

The ```p1_train.py``` script takes a protein dataset and trains the neural network models to predict the fitness of a given sequence. The arguments are as follows:
- ```--experiment-name```: Name of the training experiment for a given protein and dataset. This should be used throughout the entire process.
- ```--protein-name```: Name of the protein.
- ```--wt-files```: Path to the FASTA file for the wildtype protein sequence. Can be a single file or a comma-separated list of files for a protein complex, which should be in the same order as how the variants are formatted in the dataset csv file  (e.g. ```chain1.fasta,chain2.fasta```).
- ```--training-dataset-fname```: Path to the training dataset CSV file. It should contain two columns for the mutation and associated the property value.
- ```--wandb-key```: WandB API key for authentication.
- ```--mode```: Training mode. Options are 'test' (to test the training process for a single architecture) and 'standard' (to perform a grid search over many architectures, will take a longer time to run).

The training dataset must be in CSV format with following columns (refer to the example ```data/example_protein/example_dataset.csv```):

- ```mutation```: Variants should be formatted as ```A40P/E61Y```, or for protein complexes as ```A40P/E61Y:WT```, where ```:``` separates the individual chains (e.g. ```chain 1 mutations:chain 2 mutations```), ```/``` separates the individual mutations, and ```WT``` indicates the wildtype sequence.
- ```property_value```: the property value

```bash
p1_train.py \
--experiment-name <name> \
--protein-name <name> \
--wt-files <fasta file> \
--training-dataset-fname <csv file> \
--wandb-key <your_wandb_key> \
--mode [test|standard]
```

#### Step 2: Propose MULTI-evolve Variants:

The ```p2_propose.py``` script identifies the best performing neural network model and uses it to propose mutations. The arguments are as follows:
- ```--experiment-name```: Name of the training experiment. This should be the same as the experiment name used in Step 1.
- ```--protein-name```: Name of the protein.
- ```--wt-files```: Path to the FASTA file for the wildtype protein sequence. Can be a single file or a comma-separated list of files (e.g. ```chain1.fasta,chain2.fasta```) for a protein complex.
- ```--training-dataset```: Path to the training dataset CSV file.
- ```--mutation-pool```: Path to the mutation pool CSV file, which is a list of mutations to be used to generate the proposed combinatorial variants. Example is provided in ```data/example_protein/combo_muts.csv```.
- ```--top-muts-per-load```: Number of variants to clone per mutational load (default: 3).
- ```--export-name```: Name of the exported file containing the proposed variants.

```bash
p2_propose.py \
--experiment-name <name> \
--protein-name <name> \
--wt-files <fasta file> \
--training-dataset <csv file> \
--mutation-pool <csv file> \
--top-muts-per-load <number of mutants> \
--export-name <name>
```

The ```p2_propose.py``` script will generate a CSV file (e.g. ```multievolve_proposals.csv```) containing the proposed variants. If it is a protein complex, it will export files for each chain (e.g. ```multievolve_proposals_chain_1_mutants.csv```)

#### Step 3: Generate MULTI-assembly Mutagenic Oligos:

The ```p3_assembly_design.py``` script generates the MULTI-assembly mutagenic oligos for the proposed variants from the ```p2_propose.py``` script. The arguments are as follows:
- ```--mutations-file```: Path to the proposed variants CSV file. This is the exported file generated in Step 2. See ```data/example_protein/MULTI-assembly_input.csv``` for an example of the csv format.
- ```--wt-fasta```: Path to the FASTA file for the DNA sequence of the wildtype protein, the sequence should include overhangs for the MULTI-assembly oligos, wherein the overhangs are the same length on both ends of the DNA sequence.
- ```--overhang```: Overhang length of the FASTA file of the wild-type sequence.
- ```--species```: Species for selecting the codon usage table for the MULTI-assembly oligos. Options are 'human', 'ecoli', and 'yeast'.
- ```--oligo-direction```: Direction of the MULTI-assembly oligos. This depends on the location of the nicking sites on the MULTI-assembly vector. Options are 'top' and 'bottom'. 'top' means the oligos bind in the top strand orientation or the 5' to 3' direction of the DNA sequence. 'bottom' means the oligos bind in the bottom strand orientation or the 3' to 5' direction of the DNA sequence.
- ```--tm```: Melting temperature (in Celsius)of the MULTI-assembly oligos. Recommended value is 80.
- ```--output```: Output type. Options are 'design' (to design the oligos) and 'update' (to update the existing oligos).

```bash
p3_assembly_design.py \
--mutations-file <csv file> \
--wt-fasta <fasta file> \
--overhang <length of overhang> \
--species [human|ecoli|yeast] \
--oligo-direction [top|bottom] \
--tm <melting temperature> \
--output [design|update]
```

The ```p3_assembly_design.py``` script will generate two CSV files (```cloning_sheet.csv``` and ```oligos.csv```). The 'oligo_id' in the ```oligos.csv``` file can be updated with your own ids. By rerunning the command with the ```--output``` flag set as 'update', the script will update the 'oligo_id' in the ```cloning_sheet.csv``` file to match the updated 'oligo_id' in the ```oligos.csv``` file.

#### Test Environment and Code for Steps 0-3

We recommend running the commands below to ensure that the code and environment is working correctly.

Make sure to replace ```<your_wandb_key>``` with your actual WandB API key.

```bash
conda activate multievolve
cd data/example_protein

p1_train.py \
--experiment-name multievolve_example \
--protein-name example_protein \
--wt-files apex.fasta \
--training-dataset-fname example_dataset.csv \
--wandb-key <your_wandb_key> \
--mode test
```

```bash
p2_propose.py \
--experiment-name multievolve_example \
--protein-name example_protein \
--wt-files apex.fasta \
--training-dataset example_dataset.csv \
--mutation-pool combo_muts.csv \
--top-muts-per-load 3 \
--export-name multievolve_proposals
```

```bash
p3_assembly_design.py \
--mutations-file multievolve_proposals.csv \
--wt-fasta APEX_33overhang.fasta \
--overhang 33 \
--species human \
--oligo-direction top \
--tm 80 \
--output design
```

#### Protein Language Model Zeroshot Ensemble

To perform the protein language model zeroshot ensemble approach implemented in MULTI-evolve, use the ```plm_zeroshot_ensemble.py``` script. An example is provided below. The arguments are as follows:
- ```--wt-file```: Path to the FASTA file for the wildtype protein sequence.
- ```--pdb-files```: Path to the PDB/CIF structure file. Can be a single file or a comma-separated list of files (e.g. ```model_0.cif,model_1.cif```).
- ```--variants```: Number of variants to nominate per method in the ensemble. Note, there are a total of 4 methods in the ensemble.
- ```--excluded-positions```: Comma-separated list of positions to exclude from mutation (e.g. ```1,5,20```). If no positions are to be excluded, don't include this argument.
- ```--normalizing-method```: Method for normalizing fold-change scores to generate z-scores. Options for grouping scores for normalization are 'aa_switch' and 'new_aa'. 'aa_switch' refers to grouping scores by the specific amino acid switch (e.g. A->L). 'new_aa' refers to grouping scores by the new amino acid provided by the respective mutation (e.g. if the mutation is A10P, it is grouped with all P (Proline) mutations).

```bash
conda activate multievolve
cd data/example_protein

plm_zeroshot_ensemble.py \
--wt-file apex.fasta \
--pdb-files apex.cif \
--variants 24 \
--excluded-positions 1,14,41,112 \
--normalizing-method aa_switch
```

The script will output a CSV file (```plm_zeroshot_ensemble_nominated_mutations.csv```) containing a list of the proposed variants and the methods that nominated those variants.