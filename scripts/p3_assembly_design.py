#!/usr/bin/env python3
"""
Script to design oligos using MultiAssemblyDesigner.

Example usage:

conda activate multievolve

p3_assembly_design.py \
--mutations-file multievolve_proposals.csv \
--wt-fasta APEX_33overhang.fasta \
--overhang 33 \
--species human \
--oligo-direction bottom \
--tm 80 \
--output design
"""

import argparse
import sys
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Design oligos using MultiAssemblyDesigner')
    
    parser.add_argument(
        '-m',
        '--mutations-file',
        required=True,
        help='Path to CSV file containing mutations (no header)'
    )
    
    parser.add_argument(
        '-wt',
        '--wt-fasta',
        required=True,
        help='Path to input FASTA file'
    )
    
    parser.add_argument(
        '-ov',
        '--overhang',
        type=int,
        default=33,
        help='Overhang length (default: 33)'
    )
    
    parser.add_argument(
        '-s',
        '--species',
        choices=['human', 'ecoli', 'yeast'],
        default='human',
        help='Species (default: human)'
    )
    
    parser.add_argument(
        '-d',
        '--oligo-direction',
        choices=['top', 'bottom'],
        default='bottom',
        help='Oligo direction (default: bottom)'
    )
    
    parser.add_argument(
        '--tm',
        type=float,
        default=80.0,
        help='Melting temperature (default: 80.0)'
    )

    parser.add_argument(
        '-o',
        '--output',
        choices=['design', 'update'],
        default='design',
        help='Output type (default: design)'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Import MultiAssemblyDesigner after setting up path
    try:
        from multievolve import MultiAssemblyDesigner
    except ImportError as e:
        print(f"Error importing MultiAssemblyDesigner: {e}")
        print("Make sure the src directory path is correct and contains the required module")
        sys.exit(1)
    
    # Read mutations file
    try:
        df = pd.read_csv(args.mutations_file, header=None)
    except Exception as e:
        print(f"Error reading mutations file: {e}")
        sys.exit(1)
    
    # Create designer instance
    try:
        designer = MultiAssemblyDesigner(
            df,
            args.wt_fasta,
            args.overhang,
            args.species,
            oligo_direction=args.oligo_direction,
            tm=args.tm,
            output=args.output
        )
    except Exception as e:
        print(f"Error creating MultiAssemblyDesigner instance: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()