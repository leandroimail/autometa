#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate random samples from NYC taxi data Parquet files.
Creates sample files with different sample sizes (100, 1000, 10000) for each input file.
"""

import os
import pandas as pd
import random
import yaml
from typing import List, Tuple, Optional
from pathlib import Path


def create_sample(
    input_file: str, 
    sample_sizes: List[int] = [100, 1000, 10000],
    output_dir: Optional[str] = None,
    random_seed: int = 42
) -> List[str]:
    """
    Creates random samples from a Parquet file with specified sample sizes.
    
    Args:
        input_file: Path to the input Parquet file.
        sample_sizes: List of sample sizes to generate.
        output_dir: Directory to save the sample files (defaults to same directory as input).
        random_seed: Random seed for reproducibility.
        
    Returns:
        List of paths to the created sample files.
    """
    print(f"Processing file: {input_file}")
    
    # Set the random seed for reproducibility
    random.seed(random_seed)
    
    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the input file path
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # Create list to store output file paths
    output_files = []
    
    try:
        # For large files, we'll use efficient data loading techniques
        # First, get the number of rows in the file without loading it all
        df_info = pd.read_parquet(input_file, columns=[])
        row_count = len(df_info)
        print(f"Total rows in {base_name}: {row_count:,}")
        
        for sample_size in sample_sizes:
            # Skip if requested sample is larger than the dataset
            if sample_size >= row_count:
                print(f"Warning: Requested sample size {sample_size} is >= total rows {row_count}. Skipping.")
                continue
                
            # Create output file path
            output_file = os.path.join(output_dir, f"{base_name}_sample_{sample_size}.parquet")
            
            print(f"Creating sample with {sample_size} rows...")
            
            # Approach 1: For smaller files, load and sample
            if row_count < 1_000_000 or sample_size > row_count * 0.1:
                print(f"Using direct sampling approach (file is relatively small or sample is large)")
                # Load the dataframe and sample directly
                df = pd.read_parquet(input_file)
                sample_df = df.sample(n=sample_size, random_state=random_seed)
            
            # Approach 2: For larger files, use a more memory-efficient approach
            else:
                print(f"Using memory-efficient approach for large file")
                
                # Calculate fraction to ensure we get enough rows
                # We'll oversample slightly to account for possible duplicates
                fraction = min(sample_size * 2.0 / row_count, 0.5)  
                
                # For very large files, we may need multiple passes to avoid memory issues
                if row_count > 10_000_000 and sample_size < 1000:
                    # Take a fraction of the total rows using sparse sampling
                    # First we'll determine the fraction to read
                    fraction = sample_size / row_count
                    print(f"Using sparse sampling with fraction: {fraction:.8f}")
                    
                    # Generate random row indices to select
                    indices = sorted(random.sample(range(row_count), sample_size))
                    
                    # Read the parquet file in chunks
                    chunk_size = 1_000_000  # Read in 1M row chunks
                    sample_dfs = []
                    
                    for i in range(0, row_count, chunk_size):
                        # Check if any indices are in this chunk
                        chunk_indices = [idx for idx in indices if i <= idx < i + chunk_size]
                        if not chunk_indices:
                            continue
                        
                        # Read the chunk
                        chunk_start = i
                        chunk_end = min(i + chunk_size, row_count)
                        
                        # Mapping from global indices to chunk indices
                        local_indices = [idx - chunk_start for idx in chunk_indices]
                        
                        # Read the chunk
                        print(f"Reading chunk {i//chunk_size + 1}/{(row_count+chunk_size-1)//chunk_size}")
                        chunk = pd.read_parquet(input_file, 
                                               engine='pyarrow', 
                                               memory_map=True)
                        
                        # Select the rows
                        if local_indices:
                            chunk_sample = chunk.iloc[local_indices]
                            sample_dfs.append(chunk_sample)
                    
                    # Combine all sampled chunks
                    sample_df = pd.concat(sample_dfs, ignore_index=True)
                else:
                    # For moderately large files, we can use pandas sampling
                    print(f"Using random sampling with fraction: {fraction:.6f}")
                    df = pd.read_parquet(input_file)
                    # Sample more rows than needed to be safe
                    sample_df = df.sample(frac=fraction, random_state=random_seed)
                    
                    # If we have more rows than needed, take exactly what we need
                    if len(sample_df) > sample_size:
                        sample_df = sample_df.sample(n=sample_size, random_state=random_seed)
            
            # Ensure we have the right amount of data
            if len(sample_df) != sample_size:
                print(f"Warning: Got {len(sample_df)} rows, expected {sample_size}")
                
                # If we have more rows than needed, take exact sample
                if len(sample_df) > sample_size:
                    sample_df = sample_df.head(sample_size)
            
            # Save the sample to a new Parquet file
            print(f"Saving {len(sample_df)} rows to: {output_file}")
            sample_df.to_parquet(output_file, index=False)
            output_files.append(output_file)
            print(f"Sample saved to: {output_file}")
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        raise
        
    return output_files


def process_files(
    file_paths: List[str], 
    sample_sizes: List[int] = [100, 1000, 10000],
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Process multiple Parquet files and create samples for each.
    
    Args:
        file_paths: List of paths to the Parquet files.
        sample_sizes: List of sample sizes to generate.
        output_dir: Directory to save the sample files.
        
    Returns:
        List of paths to the created sample files.
    """
    all_output_files = []
    
    for file_path in file_paths:
        try:
            print(f"\n{'=' * 50}")
            output_files = create_sample(file_path, sample_sizes, output_dir)
            all_output_files.extend(output_files)
        except Exception as e:
            import traceback
            print(f"Failed to process file {file_path}: {str(e)}")
            traceback.print_exc()  # Print the full stack trace for debugging
    
    return all_output_files


def main() -> None:
    """Main function to generate samples from NYC taxi data files."""
    
    # Define the paths to the Parquet files
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    path_config = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    if not os.path.exists(path_config):
        raise FileNotFoundError(f"Configuration file not found: {path_config}")
    
    # Load the configuration file to get input file paths
    with open(path_config, 'r') as file:
        config = yaml.safe_load(file)
    input_files = config.get("list_full_data_samples", [])
    if not input_files:
        raise ValueError("No input files specified in the configuration file.")

    input_files = [os.path.join(file["path"]) for file in input_files]

    # Create a samples directory
    output_dir = os.path.join(data_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    print(f"Generating samples from {len(input_files)} files...")
    output_files = process_files(input_files, output_dir=output_dir)
    
    print("\nSample generation completed!")
    print(f"Created {len(output_files)} sample files in: {output_dir}")


if __name__ == "__main__":
    main()