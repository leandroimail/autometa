#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate data profiles for NYC taxi data Parquet files.
Uses the dataprofiler library to analyze files and generate reports.
"""

import os
import json
import yaml
from typing import List, Dict, Any
from dataprofiler import Data, Profiler


def generate_profile(file_path: str) -> Dict[str, Any]:
    """
    Generates a data profile for a file using dataprofiler.
    
    Args:
        file_path: Path to the file to be analyzed.
        
    Returns:
        Dictionary containing the profile report.
    """
    print(f"Generating profile for file: {file_path}")
    
    # Auto-detect and load the file (CSV, AVRO, Parquet, JSON, Text)
    try:
        data = Data(file_path)
        
        # Print the first 5 rows of data for visualization
        print("First 5 rows of data:")
        print(data.data.head(5))
        
        # Calculate statistics, entity recognition, etc.
        profile = Profiler(data)
        
        # Generate a compact report
        report = profile.report(report_options={"output_format": "compact"})
        
        return report
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise


def save_profile(profile: Dict[str, Any], file_path: str, output_dir: str = "profiles") -> None:
    """
    Saves the data profile to a JSON file.
    
    Args:
        profile: Dictionary containing the profile.
        file_path: Path to the original file.
        output_dir: Directory where the profile will be saved.
    """
    # Extract the base file name
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the full path of the output file
    output_file = os.path.join(output_dir, f"{base_name}_profile.json")
    
    print(f"Saving profile to: {output_file}")
    
    # Save the profile in JSON format
    with open(output_file, 'w') as f:
        json.dump(profile, f, indent=4)


def process_files(file_paths: List[str], output_dir: str = "profiles") -> None:
    """
    Processes a list of files, generating and saving profiles for each.
    
    Args:
        file_paths: List of paths to the files.
        output_dir: Directory where profiles will be saved.
    """
    for file_path in file_paths:
        try:
            print(f"\n{'=' * 50}")
            print(f"Processing file: {file_path}")
            
            # Generate the profile
            profile = generate_profile(file_path)
            
            # Save the profile to a JSON file
            save_profile(profile, file_path, output_dir)
            
            print(f"Profile successfully generated for: {os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"Failed to process file {file_path}: {str(e)}")


def main() -> None:
    """
    Main function that executes the profile generation process.
    """
    # Project base directory (one level above src directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data directory
    data_dir = os.path.join(base_dir, "data")
    
    # Directory to save profiles
    output_dir = os.path.join(data_dir, "profiles")
    
    path_config = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    if not os.path.exists(path_config):
        raise FileNotFoundError(f"Configuration file not found: {path_config}")
    
    # Load the configuration file to get input file paths
    with open(path_config, 'r') as file:
        config = yaml.safe_load(file)
    input_files = config.get("list_of_data_samples_profiles", [])
    if not input_files:
        raise ValueError("No input files specified in the configuration file.")

    input_files = [os.path.join(file["path"]) for file in input_files]

    
    print("Starting data profile generation...")
    process_files(input_files, output_dir)
    print("\nProcess completed. All profiles were generated.")
    print(f"Profiles were saved to: {output_dir}")


if __name__ == "__main__":
    main()