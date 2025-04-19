import os
import glob
import pandas as pd
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from functools import partial
import re
import time

'''
This file is used to convert multiple CSV files that contain only gene expressions
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Convert multiple CSV files to a single H5 file for scRNA-seq analysis using multiprocessing')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--output_file', type=str, required=True, help='Output H5 file path')
    parser.add_argument('--cell_type_pattern', type=str, default=None, 
                        help='Regex pattern to extract cell type from filename')
    parser.add_argument('--has_headers', action='store_true', help='Whether CSV files have headers')
    parser.add_argument('--processes', type=int, default=None, 
                        help='Number of processes to use, defaults to CPU core count')
    parser.add_argument('--chunk_size', type=int, default=1, 
                        help='Batch size of files to process per process')
    return parser.parse_args()

def process_csv_file(csv_file, gene_names, has_headers, cell_type_pattern):
    """Process a single CSV file and return expression vector and metadata"""
    
    # Extract cell ID from filename
    cell_id = os.path.basename(csv_file).split('.')[0]
    
    # Try to extract cell type from filename if pattern is provided
    if cell_type_pattern:
        match = re.search(cell_type_pattern, cell_id)
        cell_type = match.group(1) if match else "unknown"
    else:
        # If no pattern, use the directory name as cell type
        cell_type = os.path.basename(os.path.dirname(csv_file))
    
    try:
        # Read the CSV file
        if has_headers:
            df = pd.read_csv(csv_file)
            # Assuming each CSV has the same columns/genes
            expression_vector = df.iloc[0].values  # Take the first row
        else:
            df = pd.read_csv(csv_file, header=None)
            # Assuming first column is gene names and second column is expression values
            # We need to map gene indices correctly
            expression_vector = np.zeros(len(gene_names), dtype=np.float32)
            gene_indices = {gene: idx for idx, gene in enumerate(gene_names)}
            for _, row in df.iterrows():
                gene_name = row.iloc[0]
                expression_value = row.iloc[1]
                if gene_name in gene_indices:
                    expression_vector[gene_indices[gene_name]] = expression_value
        
        return {
            'cell_id': cell_id,
            'cell_type': cell_type,
            'expression': expression_vector
        }
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        # Return empty vector as fallback
        return {
            'cell_id': cell_id,
            'cell_type': cell_type,
            'expression': np.zeros(len(gene_names), dtype=np.float32)
        }

def process_csv_batch(file_batch, gene_names, has_headers, cell_type_pattern):
    """Process a batch of CSV files"""
    results = []
    for csv_file in file_batch:
        result = process_csv_file(csv_file, gene_names, has_headers, cell_type_pattern)
        results.append(result)
    return results

def main():
    start_time = time.time()
    args = parse_args()
    
    # Set up number of processes
    num_processes = args.processes if args.processes else mp.cpu_count()
    print(f"Using {num_processes} processes")
    
    # Step 1: Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("No CSV files found in the input directory")
        return
    
    # Step 2: Read the first file to get gene names (assuming all files have the same genes)
    print("Reading first file to identify genes...")
    if args.has_headers:
        first_file = pd.read_csv(csv_files[0])
        gene_names = first_file.columns.tolist()
    else:
        # If no headers, read the first column as gene names
        first_file = pd.read_csv(csv_files[0], header=None)
        gene_names = first_file.iloc[:, 0].tolist()
    
    # Step 3: Split files into batches
    batch_size = args.chunk_size
    file_batches = [csv_files[i:i+batch_size] for i in range(0, len(csv_files), batch_size)]
    print(f"Splitting {len(csv_files)} files into {len(file_batches)} batches for processing")
    
    # Step 4: Set up multiprocessing pool and process each batch
    print("Starting parallel processing of CSV files...")
    process_func = partial(process_csv_batch, 
                          gene_names=gene_names, 
                          has_headers=args.has_headers, 
                          cell_type_pattern=args.cell_type_pattern)
    
    with mp.Pool(processes=num_processes) as pool:
        # Use imap to process batches and show progress
        all_results = []
        for batch_results in tqdm(pool.imap(process_func, file_batches), 
                                  total=len(file_batches)):
            all_results.extend(batch_results)
    
    # Step 5: Organize the data
    print("Consolidating data...")
    cell_count = len(all_results)
    gene_count = len(gene_names)
    
    # Create expression matrix
    expression_matrix = np.zeros((cell_count, gene_count), dtype=np.float32)
    cell_ids = []
    cell_types = []
    
    for i, result in enumerate(all_results):
        expression_matrix[i] = result['expression']
        cell_ids.append(result['cell_id'])
        cell_types.append(result['cell_type'])
    
    # Step 6: Encode cell types as integers
    print("Encoding cell types...")
    encoder = LabelEncoder()
    cell_type_labels = encoder.fit_transform(cell_types)
    
    # Print statistics
    unique_cell_types = np.unique(cell_types)
    print(f"Found {len(unique_cell_types)} unique cell types: {unique_cell_types}")
    
    # Step 7: Save to H5 file
    print(f"Saving to {args.output_file}...")
    with h5py.File(args.output_file, 'w') as h5file:
        h5file.create_dataset('X', data=expression_matrix)
        h5file.create_dataset('Y', data=cell_type_labels)
        
        # Create a dataset to store gene names
        gene_names_array = np.array(gene_names, dtype='S')
        h5file.create_dataset('gene_names', data=gene_names_array)
        
        # Create a dataset to store cell IDs
        cell_ids_array = np.array(cell_ids, dtype='S')
        h5file.create_dataset('cell_ids', data=cell_ids_array)
        
        # Store cell type mapping
        cell_type_mapping = np.array(list(zip(list(encoder.classes_), range(len(encoder.classes_)))), 
                                    dtype=[('cell_type', 'S50'), ('label', 'i4')])
        h5file.create_dataset('cell_type_mapping', data=cell_type_mapping)
    
    end_time = time.time()
    print("Conversion complete!")
    print(f"Created H5 file with shape: {expression_matrix.shape}")
    print(f"Output file saved to: {args.output_file}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()