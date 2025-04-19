import os
import glob
import pandas as pd
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Merge multiple scRNA-seq CSV files from a directory into a single H5 file')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files to merge')
    parser.add_argument('--output_file', type=str, required=True, help='Output H5 file path')
    parser.add_argument('--cell_id_col', type=int, default=0, help='Column index for cell ID (default: 0)')
    parser.add_argument('--barcode_col', type=int, default=1, help='Column index for barcode (default: 1)')
    parser.add_argument('--label_col', type=int, default=2, help='Column index for cell type labels (default: 2)')
    parser.add_argument('--gene_start_col', type=int, default=3, help='Starting column index for gene expression (default: 3)')
    parser.add_argument('--file_pattern', type=str, default='*.csv', help='Pattern to match CSV files (default: "*.csv")')
    parser.add_argument('--encode_labels', action='store_true', help='Whether to encode labels as integers (default: False)')
    parser.add_argument('--gene_handling', type=str, choices=['intersection', 'union'], default='intersection', 
                        help='How to handle different gene sets: keep only common genes (intersection) or all genes (union)')
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist")
        return
    
    # Find all CSV files in the directory
    file_pattern = os.path.join(args.input_dir, args.file_pattern)
    csv_files = glob.glob(file_pattern)
    
    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {args.input_dir} matching pattern {args.file_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")
    
    # First pass: collect all gene names and check for differences
    print("Analyzing gene sets across files...")
    all_gene_sets = []
    file_gene_counts = {}  # Dictionary to store gene counts for each file
    
    for file_path in tqdm(csv_files, desc="Scanning gene sets"):
        df = pd.read_csv(file_path)
        genes = df.columns[args.gene_start_col:].tolist()
        gene_count = len(genes)
        file_gene_counts[os.path.basename(file_path)] = gene_count
        all_gene_sets.append(set(genes))
    
    # Print gene counts for each file
    print("\nGene counts per file:")
    for file_name, count in sorted(file_gene_counts.items()):
        print(f"  {file_name}: {count} genes")
    
    # Find files with the most and least genes
    max_gene_file = max(file_gene_counts.items(), key=lambda x: x[1])
    min_gene_file = min(file_gene_counts.items(), key=lambda x: x[1])
    print(f"\nFile with most genes: {max_gene_file[0]} ({max_gene_file[1]} genes)")
    print(f"File with least genes: {min_gene_file[0]} ({min_gene_file[1]} genes)")
    
    # Calculate some statistics
    gene_counts = list(file_gene_counts.values())
    avg_genes = sum(gene_counts) / len(gene_counts)
    median_genes = sorted(gene_counts)[len(gene_counts) // 2]
    print(f"Average genes per file: {avg_genes:.2f}")
    print(f"Median genes per file: {median_genes}")
    
    # Check if all files have the same number of genes
    if len(set(gene_counts)) == 1:
        print("All files have the same number of genes.")
    else:
        print("Files have different numbers of genes.")
    
    # Find union and intersection of gene sets
    union_genes = set.union(*all_gene_sets)
    intersection_genes = set.intersection(*all_gene_sets)
    
    print(f"\nTotal unique genes across all files: {len(union_genes)}")
    print(f"Genes common to all files: {len(intersection_genes)}")
    print(f"Genes in union but not in intersection: {len(union_genes) - len(intersection_genes)}")
    
    # Determine which genes to keep based on user preference
    if args.gene_handling == 'intersection':
        final_genes = sorted(list(intersection_genes))
        print(f"Using intersection of gene sets ({len(final_genes)} genes)")
    else:  # union
        final_genes = sorted(list(union_genes))
        print(f"Using union of gene sets ({len(final_genes)} genes)")
    
    # Create gene index mapping for efficient lookups
    gene_indices = {gene: idx for idx, gene in enumerate(final_genes)}
    
    # Initialize lists to store data
    all_cells = []
    all_barcodes = []
    all_labels = []
    
    # Pre-allocate expression matrix
    total_cells = 0
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        total_cells += len(df)
    
    gene_expression_matrix = np.zeros((total_cells, len(final_genes)), dtype=np.float32)
    
    # Second pass: process each file and extract data
    print("\nProcessing files and extracting data...")
    current_row = 0
    for file_path in tqdm(csv_files, desc="Processing files"):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            num_cells = len(df)
            
            # Extract cell IDs, barcodes, and labels
            cell_ids = df.iloc[:, args.cell_id_col].values
            barcodes = df.iloc[:, args.barcode_col].values
            labels = df.iloc[:, args.label_col].values
            
            # Append to metadata lists
            all_cells.extend(cell_ids)
            all_barcodes.extend(barcodes)
            all_labels.extend(labels)
            
            # Extract gene expression data - this part is improved with tqdm
            file_genes = df.columns[args.gene_start_col:].tolist()
            gene_progress = tqdm(range(num_cells), desc=f"Processing cells in {os.path.basename(file_path)}", leave=False)
            
            for cell_idx in gene_progress:
                for gene_idx, gene in enumerate(file_genes):
                    if gene in gene_indices:
                        gene_expression_matrix[current_row + cell_idx, gene_indices[gene]] = df.iloc[cell_idx, args.gene_start_col + gene_idx]
                # Update progress description with current cell number
                gene_progress.set_description(f"Cell {cell_idx+1}/{num_cells} in {os.path.basename(file_path)}")
            
            current_row += num_cells
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Verify the data was correctly processed
    print(f"Processed {current_row} cells")
    
    # Trim the matrix if we didn't use all pre-allocated rows
    if current_row < total_cells:
        gene_expression_matrix = gene_expression_matrix[:current_row, :]
    
    # Convert lists to numpy arrays
    all_cells = np.array(all_cells)
    all_barcodes = np.array(all_barcodes)
    all_labels = np.array(all_labels)
    
    X = gene_expression_matrix
    
    print(f"Combined data shape: {X.shape}")
    print(f"Number of cells: {len(all_cells)}")
    
    # Handle labels based on user preference
    if args.encode_labels:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        Y = encoder.fit_transform(all_labels)
        print("Labels encoded as integers")
        
        # Create mapping for reference
        label_mapping = {i: label for i, label in enumerate(encoder.classes_)}
        print("Label mapping:")
        for i, label in sorted(label_mapping.items()):
            print(f"  {i}: {label}")
    else:
        # Store original labels for sc-DMAE to encode later
        # First check if all labels are already integers
        try:
            Y = np.array([int(label) for label in all_labels])
            print("Labels already in integer format, using as-is")
        except ValueError:
            # If not integers, store as strings and let sc-DMAE handle encoding
            Y = all_labels
            print("Labels kept in original format for sc-DMAE to encode")
    
    # Print label statistics
    unique_labels = np.unique(all_labels)
    label_counts = {label: np.sum(all_labels == label) for label in unique_labels}
    print(f"Found {len(unique_labels)} unique cell types:")
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count} cells")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to H5 file
    print(f"Saving to {args.output_file}...")
    with h5py.File(args.output_file, 'w') as h5file:
        # Save main datasets
        h5file.create_dataset('X', data=X)
        
        # Save Y data appropriately based on its type
        if isinstance(Y[0], (np.integer, int)) or (isinstance(Y[0], (str, np.str_)) and Y[0].isdigit()):
            # If Y contains integers or strings that can be interpreted as integers
            Y_final = np.array(Y, dtype=np.int64)
            h5file.create_dataset('Y', data=Y_final)
        else:
            # If Y contains non-integer strings, store as string array
            # Note: sc-DMAE will encode these later
            Y_str = np.array(Y, dtype='S')
            h5file.create_dataset('Y', data=Y_str)
        
        # Save original labels separately for reference
        original_labels = np.array(all_labels, dtype='S')
        h5file.create_dataset('original_labels', data=original_labels)
        
        # Save gene names
        gene_names_array = np.array(final_genes, dtype='S')
        h5file.create_dataset('gene_names', data=gene_names_array)
        
        # Save cell IDs and barcodes
        cell_ids_array = np.array(all_cells, dtype='S')
        h5file.create_dataset('cell_ids', data=cell_ids_array)
        
        barcode_array = np.array(all_barcodes, dtype='S')
        h5file.create_dataset('barcodes', data=barcode_array)
        
        # Save file gene counts
        dt = h5py.special_dtype(vlen=str)
        file_count_group = h5file.create_group('file_gene_counts')
        for i, (file_name, count) in enumerate(file_gene_counts.items()):
            file_count_group.attrs[file_name] = count
        
        # Save label mapping if we encoded the labels
        if args.encode_labels and 'label_mapping' in locals():
            dt = h5py.special_dtype(vlen=str)
            mapping_dataset = h5file.create_dataset('label_mapping', (len(label_mapping),), dtype=dt)
            for i, label in label_mapping.items():
                mapping_dataset[i] = str(label)
    
    end_time = time.time()
    print("Conversion complete!")
    print(f"Created H5 file with {X.shape[0]} cells and {X.shape[1]} genes")
    print(f"Output file saved to: {args.output_file}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()