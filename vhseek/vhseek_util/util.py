import os
import torch
import json
import pickle
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from logzero import logger
from vhseek_util.evaluation import LABEL_LEVELS, OTHER_LEVEL

def get_index_protein_dic(protein_list):
    return {index: protein for index, protein in enumerate(protein_list)}

def get_protein_index_dic(protein_list):
    return {protein: index for index, protein in enumerate(protein_list)}

def make_parent_dir(path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

def tensor_to_list(tensor):
    decimals = 4
    numpy_array = tensor.cpu().numpy()
    return np.round(numpy_array, decimals=decimals).tolist()

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - config_dict (dict): Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def update_args_from_config(args, config_dict):
    """
    Update argparse Namespace with values from config dictionary.

    Parameters:
    - args (argparse.Namespace): Parsed argparse arguments.
    - config_dict (dict): Dictionary containing configuration parameters.

    Returns:
    - args (argparse.Namespace): Updated argparse arguments.
    """
    for key, value in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            logger.warning(f"Ignoring unknown config parameter: {key}")
    return args

def load_label_index(label_index_path):
    """
    Load the label index from a file in the specified directory.
    Assumes each line in the file is formatted as "host_name\tindex".

    Parameters:
    - directory (str): The directory where the label index file is located.
                       Defaults to "vhseek_data/vhdb/host_label_index".

    Returns:
    - label_index_path (dict): A dictionary mapping each host label to its unique index.
    """
    label_index = {}
    file_path = Path(label_index_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label index file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Warning: Line {line_num} in {file_path} is malformed. Skipping.")
                continue
            host, idx_str = parts
            try:
                idx = int(idx_str)
            except ValueError:
                print(f"Warning: Invalid index on line {line_num} in {file_path}. Skipping.")
                continue
            label_index[host] = idx
    
    return label_index

def get_virus_name(virusprotein_fullname):
    # Assume that virusprotein_fullnames are in the format 'virusprotein_name virus_name'
    virus_name = None
    parts = virusprotein_fullname.split(' ')
    if len(parts) > 1:
        virus_name = ' '.join(parts[1:])
    else:
        assert("bad virus_name")
    return virus_name

def load_embeddings(embedding_path):
    """
    Load embeddings from a pickle file.

    Parameters:
    - embedding_path (str): Path to the embedding file.

    Returns:
    - embeddings (dict): Dictionary mapping sample IDs to embedding vectors.
    """
    if not os.path.exists(embedding_path):
        logger.error(f"Embedding file {embedding_path} not found!")
        raise FileNotFoundError(f"Embedding file {embedding_path} not found!")
    with open(embedding_path, 'rb') as handle:
        embeddings = pickle.load(handle)
    logger.info(f'Embeddings Loaded from {embedding_path}.')
    return embeddings

def merge_embedding(dna_embedding_file, protein_embedding_file, merged_embedding_path):
    """
    Merge two embedding dictionaries by concatenating their vectors,
    and immediately save the merged result to disk.

    Parameters:
    - dna_embedding_file (str): Path to second embedding pickle.
    - protein_embedding_file (str): Path to first embedding pickle.
    - merged_embedding_path (str): Path where the merged embedding pickle will be written.
    """
    # Load both embeddings
    dna_embedding = load_embeddings(dna_embedding_file)
    protein_embedding = load_embeddings(protein_embedding_file)

    # Ensure keys match
    keys1 = set(protein_embedding.keys())
    keys2 = set(dna_embedding.keys())
    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        raise ValueError(
            f"Key mismatch:\n"
            f"  Missing in second embedding: {missing_in_2}\n"
            f"  Missing in first embedding: {missing_in_1}"
        )

    # Concatenate embeddings
    merged = {}
    for key in protein_embedding:
        vec1 = protein_embedding[key]
        vec2 = dna_embedding[key]
        # Check both are 1D tensors
        if vec1.dim() != 1 or vec2.dim() != 1:
            raise ValueError(f"Embedding for {key} is not a 1D tensor")
        merged[key] = torch.cat([vec1, vec2], dim=0)

    # Sanity check
    sample = next(iter(merged))
    print(f"Total merged embeddings: {len(merged)}")
    print(f"Shape of merged vector for '{sample}': {merged[sample].shape}")

    # Directly save merged dict
    out_path = Path(merged_embedding_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(merged, f)
    logger.info(f"Merged embeddings saved to {merged_embedding_path}.")

def load_ground_truth(key_ids, key_host_dict_path, label_index):
    """
    Load virus-level ground truth labels.

    Parameters:
    - key_ids (list): List of virus identifiers.
    - key_host_dict_path (str): Path to the key-host annotation file.
    - label_index (dict): Mapping from host labels to indices.

    Returns:
    - ground_truth (numpy.ndarray): Binary matrix of shape [num_keys, num_labels].
    """
    # Initialize the binary matrix
    num_keys = len(key_ids)
    num_labels = len(label_index)
    ground_truth = np.zeros((num_keys, num_labels), dtype=int)

    # Create a mapping from key to its hosts
    key_to_hosts = {}
    with open(key_host_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                key = parts[0]
                hosts = parts[1:]
                key_to_hosts[key] = hosts

    # Populate the binary matrix
    for i, key in enumerate(key_ids):
        hosts = key_to_hosts.get(key, [])
        for host in hosts:
            if host in label_index:
                label_idx = label_index[host]
                ground_truth[i, label_idx] = 1
            else:
                logger.warning(f"Host '{host}' for virus '{key}' not found in label_index.")

    return ground_truth

def get_topk_sparse(scores: np.ndarray, top: int):
    """
    Convert scores to a sparse matrix keeping only the top K non-zero scores per sample.
    If a sample has fewer than K non-zero scores, only those non-zero scores are kept.
    """
    n_samples, n_labels = scores.shape
    if top >= n_labels:
        return ssp.csr_matrix(scores)

    indices = np.argpartition(-scores, top, axis=1)[:, :top]
    rows = np.repeat(np.arange(n_samples), top)
    cols = indices.flatten()

    data = scores[rows, cols]

    non_zero_mask = data != 0

    filtered_data = data[non_zero_mask]
    filtered_rows = rows[non_zero_mask]
    filtered_cols = cols[non_zero_mask]

    return ssp.csr_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=scores.shape)

def get_deepest_toxonomy(taxonomy_file):
    # 1. Read host_taxonomy and determine the deepest level of each host
    host_deepest_level = {}  # key: host, value: deepest_level
    
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        # Skip the header line if it exists
        header = f.readline().strip().split('\t')
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue  # skip malformed lines
            
            host, infra, sp, ge, fa, od, cl, ph, ki = parts
            
            taxonomy_list = [infra, sp, ge, fa, od, cl, ph, ki]
            deepest = OTHER_LEVEL

            # Find the first level (from species to kingdom) that is not "_"
            # and use that as the deepest level.
            for value, level_name in zip(taxonomy_list, LABEL_LEVELS):
                if host == value:
                    deepest = level_name
                    break
            host_deepest_level[host] = deepest
    return host_deepest_level

def load_host_label_taxonomy(taxonomy_file):
    """
    Load host label taxonomy from a TSV file.

    Args:
        filepath (str): Path to the host_label_taxonomy file.

    Returns:
        dict: A nested dict mapping host_name -> level -> taxonomy_name or None.
    """
    
    # This will be our output
    host_label_taxonomy = {}
    
    # Open the file for reading (assuming UTF-8 encoding)
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        # Read and discard the header line
        header = f.readline()
        
        # Process each subsequent line
        for line in f:
            # Split line into columns by tab
            parts = line.strip().split('\t')
            
            # Skip malformed lines
            if len(parts) < len(LABEL_LEVELS) + 1:
                continue
            
            # First column is the host name
            host = parts[0]
            
            host_label_taxonomy[host] = {lvl: None for lvl in LABEL_LEVELS}
            
            # Fill in each level if it's not "_"
            for idx, lvl in enumerate(LABEL_LEVELS, start=1):
                value = parts[idx]
                if value != "_":
                    host_label_taxonomy[host][lvl] = value
    
    return host_label_taxonomy
