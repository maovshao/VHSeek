import os
import re
import csv
from tqdm import tqdm
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.sparse as ssp
from collections import Counter, defaultdict
from logzero import logger
from typing import Dict, List, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from vhseek_util.evaluation import evaluate_metrics, STAT_LEVELS, LABEL_LEVELS, OTHER_LEVEL
from vhseek_util.util import (
    load_label_index, get_virus_name, load_embeddings, make_parent_dir,
    load_ground_truth, load_host_label_taxonomy, get_deepest_toxonomy
)
from sklearn import manifold
import math
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import torch
import matplotlib.colors as mcolors

from bokeh.io import show, save, output_file
from bokeh.models import (
    ColumnDataSource, LogColorMapper, ColorBar, FixedTicker,
    HoverTool, FactorRange
)
from bokeh.plotting import figure
from bokeh.palettes import Blues256
from contextlib import ContextDecorator
from typing import Optional, Callable, Dict

# =============================================================================
# Figure saving helpers (vector-friendly PDF/PNG)
# =============================================================================

_SAVE_ROOT = Path("vhseek_data/experiment/test_result/notebook")
_PDF_DIR = _SAVE_ROOT / "pdf"
_PNG_DIR = _SAVE_ROOT / "png"
_TSV_DIR = _SAVE_ROOT / "tsv"
_PDF_DIR.mkdir(parents=True, exist_ok=True)
_PNG_DIR.mkdir(parents=True, exist_ok=True)
_TSV_DIR.mkdir(parents=True, exist_ok=True)

_PREFIX_STACK: list[str] = []

_SLUG_COUNT: Dict[str, int] = {}
_LAST_EXTENSION_PAIRS: Optional[Dict[str, float]] = None
_LAST_LABELED_EXTENSION_PAIRS: Optional[Dict[str, float]] = None
_LAST_UNLABELED_EXTENSION_PAIRS: Optional[Dict[str, float]] = None

def _slugify(name: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "figure"

def _save_one(fig: 'plt.Figure', prefix: str, index: int) -> None:
    base = f"{prefix}_{index}"
    fig.savefig(_PDF_DIR / f"{base}.pdf", bbox_inches='tight', facecolor='white', edgecolor='white')
    fig.savefig(_PNG_DIR / f"{base}.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')

class figure_capture(ContextDecorator):
    """Context manager that captures newly created matplotlib figures and
    saves them as PDF+PNG with an index that restarts per call.

    Usage:
        with figure_capture("FDR_Control"):
            ... plotting code ...
    """
    def __init__(self, title: str, close: bool = True):
        self.prefix = _slugify(title)
        self.close = close
        self._before_ids: set[int] = set()
        self._save_index = 0
        self._orig_close = None

    def __enter__(self):
        # Record figures present before entering
        managers = list(plt._pylab_helpers.Gcf.get_all_fig_managers())
        self._before_ids = {id(m.canvas.figure) for m in managers}
        # Wrap plt.close to guarantee save-on-close inside the block
        self._orig_close = plt.close
        if self.prefix:
            _PREFIX_STACK.append(self.prefix)

        def _wrapped_close(*args, **kwargs):
            # Save all not-yet-saved new figures then close
            self._save_new_figs()
            return self._orig_close(*args, **kwargs)

        plt.close = _wrapped_close  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._save_new_figs()
        finally:
            if self.prefix and _PREFIX_STACK:
                _PREFIX_STACK.pop()
            if self._orig_close is not None:
                plt.close = self._orig_close  # restore
        return False  # do not suppress exceptions

    def _save_new_figs(self):
        managers = list(plt._pylab_helpers.Gcf.get_all_fig_managers())
        new = [m for m in managers if id(m.canvas.figure) not in self._before_ids]
        if not new:
            return
        for i, m in enumerate(new, 1):
            self._save_index += 1
            _save_one(m.canvas.figure, self.prefix, self._save_index)
        if self.close:
            for m in new:
                try:
                    self._orig_close(m.canvas.figure)  # type: ignore[misc]
                except Exception:
                    pass

def auto_save_plots(func: Callable):
    """Decorator to auto-capture matplotlib figures.

    Behavior:
    - If caller provides `save_prefix=...`, run inside `figure_capture(save_prefix)`.
    - Else, if the function defines a default prefix via attribute
      `func._default_save_prefix`, run inside that capture context.
    - Otherwise, just call the function (no auto-capture).
    """
    def wrapper(*args, **kwargs):
        prefix = kwargs.pop('save_prefix', None)
        if not prefix:
            prefix = getattr(func, '_default_save_prefix', None)
        if prefix:
            with figure_capture(prefix):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


def _ensure_tsv_dir() -> None:
    _TSV_DIR.mkdir(parents=True, exist_ok=True)


def _current_prefix() -> Optional[str]:
    return _PREFIX_STACK[-1] if _PREFIX_STACK else None


def _write_tsv(
    df: pd.DataFrame,
    filename: str,
    *,
    index: bool = True,
    float_format: Optional[str] = None
) -> None:
    _ensure_tsv_dir()
    path = _TSV_DIR / filename
    df.to_csv(path, sep='\t', index=index, float_format=float_format)


def _format_decimal(value: Any, decimals: int = 3, na_str: str = "NA") -> str:
    try:
        if pd.isna(value):
            return na_str
    except TypeError:
        return str(value)
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_pair(a: Any, b: Any, decimals: int = 3) -> str:
    return "/".join([
        _format_decimal(a, decimals),
        _format_decimal(b, decimals),
    ])


def _format_triplet(a: Any, b: Any, c: Any, decimals: int = 3) -> str:
    return "/".join([
        _format_decimal(a, decimals),
        _format_decimal(b, decimals),
        _format_decimal(c, decimals),
    ])


def _update_tsne_summary(prefix: str, label_counts: Counter) -> None:
    if not label_counts:
        return

    _ensure_tsv_dir()
    tsne_path = _TSV_DIR / "Virus_tsne.tsv"

    existing_sections: list[tuple[str, list[str]]] = []
    if tsne_path.exists():
        with open(tsne_path, 'r', encoding='utf-8') as fh:
            current_name: Optional[str] = None
            buffer: list[str] = []
            for raw_line in fh:
                line = raw_line.rstrip('\n')
                if current_name is None:
                    if not line:
                        continue
                    current_name = line
                    buffer = []
                else:
                    if line == "":
                        existing_sections.append((current_name, buffer))
                        current_name = None
                        buffer = []
                    else:
                        buffer.append(line)
            if current_name is not None:
                existing_sections.append((current_name, buffer))

    items = label_counts.most_common()
    labels_line = "\t".join(label for label, _ in items)
    counts_line = "\t".join(str(count) for _, count in items)
    new_block = [labels_line, counts_line]

    updated = False
    for idx, (name, _) in enumerate(existing_sections):
        if name == prefix:
            existing_sections[idx] = (prefix, new_block)
            updated = True
            break

    if not updated:
        existing_sections.append((prefix, new_block))

    preferred_order = [
        'Virus_Genome_Type_VHSeek',
        'Virus_Kingdom_VHSeek'
    ]
    preferred_sections = [sec for sec in existing_sections if sec[0] in preferred_order]
    preferred_sections.sort(key=lambda sec: preferred_order.index(sec[0]))
    other_sections = [sec for sec in existing_sections if sec[0] not in preferred_order]
    ordered_sections = preferred_sections + other_sections

    with open(tsne_path, 'w', encoding='utf-8') as fh:
        for idx, (name, lines) in enumerate(ordered_sections):
            fh.write(f"{name}\n")
            for line in lines:
                fh.write(f"{line}\n")
            if idx != len(ordered_sections) - 1:
                fh.write("\n")

def taxonomy_analyze(taxonomy_file):
    """
    Analyze host taxonomy and virus-host relationships.
    
    :param taxonomy_file: Path to the host_taxonomy file
    :param dict_file: Path to the virus_host_dict file
    :return: A dictionary containing the statistics of host_taxonomy 
             and virus_host_dict analysis results.
    """
    host_deepest_level = get_deepest_toxonomy(taxonomy_file)
    
    # Dictionary to count the number of unique hosts at each level in taxonomy
    taxonomy_label_dictionary = {}
    for single_level in LABEL_LEVELS:
        taxonomy_label_dictionary[single_level] = 0
    taxonomy_label_dictionary[OTHER_LEVEL] = 0
    
    # Count unique hosts in host_taxonomy
    total_unique_host_in_taxonomy = len(host_deepest_level)
    
    # Count each level (for unique hosts)
    for host, level in host_deepest_level.items():
        taxonomy_label_dictionary[level] += 1
    
    # 3. Prepare final stats
    result = {
        "taxonomy_total_unique_hosts": total_unique_host_in_taxonomy,
        "taxonomy_label_dictionary": dict(taxonomy_label_dictionary)
    }
    
    return result

# ---------------------------------------------------------------------------
# New helper: 3-set Venn diagram for host coverage across splits
# ---------------------------------------------------------------------------
@auto_save_plots
def plot_host_split_venn(root_path: str, **kwargs):
    prefix = _current_prefix()
    from matplotlib_venn import venn3, venn3_circles
    # Keep the same stats/printing behavior as the original Notebook: use multi_label_analyze
    plt.rcParams.update(plt.rcParamsDefault)

    # Infer split paths from a root directory that contains subfolders: all, train, validation, test
    root = root_path.rstrip('/')
    all_path = f"{root}/all"
    train_path = f"{root}/train"
    val_path = f"{root}/validation"
    test_path = f"{root}/test"

    print("Label Distribution Analysis for All")
    _ = multi_label_analyze(all_path)
    print("Label Distribution Analysis for Train")
    train_host = multi_label_analyze(train_path)
    print("Label Distribution Analysis for Validation")
    val_host = multi_label_analyze(val_path)
    print("Label Distribution Analysis for Test")
    test_host = multi_label_analyze(test_path)

    set_train = set(train_host.keys())
    set_validation = set(val_host.keys())
    set_test = set(test_host.keys())

    if prefix:
        venn_counts = {
            'Train': len(set_train - set_validation - set_test),
            'Validation': len(set_validation - set_train - set_test),
            'Test': len(set_test - set_train - set_validation),
            'Train & Validation': len((set_train & set_validation) - set_test),
            'Validation & Test': len((set_validation & set_test) - set_train),
            'Train & Test': len((set_train & set_test) - set_validation),
            'Train & Validation & Test': len(set_train & set_validation & set_test),
        }

        # Save TSV per-prefix for consistency with figure naming
        filename = f"{prefix}.tsv" if prefix else "Venn.tsv"
        venn_path = _TSV_DIR / filename
        ordered_columns = [
            'Train',
            'Validation',
            'Test',
            'Train & Validation',
            'Validation & Test',
            'Train & Test',
            'Train & Validation & Test'
        ]

        if venn_path.exists():
            venn_df = pd.read_csv(venn_path, sep='\t', index_col=0)
        else:
            venn_df = pd.DataFrame(columns=ordered_columns)

        venn_df.loc[prefix] = [venn_counts[col] for col in ordered_columns]

        preferred_order = [
            'IPHoP_Venn_Without_Label_Transfer',
            'IPHoP_Venn_Label_Transfer',
            'VHDB_Venn_Without_Label_Transfer',
            'VHDB_Venn_Label_Transfer'
        ]
        ordered_index = [name for name in preferred_order if name in venn_df.index]
        ordered_index += [name for name in venn_df.index if name not in ordered_index]
        venn_df = venn_df.loc[ordered_index, ordered_columns]

        _write_tsv(venn_df, filename, index=True, float_format=None)

    print("Plotting Venn Diagram for Train, Validation, and Test Hosts")
    # Force a stable, explicit color palette and background to avoid global style side effects
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor('white')
    fig = plt.gcf()
    fig.patch.set_facecolor('white')

    # Explicit colors ensure stability across sessions (no dependence on seaborn/mpl global state)
    venn3(
        [set_train, set_validation, set_test],
        ('Train Hosts', 'Validation Hosts', 'Test Hosts'),
        set_colors=('red', 'green', 'blue'),
        alpha=0.4
    )
    venn3_circles([set_train, set_validation, set_test], linestyle='dotted')
    # Optional custom title from caller; default to a generic title
    title = kwargs.pop('title', None)
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("Venn Diagram of Train, Validation, and Test Hosts", fontsize=16)
    plt.show()
    plt.close()

# Default save prefix for Venn TSV/plots
plot_host_split_venn._default_save_prefix = 'Venn'

@auto_save_plots
def plot_taxonomy_comparison(before_stats: Dict, after_stats: Dict, count_key: str, **kwargs) -> None:
    prefix = _current_prefix()
    # Prepare data for plotting (capitalize x labels like 'Species', 'Genus', ...)
    data = []
    x_levels_cap = [lvl.capitalize() for lvl in STAT_LEVELS]
    for level in STAT_LEVELS:
        data.append({
            'Level': level.capitalize(),
            'Count': before_stats[count_key][level],
            'Dataset': 'Without Label Transfer'
        })
        data.append({
            'Level': level.capitalize(),
            'Count': after_stats[count_key][level],
            'Dataset': 'With Label Transfer'
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot barplot with different colors for each dataset
    sns.barplot(
        x='Level',
        y='Count',
        hue='Dataset',
        data=df,
        palette={'Without Label Transfer': '#1f77b4', 'With Label Transfer': '#ff7f0e'},
        order=x_levels_cap
    )
    
    # Set logarithmic scale for y-axis
    plt.yscale('log')
    
    # Customize y-axis ticks to show 10^1, 10^2, etc.
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))
    
    # Set labels and title
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Taxonomy Level', fontsize=14)
    plt.ylabel('Number of Labels', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title=None, fontsize=14)

    # Save the plot
    plt.show()
    plt.close()

    if prefix:
        level_headers = [lvl.capitalize() for lvl in STAT_LEVELS]
        export_df = pd.DataFrame(
            [
                [before_stats[count_key][lvl] for lvl in STAT_LEVELS],
                [after_stats[count_key][lvl] for lvl in STAT_LEVELS],
            ],
            index=[
                'Without Label Transfer',
                'With Label Transfer'
            ],
            columns=level_headers
        )
        _write_tsv(export_df, f"{prefix}_1.tsv", index=True, float_format=None)

# Default save prefix so plot_taxonomy_comparison auto-saves figures
plot_taxonomy_comparison._default_save_prefix = 'Taxonomy_Comparison'

def multi_label_analyze(dict_file):
    plt.rcParams.update(plt.rcParamsDefault)
    host_dict = {}  # List to store number of host per virus
    virus_dict = {}  # Dictionary to store number of viruses per host

    # Read the virusprotein_host_pairs file
    with open(dict_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')

            # Get the host (starting from the second column)
            virus = parts[0]
            host = parts[1:]

            for single_host in host:
                host_dict.setdefault(virus,set())
                host_dict[virus].add(single_host)
                virus_dict.setdefault(single_host,set())
                virus_dict[single_host].add(virus)

    # Print results
    print(f"Total number of unique host: {len(virus_dict)}")
    return virus_dict

@auto_save_plots
def top_k_confidence_score(
    input_file: str,
    method_name: str,
    k: int = 10,
) -> None:
    """Aggregate and plot average Top-k confidence scores across viruses."""
    df = pd.read_csv(input_file, sep='\t', header=None,
                     names=['virus_name', 'host_name', 'confidence_score'])
    grouped = df.groupby('virus_name', sort=False)

    top_k_rows = []
    for _, group_df in grouped:
        group_df = group_df.reset_index(drop=True)
        scores = [group_df.loc[i, 'confidence_score'] if i < len(group_df) else None for i in range(k)]
        top_k_rows.append(scores)

    top_k_df = pd.DataFrame(top_k_rows).T
    avg_scores = top_k_df.mean(axis=1, skipna=True).values
    avg_strings = [_format_decimal(v) if not np.isnan(v) else 'NA' for v in avg_scores]

    rank_labels = [f"Top{i+1}" for i in range(k)]
    # Save per-call TSV using current prefix (consistent with figure naming)
    prefix = _current_prefix()
    out_df = pd.DataFrame([avg_strings], columns=rank_labels, index=[method_name])
    out_df.index.name = 'Method'
    filename = f"{prefix}.tsv" if prefix else "Top_k_confidence_score.tsv"
    _write_tsv(out_df, filename, index=True, float_format=None)

    plot_data = pd.DataFrame({
        'rank': rank_labels,
        'confidence_score': avg_scores
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='rank', y='confidence_score', color='skyblue', label='Confidence Score')
    sns.lineplot(data=plot_data, x='rank', y='confidence_score', color='red', marker='o', legend=False)
    plt.xlabel('Ranking', fontsize=18)
    plt.ylabel('Confidence Score', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()
    plt.close()


top_k_confidence_score._default_save_prefix = 'Top_k_confidence_score'

# Plot T-SNE
# Modify labels according to the specified rules
def process_labels(labels_dict):
    processed_labels = {}
    for key, label in labels_dict.items():
        if 'ssRNA-RT' in label:
            new_label = 'ssRNA-RT'
        elif 'dsDNA-RT' in label:
            new_label = 'dsDNA-RT'
        elif 'ssRNA' in label:
            new_label = 'ssRNA'
        elif 'dsRNA' in label:
            new_label = 'dsRNA'
        elif 'ssDNA' in label:
            new_label = 'ssDNA'
        elif 'dsDNA' in label:
            new_label = 'dsDNA'
        else:
            new_label = 'other'
        processed_labels[key] = new_label
    return processed_labels

def load_classification(classification_path):
    """
    Load classification labels from a file.

    Parameters:
    - classification_path (str): Path to the classification file.

    Returns:
    - classification_dict (dict): Dictionary mapping sample IDs to classifications.
    """
    classification_dict = {}
    with open(classification_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sample_id = parts[0]
                classification = parts[1]
                classification_dict[sample_id] = classification
    return classification_dict

@auto_save_plots
def create_tsne_plot(embedding_path, classification_path, title, output_dir, **kwargs):
    """
    Create a t-SNE plot for the given embeddings and labels.

    Parameters:
    - embedding_path (dict): Dictionary mapping sample IDs to embeddings.
    - classification_path (dict): Dictionary mapping sample IDs to class labels.
    - title (str): Title of the plot.
    - figure_path: Path to save figure
    """
    prefix = _current_prefix()

    # Load embeddings
    embeddings = load_embeddings(embedding_path)

    # Load classifications
    classification = load_classification(classification_path)

    # Processed classifications
    if (title[1] == 'Genome Type'):
        processed_classification = process_labels(classification)
    else:
        processed_classification = classification

    #########
    # Count the number of viruses per label
    # Keep only labels with more than 10 samples
    label_counts = Counter(processed_classification.values())
    processed_classification = {
        key: label for key, label in processed_classification.items()
        if label_counts[label] > 10
    }
    label_counts = Counter(processed_classification.values())
    logger.info(f"Filtered virus count: {label_counts}")

    # Choose a suitable seaborn color palette and fix the color for each label
    labels_list = list(label_counts.keys())
    palette_colors = sns.color_palette('tab10', n_colors=len(labels_list))
    label_to_color = dict(zip(labels_list, palette_colors))

    ###########
    # 1. Virus embeddings t-SNE plot (MLP) with processed labels
    # Filter out embeddings without classification
    filtered_embeddings = {}
    if (title[0] == 'Virus'):
        filtered_embeddings = {key: embeddings[key] for key in processed_classification}
    else:
        all_keys = list(processed_classification.keys())
        subset_size = max(1, len(all_keys) // 10)
        subset_keys = set(all_keys[:subset_size])
        
        filtered_embeddings = {
            key: embeddings[key]
            for key in embeddings
            if get_virus_name(key) in subset_keys
        }

    # Extract sample IDs and corresponding embeddings
    sample_ids = list(filtered_embeddings.keys())
    embedding_vectors = np.array([filtered_embeddings[sid] for sid in sample_ids])
    logger.info(f"Embedding vectors shape: {embedding_vectors.shape}")

    if prefix:
        _update_tsne_summary(prefix, label_counts)

    if (title[0] == 'Virus'):
        class_labels = [processed_classification[sid] for sid in sample_ids]
    else:
        class_labels = [processed_classification[get_virus_name(sid)] for sid in sample_ids]

    # Perform t-SNE dimensionality reduction
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(embedding_vectors)
    
    # Normalize the t-SNE embeddings
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'X': X_norm[:, 0],
        'Y': X_norm[:, 1],
        'Class': class_labels
    })

    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='X', y='Y', hue='Class', s=10, palette=label_to_color)

    plt.title(title, fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Classification', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=18, title_fontsize=18)
    plt.tight_layout()

    # Ensure the output directory exists
    plt.show()
    plt.close()

@auto_save_plots
def min_distance_plot(distance_file, **kwargs):
    # Restore matplotlib default parameters
    plt.rcParams.update(plt.rcParamsDefault)

    # Dictionary to store virus -> Min distance
    virus_distance_dict = {}

    # Step 1: Read distance_file and collect Min distance
    try:
        with open(distance_file, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue  # Need at least virus_name, Min distance
                virus_name = cols[0].strip()
                try:
                    min_distance_val = float(cols[1])
                except ValueError:
                    continue  # Skip lines where distance is not valid float
                # Store Min distance (assume each virus_name appears only once)
                virus_distance_dict[virus_name] = min_distance_val
    except FileNotFoundError:
        print(f"File not found: {distance_file}")
        return
    except Exception as e:
        print(f"Error reading {distance_file}: {e}")
        return

    # Step 3: Prepare data for statistics and plotting
    # Create a list of all Min distance values
    all_distances = list(virus_distance_dict.values())

    # Print summary statistics
    if all_distances:
        dist_mean = statistics.mean(all_distances)
        dist_max = max(all_distances)
        dist_min = min(all_distances)
    else:
        dist_mean = 0
        dist_max = 0
        dist_min = 0

    print("Min Distance Distribution:")
    print(f"  Mean: {dist_mean}")
    print(f"  Max: {dist_max}")
    print(f"  Min: {dist_min}\n")

    # Step 4: Violin plot for Min distance distribution
    plot_data = {
        'MinDistance': all_distances,
        'Dataset': ['Min distance'] * len(all_distances)
    }

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=plot_data, x='Dataset', y='MinDistance', inner='box', color="skyblue")
    # Empty x ticks and x label
    plt.xticks([])
    plt.yticks(fontsize=14)
    plt.xlabel('Virus List', fontsize=14)
    plt.ylabel('Min Distance', fontsize=14)
    plt.show()
    plt.close()

    # Step 5: Bar plot (Y-axis range set to [-0.1, 1.1], no log scale)
    vd_sorted = sorted(virus_distance_dict.items(), key=lambda x: x[1], reverse=True)
    vd_array = np.array([item[1] for item in vd_sorted])
    x_vd = 0.5 + np.arange(len(vd_array))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_vd, vd_array, width=1, linewidth=0, color="lightgreen")
    ax.set_xlabel('Virus List Sorted by Min Distance', fontsize=16)
    ax.set_ylabel('Min Distance', fontsize=16)

    ax.set(
        xlim=(0, len(vd_array)),
        ylim=(-0.1, 1.1),
        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    plt.close()

# default capture prefix when user doesn't supply one
min_distance_plot._default_save_prefix = 'Min_Distance'

### MAJOR TEST ###
def get_viruses_from_ground_truth_file(filepath: str) -> List[str]:
    virus_ids: List[str]          = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts and parts[0]:
                    virus_ids.append(parts[0])
    except FileNotFoundError:
        logger.error(f"Virus ground truth file not found when trying to extract virus list: {filepath}")
    return virus_ids

def major_test(
    prediction_path: str,
    truth_paths: Dict[str, str],
):
    """
    Runs the full evaluation pipeline for a single prediction file.
    Returns all metrics per level.
    """
    method_name = os.path.basename(prediction_path)
    logger.info(f"--- Starting evaluation for: {method_name} (major_test) ---")

    # 1. Load ground truth
    logger.info("Loading ground truth definitions...")
    label_taxonomy = load_host_label_taxonomy(truth_paths["label_taxonomy_path"])
    taxonomy_index = load_label_index(truth_paths["label_index_path"])
    
    logger.info(f"Loading host label_index from {truth_paths['label_index_path']}")
    label_index = load_label_index(truth_paths["label_index_path"])
    if not label_index:
        logger.error(f"Failed to load host label_index from {truth_paths['label_index_path']} or it's empty. Aborting for {method_name}.")
        return {} # Return empty dict on critical error
    n_hosts = len(label_index)
    logger.info(f"Loaded {n_hosts} unique host labels from host label_index file.")

    logger.info(f"Determining all target viruses from ground truth file: {truth_paths['virus_test_path']}...")
    virus_ids = get_viruses_from_ground_truth_file(truth_paths['virus_test_path'])
    virus_ids_index = {virus: i for i, virus in enumerate(virus_ids)}
    n_target_viruses = len(virus_ids_index)
    logger.info(f"Found {n_target_viruses} unique viruses in the ground truth set to be used as master virus list.")

    # 2. Load predictions
    logger.info(f"Loading predictions from {prediction_path}...")
    try:
        df_pred = pd.read_csv(prediction_path, sep='\t', header=None, names=['virus', 'host', 'score'])
        if df_pred.empty:
            logger.warning(f"Prediction file {prediction_path} is empty. No scores to evaluate.")
            df_pred = pd.DataFrame(columns=['virus', 'host', 'score'])
    except pd.errors.EmptyDataError:
        logger.warning(f"Prediction file {prediction_path} is empty (pd.errors.EmptyDataError). No scores to evaluate.")
        df_pred = pd.DataFrame(columns=['virus', 'host', 'score']) 
    except FileNotFoundError:
        logger.error(f"Prediction file {prediction_path} not found. Skipping evaluation for this file.")
        return {}
        
    # 3. Build the scores matrix
    logger.info("Building scores matrix...")
    score_rows, score_cols, score_data = [], [], []
    # ... (rest of score matrix building logic as provided by user - unchanged) ...
    predictions_in_master_list = 0
    predictions_skipped_host_not_in_index = 0
    predictions_skipped_virus_not_in_master = 0
    for _, row in df_pred.iterrows():
        virus_name = row['virus']
        host_name = row['host']
        if virus_name in virus_ids_index:
            if host_name in label_index: 
                score_rows.append(virus_ids_index[virus_name])
                score_cols.append(label_index[host_name])
                score_data.append(row['score'])
                predictions_in_master_list += 1
            else:
                predictions_skipped_host_not_in_index +=1
        else:
            predictions_skipped_virus_not_in_master +=1
    # if predictions_skipped_host_not_in_index > 0:
    #     logger.debug(f"Skipped {predictions_skipped_host_not_in_index} predictions...")
    # if predictions_skipped_virus_not_in_master > 0:
    #     logger.warning(f"Skipped {predictions_skipped_virus_not_in_master} predictions...")
    logger.info(f"Processed {predictions_in_master_list} valid prediction entries...")
    scores_matrix = ssp.csr_matrix(
        (score_data, (score_rows, score_cols)),
        shape=(n_target_viruses, n_hosts), 
        dtype=np.float32
    )
    logger.info(f"Built scores_matrix with shape {scores_matrix.shape} for {method_name}")

    # 4. Build the targets matrix
    logger.info("Building targets matrix using load_ground_truth...")
    targets_matrix_np = load_ground_truth(
        key_ids=virus_ids, 
        key_host_dict_path=truth_paths["virus_test_path"],
        label_index=label_index 
    )
    if targets_matrix_np.shape != (n_target_viruses, n_hosts):
        logger.error(f"Shape mismatch from load_ground_truth! Expected: {(n_target_viruses, n_hosts)}, Got: {targets_matrix_np.shape}.")
        return {} 
    targets_matrix = ssp.csr_matrix(targets_matrix_np)
    logger.info(f"Built targets_matrix with shape {targets_matrix.shape} for {method_name}")

    if targets_matrix.shape[0] == 0: 
        logger.error(f"Targets matrix is empty for {method_name}, cannot proceed with evaluation.")
        return {}
    if scores_matrix.shape != targets_matrix.shape:
        logger.error(f"CRITICAL SHAPE MISMATCH for {method_name}: Scores {scores_matrix.shape}, Targets {targets_matrix.shape}. Evaluation will be incorrect.")
        return {}

    # 5. Call the main evaluation function
    logger.info(f"Calling evaluate_metrics for {method_name}...")
    # --- MODIFICATION: Capture prec_dict and rec_dict ---
    result = evaluate_metrics(
        targets=targets_matrix,
        scores=scores_matrix,
        label_index=label_index, 
        label_taxonomy=label_taxonomy,
        taxonomy_index=taxonomy_index,
        with_log=True # Or pass a parameter to control this
    )
    
    result["method_name_raw"] = method_name
    result["virus_ids"] = virus_ids

    return result

def fmax_plot(
    fmax_plot_data: Dict[str, float],
    method_colors: Dict[str, Any],
    method_order: List[str]
):
    """
    Plot Fmax bars (Plot 1) for a specific taxonomic level.

    Args:
        fmax_plot_data: Dictionary mapping method display name to fmax_score.
        method_colors: Dict mapping method names to colors.
        method_order: List defining the order of methods on the plot.
    """
    methods_to_display = [m for m in method_order if m in fmax_plot_data]
    fmax_values = [fmax_plot_data[method_name] for method_name in methods_to_display]
    
    # Define colors for the methods to be displayed
    # Assuming method_colors contains all methods in methods_to_display from master_color_map
    # Fallback to a default color generation if a method is missing, though ideally it shouldn't happen.
    colors = []
    cmap_fallback = plt.cm.get_cmap('tab10')
    fallback_color_idx = 0
    for i, method_name in enumerate(methods_to_display):
        color = method_colors.get(method_name)
        if color is None:
            logger.warning(f"Color for method '{method_name}' not found. Using fallback color for Fmax plot.")
            color = cmap_fallback(fallback_color_idx % cmap_fallback.N)
            fallback_color_idx +=1
        colors.append(color)

    # --- Plot 1: Fmax bars ---
    # (X-axis: methods, Y-axis: Fmax value)
    fig_fmax, ax_fmax = plt.subplots(figsize=(max(6, len(methods_to_display) * 0.8), 5)) # Adjusted figsize
    method_indices = np.arange(len(methods_to_display))
    
    bars = ax_fmax.bar(method_indices, fmax_values, color=colors, width=0.5)
    
    ax_fmax.set_ylabel('Fmax', fontsize=14)
    ax_fmax.set_xticks(method_indices)
    ax_fmax.set_xticklabels(methods_to_display, rotation=45, ha="right", fontsize=14)
    ax_fmax.tick_params(axis='y', labelsize=14)
    # Adjust y-lim dynamically, ensure it handles cases with zero or very small fmax_values
    max_fmax_val = 0
    if fmax_values: # Check if fmax_values is not empty
        max_fmax_val = np.max(fmax_values) if len(fmax_values) > 0 else 0.0

    ax_fmax.set_ylim(0, max(0.01, min(1.0, max_fmax_val * 1.1 if max_fmax_val > 0 else 0.1)))
    ax_fmax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Create legend handles for Fmax plot
    legend_handles_fmax = [mpatches.Patch(color=colors[i], label=methods_to_display[i]) for i in range(len(methods_to_display))]
    ax_fmax.legend(handles=legend_handles_fmax, fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.show()
    plt.close(fig_fmax)

def aupr_plot(
    aupr_plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    method_colors: Dict[str, Any],
    method_order: List[str]
):
    """
    Plot AUPR precision-recall curves (Plot 2) for a specific taxonomic level,
    only for specified methods.

    Args:
        aupr_plot_data: Dictionary mapping method display name to a tuple of
                        (precision_values_array, recall_values_array).
        method_colors: Dict mapping method names to colors.
        method_order: List of method display names to be plotted, in the desired order.
    """
    
    methods_for_curves = []
    precision_curves_plot = {}
    recall_curves_plot = {}

    for method_name in method_order:
        if method_name in aupr_plot_data:
            p_vals, r_vals = aupr_plot_data[method_name]
            if isinstance(p_vals, np.ndarray) and isinstance(r_vals, np.ndarray) and \
               p_vals.size > 0 and r_vals.size > 0:
                methods_for_curves.append(method_name)
                precision_curves_plot[method_name] = p_vals
                recall_curves_plot[method_name] = r_vals
            else:
                logger.warning(f"Method '{method_name}' from top methods list has invalid or empty P/R arrays. Skipping for AUPR plot.")
        # else: Method was in method_order, but no data for it in aupr_plot_data dict.
        # This case should ideally be handled by the caller by pre-filtering aupr_plot_data.

    # Define colors for the methods that will actually be plotted
    # Assuming method_colors contains all methods in methods_for_curves from master_color_map
    plot_colors = []
    cmap_fallback = plt.cm.get_cmap('tab10')
    fallback_color_idx = 0
    for i, method_name in enumerate(methods_for_curves):
        color = method_colors.get(method_name)
        if color is None:
            logger.warning(f"Color for method '{method_name}' not found. Using fallback color for AUPR plot.")
            color = cmap_fallback(fallback_color_idx % cmap_fallback.N)
            fallback_color_idx += 1
        plot_colors.append(color)

    # --- Plot 2: AUPR curves ---
    # (X-axis: Recall, Y-axis: Precision)
    fig_aupr, ax_aupr = plt.subplots(figsize=(8, 6)) # Adjusted figsize
    for i, method_name in enumerate(methods_for_curves): # Iterate in the order of methods_for_curves
        r_vals = recall_curves_plot[method_name]
        p_vals = precision_curves_plot[method_name]
        
        ax_aupr.plot(
            r_vals,
            p_vals,
            color=plot_colors[i], # Use determined color
            label=f'{method_name}', # Requested label format
            lw=2,
            #drawstyle='steps-post' # Common for PR curves, uncomment if desired
        )

    ax_aupr.set_xlabel('Recall', fontsize=18)
    ax_aupr.set_ylabel('Precision', fontsize=18)
    ax_aupr.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax_aupr.set_xlim(-0.05, 1.05)
    ax_aupr.set_ylim(-0.05, 1.05)
    ax_aupr.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    plt.close(fig_aupr)

@auto_save_plots
def plot_cina_fractions(
    cina_data_dict: Dict[str, Tuple[float, float, float]], 
    method_colors: Dict[str, Any],
    method_order: List[str],
    **kwargs
):
    """
    Plot stacked bar chart for Correct, Incorrect, No Answer fractions,
    styled similarly to plot_iphop_result.
    'Correct' part of each method gets a distinct color. 'Incorrect' is gray. 'No Answer' is lightsteelblue.

    Args:
        cina_data_dict: Dictionary mapping method display name to (CR, IR, NA) scores.
        method_colors: Optional dict mapping method names to colors for the 'Correct' portion.
    """
    methods = [m for m in method_order if m in cina_data_dict]

    cr_values = np.array([cina_data_dict[m][0] for m in methods])
    ir_values = np.array([cina_data_dict[m][1] for m in methods])
    na_values = np.array([cina_data_dict[m][2] for m in methods])

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 0.9), 6)) # Adjusted figsize
    x_positions = np.arange(len(methods))

    # Colors
    default_palette = plt.cm.get_cmap('tab10')
    correct_bar_colors = []
    if method_colors:
        correct_bar_colors = [method_colors.get(method, default_palette(i % default_palette.N)) for i, method in enumerate(methods)]
    else:
        correct_bar_colors = [default_palette(i % default_palette.N) for i in range(len(methods))]
    
    color_incorrect = 'darkgrey'  # Darker gray (#696969)
    color_no_answer = 'lightgrey'   # Lighter gray (#C0C0C0)

    # Plot stacked bars
    ax.bar(x_positions, cr_values, color=correct_bar_colors, width=0.6, label="Correct (varies by method)") # Label for doc, not direct legend
    ax.bar(x_positions, ir_values, bottom=cr_values, color=color_incorrect, width=0.6, label="Incorrect")
    ax.bar(x_positions, na_values, bottom=cr_values + ir_values, color=color_no_answer, width=0.6, label="No Answer")

    ax.set_ylabel("Fraction", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Custom Legend (similar to plot_iphop_result)
    legend_patches = []
    for i, method_name in enumerate(methods):
        legend_patches.append(mpatches.Patch(color=correct_bar_colors[i], label=f'{method_name}'))
    legend_patches.append(mpatches.Patch(color=color_incorrect, label='Incorrect'))
    legend_patches.append(mpatches.Patch(color=color_no_answer, label='No Answer'))
    
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=14)

    plt.show()
    plt.close(fig)

@auto_save_plots
def plot_top_methods_level(
    level_cr_plot_data: Dict[str, Dict[str, float]],
    level_ir_plot_data: Dict[str, Dict[str, float]],
    levels_order: List[str],
    method_colors: Dict[str, Any],
    method_order: List[str],
    **kwargs
):
    """
    Plot Top-1 Recall (TPR) and Top-1 FDR curves for each method.

    Parameters
    ----------
    level_cr_plot_data : dict
        method -> level -> Top-1 Recall (n1/100).
    level_ir_plot_data : dict
        method -> level -> Top-1 error rate (n2/100).
    levels_order : list
        Desired order of taxonomic levels.
    method_colors : dict
        Method name -> color.
    method_order : list
        Order in which methods should appear.
    """
    x = np.arange(len(levels_order))
    x_labels = [lvl.capitalize() for lvl in levels_order]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # leave 25 % of the figure width on the right for legends
    fig.subplots_adjust(right=0.75)

    if not method_colors:
        palette = sns.color_palette(n_colors=len(method_order))
        method_colors = {m: palette[i] for i, m in enumerate(method_order)}

    for method in method_order:
        recall_vals = [level_cr_plot_data[method].get(lvl, np.nan)
                       for lvl in levels_order]
        fdr_vals = []
        for lvl in levels_order:
            cr = level_cr_plot_data[method].get(lvl, np.nan)
            ir = level_ir_plot_data[method].get(lvl, np.nan)
            denom = cr + ir
            fdr_vals.append(ir / denom if denom > 0 else np.nan)

        color = method_colors[method]
        ax.plot(x, recall_vals, marker='o', linestyle='-',  color=color, lw=2)
        ax.plot(x, fdr_vals,    marker='^', linestyle='--', color=color, lw=2)

    # axis decorations
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlabel("Taxonomic Level", fontsize=14)
    ax.set_ylabel("Top-1 TPR/FDR", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # ---------- build two legend groups ----------
    # 1) colors = methods
    method_handles = [Line2D([0], [0], color=method_colors[m], lw=3)
                      for m in method_order]
    # 2) styles = metrics
    metric_handles = [
        Line2D([0], [0], color='black', marker='o', linestyle='-',  lw=2),
        Line2D([0], [0], color='black', marker='^', linestyle='--', lw=2)
    ]

    # put legends on the right within the reserved 25 % strip
    # anchor coords are in figure fraction units (0-1)
    fig.legend(method_handles, method_order,
               #frameon=True,
               loc='upper left', bbox_to_anchor=(0.77, 0.85),
               fontsize=14, borderaxespad=0.0,
               bbox_transform=fig.transFigure)

    fig.legend(metric_handles, ['Top-1 TPR', 'Top-1 FDR'],
               #title="Metric", frameon=True,
               loc='upper left', bbox_to_anchor=(0.77, 0.2),
               fontsize=14, borderaxespad=0.0,
               bbox_transform=fig.transFigure)

    plt.show()
    plt.close()

@auto_save_plots
def major_plot(
    result_dict,
    order_list,
    top_methods_order_list,
    **kwargs
):
    logger.info(f"Starting evaluation process for level(s): {', '.join(STAT_LEVELS)}")
    prefix = _current_prefix()

    full_palette = sns.color_palette("tab20", n_colors=len(order_list) + 2)
    filtered_palette = [color for i, color in enumerate(full_palette) if i not in [14, 15]][::-1]

    master_color_map = {key: filtered_palette[i] for i, key in enumerate(order_list)}
    
    # --- Prepare data for each plot type ---
    # (This logic remains as modified in the previous correct response)
    fmax_data_genus = {}
    aupr_data_genus = {}
    cina_data_genus = {}
    level_cr_plot_data = { 
        method_disp_name: {level: np.nan for level in STAT_LEVELS}
        for method_disp_name in top_methods_order_list 
    }
    level_ir_plot_data = { 
        method_disp_name: {level: np.nan for level in STAT_LEVELS}
        for method_disp_name in top_methods_order_list 
    }
    
    tsv_data_output = {name: {} for name in order_list}
    metrics_for_tsv = ['Fmax', 'AUPR', 'CR', 'IR', 'NA']

    for key in result_dict:
        result_item = result_dict[key]

        fmax_per_level = result_item.get("fmax_per_level", {})
        aupr_per_level = result_item.get("aupr_per_level", {})
        prec_per_level = result_item.get("precision_per_level", {})
        rec_per_level = result_item.get("recall_per_level", {})
        if 'genus' in fmax_per_level:
            fmax_data_genus[key] = fmax_per_level['genus']
            if key in top_methods_order_list:
                aupr_data_genus[key] = (
                    prec_per_level.get('genus', np.array([])), 
                    rec_per_level.get('genus', np.array([]))
                )

        cr_per_level = result_item.get("cr_per_level", {})
        ir_per_level = result_item.get("ir_per_level", {})
        na_per_level = result_item.get("na_per_level", {})
        if 'genus' in cr_per_level:
            cina_data_genus[key] = (
                cr_per_level['genus'],
                ir_per_level.get('genus', np.nan),
                na_per_level.get('genus', np.nan)
            )

        if key in top_methods_order_list:
            for level_key in STAT_LEVELS: 
                cr_value = cr_per_level.get(level_key, np.nan)
                ir_value = ir_per_level.get(level_key, np.nan)
                if not pd.isna(cr_value) and isinstance(cr_value, (float, int, np.number)):
                    level_cr_plot_data[key][level_key] = float(cr_value)
                if not pd.isna(ir_value) and isinstance(ir_value, (float, int, np.number)):
                    level_ir_plot_data[key][level_key] = float(ir_value)
        
        for level_key in STAT_LEVELS:
            tsv_data_output[key][(level_key, 'Fmax')] = fmax_per_level.get(level_key, np.nan)
            tsv_data_output[key][(level_key, 'AUPR')] = aupr_per_level.get(level_key, np.nan)
            tsv_data_output[key][(level_key, 'CR')] = cr_per_level.get(level_key, np.nan)
            tsv_data_output[key][(level_key, 'IR')] = ir_per_level.get(level_key, np.nan)
            tsv_data_output[key][(level_key, 'NA')] = na_per_level.get(level_key, np.nan)

    if prefix:
        level_headers = [lvl.capitalize() for lvl in STAT_LEVELS]

        def _collect_metric(metric: str) -> pd.DataFrame:
            rows = []
            for method in order_list:
                row = {}
                method_dict = tsv_data_output.get(method, {})
                for lvl, header in zip(STAT_LEVELS, level_headers):
                    row[header] = method_dict.get((lvl, metric), np.nan)
                rows.append(pd.Series(row, name=method))
            df_metric = pd.DataFrame(rows)
            df_metric.index.name = "Method"
            return df_metric

        # Table 1: Fmax per taxonomy level
        fmax_table = _collect_metric('Fmax')
        formatted_fmax = fmax_table.map(lambda v: _format_decimal(v))
        _write_tsv(formatted_fmax, f"{prefix}_1.tsv", index=True, float_format=None)

        # Table 2: Correct/Incorrect/No-Answer triplets
        triplet_rows = []
        for method in order_list:
            row = {}
            method_dict = tsv_data_output.get(method, {})
            for lvl, header in zip(STAT_LEVELS, level_headers):
                cr_val = method_dict.get((lvl, 'CR'), np.nan)
                ir_val = method_dict.get((lvl, 'IR'), np.nan)
                na_val = method_dict.get((lvl, 'NA'), np.nan)
                row[header] = _format_triplet(cr_val, ir_val, na_val)
            triplet_rows.append(pd.Series(row, name=method))
        triplet_table = pd.DataFrame(triplet_rows)
        triplet_table.index.name = "Method"
        _write_tsv(triplet_table, f"{prefix}_2.tsv", index=True, float_format=None)

        # Table 3: AUPR per taxonomy level
        aupr_table = _collect_metric('AUPR')
        formatted_aupr = aupr_table.map(lambda v: _format_decimal(v))
        _write_tsv(formatted_aupr, f"{prefix}_3.tsv", index=True, float_format=None)

        # Table 4: Top-1 TPR/FDR per level
        ratio_rows = []
        for method in order_list:
            row = {}
            method_dict = result_dict.get(method)
            cr_dict = method_dict.get('cr_per_level', {}) if method_dict else {}
            ir_dict = method_dict.get('ir_per_level', {}) if method_dict else {}
            for lvl, header in zip(STAT_LEVELS, level_headers):
                cr_val = cr_dict.get(lvl, np.nan)
                ir_val = ir_dict.get(lvl, np.nan)
                denom = None
                if not pd.isna(cr_val) or not pd.isna(ir_val):
                    denom = (cr_val if not pd.isna(cr_val) else 0.0) + (ir_val if not pd.isna(ir_val) else 0.0)
                fdr_val = ir_val / denom if denom else np.nan
                row[header] = _format_pair(cr_val, fdr_val)
            ratio_rows.append(pd.Series(row, name=method))
        ratio_table = pd.DataFrame(ratio_rows)
        ratio_table.index.name = 'Method'
        _write_tsv(ratio_table, f"{prefix}_4.tsv", index=True, float_format=None)

    # --- Generate Plots ---
    # (This logic for calling new plot functions remains as modified)

    # Data for Fmax plot (Plot 1)
    # fmax_data_for_plot_genus = {
    #     method: data[0] for method, data in fmax_aupr_data_genus.items() if data is not None
    # }
    if fmax_data_genus:
        logger.info("Generating Fmax plot for 'genus' level...")
        fmax_plot( # New function call
            fmax_plot_data=fmax_data_genus,
            method_colors=master_color_map,
            method_order=order_list
        )
    else:
        logger.info("No data for 'genus' level Fmax plot.")

    if cina_data_genus:
        logger.info("Generating Correct/Incorrect/No Answer plot for 'genus' level...")
        plot_cina_fractions(
            cina_data_dict=cina_data_genus, 
            method_colors=master_color_map,
            method_order=order_list
        )
    else:
        logger.info("No data for 'genus' level C/I/NA plot.")

    if aupr_data_genus:
        logger.info("Generating AUPR plot for 'genus' level (top methods)...")
        aupr_plot( # New function call
            aupr_plot_data=aupr_data_genus, 
            method_colors=master_color_map,
            method_order=top_methods_order_list
        )
    else:
        logger.info("No data for 'genus' level AUPR plot for top methods, or top methods lack P/R data.")

    if level_cr_plot_data and level_ir_plot_data:
        plot_top_methods_level(
            level_cr_plot_data=level_cr_plot_data,
            level_ir_plot_data=level_ir_plot_data,
            levels_order=STAT_LEVELS, 
            method_colors=master_color_map,
            method_order=top_methods_order_list
        )
    else:
        logger.info("No data or no target methods with data for the top methods summary plot (Correct Rate).")

    # --- Output TSV File ---
    # (This logic remains as modified in the previous correct response)
    print("====================================================")
    logger.info("Evaluation and plotting script finished.")

@auto_save_plots
def fdr_control_plot(
    metrics_results: Dict,
    target_precision: float,
    reference_level: str = 'genus',
    **kwargs
):
    """
    Calculates and plots FDR based on precision-recall curves from evaluate_metrics results.
    It also identifies the threshold for a target FDR (1 - target_precision) on a reference level.

    Parameters
    ----------
    metrics_results : dict
        The dictionary returned by the evaluate_metrics function.
    target_precision : float
        Desired precision target to control FDR (FDR = 1 - Precision), e.g. 0.9.
    reference_level : str, optional
        The taxonomic level used as a reference to determine the control threshold. Defaults to 'genus'.
    """
    
    # --- 1. Data Preparation and Target Threshold Calculation ---
    
    prefix = _current_prefix()

    plot_data = []
    target_precisions = [float(target_precision)]
    # Storage for per-target stats
    target_stats: list[dict] = []

    # This loop prepares data for all levels for plotting
    for level in STAT_LEVELS:
        if level == 'infraspecies':
            continue
        precision = metrics_results['precision_per_level'].get(level)
        # MODIFICATION 4: Get recall for later interpolation
        recall = metrics_results['recall_per_level'].get(level) 
        # NOTE: Assuming the key for thresholds is 'aupr_thresholds_per_level' as per your code
        thresholds = metrics_results.get('aupr_thresholds_per_level', {}).get(level)
        
        if precision is None or thresholds is None or len(precision) < 2:
            print(f"Skipping level '{level}' due to insufficient data.")
            continue

        fdr = 1.0 - precision[:-1]
        
        # MODIFICATION: Capitalize the level name before adding it to the plot data
        capitalized_level = level.capitalize()
        for t, f in zip(thresholds, fdr):
            plot_data.append({'Threshold': t, 'FDR': f, 'Level': capitalized_level})
            
        if level == reference_level:
            for p in target_precisions:
                idx = np.where(precision >= p)[0]
                if idx.size > 0:
                    first = idx[0]
                    thr = thresholds[first] if first < len(thresholds) else thresholds[-1]
                else:
                    thr = None

                if thr is not None:
                    ref_precision = metrics_results['precision_per_level'][reference_level]
                    ref_recall = metrics_results['recall_per_level'][reference_level]
                    ref_thresholds = metrics_results['aupr_thresholds_per_level'][reference_level]
                    sort_idx = np.argsort(ref_thresholds)
                    prec_val = float(np.interp(thr, ref_thresholds[sort_idx], ref_precision[:-1][sort_idx]))
                    rec_val = float(np.interp(thr, ref_thresholds[sort_idx], ref_recall[:-1][sort_idx]))
                    fdr_val = 1.0 - prec_val
                else:
                    prec_val = rec_val = fdr_val = float('nan')

                target_stats.append({
                    'precision_target': p,
                    'threshold': thr,
                    'precision': prec_val,
                    'recall': rec_val,
                    'fdr': fdr_val,
                })

    # Print summaries for each requested precision target
    for s in target_stats:
        p = s['precision_target']
        print(f"Calculating threshold for {reference_level.capitalize()} level at Precision >= {p:.2f} (FDR <= {1 - p:.2f})...")
        thr = s['threshold']
        if thr is None or np.isnan(s['precision']):
            print(f"Warning: Could not find a threshold for {reference_level.capitalize()} that meets Precision >= {p:.2f}.")
            continue
        print(f"Found Threshold: {thr:.3f}")
        print(f"  - At this threshold, for {reference_level.capitalize()} level:")
        print(f"  - Precision: {s['precision']:.3f}")
        print(f"  - Recall:    {s['recall']:.3f}")
        print(f"  - FDR:       {s['fdr']:.3f}")

    df_plot = pd.DataFrame(plot_data)

    # --- 2. Print Table for Key Thresholds on Genus Level Only ---
    
    # MODIFICATION 2: Changed title and table format
    print(f"\n--- Performance at Key Thresholds on {reference_level.capitalize()} Level ---")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'FDR':<12}")
    print("-" * 48)

    # Directly get data for the reference level
    precision = metrics_results['precision_per_level'].get(reference_level)
    recall = metrics_results['recall_per_level'].get(reference_level)
    thresholds = metrics_results['aupr_thresholds_per_level'].get(reference_level)

    report_rows = []
    if precision is not None and thresholds is not None and len(precision) > 1:
        report_thresholds = np.arange(0, 1.01, 0.1)
        # Ensure thresholds are sorted for interpolation
        sorted_indices = np.argsort(thresholds)
        
        for t_report in report_thresholds:
            p_interp = np.interp(t_report, thresholds[sorted_indices], precision[:-1][sorted_indices])
            r_interp = np.interp(t_report, thresholds[sorted_indices], recall[:-1][sorted_indices])
            fdr_interp = 1.0 - p_interp
            print(f"{t_report:<12.1f}{p_interp:<12.3f}{r_interp:<12.3f}{fdr_interp:<12.3f}")
            report_rows.append({
                'Threshold': _format_decimal(t_report, 1),
                'Precision': _format_decimal(p_interp),
                'Recall': _format_decimal(r_interp),
                'FDR': _format_decimal(fdr_interp)
            })
    print("-" * 48)

    if prefix and report_rows:
        table_df = pd.DataFrame(report_rows)
        filename = f"{prefix}_1.tsv" if prefix else "FDR_Control_1.tsv"
        _write_tsv(table_df, filename, index=False, float_format=None)

    # --- 3. Plotting with Seaborn ---

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(
        data=df_plot,
        x='Threshold',
        y='FDR',
        hue='Level',
        # MODIFICATION: Ensure the legend order matches the capitalized names
        hue_order=[s.capitalize() for s in STAT_LEVELS if s!="infraspecies"],
        ax=ax,
        linewidth=2.5
    )

    # Draw reference lines for each requested target precision (all in red)
    proxies = []
    proxy_labels = []
    for i, s in enumerate(target_stats):
        thr = s['threshold']
        fdr_val = s['fdr']
        p = s['precision_target']
        if thr is None or np.isnan(fdr_val):
            continue
        col = 'red'
        ax.plot([thr, thr], [0, fdr_val], color=col, linestyle='--', linewidth=2)
        ax.plot([0, thr], [fdr_val, fdr_val], color=col, linestyle='--', linewidth=2)
        ax.scatter(thr, fdr_val, color=col, s=50, zorder=5)
        proxies.append(Line2D([0], [0], color=col, linestyle='--', linewidth=2))
        proxy_labels.append(f"FDR Control (Level={reference_level.capitalize()}, P≥{p:.2f}, Thr={thr:.3f})")

    ax.set_xlabel("Confidence Score Threshold", fontsize=14)
    ax.set_ylabel("False Discovery Rate (FDR)", fontsize=14)
    
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticks(np.arange(0.2, 1.0, 0.2))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend: levels + control lines
    handles, labels = ax.get_legend_handles_labels()
    if proxies:
        handles.extend(proxies)
        labels.extend(proxy_labels)
    ax.legend(handles=handles, labels=labels, fontsize=14, title_fontsize=14,
              title='Taxonomy Level', loc='upper right')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# Default save prefix for FDR control plot/TSV
fdr_control_plot._default_save_prefix = 'FDR_Control'

def host_predictions_analysis(
    metrics_results: Dict,
    dict_file: str, 
    dict_test_file: str, 
    ic_score_file: str, 
    taxonomy_file: str,
    top: Optional[int] = None
):
    """
    Calculates taxonomy-aware recall and combines it with other host metadata.
    This function consumes results from `major_test`.
    """
    logger.info("Analyzing results and calculating taxonomy-aware recall...")

    # --- Step 1: Load static host metadata ---
    host_count_dict = defaultdict(int)
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(parts := line.strip().split('\t')) >= 2:
                for h in parts[1:]: host_count_dict[h] += 1
    
    host_ic_dict = {p[0]: float(p[1]) for p in (l.strip().split('\t') for l in open(ic_score_file, 'r')) if len(p) == 2}

    host_count_test = defaultdict(int)
    virus_host_test = set()
    with open(dict_test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(parts := line.strip().split('\t')) >= 2:
                for h in parts[1:]:
                    virus_host_test.add((parts[0], h))
                    host_count_test[h] += 1
    
    # --- Step 2: Calculate new taxonomy-aware recall using results from major_test ---
    gt_tx, pr_tx = metrics_results.get('gt_tx', {}), metrics_results.get('pr_tx', {})
    virus_ids_list = metrics_results.get("virus_ids", [])
    
    if not virus_ids_list:
        logger.error("'virus_ids' not in metrics_results. Cannot calculate recall.")
        return pd.DataFrame()

    master_virus_index = {virus_str: i for i, virus_str in enumerate(virus_ids_list)}
    host_deepest_level = get_deepest_toxonomy(taxonomy_file)
    
    correct_count_dict = defaultdict(int)
    for virus_str, true_host_str in virus_host_test:
        if virus_str not in master_virus_index: continue
        
        vid = master_virus_index[virus_str]
        level = host_deepest_level.get(true_host_str)
        if not level or level not in STAT_LEVELS: continue

        ground_truth_taxa = gt_tx.get(vid, {}).get(level, set())
        predicted_taxa_list = pr_tx.get(vid, {}).get(level, [])

        # optionally truncate predictions
        if top is not None:
            predicted_taxa_list = predicted_taxa_list[:top]
        
        # if predicted_taxa_list and ground_truth_taxa and predicted_taxa_list[0][1] in ground_truth_taxa:
        #     correct_count_dict[true_host_str] += 1
        if predicted_taxa_list and ground_truth_taxa:
            if any(taxon in ground_truth_taxa for _, taxon in predicted_taxa_list):
                correct_count_dict[true_host_str] += 1

    host_recall_dict = {host: correct_count_dict.get(host, 0) / count 
                        for host, count in host_count_test.items() if count > 0}

    # --- Step 3: Assemble and return the final DataFrame ---
    processed_data = []
    for h in host_count_test.keys():
        processed_data.append({
            "host_name": h, "occurrences": host_count_dict.get(h, 0),
            "ic_score": host_ic_dict.get(h, 0.0),
            "recall": host_recall_dict.get(h, 0.0),
            "taxonomy_level": host_deepest_level.get(h, OTHER_LEVEL)
        })
    return processed_data

def _compute_host_extension_counts(
    label_transfer_path: str,
    experiment_path: str,
    taxonomy_path: str,
    *,
    high_threshold: float,
    host_filter: Optional[Set[str]] = None,
    exclude_virus_path: Optional[str] = None,
    unlabeled_virus_path: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Compute per-host original and extension counts from label-transfer inputs.

    Extension definition matches `extension_plot`:
    - high: score > high_threshold
    - unlabeled: score > high_threshold for viruses in unlabeled_virus_path
    """
    host_level_map = get_deepest_toxonomy(taxonomy_path)
    gt_level_data: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    original_counts: Dict[str, int] = defaultdict(int)

    exclude_viruses: Set[str] = set()
    if exclude_virus_path:
        with Path(exclude_virus_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                virus = line.split("\t", 1)[0].strip()
                if virus:
                    exclude_viruses.add(virus)

    unlabeled_viruses: Set[str] = set()
    if unlabeled_virus_path:
        with Path(unlabeled_virus_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                virus = line.split("\t", 1)[0].strip()
                if virus:
                    unlabeled_viruses.add(virus)
        if exclude_viruses:
            unlabeled_viruses -= exclude_viruses

    with Path(label_transfer_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            virus = parts[0].strip()
            if exclude_viruses and virus in exclude_viruses:
                continue
            for host in parts[1:]:
                host = host.strip()
                if not host:
                    continue
                level = host_level_map.get(host)
                if not level or level not in STAT_LEVELS:
                    continue
                gt_level_data[virus][level].add(host)
                if host_filter is None or host in host_filter:
                    original_counts[host] += 1

    header_re = re.compile(r"\[(.+?)\]\[(.+?)\]")
    pred_level_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    current_virus = None
    current_level = None

    with Path(experiment_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                m = header_re.match(line)
                if not m:
                    continue
                current_virus = m.group(1).strip()
                current_level = m.group(2).strip().lower()
                continue
            if current_virus is None or current_level is None:
                continue
            if exclude_viruses and current_virus in exclude_viruses:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            host, score_str = parts[0].strip(), parts[1].strip()
            if host_filter is not None and host not in host_filter:
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            prev = pred_level_data[current_virus][current_level].get(host, 0.0)
            if score > prev:
                pred_level_data[current_virus][current_level][host] = score

    high_counts: Dict[str, int] = defaultdict(int)
    mid_counts: Dict[str, int] = defaultdict(int)
    all_viruses = set(gt_level_data) | set(pred_level_data)
    for virus in all_viruses:
        if exclude_viruses and virus in exclude_viruses:
            continue
        is_unlabeled = bool(unlabeled_viruses and virus in unlabeled_viruses)
        for level, host_map in pred_level_data.get(virus, {}).items():
            gt_hosts = gt_level_data.get(virus, {}).get(level, set())
            for host, score in host_map.items():
                if host in gt_hosts:
                    continue
                if is_unlabeled:
                    if score > high_threshold:
                        mid_counts[host] += 1
                    continue
                if score > high_threshold:
                    high_counts[host] += 1

    return dict(original_counts), dict(high_counts), dict(mid_counts)

@auto_save_plots
def host_predictions_plot(
    processed_data: List,
    **kwargs
):
    prefix = _current_prefix()
    # --- Step 4: Filter, sort, and plot hosts ---
    # Convert to DataFrame and filter by STAT_LEVELS
    df = pd.DataFrame(processed_data)
    df = df[df['taxonomy_level'].isin(STAT_LEVELS)]
    if df.empty:
        logger.warning("No hosts with taxonomy_level in STAT_LEVELS. Nothing to plot.")
        return

    # Sort by occurrences descending
    df = df.sort_values(by='occurrences', ascending=False).reset_index(drop=True)
    num_hosts = len(df)

    if prefix:
        export_df = df[['host_name', 'taxonomy_level', 'occurrences', 'ic_score', 'recall']].copy()
        export_df.insert(0, 'Index', np.arange(1, len(export_df) + 1))
        export_df.rename(columns={
            'host_name': 'Host Name',
            'taxonomy_level': 'Host Taxonomy Level',
            'occurrences': 'Number of Annotated Viruses',
            'ic_score': 'IC Score',
            'recall': 'Recall'
        }, inplace=True)
        export_df['IC Score'] = export_df['IC Score'].apply(lambda v: _format_decimal(v))
        export_df['Recall'] = export_df['Recall'].apply(lambda v: _format_decimal(v))
        filename = f"{prefix}.tsv" if prefix else "Host_Predictions.tsv"
        _write_tsv(export_df, filename, index=False, float_format=None)

    # Prepare arrays
    occ_array = df['occurrences'].to_numpy()
    ic_array = df['ic_score'].to_numpy()
    recall_array = df['recall'].to_numpy()

    # Setup plot style
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    x_positions = np.arange(num_hosts)

    # Top: piece-wise transform for bar height
    top_heights = np.where(
        occ_array == 0,
        0.0,                             # occ = 0  → y = 0
        np.log10(occ_array) + 1.0        # occ > 0 → y = log10(occ) + 1
    )
    max_log_occ = np.nanmax(top_heights) if num_hosts else 1

    # Bottom: IC score scaled to match -max_log_occ
    max_ic = np.max(ic_array) if num_hosts else 1
    scale = max_log_occ / max_ic if max_ic else 1
    bottom_heights = -ic_array * scale

    # Set y-limits
    ax.set_ylim(-max_log_occ, max_log_occ)

    # Create custom colormap
    original_cmap = plt.colormaps['RdPu']
    colors = [original_cmap(0.5 + 0.5 * i/255) for i in range(256)]
    new_cmap = mcolors.ListedColormap(colors)
    norm = plt.Normalize(0, 1)

    # Plot bars
    for i in range(num_hosts):
        color = new_cmap(norm(recall_array[i]))
        ax.bar(x_positions[i], top_heights[i], width=0.8, color=color, linewidth=0)
        ax.bar(x_positions[i], bottom_heights[i], width=0.8, color=color, linewidth=0)

    # Y-ticks: top for occurrences, bottom for IC scores
    top_ticks  = list(range(0, int(math.floor(max_log_occ)) + 1))   # 0,1,2,…
    top_labels = ['0'] + [f'$10^{t-1}$' for t in top_ticks[1:]]     # 0,10^0,10^1,…

    num_bottom_ticks = 5
    ic_vals = np.linspace(0, max_ic, num_bottom_ticks+1)
    ic_positions = -ic_vals * scale
    ic_labels = [f'{v:.1f}' for v in ic_vals]
    ic_positions = ic_positions[::-1]
    ic_labels = ic_labels[::-1]

    combined_positions = list(ic_positions[:-1]) + top_ticks
    combined_labels = ic_labels[:-1] + top_labels
    ax.set_yticks(combined_positions)
    ax.set_yticklabels(combined_labels)

    ax.axhline(0, color='black', linewidth=1)

    # X-axis: show count of hosts at end
    ax.set_xlim(-0.5, num_hosts)
    ax.set_xticks([num_hosts])
    ax.set_xticklabels([f'{num_hosts}'], fontsize=18)
    ax.set_xlabel('Host List', fontsize=18)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label('Recall', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # Axis labels
    ax.set_ylabel('')
    ax.text(-0.07, 1.0, 'Number of Annotated Virus', rotation=90,
            ha='right', va='top', transform=ax.transAxes, fontsize=16)
    ax.text(-0.07, 0.3, 'IC Score', rotation=90,
            ha='right', va='top', transform=ax.transAxes, fontsize=16)

    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.tick_params(axis='both', labelsize=16)

    plt.show()
    plt.close()

    # Default save prefix for host predictions outputs
    host_predictions_plot._default_save_prefix = 'Host_Predictions'

    # --- Step 5: Scatter plot with log-scaled y-axis and custom ticks --------------
    # Re-uses: df, new_cmap, norm, STAT_LEVELS created above
    # Purpose : each host is a point; x = taxonomy level (with jitter), 
    #           y = log10(occurrences + 1); color = recall

    sns.set_theme(style="whitegrid")
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    level_pos = {lvl: i for i, lvl in enumerate(STAT_LEVELS)}
    x_ticks   = np.arange(len(STAT_LEVELS))
    x_labels  = [lvl.capitalize() for lvl in STAT_LEVELS]

    rng = np.random.default_rng(42)
    log_occ_values = []

    def occ_to_y(occ: int) -> float:
        """Map occurrences to y: 0 ↦ 0, else log10(occ)+1."""
        return 0.0 if occ == 0 else np.log10(occ) + 1.0

    for _, row in df.iterrows():
        x_val = level_pos[row['taxonomy_level']] + rng.uniform(-0.25, 0.25)
        y_val = occ_to_y(row['occurrences'])
        log_occ_values.append(y_val)

        ax2.scatter(
            x_val, y_val,
            color=new_cmap(norm(row['recall'])),
            s=40, alpha=0.85, edgecolors="none"
        )

    # Axis formatting -------------------------------------------------------------
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, fontsize=16)
    ax2.set_xlabel("Taxonomy Level", fontsize=16)

    max_pos = int(np.ceil(max(log_occ_values))) if log_occ_values else 1
    y_tick_positions = np.arange(0, max_pos + 1)          # 0,1,2,…
    y_tick_labels    = ['0'] + [fr'$10^{k-1}$' for k in y_tick_positions[1:]]

    ax2.set_yticks(y_tick_positions)
    ax2.set_yticklabels(y_tick_labels, fontsize=16)
    ax2.set_ylabel("Number of Annotated Virus", fontsize=16)

    ax2.set_ylim(-0.1, max_pos + 0.5)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Color bar -------------------------------------------------------------------
    sm2 = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.03, pad=0.03)
    cbar2.set_label("Recall", fontsize=16)
    cbar2.ax.tick_params(labelsize=16)

    plt.show()
    plt.close()

# ===============================================================
# distance_control_plot  ——  uses internal colour mapping
# ===============================================================
@auto_save_plots
def distance_control_plot(
    result_dict: Dict[str, Dict],
    virus_distance_path: str,
    thresholds: List[float],
    method_order: List[str],
    full_order_list: List[str],
    level: str,
    **kwargs
):
    """
    Draw Top-1 TPR / FDR curves (Genus level) for the selected methods
    while progressively filtering viruses whose Mash distance is below
    each threshold.

    Colours are generated internally so they match those used in
    major_plot (Tab20 palette minus indices 14 & 15, reversed).
    """
    # ---------- read mash distances ----------
    prefix = _current_prefix()

    virus2dist: Dict[str, float] = {}
    with open(virus_distance_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            cols = ln.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            try:
                virus2dist[cols[0]] = float(cols[1])
            except ValueError:
                logger.warning(f"[mash] invalid distance: {cols[:2]}")

    if not virus2dist:
        logger.error("Mash distance file is empty or unreadable.")
        return

    # ---------- consistent colour map ----------
    full_palette = sns.color_palette("tab20", n_colors=len(full_order_list) + 2)
    filtered_palette = [c for i, c in enumerate(full_palette) if i not in (14, 15)][::-1]
    colour_map = {m: filtered_palette[i] for i, m in enumerate(full_order_list)}

    # virus name list from any method result
    sample_result = next(iter(result_dict.values()))
    vid2name = {vid: name for vid, name in enumerate(sample_result["virus_ids"])}

    # ---------- compute metrics ----------
    metrics: Dict[str, Dict[float, Dict[str, float]]] = {m: {thr: {} for thr in thresholds} for m in method_order}

    for thr in thresholds:
        vids_kept = {
            vid for vid, vname in vid2name.items()
            if virus2dist.get(vname, float("inf")) >= thr
        }
        if not vids_kept:
            logger.warning(f"[{thr}] no viruses kept; skipping.")
            continue
        else:
            logger.info(f"Virus with distance >= {thr}: [{len(vids_kept)}/{len(vid2name)}] ")

        for method in method_order:
            res = result_dict.get(method)
            if res is None:
                continue
            gt_tx, pr_tx = res["gt_tx"], res["pr_tx"]

            n1 = n2 = n3 = 0
            for vid in vids_kept:
                true_set = gt_tx.get(vid, {}).get(level, set())
                if not true_set:
                    continue
                pred = pr_tx.get(vid, {}).get(level, [])
                if not pred:
                    n3 += 1
                    continue
                if pred[0][1] in true_set:
                    n1 += 1
                else:
                    n2 += 1

            tot = n1 + n2 + n3
            recall = n1 / tot if tot else float("nan")
            fdr = n2 / (n1 + n2) if (n1 + n2) else float("nan")
            metrics[method][thr]["recall"] = recall
            metrics[method][thr]["fdr"] = fdr

    # ---------- plotting (same style as plot_top_methods_level) ----------
    x = np.arange(len(thresholds))
    x_labels = [f"{t:.2g}" for t in thresholds]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(right=0.75)

    for method in method_order:
        rec_vals = [metrics[method][t]["recall"] for t in thresholds]
        fdr_vals = [metrics[method][t]["fdr"]  for t in thresholds]
        col = colour_map.get(method, "grey")
        ax.plot(x, rec_vals, marker="o", linestyle="-",  color=col, lw=2)
        ax.plot(x, fdr_vals, marker="^", linestyle="--", color=col, lw=2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlabel('Minimal Mash Distance', fontsize=14)
    ax.set_ylabel("Top-1 TPR / FDR", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # legends
    method_handles = [Line2D([0], [0], color=colour_map[m], lw=3) for m in method_order]
    metric_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-",  lw=2),
        Line2D([0], [0], color="black", marker="^", linestyle="--", lw=2),
    ]
    fig.legend(method_handles, method_order,
               loc="upper left", bbox_to_anchor=(0.77, 0.85),
               fontsize=14, borderaxespad=0.0,
               bbox_transform=fig.transFigure)
    fig.legend(metric_handles, ["Top-1 TPR", "Top-1 FDR"],
               loc="upper left", bbox_to_anchor=(0.77, 0.2),
               fontsize=14, borderaxespad=0.0,
               bbox_transform=fig.transFigure)

    plt.show()
    plt.close()

    # ---------- write TSV with distance-controlled metrics ----------
    if prefix:
        columns = [f"{thr:g}" for thr in thresholds]
        rows = []
        for method in method_order:
            row = {}
            for thr, col in zip(thresholds, columns):
                rec_val = metrics.get(method, {}).get(thr, {}).get("recall", np.nan)
                fdr_val = metrics.get(method, {}).get(thr, {}).get("fdr", np.nan)
                row[col] = _format_pair(rec_val, fdr_val)
            rows.append(pd.Series(row, name=method))
        table = pd.DataFrame(rows)
        table.index.name = "Method"
        _write_tsv(table, f"{prefix}_1.tsv", index=True, float_format=None)

@auto_save_plots
def host_predictions_head_to_head_compare(
    data1: list[dict],
    data2: list[dict],
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
    **kwargs,
) -> None:
    prefix = _current_prefix()
    # Merge two result sets on host_name
    df1 = pd.DataFrame(data1).set_index("host_name")
    df2 = pd.DataFrame(data2).set_index("host_name")
    df = pd.DataFrame({
        "recall_1": df1["recall"],
        "recall_2": df2["recall"],
        "occurrences": df1["occurrences"]
    })

    # Remove hosts with zero recall in both runs
    df = df[~((df["recall_1"] == 0) & (df["recall_2"] == 0))]
    if df.empty:
        logger.warning("All hosts have zero recall in both runs - nothing to plot.")
        return

    if prefix:
        recall_col_1 = f"Recall of {method1_name}"
        recall_col_2 = f"Recall of {method2_name}"
        export_df = df.reset_index().rename(columns={
            'host_name': 'Host Name',
            'occurrences': 'Host Occurrences',
            'recall_1': recall_col_1,
            'recall_2': recall_col_2,
        })
        export_df[recall_col_1] = export_df[recall_col_1].apply(lambda v: _format_decimal(v))
        export_df[recall_col_2] = export_df[recall_col_2].apply(lambda v: _format_decimal(v))
        export_df.insert(0, 'Index', np.arange(1, len(export_df) + 1))
        filename = f"{prefix}_1.tsv" if prefix else "Head_to_Head_1.tsv"
        _write_tsv(export_df, filename, index=False, float_format=None)

    # Statistics
    better_1 = df[df["recall_1"] > df["recall_2"]]
    better_2 = df[df["recall_2"] > df["recall_1"]]
    logger.info(
        f"{method1_name} > {method2_name}: {len(better_1)} hosts, "
        f"mean occurrences = {better_1['occurrences'].mean():.2f}"
    )
    logger.info(
        f"{method2_name} > {method1_name}: {len(better_2)} hosts, "
        f"mean occurrences = {better_2['occurrences'].mean():.2f}"
    )

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    base_cmap = plt.colormaps["RdPu"]
    colors = [base_cmap(0.2 + 0.8 * i / 255) for i in range(256)]
    new_cmap = mcolors.ListedColormap(colors)

    log_occ = np.clip(np.log10(df["occurrences"].replace(0, 1)), 1, 4)
    norm = mcolors.Normalize(vmin=1, vmax=4)

    sc = ax.scatter(
        df["recall_1"],
        df["recall_2"],
        c=log_occ,
        cmap=new_cmap,
        norm=norm,
        s=50,
        edgecolors="none",
        alpha=0.85
    )

    # Reference line y = x
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"Recall of {method1_name}", fontsize=14)
    ax.set_ylabel(f"Recall of {method2_name}", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Host Occurrences", fontsize=14)
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(["10¹", "10²", "10³", "10⁴"], fontsize=14)

    sns.despine(trim=True)
    plt.show()
    plt.close()

# Default save prefix for head-to-head comparison outputs
host_predictions_head_to_head_compare._default_save_prefix = 'Head_to_Head'

@auto_save_plots
def ecoli_test(
    ground_truth: str,
    test_result: str
) -> None:
    """
    Swarm plot of Escherichia coli counts vs recall status, with natural
    vertical spread of points.

    - Recalled : light-blue circles
    - Missed   : orange X's
    - Points are separated vertically by swarmplot's algorithm

    For each virus, parse the species block to find whether "Escherichia coli"
    appears and what its confidence score is. Rules:
      - If species contains "Escherichia coli" and it is the first (top-1),
        Type = Recalled, Confidence Score = that score.
      - If species contains "Escherichia coli" but not first, Type = Missed,
        Confidence Score = that score.
      - If species block does not contain "Escherichia coli", Type = Missed,
        Confidence Score = 0.
    Plot uses x = Escherichia Coli Association Count and y = Confidence Score (0-1).
    """
    prefix = _current_prefix()

    # 2. Read counts
    count_pat = re.compile(r"Escherichia coli\((\d+)\)")
    counts = {}
    with open(ground_truth, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            virus, token = line.split("\t")
            m = count_pat.search(token)
            counts[virus] = int(m.group(1)) if m else 0

    # 3. Determine recall based on the first species prediction only
    header_pat = re.compile(r"\[(.+?)\]\[(.+?)\]")
    # Track status, confidence, and rank per virus
    status_map: Dict[str, str] = {}
    conf_map: Dict[str, float] = {}
    rank_map: Dict[str, Optional[int]] = {}
    missed_list: list[str] = []

    current_virus: Optional[str] = None
    current_level: Optional[str] = None
    # State for current species block
    top_label: Optional[str] = None
    top_score: Optional[float] = None
    ecoli_present: bool = False
    ecoli_score: Optional[float] = None
    ecoli_rank: Optional[int] = None
    species_rank: int = 0

    def finalize_species_block() -> None:
        nonlocal current_virus, top_label, top_score, ecoli_present, ecoli_score, ecoli_rank
        if current_virus is None:
            return
        if ecoli_present:
            if top_label == "Escherichia coli":
                status_map[current_virus] = "Recalled"
                conf_map[current_virus] = float(ecoli_score or 0.0)
                rank_map[current_virus] = int(ecoli_rank or 1)
            else:
                status_map[current_virus] = "Missed"
                conf_map[current_virus] = float(ecoli_score or 0.0)
                rank_map[current_virus] = int(ecoli_rank or 2)
                # E. coli appears but not top-1 -> treat as missed
                missed_list.append(current_virus)
        else:
            status_map[current_virus] = "Missed"
            conf_map[current_virus] = 0.0
            rank_map[current_virus] = None
            missed_list.append(current_virus)

    with open(test_result, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue

            hdr = header_pat.fullmatch(line.strip())
            if hdr:
                # finalize previous species block if needed
                if current_level == "species":
                    finalize_species_block()

                current_virus, current_level = hdr.group(1), hdr.group(2).lower()
                # reset species block state
                if current_level == "species":
                    top_label = None
                    top_score = None
                    ecoli_present = False
                    ecoli_score = None
                    ecoli_rank = None
                    species_rank = 0
                continue

            # Species block content
            if current_level == "species" and current_virus:
                try:
                    label, score_str = line.split("\t")
                    score = float(score_str)
                except ValueError:
                    # malformed line, skip
                    continue

                species_rank += 1

                if top_label is None:
                    top_label = label
                    top_score = score
                if label == "Escherichia coli":
                    ecoli_present = True
                    ecoli_score = score
                    ecoli_rank = species_rank

    # finalize last block at EOF
    if current_level == "species":
        finalize_species_block()

    # 4. Build DataFrame for plotting
    rows = []
    for virus, cnt in counts.items():
        status = status_map.get(virus, "Missed")
        conf = conf_map.get(virus, 0.0)
        rnk = rank_map.get(virus, None)
        rows.append({"virus": virus, "count": cnt, "status": status, "confidence": conf, "rank": rnk})
    df = pd.DataFrame(rows)

    if prefix:
        export_df = df[['virus', 'count', 'status', 'confidence', 'rank']].copy()
        export_df.rename(columns={
            'virus': 'Virus Name',
            'count': 'Escherichia Coli Association Count',
            'status': 'Type',
            'confidence': 'Confidence Score',
            'rank': 'Rank'
        }, inplace=True)
        # Convert missing ranks (None/NaN) to string 'None' explicitly
        if 'Rank' in export_df.columns:
            export_df['Rank'] = export_df['Rank'].apply(
                lambda v: 'None' if (v is None or pd.isna(v)) else int(v)
            )
        export_df.insert(0, 'Index', np.arange(1, len(export_df) + 1))
        filename = f"{prefix}_1.tsv" if prefix else "Ecoli_Test_1.tsv"
        _write_tsv(export_df, filename, index=False, float_format=None)

    # 5. Compute and print recall stats and missed names
    total = len(df)
    rec_num = df["status"].eq("Recalled").sum()
    miss_num = total - rec_num
    recall_rate = rec_num / total if total else 0.0

    print(f"Recall:{recall_rate:.3f}")
    print(f"({rec_num}/{total} recalled)")
    print(f"({miss_num}/{total} missed)")
    # Use the tracked missed_list (includes not-present and not-top-1 cases)
    print("Missed viruses:", ", ".join(sorted(set(missed_list))))

    # 6. Plot: x = count, y = confidence (0-1)
    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(8.5, 4.5))
    ax = plt.gca()
    # Recalled
    sub = df[df.status == "Recalled"]
    ax.scatter(sub["count"], sub["confidence"], s=50, c="#9ecae1", marker="o", label="Recalled", alpha=0.8)
    # Missed
    sub = df[df.status == "Missed"]
    ax.scatter(sub["count"], sub["confidence"], s=50, c="#f28e2b", marker="x", label="Missed", alpha=0.9)

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Confidence Score")
    ax.set_xlabel("Escherichia Coli Association Count")
    ax.legend(loc="upper right")

    plt.show()
    plt.close()

# Default save prefix for E. coli test outputs
ecoli_test._default_save_prefix = 'Ecoli_Test'

@auto_save_plots
def ecoli_plot(
    virus_file: str,
    score_file: str,
    taxonomy_file: str,
    output_dir: str
) -> None:
    def hex_to_rgb(hex_str):
        """Convert hex color code to an (R, G, B) tuple."""
        h = hex_str.lstrip('#')
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


    def rgb_to_hex(rgb):
        """Convert (R, G, B) tuple back to hex color code."""
        return "#{:02x}{:02x}{:02x}".format(*rgb)


    def blend_with_white(base_rgb, factor):
        """
        Blend the given RGB color with white.
        factor = 0 keeps the original color; factor = 1 returns white.
        """
        return tuple(int(c + (255 - c) * factor) for c in base_rgb)
    # --------------------------------------------------------------------------- #
    # 2. Virus metadata: Virus, Genus, Morphotype
    # --------------------------------------------------------------------------- #
    virus_df = pd.read_csv(virus_file, sep="\t", usecols=["Virus", "Genus", "Morphotype"])

    # --------------------------------------------------------------------------- #
    # 3. Base colors for each Morphotype (non-blue, to avoid confusion)
    # --------------------------------------------------------------------------- #
    morph_base_color = {
        "Siphoviridae": "#e41a1c",  # red
        "Podoviridae":  "#4daf4a",  # green
        "Myoviridae":   "#984ea3"   # purple
    }
    default_color = "#999999"

    # --------------------------------------------------------------------------- #
    # 4. Shades for each Genus under its Morphotype
    # --------------------------------------------------------------------------- #
    morph_order = ["Siphoviridae", "Podoviridae", "Myoviridae"]
    genus_by_morph = {m: [] for m in morph_order}

    for _, row in virus_df.iterrows():
        morph, genus = row["Morphotype"], row["Genus"]
        if morph in genus_by_morph and genus not in genus_by_morph[morph]:
            genus_by_morph[morph].append(genus)

    genus_color = {}
    for morph in morph_order:
        base_rgb = hex_to_rgb(morph_base_color.get(morph, default_color))
        sub_genera = genus_by_morph[morph]
        n = len(sub_genera)
        for idx, genus in enumerate(sub_genera):
            # Lighten progressively toward white (factor ≤ 0.6)
            factor = idx / max(1, n - 1) * 0.6
            genus_color[genus] = rgb_to_hex(blend_with_white(base_rgb, factor))

    virus_color = {
        row["Virus"]: genus_color.get(row["Genus"], default_color)
        for _, row in virus_df.iterrows()
    }

    # --------------------------------------------------------------------------- #
    # 5. X-axis order: Morphotype → Genus → original order
    # --------------------------------------------------------------------------- #
    virus_order = []
    for morph in morph_order:
        sub = virus_df[virus_df["Morphotype"] == morph]
        for genus in genus_by_morph[morph]:
            virus_order.extend(sub[sub["Genus"] == genus]["Virus"].tolist())
    for v in virus_df["Virus"].tolist():
        if v not in virus_order:
            virus_order.append(v)

    # --------------------------------------------------------------------------- #
    # 6. Host taxonomy: keep original host order
    # --------------------------------------------------------------------------- #
    taxonomy_df = pd.read_csv(taxonomy_file, sep="\t")
    host_to_species = dict(zip(taxonomy_df["host"], taxonomy_df["species"]))
    host_order = taxonomy_df["host"].tolist()

    # --------------------------------------------------------------------------- #
    # 7. Read virus-bacteria score matrix
    # --------------------------------------------------------------------------- #
    pattern = re.compile(r"^(.*)\((\d+)\)$")
    score_matrix = {}
    with open(score_file, encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split("\t")
            virus_name = parts[0]
            for item in parts[1:]:
                m = pattern.match(item)
                if not m:
                    continue
                bacterium, score = m.group(1), int(m.group(2))
                score_matrix.setdefault(bacterium, {})[virus_name] = score

    # --------------------------------------------------------------------------- #
    # 8. Filter bacteria that actually appear in the matrix
    # --------------------------------------------------------------------------- #
    bacteria_list = [h for h in host_order if h in score_matrix]

    # --------------------------------------------------------------------------- #
    # 9. Species-level palette (non-blue)
    # --------------------------------------------------------------------------- #
    custom_palette = [
        "#ff7f00", "#ffff33", "#a65628", "#f781bf",
        "#8dd3c7", "#bebada", "#fc8d62", "#b3de69"
    ]
    species_set = sorted({host_to_species.get(b, "_") for b in bacteria_list})
    species_base = {
        sp: custom_palette[i % len(custom_palette)]
        for i, sp in enumerate(species_set)
    }

    bacterium_color = {}
    for sp in species_set:
        group = [b for b in bacteria_list if host_to_species.get(b, "_") == sp]
        base_rgb = hex_to_rgb(species_base[sp])
        n = len(group)
        for idx, bacterium in enumerate(group):
            factor = idx / max(1, n - 1) * 0.6
            bacterium_color[bacterium] = rgb_to_hex(blend_with_white(base_rgb, factor))

    # --------------------------------------------------------------------------- #
    # 10. Build DataFrame for plotting
    # --------------------------------------------------------------------------- #
    df = pd.DataFrame(0, index=bacteria_list, columns=virus_order)
    for bacterium, virus_scores in score_matrix.items():
        for virus, score in virus_scores.items():
            if virus in df.columns:
                df.at[bacterium, virus] = score

    prefix = _current_prefix()
    if prefix:
        export_df = df.copy()
        export_df.index.name = 'Host'
        filename = f"{prefix}.tsv" if prefix else "bokeh_plot.tsv"
        _write_tsv(export_df, filename, index=True, float_format=None)

    # --------------------------------------------------------------------------- #
    # 11. Flatten data for ColumnDataSource (cap at 100 -> 10², +1 to avoid log(0))
    # --------------------------------------------------------------------------- #
    x_vals, y_vals, color_vals, raw_vals = [], [], [], []
    for virus in virus_order:
        for bacterium in reversed(bacteria_list):
            x_vals.append(virus)
            y_vals.append(bacterium)
            raw = df.at[bacterium, virus]
            capped = min(raw, 100)
            color_vals.append(capped + 1)
            raw_vals.append(capped)

    source = ColumnDataSource(dict(
        x=x_vals, y=y_vals,
        score=raw_vals,          # shown in hover
        score_col=color_vals     # used for coloring
    ))

    # --------------------------------------------------------------------------- #
    # 12. Log-scale color mapper (Blues256 reversed for light→dark)
    # --------------------------------------------------------------------------- #
    color_mapper = LogColorMapper(
        palette=Blues256[::-1],  # reverse palette
        low=1, high=101
    )

    # --------------------------------------------------------------------------- #
    # 13. Build figure and render
    # --------------------------------------------------------------------------- #
    p = figure(
        width=40 + 40 * len(virus_order),
        height=40 + 20 * len(bacteria_list),
        x_range=FactorRange(factors=virus_order),
        y_range=FactorRange(factors=list(reversed(bacteria_list))),
        title="Virus-Host Association",
        tools="hover,save,reset,pan,box_zoom,wheel_zoom"
    )

    # Heatmap rectangles
    p.rect(
        "x", "y", width=1, height=1, source=source,
        line_color=None,
        fill_color={"field": "score_col", "transform": color_mapper}
    )

    # Hover tooltip
    p.select_one(HoverTool).tooltips = [
        ("Virus", "@x"), ("Host", "@y"), ("MLC score", "@score")
    ]

    # Axis & grid styling
    p.xaxis.major_label_orientation = 0.8
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Color bar: logarithmic, ticks at 1, 11, 101
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=FixedTicker(ticks=[1, 11, 101]),
        major_label_overrides={1: "0", 11: "10¹", 101: "10²+"},
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Association count"
    )
    p.add_layout(color_bar, 'right')

    # --------------------------------------------------------------------------- #
    # 14. Virus swatch (below) and bacteria swatch (left)
    # --------------------------------------------------------------------------- #
    swatch_y = -1  # row just below the matrix
    p.rect(
        x=virus_order, y=[swatch_y] * len(virus_order),
        width=1, height=0.5,
        fill_color=[virus_color[v] for v in virus_order],
        line_color=None
    )
    p.rect(
        x=[-1] * len(bacteria_list),
        y=list(reversed(bacteria_list)),
        width=0.5, height=1,
        fill_color=[bacterium_color[b] for b in reversed(bacteria_list)],
        line_color=None
    )

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "ecoli_plot.html")
    output_file(html_path)
    save(p)

# default capture prefix when user doesn't supply one
ecoli_plot._default_save_prefix = 'Ecoli_Plot'

def draw_taxonomy_diagram(
    host_name,
    infraspecies,
    species,
    genus,
    family,
    order,
    class_,
    phylum,
    title: str | None = None,
    char_scale: float = 0.5,
    min_box_width: float = 1.0,
    units_to_inches: float = 0.35,
):
    """
    Draws a one-line hierarchical taxonomy diagram using Matplotlib.

    Parameters
    ----------
    host_name : str
        The host name.
    infraspecies : str
        Infraspecies name.
    species : str
        Species name.
    genus : str
        Genus name.
    family : str
        Family name.
    order : str
        Order name.
    class_ : str
        Class name.
    phylum : str
        Phylum name.

    Returns
    -------
    None
    """

    # Prepare the taxonomy levels from species to kingdom
    taxonomy_levels = [
        ("Infraspecies", infraspecies),
        ("Species", species),
        ("Genus", genus),
        ("Family", family),
        ("Order", order),
        ("Class", class_),
        ("Phylum", phylum),
    ]

    # --- Dynamically determine figure width from content length ---
    # Precompute widths for all taxonomy boxes based on text length
    pre_box_widths = []
    for level_name, level_value in taxonomy_levels:
        max_text_len = max(len(level_name), len(level_value))
        pre_box_widths.append(max(min_box_width, char_scale * max_text_len))

    # Layout constants (Host box removed)
    left_margin_units = 2.0
    box_gap = 1.5

    # Total logical width in axis units (no Host box)
    total_units = left_margin_units + sum(pre_box_widths) + max(0, len(pre_box_widths) - 1) * box_gap
    margin_units = 4
    total_units_with_margin = total_units + margin_units

    # Map logical units to inches, clamp to a reasonable range
    fig_w = max(10, min(30, units_to_inches * total_units_with_margin))
    fig_h = 2

    # Set up the figure size and axes bounds
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    ax.set_xlim(0, total_units_with_margin)
    ax.set_ylim(0, 5)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, pad=10)

    # ====================
    # (1) Draw the horizontal chain (species -> genus -> family -> ...)
    # ====================
    # Start coordinates after a left margin (Host box removed)
    start_y = 2
    start_x = left_margin_units

    # Lighter/pastel color map for boxes
    pastel_colors = plt.cm.Pastel1(np.linspace(0, 1, len(taxonomy_levels)))

    # Default heights and gap
    box_height = 0.8
    # box_gap already defined above to synchronize sizing

    # Keep track of previous box center for arrows
    prev_center_x = None
    prev_center_y = None

    for i, (level_name, level_value) in enumerate(taxonomy_levels):
        # Calculate box width dynamically based on text length
        # We use the longer length between level name and level value
        max_text_len = max(len(level_name), len(level_value))
        # Scale factor for each character; adjust as needed
        # The minimum box width is 3
        box_width = max(min_box_width, char_scale * max_text_len)

        # Current x and y coordinates for the box
        cur_x = start_x
        cur_y = start_y

        # Draw the box
        box = patches.FancyBboxPatch(
            (cur_x, cur_y),
            width=box_width,
            height=box_height,
            boxstyle="round,pad=0.3",
            edgecolor='black',
            facecolor=pastel_colors[i]  # Use lighter pastel color
        )
        ax.add_patch(box)

        # Text above the box (taxonomy level label)
        ax.text(
            cur_x + box_width / 2,
            cur_y + box_height + 0.3,
            level_name,
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

        # Text inside the box (actual taxonomy value)
        ax.text(
            cur_x + box_width / 2,
            cur_y + box_height / 2,
            level_value,
            ha='center',
            va='center',
            fontsize=14
        )

        # Draw an arrow from the previous box if this is not the first taxonomy level
        if i > 0:
            ax.arrow(
                prev_center_x,
                prev_center_y,
                (cur_x - prev_center_x),
                0,
                length_includes_head=True,
                head_width=0.2,
                head_length=0.6,
                fc='k',
                ec='k'
            )

        # Update the "previous box center" position
        prev_center_x = cur_x + box_width
        prev_center_y = cur_y + box_height / 2

        # Update start_x so that the next box is placed to the right
        start_x += box_width + box_gap

    plt.tight_layout()
    plt.show()

@auto_save_plots
def analyze_and_visualize(label_transfer_file: str, taxonomy_file: str) -> None:
    """
    Parse a vhseek label-transfer file, print summaries and draw a non-overlapping
    horizontal taxonomy tree. Independent sub-trees (separate phyla)
    get disjoint y-ranges, eliminating visual overlap.

    Parameters
    ----------
    label_transfer_file : str
        Path to vhseek label-transfer output.
    taxonomy_file : str
        TSV with columns:
        host, infraspecies, species, genus, family, order, class, phylum, kingdom
    """
    levels = STAT_LEVELS
    df_tax = pd.read_csv(taxonomy_file, sep="\t", dtype=str).fillna("_")

    # ------------ 1. Parse label-transfer blocks → nested dict -------------
    virus_dict: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    with open(label_transfer_file, encoding="utf-8") as fh:
        cur_virus = cur_level = None
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"\[(.+)]\[(\w+)]", line)
            if m:
                cur_virus, lvl = m.group(1), m.group(2)
                cur_level = lvl if lvl in levels else None
                if cur_level:
                    virus_dict.setdefault(cur_virus, {}).setdefault(cur_level, [])
                continue

            if cur_virus and cur_level:
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    host, sc = parts
                    try:
                        virus_dict[cur_virus][cur_level].append((host, float(sc)))
                    except ValueError:
                        pass  # ignore bad score

    # ---------------- 2. Process every virus -----------------
    for index, (virus, lvl_data) in enumerate(virus_dict.items()):
        deepest = next((lvl for lvl in levels if lvl in lvl_data), None)
        if deepest is None:
            print(f"[WARNING] No taxonomy for '{virus}'. Skipping.")
            continue

        top_host, top_score = max(lvl_data[deepest], key=lambda x: x[1])
        row_top = df_tax[df_tax["host"] == top_host]
        if row_top.empty:
            print(f"[WARNING] Host '{top_host}' not found for '{virus}'.")
            continue
        tax_vals = [row_top.iloc[0][lvl] for lvl in levels]

        # ---- summary ----
        print(f"Virus: {virus}")
        print(f"Deepest level with data: {deepest}")
        print(f"Top host: {top_host} (score = {top_score})")
        print("Taxonomy:")
        for l, v_ in zip(levels, tax_vals):
            print(f"  {l}: {v_}")
        print("")
        draw_taxonomy_diagram(top_host, *tax_vals, title=f"Top-1 Result for {virus}")

        # ----------- 3. Build edges / node meta --------------
        edges = set()
        node_level: Dict[str, str] = {}
        node_score: Dict[str, float] = {}

        for lvl, pairs in lvl_data.items():
            for host, sc in pairs:
                node_level[host] = lvl
                node_score[host] = sc
                # walk up to phylum
                row_host = df_tax[df_tax[lvl] == host]
                if row_host.empty:
                    continue
                tax_row = row_host.iloc[0]
                idx = levels.index(lvl)
                prev = host
                for higher in levels[idx + 1 :]:
                    parent = tax_row[higher]
                    edges.add((prev, parent))
                    node_level[parent] = higher
                    prev = parent

        # children mapping
        children = defaultdict(list)
        for child, parent in edges:
            children[parent].append(child)

        # -------------- 4. Group leaves by phylum --------------
        def gather_leaves(node: str) -> List[str]:
            """Recursively gather leaves under node."""
            if node not in children:
                return [node]
            res: List[str] = []
            for ch in children[node]:
                res.extend(gather_leaves(ch))
            return res

        # roots == nodes at 'phylum' level (they never appear as child)
        roots = [n for n, lvl in node_level.items() if lvl == "phylum"]

        leaves_order: List[str] = []
        for root in roots:  # preserve root order
            leaves_order.extend(gather_leaves(root))

        # unique while keeping order
        seen = set()
        leaves_seq = [x for x in leaves_order if not (x in seen or seen.add(x))]
        n_leaf = len(leaves_seq)
        y_map = {
            leaf: i / (n_leaf - 1) if n_leaf > 1 else 0.5
            for i, leaf in enumerate(leaves_seq)
        }

        # -------------- 5. Assign y to all nodes --------------
        def assign_y(node: str) -> float:
            if node in y_map:
                return y_map[node]
            ys = [assign_y(c) for c in children.get(node, [])]
            y_val = sum(ys) / len(ys) if ys else 0.5
            y_map[node] = y_val
            return y_val

        for n in node_level:
            assign_y(n)
        
        # ---------------- 6. Matplotlib / Seaborn drawing ----------------
        sns.set_theme(style="white")

        # Figure size adapts to tree size
        fig_h = max(4, 0.25 * n_leaf)          # n_leaf was computed earlier
        fig_w = 2 * len(levels)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # ---- draw edges first (under labels) ----
        for child, parent in edges:
            x0, x1 = levels.index(node_level[child]), levels.index(node_level[parent])
            y0, y1 = y_map[child], y_map[parent]
            ax.plot([x0, x1], [y0, y1],
                    color="gray", linewidth=2, zorder=1)

        # ---- draw label rectangles (no scatter points) ----
        for node, lvl in node_level.items():
            x, y = levels.index(lvl), y_map[node]

            # Compose label with score if available
            score = node_score.get(node)
            label = f"{node}\n({score:.3f})" if score is not None else node

            # Centered text with rounded rectangle background
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=10,
                zorder=2,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="aliceblue",          # fill color
                    ec="steelblue",          # edge color
                    lw=0.8
                )
            )

        # ---- axis formatting ----
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([lvl.capitalize() for lvl in levels], fontsize=10)

        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([])                      # hide y ticks

        sns.despine(left=True, bottom=True)

        ax.set_title(f"Taxonomy Tree for {virus}", fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()

# default capture prefixes for automatic export
analyze_and_visualize._default_save_prefix = 'Label_Transfer_Taxonomy'

def specific_host_analysis(
    label_transfer_path: str,
    experiment_path: str,
    host_name: Optional[str] = None,
    taxonomy_path: Optional[str] = None,
    exclude_virus_path: Optional[str] = None,
    unlabeled_virus_path: Optional[str] = None,
    return_unlabeled_scores: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, List[float]], Optional[Dict[str, List[float]]]]:
    """
    Compare two label-transfer files and return TP / FP / FN dictionaries.
    
    Parameters
    ----------
    label_transfer_path : str
        Path to file #1 (virus + host labels, tab-separated).
    experiment_path : str
        Path to file #2 (block format: [virus][level] then host/score lines).
    host_name : Optional[str]
        When provided, analyze this host only (exact, case-sensitive).
        When None, analyze ALL virus–host predictions in both files.
    exclude_virus_path : Optional[str]
        Optional TSV/line list of virus IDs to exclude (first column used).
    unlabeled_virus_path : Optional[str]
        Optional TSV/line list of virus IDs considered "unlabeled" (first column used).
        When provided, extension scores from these viruses are separated into
        `unlabeled_level_scores` for downstream plotting.
    return_unlabeled_scores : bool
        When True, returns an extra `unlabeled_level_scores` dict as the last value.
    
    Returns
    -------
    tp_dict : Dict[str, float]
        If `host_name` is provided: {virus -> score} where virus has
        `host_name` in BOTH files; score from file #2.
        If `host_name` is None: {"virus | host" -> score} for all virus–host
        pairs present in BOTH files; score from file #2.
    fp_dict : Dict[str, float]
        If `host_name` is provided: {virus -> score} where virus has
        `host_name` ONLY in file #2; score from file #2.
        If `host_name` is None: {"virus | host" -> score} for all virus–host
        pairs present ONLY in file #2; score from file #2.
    fn_dict : Dict[str, float]
        If `host_name` is provided: {virus -> 0.0} where virus has
        `host_name` ONLY in file #1.
        If `host_name` is None: {"virus | host" -> 0.0} for all virus–host
        pairs present ONLY in file #1.
    unlabeled_level_scores : Optional[Dict[str, List[float]]]
        Only returned when `return_unlabeled_scores=True`. Scores from
        unlabeled viruses grouped by taxonomy level.
        When `return_unlabeled_scores=False`, the function returns the first
        five values only.
    """
    # ----------------------- parse file #1 (GT) -----------------------
    gt_flat: Dict[str, Set[str]] = defaultdict(set)
    gt_level_data: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    
    host_level_map = get_deepest_toxonomy(taxonomy_path) if taxonomy_path else {}

    with Path(label_transfer_path).open(encoding="utf-8") as f1:
        for line in f1:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue
            virus = parts[0].strip()
            hosts = {h.strip() for h in parts[1:] if h.strip()}
            if not virus or not hosts:
                continue

            gt_flat[virus].update(hosts)
            if host_level_map:
                for h in hosts:
                    lvl = host_level_map.get(h)
                    if lvl:
                        gt_level_data[virus][lvl.lower()].add(h)

    # ----------------------- parse file #2 (Pred) -----------------------
    header_re = re.compile(r"\[(.+?)\]\[(.+?)\]")
    pred_flat: Dict[str, Dict[str, float]] = defaultdict(dict)
    pred_level_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    
    current_virus = None
    current_level = None

    with Path(experiment_path).open(encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue

            if line.startswith("["):
                m = header_re.match(line)
                if not m:
                    continue
                current_virus = m.group(1).strip()
                current_level = m.group(2).strip().lower()
                continue

            parts = line.split("\t")
            if len(parts) != 2 or current_virus is None or current_level is None:
                continue
            host, score_str = parts[0].strip(), parts[1].strip()
            score = float(score_str)

            prev = pred_flat[current_virus].get(host, 0.0)
            if score > prev:  # keep maximum score for duplicate host entries
                pred_flat[current_virus][host] = score
            
            if current_level:
                prev_lvl = pred_level_data[current_virus][current_level].get(host, 0.0)
                if score > prev_lvl:
                    pred_level_data[current_virus][current_level][host] = score

    exclude_viruses: Set[str] = set()
    if exclude_virus_path:
        with Path(exclude_virus_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                virus = line.split("\t", 1)[0].strip()
                if virus:
                    exclude_viruses.add(virus)

    unlabeled_viruses: Set[str] = set()
    if unlabeled_virus_path:
        with Path(unlabeled_virus_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                virus = line.split("\t", 1)[0].strip()
                if virus:
                    unlabeled_viruses.add(virus)
        if exclude_viruses:
            unlabeled_viruses -= exclude_viruses
        if unlabeled_viruses and not return_unlabeled_scores:
            logger.warning(
                "unlabeled_virus_path provided but return_unlabeled_scores=False; "
                "unlabeled extension scores will not be returned."
            )

    # ----------------------- build TP / FP / FN -----------------------
    if host_name is not None:
        # Viruses associated with host_name in GT or Pred
        target_viruses = {v for v, hosts in gt_flat.items() if host_name in hosts}
        target_viruses.update({v for v, h_map in pred_flat.items() if host_name in h_map})
    else:
        target_viruses = set(gt_flat.keys()) | set(pred_flat.keys())
    if exclude_viruses:
        target_viruses -= exclude_viruses

    tp_dict, fp_dict, fn_dict = {}, {}, {}
    original_level_counts = defaultdict(int)
    # Only keep prediction scores for labels not present in GT at the same level.
    experiment_level_scores = defaultdict(list)
    unlabeled_level_scores = defaultdict(list)

    for virus in target_viruses:
        # 1. TP/FP/FN Analysis
        if host_name is not None:
            in_gt = host_name in gt_flat.get(virus, set())
            in_pred = host_name in pred_flat.get(virus, {})
            score = pred_flat.get(virus, {}).get(host_name, 0.0)

            if in_gt and in_pred:
                tp_dict[virus] = score
            elif not in_gt and in_pred:
                fp_dict[virus] = score
            elif in_gt and not in_pred:
                fn_dict[virus] = 0.0
        else:
            gt_hosts = gt_flat.get(virus, set())
            pred_map = pred_flat.get(virus, {})

            for host, score in pred_map.items():
                key = f"{virus} | {host}"
                if host in gt_hosts:
                    tp_dict[key] = score
                else:
                    fp_dict[key] = score

            for host in gt_hosts - set(pred_map.keys()):
                key = f"{virus} | {host}"
                fn_dict[key] = 0.0
        
        # 2. Level Counts/Scores (Extension Plot)
        if virus in gt_level_data:
            for lvl, hosts in gt_level_data[virus].items():
                if host_name is not None:
                    if host_name in hosts:
                        original_level_counts[lvl] += 1
                else:
                    original_level_counts[lvl] += len(hosts)
        
        if virus in pred_level_data:
            is_unlabeled = bool(unlabeled_viruses and virus in unlabeled_viruses)
            for lvl, host_map in pred_level_data[virus].items():
                gt_hosts_lvl = gt_level_data.get(virus, {}).get(lvl, set())
                if host_name is not None:
                    if host_name in host_map and host_name not in gt_hosts_lvl:
                        if is_unlabeled:
                            unlabeled_level_scores[lvl].append(host_map[host_name])
                        else:
                            experiment_level_scores[lvl].append(host_map[host_name])
                else:
                    for host, score in host_map.items():
                        if host not in gt_hosts_lvl:
                            if is_unlabeled:
                                unlabeled_level_scores[lvl].append(score)
                            else:
                                experiment_level_scores[lvl].append(score)

    logger.info(f"TP Number = {len(tp_dict)}")
    logger.info(f"FP Number = {len(fp_dict)}")
    logger.info(f"FN Number = {len(fn_dict)}")

    global _LAST_EXTENSION_PAIRS, _LAST_LABELED_EXTENSION_PAIRS, _LAST_UNLABELED_EXTENSION_PAIRS
    if host_name is None:
        _LAST_EXTENSION_PAIRS = dict(fp_dict)
        if unlabeled_viruses:
            labeled_pairs: Dict[str, float] = {}
            unlabeled_pairs: Dict[str, float] = {}
            for pair, score in fp_dict.items():
                virus = pair.split(" | ", 1)[0]
                if virus in unlabeled_viruses:
                    unlabeled_pairs[pair] = score
                else:
                    labeled_pairs[pair] = score
            _LAST_LABELED_EXTENSION_PAIRS = labeled_pairs
            _LAST_UNLABELED_EXTENSION_PAIRS = unlabeled_pairs
        else:
            _LAST_LABELED_EXTENSION_PAIRS = None
            _LAST_UNLABELED_EXTENSION_PAIRS = None
    else:
        _LAST_EXTENSION_PAIRS = None
        _LAST_LABELED_EXTENSION_PAIRS = None
        _LAST_UNLABELED_EXTENSION_PAIRS = None

    if return_unlabeled_scores:
        return tp_dict, fp_dict, fn_dict, original_level_counts, experiment_level_scores, unlabeled_level_scores
    return tp_dict, fp_dict, fn_dict, original_level_counts, experiment_level_scores

def _export_specific_host_tables(tp: Dict[str, float], fp: Dict[str, float], fn: Dict[str, float] | None, *, threshold: float | None) -> None:
    """Common TSV exports for specific-host plots.

    - Always exports the full TP/FP/FN table as `{prefix}_1.tsv` when a prefix is active.
    - If `threshold` is provided, also exports FP rows with score > threshold as `{prefix}_2.tsv`.
    """
    prefix = _current_prefix()
    if fn is None:
        fn = {}
    if not prefix:
        return
    table_rows: list[dict] = []
    for virus, score in tp.items():
        table_rows.append({"Virus Name": virus, "Confidence Score": score, "Type": "TP"})
    for virus, score in fp.items():
        table_rows.append({"Virus Name": virus, "Confidence Score": score, "Type": "FP"})
    for virus, score in fn.items():
        table_rows.append({"Virus Name": virus, "Confidence Score": score, "Type": "FN"})
    if table_rows:
        export_df = pd.DataFrame(table_rows)
        export_df.sort_values(by="Confidence Score", ascending=False, inplace=True)
        export_df['Confidence Score'] = export_df['Confidence Score'].apply(lambda v: _format_decimal(v))
        export_df.insert(0, 'Index', np.arange(1, len(export_df) + 1))
        filename = f"{prefix}_1.tsv" if prefix else "Human_Health_1.tsv"
        _write_tsv(export_df, filename, index=False, float_format=None)

        if threshold is not None:
            fp_rows = [
                {"Virus Name": virus, "Confidence Score": score, "Type": "FP"}
                for virus, score in fp.items()
                if score > threshold
            ]
            fp_export_df = pd.DataFrame(fp_rows)
            if not fp_export_df.empty:
                fp_export_df.sort_values(by="Confidence Score", ascending=False, inplace=True)
                fp_export_df['Confidence Score'] = fp_export_df['Confidence Score'].apply(lambda v: _format_decimal(v))
                fp_export_df.insert(0, 'Index', np.arange(1, len(fp_export_df) + 1))
                filename_fp = f"{prefix}_2.tsv" if prefix else "Human_Health_2.tsv"
                _write_tsv(fp_export_df, filename_fp, index=False, float_format=None)


@auto_save_plots
def specific_host_scatter_plot(
    tp: Dict[str, float],
    fp: Dict[str, float],
    fn: Optional[Dict[str, float]] = None,
    threshold: float = 0.835,
) -> None:
    """
    Scatter plot version: one threshold controlling color mapping.
    - TP above threshold: blue; below: grey
    - FP above threshold: red; below: grey
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from logzero import logger

    # Export TSVs
    _export_specific_host_tables(tp, fp, fn, threshold=threshold)

    # Data
    rng = np.random.default_rng(42)
    y_tp = list(tp.values())
    y_fp = list(fp.values())

    # Colors
    color_tp_above = (100/255, 150/255, 255/255, 0.8)      # light blue
    color_fp_above = (255/255, 100/255, 100/255, 0.8)      # light red
    color_below    = (160/255, 160/255, 160/255, 0.7)      # light grey

    # Stats/logs
    tp_above = sum(y > threshold for y in y_tp)
    fp_above = sum(y > threshold for y in y_fp)
    tp_ratio = tp_above / len(y_tp) if y_tp else 0.0
    fp_ratio = fp_above / len(y_fp) if y_fp else 0.0
    logger.info(f"TP above threshold: {tp_above}/{len(y_tp)} ({tp_ratio:.1%})")
    logger.info(f"FP above threshold: {fp_above}/{len(y_fp)} ({fp_ratio:.1%})")

    fp_above_list = [(v, s) for v, s in fp.items() if s > threshold]
    if fp_above_list:
        logger.info("\nFP virus above threshold:")
        for v, s in fp_above_list:
            logger.info(f"  {v}\t{s:.3f}")
    else:
        logger.info("\nNo FP virus above threshold.")

    # Plot
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (8.2, 5.2),
        "axes.facecolor": (0.94, 0.94, 0.98, 0.8)
    })
    fig, ax = plt.subplots()

    x_tp = rng.uniform(-0.2, 0.2, size=len(y_tp)) + 0        # TP at x≈0
    x_fp = rng.uniform(-0.2, 0.2, size=len(y_fp)) + 1        # FP at x≈1
    colors_tp = [color_tp_above if y > threshold else color_below for y in y_tp]
    colors_fp = [color_fp_above if y > threshold else color_below for y in y_fp]

    ax.scatter(x_tp, y_tp, s=120, c=colors_tp, linewidths=0, zorder=3, label="TP")
    ax.scatter(x_fp, y_fp, s=120, c=colors_fp, linewidths=0, zorder=3, label="FP")

    # Reference threshold line
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, zorder=2)

    # Axis formatting
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TP", "FP"], fontsize=20)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylabel("Score", fontsize=20)
    ax.set_xlabel("", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    sns.despine(ax=ax, top=True, right=True)
    plt.show()


@auto_save_plots
def extension_plot(
    original_counts: Dict[str, int],
    experiment_scores: Dict[str, List[float]],
    high_threshold: float,
    levels: Optional[List[str]] = None,
    unlabeled_scores: Optional[Dict[str, List[float]]] = None,
    label_high: Optional[str] = None,
    label_mid: Optional[str] = None,
    **kwargs
) -> None:
    prefix = _current_prefix()
    if levels is None:
        levels = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]

    label_high = label_high or "Extension (Labeled)"
    label_mid = label_mid or "Extension (Unlabeled)"
    
    # Prepare data
    grey_counts = []
    high_counts = []
    mid_counts = []
    
    for lvl in levels:
        grey_counts.append(original_counts.get(lvl, 0))
        
        scores = experiment_scores.get(lvl, [])
        high = len([s for s in scores if s > high_threshold])
        if unlabeled_scores is not None:
            unlabeled_level_scores = unlabeled_scores.get(lvl, [])
            mid = len([s for s in unlabeled_level_scores if s > high_threshold])
        else:
            mid = 0

        high_counts.append(high)
        mid_counts.append(mid)

    if prefix:
        table_df = pd.DataFrame({
            'Level': levels,
            'Original': grey_counts,
            'New (High)': high_counts,
            'New (Mid)': mid_counts,
        })
        table_df.insert(0, 'Index', np.arange(1, len(table_df) + 1))
        filename = f"{prefix}_1.tsv" if prefix else "Association_Extension_1.tsv"
        _write_tsv(table_df, filename, index=False, float_format=None)

        extension_pairs = _LAST_EXTENSION_PAIRS
        if unlabeled_scores is not None and (_LAST_LABELED_EXTENSION_PAIRS or _LAST_UNLABELED_EXTENSION_PAIRS):
            pair_rows = []
            if _LAST_LABELED_EXTENSION_PAIRS:
                for pair, score in _LAST_LABELED_EXTENSION_PAIRS.items():
                    if score > high_threshold:
                        pair_rows.append({
                            "Virus Name": pair,
                            "Confidence Score": score,
                            "Type": label_high,
                        })
            if _LAST_UNLABELED_EXTENSION_PAIRS:
                for pair, score in _LAST_UNLABELED_EXTENSION_PAIRS.items():
                    if score > high_threshold:
                        pair_rows.append({
                            "Virus Name": pair,
                            "Confidence Score": score,
                            "Type": label_mid,
                        })
            if pair_rows:
                pair_df = pd.DataFrame(pair_rows)
                pair_df.sort_values(by="Confidence Score", ascending=False, inplace=True)
                pair_df["Confidence Score"] = pair_df["Confidence Score"].apply(lambda v: _format_decimal(v))
                pair_df.insert(0, "Index", np.arange(1, len(pair_df) + 1))
                pair_filename = f"{prefix}_2.tsv" if prefix else "Association_Extension_2.tsv"
                _write_tsv(pair_df, pair_filename, index=False, float_format=None)
        elif extension_pairs:
            pair_rows = []
            for pair, score in extension_pairs.items():
                if score > high_threshold:
                    pair_rows.append({
                        "Virus Name": pair,
                        "Confidence Score": score,
                        "Type": label_high,
                    })
            if pair_rows:
                pair_df = pd.DataFrame(pair_rows)
                pair_df.sort_values(by="Confidence Score", ascending=False, inplace=True)
                pair_df["Confidence Score"] = pair_df["Confidence Score"].apply(lambda v: _format_decimal(v))
                pair_df.insert(0, "Index", np.arange(1, len(pair_df) + 1))
                pair_filename = f"{prefix}_2.tsv" if prefix else "Association_Extension_2.tsv"
                _write_tsv(pair_df, pair_filename, index=False, float_format=None)
        
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(levels))
    width = 0.6
    
    # Colors for extension plots
    color_grey = (160/255, 160/255, 160/255, 0.7)
    color_green = (80/255, 180/255, 80/255, 0.85)
    color_light_green = (180/255, 230/255, 180/255, 0.85)
    
    # Stacked bars
    # Grey (Base)
    ax.bar(x, grey_counts, width, color=color_grey, label='Original')
    # Green (High) on Grey
    ax.bar(x, high_counts, width, bottom=grey_counts, color=color_green, label=label_high)
    # Light Green (Mid) on Grey + High
    if any(mid_counts):
        bottom_mid = [g + h for g, h in zip(grey_counts, high_counts)]
        ax.bar(x, mid_counts, width, bottom=bottom_mid, color=color_light_green, label=label_mid)
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in levels], fontsize=12)
    ax.set_ylabel("Number of Predictions", fontsize=14)
    
    # Add counts text
    for i in range(len(levels)):
        # Grey
        if grey_counts[i] > 0:
            ax.text(i, grey_counts[i]/2, str(grey_counts[i]), ha='center', va='center', fontsize=10)
        # Green (High)
        if high_counts[i] > 0:
            ax.text(i, grey_counts[i] + high_counts[i]/2, str(high_counts[i]), ha='center', va='center', fontsize=10)
        # Light Green (Mid)
        if mid_counts[i] > 0:
            ax.text(i, grey_counts[i] + high_counts[i] + mid_counts[i]/2, str(mid_counts[i]), ha='center', va='center', fontsize=10)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=12,
    )
    legend.get_frame().set_alpha(0.9)
            
    plt.show()
    plt.close()

# default capture prefixes for host-specific plots
specific_host_scatter_plot._default_save_prefix = 'Human_Health'


@auto_save_plots
def specific_host_wmw(tsv_path: str) -> Dict[str, float]:
    """
    Compute a one-sided Wilcoxon–Mann–Whitney (Mann–Whitney U) test comparing
    TP vs FP confidence scores from a TSV exported by `specific_host_plot`.

    This function takes only one parameter (`tsv_path`). The caller may pass a
    `save_prefix` keyword (intercepted by `@auto_save_plots`) to control output
    naming and the TSV destination directory is consistent with other outputs.

    Parameters
    ----------
    tsv_path : str
        Path to a TSV with columns: Index, Virus Name, Confidence Score, Type.

    Returns
    -------
    Dict[str, float]
        Summary including U, pvalue, auc, n_tp, n_fp, median_tp, median_fp.
    """
    import math
    import csv

    # ---- Load TP/FP scores robustly using csv ----
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    score_col = "Confidence Score"
    type_col = "Type"
    group_a = "TP"
    group_b = "FP"

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter='\t')
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Empty TSV file")

        col_index = {name.strip(): i for i, name in enumerate(header)}
        if score_col not in col_index or type_col not in col_index:
            raise ValueError(
                f"Expected columns '{score_col}' and '{type_col}' in TSV. Found: {header}"
            )
        i_score = col_index[score_col]
        i_type = col_index[type_col]

        tp_scores: list[float] = []
        fp_scores: list[float] = []

        for row in reader:
            if not row:
                continue
            try:
                grp = row[i_type].strip()
            except IndexError:
                continue
            if grp not in (group_a, group_b):
                continue
            try:
                score = float(row[i_score])
            except (ValueError, IndexError):
                continue
            if grp == group_a:
                tp_scores.append(score)
            else:
                fp_scores.append(score)

    a = np.asarray(tp_scores, dtype=float)
    b = np.asarray(fp_scores, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError(
            f"No data for one or both groups: TP (n={a.size}), FP (n={b.size})."
        )

    # ---- Mann–Whitney U test using SciPy ----
    from scipy.stats import mannwhitneyu as _mw  # type: ignore

    alternative = "greater"  # Test direction: TP > FP
    res = _mw(a, b, alternative=alternative, method="auto")  # type: ignore
    U = float(res.statistic)
    p = float(res.pvalue)

    n_tp = int(a.size)
    n_fp = int(b.size)
    median_tp = float(np.median(a))
    median_fp = float(np.median(b))
    auc = U / (n_tp * n_fp)

    # ---- Logging (aligns with existing style) ----
    logger.info("Directional Mann–Whitney U test (TP > FP)")
    logger.info(f"TP Number = {n_tp}")
    logger.info(f"FP Number = {n_fp}")
    logger.info(f"Median(TP) = {median_tp:.3f}; Median(FP) = {median_fp:.3f}")
    logger.info(f"U statistic = {U:.3f}")
    logger.info(f"p-value = {p:.3e}")
    logger.info(f"Effect size AUC = {auc:.3f}")

    # ---- Save summary TSV under the current prefix ----
    prefix = _current_prefix() or "Specific_Host"
    out_df = pd.DataFrame([
        {
            "Input TSV": str(path),
            "Group A": group_a,
            "Group B": group_b,
            "Alternative": alternative,
            "n_TP": n_tp,
            "n_FP": n_fp,
            "Median_TP": median_tp,
            "Median_FP": median_fp,
            "U": U,
            "pvalue": p,
            "AUC": auc,
        }
    ])
    # File name incorporates the prefix for consistency with other outputs
    filename = f"{prefix}.tsv" if prefix else "Association_Extension_WMW.tsv"
    _write_tsv(out_df, filename, index=False, float_format=None)

    return {
        "U": U,
        "pvalue": p,
        "auc": auc,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "median_tp": median_tp,
        "median_fp": median_fp,
    }

# default capture prefix for WMW test
specific_host_wmw._default_save_prefix = 'Association_Extension_WMW'

# ---------------------------------------------------------------------------
# Contig-level statistics: length-stratified Top-1 performance
# ---------------------------------------------------------------------------
@auto_save_plots
def contig_statistics(
    prediction_path: str,
    truth_paths: Dict[str, str],
    *,
    length_path: str,
    method_name: Optional[str] = None,
    bins: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Summarise contig-level predictions with length-aware accuracy plots.

    Parameters
    ----------
    prediction_path : str
        TSV file with three columns: contig_id, host_label, score.
    truth_path : str
        TSV file with ground-truth host labels (deepest → broadest) per contig.
    taxonomy_path : str
        Host taxonomy lookup table (same format used in major_test).
    length_path : str
        TSV file mapping contig_id -> length (bp), one per line as
        "{contig_id}\\t{length}".
    method_name : str, optional
        Label for logging/legend purposes.
    bins : list of float, optional
        Custom length breakpoints. If None, defaults to:
        [0, 10000, 25000, 50000, inf].

    Returns
    -------
    Dict[str, Any]
        Dictionary containing per-level, per-bin aggregates consistent with
        the Top-1 metrics used in `major_plot`.
    """

    if not os.path.exists(prediction_path):
        logger.error(f"Prediction file not found: {prediction_path}")
        return {}
    missing_truth = [k for k, v in truth_paths.items() if not os.path.exists(v)]
    if missing_truth:
        logger.error(f"Missing truth resources: {missing_truth}")
        return {}

    if not length_path or not os.path.exists(length_path):
        logger.error(f"Length file not found: {length_path}")
        return {}

    method_name = method_name or os.path.basename(prediction_path)
    logger.info(f"[contig_statistics] Method: {method_name}")
    logger.info(f"[contig_statistics] Prediction: {prediction_path}")
    logger.info(
        "[contig_statistics] Truth resources: %s",
        {k: truth_paths[k] for k in sorted(truth_paths)}
    )

    # Load contig lengths from the provided file
    length_map: Dict[str, int] = {}
    try:
        with open(length_path, "r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                cid, length_str = parts
                try:
                    length_val = int(length_str)
                except ValueError:
                    continue
                length_map[cid] = length_val
        logger.info(
            "[contig_statistics] Loaded %d contig lengths from %s",
            len(length_map),
            length_path,
        )
    except OSError as exc:
        logger.error(f"Failed to read length file {length_path}: {exc}")
        return {}

    # ------------------------------------------------------------------
    # Load GT taxonomy (label transfer) and Top-1 predictions per level
    # ------------------------------------------------------------------
    levels_to_plot = [lvl for lvl in STAT_LEVELS if lvl != "infraspecies"]
    label_taxonomy = load_host_label_taxonomy(truth_paths["label_taxonomy_path"])
    label_index = load_label_index(truth_paths["label_index_path"])
    valid_labels = set(label_index.keys())

    # 1) Ground-truth taxonomy sets per contig + level
    virus_ids: List[str] = []
    gt_tx: Dict[str, Dict[str, Set[str]]] = {}
    try:
        with open(truth_paths["virus_test_path"], "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if not parts or not parts[0]:
                    continue
                virus_id = parts[0]
                virus_ids.append(virus_id)

                level_sets: Dict[str, Set[str]] = {}
                for host_label in parts[1:]:
                    if host_label not in valid_labels:
                        continue
                    tx_path = label_taxonomy.get(host_label, {})
                    if not tx_path:
                        continue
                    for lvl in levels_to_plot:
                        tx = tx_path.get(lvl)
                        if tx:
                            level_sets.setdefault(lvl, set()).add(tx)
                gt_tx[virus_id] = level_sets
    except OSError as exc:
        logger.error(f"Failed to read ground truth file {truth_paths['virus_test_path']}: {exc}")
        return {}

    if not virus_ids:
        logger.error("No virus IDs found in the ground truth set; aborting contig statistics.")
        return {}

    virus_id_set = set(virus_ids)

    # 2) Top-1 predicted taxonomy term per contig + level (max score after transfer)
    pred_top1: Dict[str, Dict[str, Tuple[float, str]]] = defaultdict(dict)
    try:
        with open(prediction_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                parts = raw.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                virus_id, host_label, score_str = parts
                if virus_id not in virus_id_set:
                    continue
                if host_label not in valid_labels:
                    continue
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                tx_path = label_taxonomy.get(host_label, {})
                if not tx_path:
                    continue
                for lvl in levels_to_plot:
                    tx = tx_path.get(lvl)
                    if not tx:
                        continue
                    current = pred_top1[virus_id].get(lvl)
                    if current is None or score > current[0]:
                        pred_top1[virus_id][lvl] = (score, tx)
    except OSError as exc:
        logger.error(f"Failed to read prediction file {prediction_path}: {exc}")
        return {}

    # ------------------------------------------------------------------
    # Set up bins
    # ------------------------------------------------------------------
    # Default bins; allow user override.
    if bins:
        bins = list(bins)
    else:
        bins = [0, 20000, 40000, 60000, 80000, float("inf")]
    if bins[0] > 0:
        bins = [0.0] + bins

    bin_pairs = list(zip(bins[:-1], bins[1:]))
    bin_labels = [
        "0-20000",
        "20000-40000",
        "40000-60000",
        "60000-80000",
        ">80000",
    ]
    if len(bin_labels) != len(bin_pairs):
        bin_labels = [f"{int(lo)}-{int(hi)}" if np.isfinite(hi) else f">{int(lo)}" for lo, hi in bin_pairs]

    def assign_bin(length_val: int) -> Tuple[str, int]:
        for idx, (lo, hi) in enumerate(bin_pairs):
            if idx == 0:
                if length_val <= hi:
                    return bin_labels[idx], idx
            elif not np.isfinite(hi):
                if length_val > lo:
                    return bin_labels[idx], idx
            else:
                if lo < length_val <= hi:
                    return bin_labels[idx], idx
        return bin_labels[-1], len(bin_labels) - 1

    # ------------------------------------------------------------------
    # Aggregate per (level, length_bin): Correct / Incorrect / No-answer
    # ------------------------------------------------------------------
    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "no_answer": 0}
    )

    missing_length = 0
    nonpositive_length = 0

    for virus_id in virus_ids:
        length_val = length_map.get(virus_id)
        if length_val is None:
            missing_length += 1
            continue
        if length_val <= 0:
            nonpositive_length += 1
            continue

        bin_label, _ = assign_bin(int(length_val))

        gt_levels = gt_tx.get(virus_id, {})
        pred_levels = pred_top1.get(virus_id, {})

        for lvl in levels_to_plot:
            gt_set = gt_levels.get(lvl)
            if not gt_set:
                continue  # consistent with major_plot: only viruses with GT at this level

            top1 = pred_levels.get(lvl)
            if top1 is None:
                status = "no_answer"
            else:
                _, tx = top1
                status = "correct" if tx in gt_set else "incorrect"

            counts[(lvl, bin_label)][status] += 1

    if missing_length:
        logger.warning(
            "[contig_statistics] %d contigs missing from length file; skipped.",
            missing_length
        )
    if nonpositive_length:
        logger.warning(
            "[contig_statistics] %d contigs have non-positive length; skipped.",
            nonpositive_length
        )

    # Normalise counts into nested dicts with all bins present
    counts_by_level: Dict[str, Dict[str, Dict[str, int]]] = {
        lvl: {
            bin_label: {"correct": 0, "incorrect": 0, "no_answer": 0}
            for bin_label in bin_labels
        }
        for lvl in levels_to_plot
    }
    for (lvl, bin_label), c in counts.items():
        if lvl in counts_by_level and bin_label in counts_by_level[lvl]:
            counts_by_level[lvl][bin_label].update(c)

    # Precompute Top-1 metrics per level + bin (CR/IR/NA and FDR)
    metrics_by_level: Dict[str, Dict[str, Dict[str, float]]] = {
        lvl: {} for lvl in levels_to_plot
    }
    for lvl in levels_to_plot:
        for bin_label in bin_labels:
            c = counts_by_level[lvl][bin_label]
            correct = int(c["correct"])
            incorrect = int(c["incorrect"])
            no_answer = int(c["no_answer"])
            total = correct + incorrect + no_answer

            cr = correct / total if total else float("nan")
            ir = incorrect / total if total else float("nan")
            na = no_answer / total if total else float("nan")
            denom = correct + incorrect
            fdr = incorrect / denom if denom else float("nan")

            metrics_by_level[lvl][bin_label] = {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "no_answer": no_answer,
                "cr": cr,
                "ir": ir,
                "na": na,
                "fdr": fdr,
            }

    # ------------------------------------------------------------------
    # Plot 1: length × taxonomy-level grid (stacked proportions)
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(right=0.75)

    color_map = {
        "correct": (126/255, 196/255, 234/255, 0.85),   # light blue
        "incorrect": (235/255, 154/255, 154/255, 0.85), # light red
        "no_answer": (215/255, 215/255, 215/255, 0.6),  # lighter grey
    }

    # Slightly increase vertical spacing between taxonomy levels
    row_scale = 1.5  # spacing multiplier (>1 enlarges gaps)
    level_positions = {lvl: idx * row_scale for idx, lvl in enumerate(levels_to_plot)}
    y_positions = np.arange(len(levels_to_plot)) * row_scale

    bin_label_to_idx = {label: idx for idx, label in enumerate(bin_labels)}

    bar_width = 0.7
    for lvl in levels_to_plot:
        for bin_label in bin_labels:
            cell = counts_by_level[lvl][bin_label]
            raw_total = int(cell["correct"] + cell["incorrect"] + cell["no_answer"])
            if raw_total <= 0:
                continue
            total_here = max(raw_total, 1)  # avoid zero-height
            x_idx = bin_label_to_idx[bin_label]
            base_line = level_positions[lvl]
            # Anchor the stack bottom at the taxonomy level line; use 0.8 height for bars
            # so labels can sit just above.
            y_base = base_line

            heights = {k: cell[k] / total_here for k in ("correct", "incorrect", "no_answer")}

            bottom = 0.0
            # Draw from bottom→top as grey(no_answer), red(incorrect), blue(correct)
            for status in ("no_answer", "incorrect", "correct"):
                h = heights[status]
                if h <= 0:
                    continue
                ax.bar(
                    x_idx,
                    h * 0.7 * row_scale,
                    width=bar_width,
                    bottom=y_base + bottom * 0.7 * row_scale,
                    color=color_map[status],
                    edgecolor="none"
                )
                bottom += h

            ax.text(
                x_idx,
                y_base + bottom * 0.7 * row_scale + 0.05 * row_scale,
                f"{raw_total}",
                color="black",
                fontsize=14,
                ha="center",
                va="bottom"
            )

    ax.set_xticks(np.arange(len(bin_pairs)))
    ax.set_xticklabels(bin_labels, rotation=15, ha="right", fontsize=16)
    ax.set_xlabel("Contig length bins (bp)", fontsize=16)

    # Expand limits so stacks and labels are fully visible; add extra headroom on top.
    y_min = -0.5 * row_scale
    y_max = row_scale * (len(levels_to_plot) + 0.35)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lvl.capitalize() for lvl in levels_to_plot], fontsize=16)
    ax.set_ylabel("Taxonomy level", fontsize=16)
    ax.grid(True, linestyle="-", alpha=0.6)

    scatter_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color_map["correct"], markeredgecolor="none", markersize=12, label="Correct"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color_map["incorrect"], markeredgecolor="none", markersize=12, label="Incorrect"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color_map["no_answer"], markeredgecolor="none", markersize=12, label="No answer"),
    ]

    legend_status = ax.legend(
        handles=scatter_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=16
    )
    legend_status.get_frame().set_alpha(0.9)

    ax.tick_params(axis="x", labelsize=16)

    plt.tight_layout()
    plt.show()

    # # ------------------------------------------------------------------
    # # Plot 2: per-level Top-1 TPR/FDR vs length bins
    # # ------------------------------------------------------------------
    # fig_line, ax_line = plt.subplots(figsize=(14, 8))
    # fig_line.subplots_adjust(right=0.75)

    # x_positions = np.arange(len(bin_labels))

    # # Level colors reference fdr_control_plot (seaborn hue order over STAT_LEVELS)
    # level_palette = sns.color_palette(n_colors=len(levels_to_plot))
    # level_colors = {lvl: level_palette[i] for i, lvl in enumerate(levels_to_plot)}

    # line_width = 2.5
    # marker_size = 9

    # for lvl in levels_to_plot:
    #     tpr_values = [metrics_by_level[lvl][label]["cr"] for label in bin_labels]
    #     fdr_values = [metrics_by_level[lvl][label]["fdr"] for label in bin_labels]
    #     color = level_colors[lvl]
    #     ax_line.plot(
    #         x_positions,
    #         tpr_values,
    #         color=color,
    #         linestyle="-",
    #         marker="o",
    #         linewidth=line_width,
    #         markersize=marker_size,
    #     )
    #     ax_line.plot(
    #         x_positions,
    #         fdr_values,
    #         color=color,
    #         linestyle="--",
    #         marker="^",
    #         linewidth=line_width,
    #         markersize=marker_size,
    #     )

    # ax_line.set_xticks(x_positions)
    # ax_line.set_xticklabels(bin_labels, rotation=15, ha="right", fontsize=16)
    # ax_line.set_xlabel("Contig length bins (bp)", fontsize=16)
    # ax_line.set_ylabel("Top-1 TPR / FDR", fontsize=16)
    # ax_line.set_ylim(-0.05, 1.05)
    # ax_line.set_yticks(np.arange(0, 1.01, 0.2))
    # ax_line.grid(True, linestyle="--", alpha=0.7)
    # ax_line.tick_params(axis="y", labelsize=16)

    # # Legends: colors=levels, styles=metrics (TPR/FDR)
    # level_handles = [
    #     Line2D([0], [0], color=level_colors[lvl], lw=3)
    #     for lvl in levels_to_plot
    # ]
    # legend_levels = ax_line.legend(
    #     handles=level_handles,
    #     labels=[lvl.capitalize() for lvl in levels_to_plot],
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1.0),
    #     frameon=True,
    #     fontsize=16,
    #     title="Level"
    # )
    # legend_levels.get_frame().set_alpha(0.9)
    # ax_line.add_artist(legend_levels)

    # metric_handles = [
    #     Line2D([0], [0], color="black", marker="o", linestyle="-", lw=line_width),
    #     Line2D([0], [0], color="black", marker="^", linestyle="--", lw=line_width),
    # ]
    # legend_metrics = ax_line.legend(
    #     handles=metric_handles,
    #     labels=["Top-1 TPR", "Top-1 FDR"],
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 0.35),
    #     frameon=True,
    #     fontsize=16,
    #     title="Metric"
    # )
    # legend_metrics.get_frame().set_alpha(0.9)

    # plt.tight_layout()
    # plt.show()

    summary = {
        "counts_by_level": counts_by_level,
        "metrics_by_level": metrics_by_level,
        "levels": levels_to_plot,
        "bins": bin_labels,
        "skipped": {
            "missing_length": missing_length,
            "nonpositive_length": nonpositive_length,
        }
    }

    return summary

contig_statistics._default_save_prefix = 'Contig_Statistics'


@auto_save_plots
def host_specific_extension_plot(
    processed_data: List,
    label_transfer_path: str,
    experiment_path: str,
    taxonomy_path: str,
    high_threshold: float = 0.835,
    top: Optional[int] = 10,
    exclude_virus_path: Optional[str] = None,
    unlabeled_virus_path: Optional[str] = None,
    label_high: Optional[str] = None,
    label_mid: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot per-host extension counts using the same definition as extension_plot."""
    prefix = _current_prefix()

    label_high = label_high or "Extension (Labeled)"
    label_mid = label_mid or "Extension (Unlabeled)"

    df = pd.DataFrame(processed_data)
    df = df[df['taxonomy_level'].isin(STAT_LEVELS)]
    if df.empty:
        logger.warning("No hosts with taxonomy_level in STAT_LEVELS. Nothing to plot.")
        return

    df = df.sort_values(by='occurrences', ascending=False).reset_index(drop=True)
    num_hosts = len(df)

    host_filter = set(df['host_name'])
    original_counts, high_counts, mid_counts = _compute_host_extension_counts(
        label_transfer_path=label_transfer_path,
        experiment_path=experiment_path,
        taxonomy_path=taxonomy_path,
        high_threshold=high_threshold,
        host_filter=host_filter,
        exclude_virus_path=exclude_virus_path,
        unlabeled_virus_path=unlabeled_virus_path,
    )

    original_array = np.array([original_counts.get(h, 0) for h in df['host_name']], dtype=int)
    high_array = np.array([high_counts.get(h, 0) for h in df['host_name']], dtype=int)
    mid_array = np.array([mid_counts.get(h, 0) for h in df['host_name']], dtype=int)

    if prefix:
        extension_df = pd.DataFrame({
            'Host Name': df['host_name'],
            'Original': original_array,
            'New (High)': high_array,
            'New (Mid)': mid_array,
        })
        extension_df.insert(0, 'Index', np.arange(1, len(extension_df) + 1))
        filename = f"{prefix}.tsv" if prefix else "Host_Predictions_Extension.tsv"
        _write_tsv(extension_df, filename, index=False, float_format=None)

    plot_df = pd.DataFrame({
        'host_name': df['host_name'],
        'original': original_array,
        'high': high_array,
        'mid': mid_array,
        'occurrences': df['occurrences'],
    })
    plot_df['extension_total'] = plot_df['high'] + plot_df['mid']
    plot_df['total_count'] = plot_df['original'] + plot_df['extension_total']
    plot_df['extension_ratio'] = plot_df['extension_total'] / plot_df['total_count'].replace(0, np.nan)
    plot_df['extension_ratio'] = plot_df['extension_ratio'].fillna(0.0)

    # Step 1: pick Top-N by extension_total (high + mid)
    rank_df = plot_df.sort_values(
        by=['extension_total', 'high', 'mid', 'total_count', 'occurrences', 'host_name'],
        ascending=[False, False, False, False, False, True],
    )
    if top is not None and top > 0:
        rank_df = rank_df.head(top)

    # Step 2: order left-to-right by extension_ratio (high + mid fraction)
    plot_df = rank_df.sort_values(
        by=['extension_ratio', 'extension_total', 'high', 'mid', 'total_count', 'occurrences', 'host_name'],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)

    original_array = plot_df['original'].to_numpy(dtype=int)
    high_array = plot_df['high'].to_numpy(dtype=int)
    mid_array = plot_df['mid'].to_numpy(dtype=int)
    num_hosts = len(plot_df)

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    x_positions = np.arange(num_hosts) * 1.1
    width = 0.8

    color_grey = (160/255, 160/255, 160/255, 0.7)
    color_green = (80/255, 180/255, 80/255, 0.85)
    color_light_green = (180/255, 230/255, 180/255, 0.85)

    total_counts = original_array + high_array + mid_array
    total_counts = np.where(total_counts > 0, total_counts, 1)
    original_ratio = original_array / total_counts
    high_ratio = high_array / total_counts
    mid_ratio = mid_array / total_counts

    ax.bar(x_positions, original_ratio, width, color=color_grey, linewidth=0, label='Original')
    ax.bar(x_positions, high_ratio, width, bottom=original_ratio,
           color=color_green, linewidth=0, label=label_high)
    if np.any(mid_ratio > 0):
        ax.bar(x_positions, mid_ratio, width, bottom=original_ratio + high_ratio,
               color=color_light_green, linewidth=0, label=label_mid)

    ax.set_ylim(0, 1.0)
    y_ticks = np.linspace(0, 1.0, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_format_decimal(v, 2) for v in y_ticks], fontsize=16)

    ax.set_xlim(x_positions.min() - 0.8, x_positions.max() + 0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df['host_name'], rotation=45, ha='right', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Fraction', fontsize=16)

    for i in range(num_hosts):
        counts = [original_array[i], high_array[i], mid_array[i]]
        ratios = [original_ratio[i], high_ratio[i], mid_ratio[i]]
        bottoms = [0.0, original_ratio[i], original_ratio[i] + high_ratio[i]]
        for count, ratio, bottom in zip(counts, ratios, bottoms):
            if ratio <= 0:
                continue
            if ratio < 0.10:
                label = f"{count}"
            else:
                label = f"{count}\n({ratio:.1%})"
            ax.text(
                x_positions[i],
                bottom + ratio / 2.0,
                label,
                ha='center',
                va='center',
                fontsize=10,
                color='black',
            )
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=12,
    )
    legend.get_frame().set_alpha(0.9)

    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(False, axis='x')
    ax.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()
    plt.close()


host_specific_extension_plot._default_save_prefix = 'Host_Predictions_Extension'
