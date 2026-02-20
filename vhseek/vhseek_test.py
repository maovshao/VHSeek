import os
import sys
import pickle
import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as ssp
from logzero import logger

from vhseek_util.util import (
    load_label_index, make_parent_dir, load_embeddings, load_config,
    update_args_from_config, load_ground_truth, get_topk_sparse,
    load_host_label_taxonomy
)
from vhseek_util.vhseek_model import (
    load_dataset, vhseek_model
)

from vhseek_util.evaluation import (
    evaluate_metrics, build_prediction_taxonomy, STAT_LEVELS
)

def load_torch_model(model_path: str,
                     input_dim: int,
                     output_dim: int,
                     device: torch.device) -> torch.nn.Module:
    """
    Restore the saved MLP model from disk.
    """
    model = vhseek_model(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("PyTorch model loaded successfully.")
    return model

# --------------------------------------------------------------------------- #
# --------------------------- main evaluation loop -------------------------- #
# --------------------------------------------------------------------------- #
def vhseek_test(
    model_path: str,
    embedding_path: str,
    label_index_path: str,
    label_taxonomy_path: str,
    taxonomy_index_path: str,
    virus_test_path: str | None = None,
    top: int | None = 10,
    threshold: float | None = None,
    output_path: str | None = None,
    with_embedding: bool = False
):
    """
    Run inference + (optional) evaluation.
    The I/O contract is unchanged except that
    * host-level labels are dumped via `pred_raw`;
    * taxonomy-level labels are dumped via `pr_tx`
      under the file suffix `_virus_label_transfer`.
    """
    result = {}
    # ---------------------------------------------------------------------- #
    # 1. Data loading                                                        #
    # ---------------------------------------------------------------------- #
    esm_embedding   = load_embeddings(embedding_path)
    label_index     = load_label_index(label_index_path)
    label_taxonomy  = load_host_label_taxonomy(label_taxonomy_path)
    taxonomy_index  = load_label_index(taxonomy_index_path)

    logger.info(f"# Label size:          {len(label_index)}")
    logger.info(f"# Label-to-taxonomy:   {len(label_taxonomy)}")
    logger.info(f"# Taxonomy index size: {len(taxonomy_index)}")

    dataset = load_dataset(esm_embedding, label_index, virus_test_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1000,
                                             shuffle=False)

    # ---------------------------------------------------------------------- #
    # 2. Model restoration                                                   #
    # ---------------------------------------------------------------------- #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device.type.upper()} for prediction.")
    model  = load_torch_model(model_path,
                              input_dim=dataset.get_dim(),
                              output_dim=dataset.get_class_num(),
                              device=device)

    # ---------------------------------------------------------------------- #
    # 3. Forward pass                                                        #
    # ---------------------------------------------------------------------- #
    virus_ids: List[str]          = []
    virus_outputs: List[np.ndarray] = []
    virus_embeddings: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for x, sample_id, _ in dataloader:
            logits, emb = model(x.to(device))
            virus_outputs.append(torch.sigmoid(logits).cpu().numpy())
            virus_ids.extend(sample_id)
            for sid, e in zip(sample_id, emb):
                # store CPU tensor to avoid GPU memory leak
                virus_embeddings[sid] = e.cpu()

    virus_outputs = np.vstack(virus_outputs)

    # ---------------------------------------------------------------------- #
    # 4. Top-K pruning (sparse) & taxonomy extraction                        #
    # ---------------------------------------------------------------------- #
    # MODIFICATION: Re-structured this block to handle threshold and top filters
    
    # Make a copy to apply filters on
    filtered_outputs = virus_outputs.copy()

    # Apply threshold filter first if provided
    if threshold is not None:
        logger.info(f"Applying confidence threshold > {threshold}")
        filtered_outputs[filtered_outputs < threshold] = 0

    # Then, apply top-k pruning or convert to sparse format
    if top is not None:
        virus_outputs_sparse = get_topk_sparse(filtered_outputs, top)
    else:
        virus_outputs_sparse = ssp.csr_matrix(filtered_outputs)
    
    # Round the data of the final sparse matrix
    virus_outputs_sparse.data = np.round(virus_outputs_sparse.data, 3)


    # ---------------------------------------------------------------------- #
    # 5. Optional evaluation                                                 #
    # ---------------------------------------------------------------------- #
    if virus_test_path:
        virus_targets = load_ground_truth(virus_ids, virus_test_path, label_index)
        virus_targets_sparse = ssp.csr_matrix(virus_targets)

        result = evaluate_metrics(
            targets         = virus_targets_sparse,
            scores          = virus_outputs_sparse,
            label_index     = label_index,
            label_taxonomy  = label_taxonomy,
            taxonomy_index  = taxonomy_index,
            with_log        = True
        )

    # ---------------------------------------------------------------------- #
    # 6. Output                                                              #
    # ---------------------------------------------------------------------- #
    if not output_path:
        return result # MODIFICATION: Return result even if no output path

    # Build host-level / taxonomy-level prediction lists for later dumping
    pred_raw, pr_tx = build_prediction_taxonomy(
        virus_outputs_sparse, label_index, label_taxonomy
    )

    model_name = os.path.basename(model_path.rstrip("/"))

    # ---------- 6.1 host-level probabilities ---------- #
    prob_file = os.path.join(output_path, f"{model_name}_probability")
    make_parent_dir(prob_file)
    with open(prob_file, "w") as f_prob:
        for vid, virus_id in enumerate(virus_ids):
            pairs = pred_raw.get(vid, [])
            for prob, host in pairs:          # (prob, host)
                f_prob.write(f"{virus_id}\t{host}\t{prob:.3f}\n")
    logger.info(f"Virus-level probabilities saved to {prob_file}")

    # ---------- 6.2 taxonomy-level labels after transfer ------------------- #
    transfer_file = os.path.join(output_path, f"{model_name}_label_transfer")
    with open(transfer_file, "w") as f_tx:
        for vid, virus_id in enumerate(virus_ids):
            for lv in STAT_LEVELS:
                items = pr_tx[vid][lv] if vid in pr_tx else []
                if not items:
                    continue
                f_tx.write(f"[{virus_id}][{lv}]\n")
                for prob, tx in items:        # (prob, taxonomy)
                    f_tx.write(f"{tx}\t{prob:.3f}\n")
    logger.info(f"Transferred taxonomy labels saved to {transfer_file}")

    # ---------- 6.3 embedding pickle -------------------------------------- #
    if with_embedding:
        emb_file = os.path.join(output_path, f"{model_name}_label_embedding.pkl")
        with open(emb_file, "wb") as f_emb:
            pickle.dump(virus_embeddings, f_emb, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Virus embeddings saved to {emb_file}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for testing VHSeek model")

    # Configuration file (optional)
    parser.add_argument("-c",  "--config", type=str, help="JSON config file")

    # Required paths
    parser.add_argument("-m",  "--model_path",          type=str)
    parser.add_argument("-ep", "--embedding_path",      type=str)
    parser.add_argument("-lip","--label_index_path",    type=str)
    parser.add_argument("-ltp","--label_taxonomy_path", type=str)
    parser.add_argument("-tip","--taxonomy_index_path", type=str)

    # Optional evaluation file
    parser.add_argument("-vtp","--virus_test_path", type=str)

    # Other runtime parameters
    parser.add_argument("--top",       type=int,   default=10)
    # MODIFICATION: Added command line argument for threshold
    parser.add_argument('-th', "--threshold", type=float, default=None)
    parser.add_argument("-op","--output_path", type=str)
    parser.add_argument("--with_embedding", action="store_true")

    args = parser.parse_args()

    # Load YAML/JSON config if provided
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file {args.config} does not exist.")
            sys.exit(1)
        cfg = load_config(args.config)
        args = update_args_from_config(args, cfg)

    logger.info(
        "# Testing VHSeek\n"
        f"  model_path      = {args.model_path}\n"
        f"  embedding_path  = {args.embedding_path}\n"
        f"  output_path     = {args.output_path}\n"
        f"  top             = {args.top}\n"
        f"  threshold       = {args.threshold}\n"
        f"  With embedding  = {args.with_embedding}\n"
    )

    vhseek_test(
        model_path            = args.model_path,
        embedding_path        = args.embedding_path,
        label_index_path      = args.label_index_path,
        label_taxonomy_path   = args.label_taxonomy_path,
        taxonomy_index_path   = args.taxonomy_index_path,
        virus_test_path       = args.virus_test_path,
        top                   = args.top,
        threshold             = args.threshold,
        output_path           = args.output_path,
        with_embedding        = args.with_embedding
    )