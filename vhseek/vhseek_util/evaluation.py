###############################################################################
# --------------------------- evaluation utilities -------------------------- #
###############################################################################
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict, Set

import numpy as np
import scipy.sparse as ssp
from logzero import logger
from sklearn.metrics import average_precision_score, precision_recall_curve

# --------------------------------------------------------------------------- #
# Global constants                                                            #
# --------------------------------------------------------------------------- #
STAT_LEVELS: List[str] = [
    "infraspecies", "species", "genus", "family",
    "order", "class", "phylum"
]

LABEL_LEVELS: List[str] = [
    "infraspecies", "species", "genus", "family",
    "order", "class", "phylum", "kingdom"
]

OTHER_LEVEL: str = "other"

# --------------------------------------------------------------------------- #
# Auxiliary metric functions                                                  #
# --------------------------------------------------------------------------- #
def fmax(targets: ssp.csr_matrix, scores: ssp.csr_matrix) -> Tuple[float, float]:
    """Compute F-max and its best threshold on sparse matrices."""
    best_f, best_th = 0.0, 0.0
    for cut in (c / 100 for c in range(101)):
        binarized = scores.copy()
        binarized.data = (binarized.data >= cut).astype(int)
        binarized.eliminate_zeros()

        correct   = binarized.multiply(targets).sum()
        predicted = binarized.sum()
        actual    = targets.sum()

        prec = correct / predicted if predicted else 0.0
        rec  = correct / actual    if actual    else 0.0
        f    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f > best_f:
            best_f, best_th = f, cut
    return best_f, best_th


def pair_aupr(targets: ssp.csr_matrix, scores: ssp.csr_matrix):
    """Return AUPR together with its precision-recall curve."""
    y_true   = targets.toarray().ravel()
    y_scores = scores.toarray().ravel()
    prec, rec, th = precision_recall_curve(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    return aupr, prec, rec, th

# --------------------------------------------------------------------------- #
# NEW -- ground-truth extraction                                              #
# --------------------------------------------------------------------------- #
def build_ground_truth_taxonomy(
    targets: ssp.csr_matrix,
    label_index: Dict[str, int],
    label_taxonomy: Dict[str, Dict[str, str]],
):
    """
    Convert sparse ground-truth host labels into
    1) raw host sets (tgt_raw) and
    2) transferred taxonomy sets per LEVEL (gt_tx).

    Returns
    -------
    tgt_raw : virus-id → {host}
    gt_tx   : virus-id → {level → {taxonomy}}
    """
    inv_label_index = {idx: host for host, idx in label_index.items()}

    # -------- raw host-level GT -------- #
    tgt_raw: DefaultDict[int, Set[str]] = defaultdict(set)
    for r, c in zip(*targets.nonzero()):
        tgt_raw[r].add(inv_label_index[c])

    # -------- taxonomy-level GT -------- #
    gt_tx: Dict[int, Dict[str, Set[str]]] = {
        vid: {lv: set() for lv in STAT_LEVELS}
        for vid in tgt_raw
    }
    for vid, hosts in tgt_raw.items():
        for host in hosts:
            tx_path = label_taxonomy.get(host, {})
            for lv in STAT_LEVELS:
                tx = tx_path.get(lv)
                if tx:
                    gt_tx[vid][lv].add(tx)

    return tgt_raw, gt_tx

# --------------------------------------------------------------------------- #
# NEW -- prediction extraction                                                #
# --------------------------------------------------------------------------- #
def build_prediction_taxonomy(
    scores: ssp.csr_matrix,
    label_index: Dict[str, int],
    label_taxonomy: Dict[str, Dict[str, str]],
):
    """
    Convert prediction score matrix into
    1) raw host-level ranked list (pred_raw) and
    2) transferred taxonomy ranked list per LEVEL (pr_tx).

    Returns
    -------
    pred_raw : virus-id → [(prob, host), ...]               (descending prob)
    pr_tx    : virus-id → {level → [(prob, taxonomy), ...]} (descending prob)
    """
    inv_label_index = {idx: host for host, idx in label_index.items()}

    # -------- raw host-level prediction -------- #
    pred_raw: DefaultDict[int, List[Tuple[float, str]]] = defaultdict(list)
    scores_csr = scores.tocsr()
    n_samples = scores.shape[0]
    for vid in range(n_samples):
        row = scores_csr.getrow(vid)
        if row.nnz:
            pairs = sorted(
                [(p, inv_label_index[c]) for p, c in zip(row.data, row.indices)],
                key=lambda x: -x[0]
            )
            pred_raw[vid] = pairs

    # -------- taxonomy-level prediction -------- #
    pr_tx_tmp: Dict[int, Dict[str, Dict[str, float]]] = {
        vid: {lv: {} for lv in STAT_LEVELS}
        for vid in pred_raw
    }
    for vid, pairs in pred_raw.items():
        for prob, host in pairs:
            tx_path = label_taxonomy.get(host, {})
            for lv in STAT_LEVELS:
                tx = tx_path.get(lv)
                if tx and prob > pr_tx_tmp[vid][lv].get(tx, 0.0):
                    pr_tx_tmp[vid][lv][tx] = prob

    # Sort each taxonomy list by probability (descending)
    pr_tx: Dict[int, Dict[str, List[Tuple[float, str]]]] = {
        vid: {lv: [] for lv in STAT_LEVELS}
        for vid in pred_raw
    }
    for vid in pr_tx_tmp:
        for lv in STAT_LEVELS:
            items = sorted(
                [(p, tx) for tx, p in pr_tx_tmp[vid][lv].items()],
                key=lambda x: -x[0]
            )
            pr_tx[vid][lv] = items

    return pred_raw, pr_tx

# --------------------------------------------------------------------------- #
# --------------------------- evaluate_metrics ------------------------------ #
# --------------------------------------------------------------------------- #
# Assume imports like numpy as np, scipy.sparse as ssp, logger,
# and helper functions like build_ground_truth_taxonomy, build_prediction_taxonomy,
# fmax, pair_aupr, and STAT_LEVELS are defined elsewhere.

def evaluate_metrics(
    targets: ssp.csr_matrix,
    scores: ssp.csr_matrix,
    label_index: Dict[str, int],
    label_taxonomy: Dict[str, Dict[str, str]],
    taxonomy_index: Dict[str, int],
    with_log: bool = True
) -> Dict:
    """
    Compute F-max, AUPR and top-1 correctness on every taxonomy LEVEL.
    Logging can be suppressed by setting with_log=False.
    """
    result = {
        "fmax_per_level" : {},
        "fmax_thresholds_per_level" : {},
        "aupr_per_level" : {},
        "precision_per_level" : {},
        "recall_per_level" : {},
        "aupr_thresholds_per_level" : {},
        "cr_per_level" : {},
        "ir_per_level" : {},
        "na_per_level" : {}
        
    }
    n_samples, n_hosts = targets.shape
    assert n_hosts == scores.shape[1], "targets/scores dimension mismatch"

    # These helper functions might have their own logging, which is not controlled by this with_log parameter.
    tgt_raw, gt_tx   = build_ground_truth_taxonomy(targets, label_index, label_taxonomy)
    pred_raw, pr_tx  = build_prediction_taxonomy(scores, label_index, label_taxonomy)
    result["gt_tx"] = gt_tx
    result["pr_tx"] = pr_tx

    taxonomy_index = taxonomy_index or {}
    G = len(taxonomy_index)

    for lv in STAT_LEVELS: # Assuming STAT_LEVELS is defined globally or passed in
        valid_vids = [vid for vid in range(n_samples) if vid in gt_tx and gt_tx[vid].get(lv)]
        a = len(valid_vids)
        b = n_samples
        vid2row = {vid: i for i, vid in enumerate(valid_vids)}

        gt_rows, gt_cols, gt_data = [], [], []
        for vid in valid_vids:
            for tx in gt_tx[vid].get(lv, []): # Used .get for safety, consistent with pr_tx access
                if tx in taxonomy_index:
                    gt_rows.append(vid2row[vid])
                    gt_cols.append(taxonomy_index[tx])
                    gt_data.append(1)
                else:
                    # MODIFICATION: Conditional logging
                    if with_log:
                        logger.warning(f"Ground truth taxonomy term '{tx}' for virus ID {vid} at level '{lv}' not found in taxonomy_index. Skipping this GT term.")

        gt_csr = ssp.csr_matrix(
            (gt_data, (gt_rows, gt_cols)),
            shape=(a, G)
        )

        score_dense = np.zeros((a, G), dtype=np.float32)
        for vid in valid_vids:
            row = vid2row[vid]
            predicted_taxa_for_level = pr_tx.get(vid, {}).get(lv, [])
            for prob, tx in predicted_taxa_for_level:
                if tx in taxonomy_index:
                    col = taxonomy_index[tx]
                    score_dense[row, col] = max(score_dense[row, col], prob)
                else:
                    # MODIFICATION: Conditional logging
                    if with_log:
                        logger.debug(f"Predicted taxonomy term '{tx}' for virus ID {vid} at level '{lv}' not found in taxonomy_index. Skipping this specific prediction score.")
        score_csr = ssp.csr_matrix(score_dense)

        active_prediction_rows_mask = score_csr.getnnz(axis=1) > 0
        with_prediction_rows = np.where(active_prediction_rows_mask)[0]

        if a == 0: # This condition was already present in the input
            result["fmax_per_level"][lv], result["fmax_thresholds_per_level"][lv] = 0.0, 0.0
            result["aupr_per_level"][lv], result["precision_per_level"][lv], result["recall_per_level"][lv], result["aupr_thresholds_per_level"][lv] = 0.0, np.array([]), np.array([]), np.array([])
            result["cr_per_level"][lv], result["ir_per_level"][lv], result["na_per_level"][lv] = 0.0, 0.0, 1.0
            # MODIFICATION: Conditional logging
            if with_log:
                logger.info(
                    f"Level '{lv:<12s}' | No viruses with ground truth  "
                    f"(With GT: {a}/{b}, With Preds: {with_prediction_rows.size}/{b})" # Note: with_prediction_rows will be empty if a=0
                )
            continue

        if with_prediction_rows.size == 0: # This condition was already present
            result["fmax_per_level"][lv], result["fmax_thresholds_per_level"][lv] = 0.0, 0.0
            result["aupr_per_level"][lv], result["precision_per_level"][lv], result["recall_per_level"][lv], result["aupr_thresholds_per_level"][lv] = 0.0, np.array([]), np.array([]), np.array([])
            result["cr_per_level"][lv], result["ir_per_level"][lv], result["na_per_level"][lv] = 0.0, 0.0, 1.0
            if with_log:
                logger.info(
                    f"Level '{lv:<12s}' | No viruses with prediction  "
                    f"(With GT: {a}/{b}, With Preds: {with_prediction_rows.size}/{b})"
                )
            continue
        else:
            result["fmax_per_level"][lv], result["fmax_thresholds_per_level"][lv]  = fmax(gt_csr, score_csr) # Using unfiltered as per input
            result["aupr_per_level"][lv], result["precision_per_level"][lv], result["recall_per_level"][lv], result["aupr_thresholds_per_level"][lv] = pair_aupr(gt_csr, score_csr) # Using unfiltered

        correct = incorrect = no_ans = 0
        for vid in valid_vids: 
            predicted_taxa_for_level = pr_tx.get(vid, {}).get(lv, [])
            if not predicted_taxa_for_level:
                no_ans += 1
                continue
            top1_tx = predicted_taxa_for_level[0][1]
            if top1_tx in gt_tx[vid].get(lv, []): # Used .get for safety
                correct += 1
            else:
                incorrect += 1

        # Ensure 'a' is not zero before division for CR, IR, NA.
        # The 'if a == 0: continue' block above handles this.
        result["cr_per_level"][lv] = correct   / a
        result["ir_per_level"][lv] = incorrect / a
        result["na_per_level"][lv] = no_ans    / a

        # MODIFICATION: Conditional logging
        if with_log:
            logger.info(
                f"Level '{lv:<12s}' | "
                f"Fmax={result['fmax_per_level'][lv]:.3f}  AUPR={result['aupr_per_level'][lv]:.3f}  "
                f"CR={result['cr_per_level'][lv]:.3f}  IR={result['ir_per_level'][lv]:.3f}  NA={result['na_per_level'][lv]:.3f}  "
                f"(With GT: {a}/{b}, With Preds: {with_prediction_rows.size}/{b})"
            )

    return result
