import os
import sys
import argparse
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb  # Import W&B
from vhseek_util.util import load_label_index, load_config, update_args_from_config, load_embeddings, load_host_label_taxonomy, get_topk_sparse
from vhseek_util.vhseek_model import vhseek_model, load_dataset
from vhseek_util.evaluation import evaluate_metrics
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as ssp
import json  # For serializing IC scores
from logzero import logger

def train_ss(model, train_dataloader, device, optimizer):
    """
    Train the model for one epoch.

    Parameters:
    - model (nn.Module): The neural network model.
    - train_dataloader (DataLoader): DataLoader for training data.
    - device (torch.device): Device to run the training on.
    - optimizer (torch.optim.Optimizer): Optimizer for training.

    Returns:
    - avg_loss (float): Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_samples = 0
    for x, _, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs, _ = model(x)
        label_loss = F.binary_cross_entropy_with_logits(outputs, y)
        total_loss_batch = label_loss

        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item() * x.size(0)
        total_samples += x.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss

def eval_ss(model, dataloader, device, label_index, label_taxonomy, taxonomy_index, top=None):
    """
    Run evaluation over a DataLoader and return loss together with
    *genus-level* Fmax / AUPR (other levels are computed but not returned).

    Returns
    -------
    loss           : float
    fmax_genus     : float   -- Fmax at the 'genus' level
    aupr_genus     : float   -- AUPR at the 'genus' level
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    outs, tgts = [], []

    with torch.no_grad():
        for x, _, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            outs.append(logits.cpu())
            tgts.append(y.cpu())

    loss = total_loss / total_samples
    all_outputs = torch.sigmoid(torch.cat(outs)).numpy()
    all_targets = torch.cat(tgts).numpy()

    # Convert ground truth to sparse CSR
    all_targets_sparse = ssp.csr_matrix(all_targets)

    all_outputs_sparse = None
    # ------------------------------------------------------------------ #
    # 2.2  Top-K runing
    # ------------------------------------------------------------------ #
    if top is not None:
        all_outputs_sparse = get_topk_sparse(all_outputs.copy(), top)
    else:
        all_outputs_sparse = ssp.csr_matrix(all_outputs.copy())

    # ---- call the new evaluate_metrics (per‑level) ---- #
    result = evaluate_metrics(
        targets=all_targets_sparse,
        scores=all_outputs_sparse,
        label_index=label_index,
        label_taxonomy=label_taxonomy,
        taxonomy_index=taxonomy_index
    )

    return loss, result["fmax_per_level"].get("genus", 0.0), result["aupr_per_level"].get("genus", 0.0), result["cr_per_level"].get("genus", 0.0), result["ir_per_level"].get("genus", 0.0), result["na_per_level"].get("genus", 0.0)


def main():
    """
    Main function to train the VHSeek model and evaluate it.
    """

    parser = argparse.ArgumentParser('Script for training VHSeek')

    # Configuration file
    parser.add_argument('-c', '--config', type=str, help='Path to JSON config file')

    # Input paths
    parser.add_argument('-ep', '--embedding_path', type=str)
    parser.add_argument('-trp', '--train_path', type=str)
    parser.add_argument('-vap', '--validation_path', type=str)
    parser.add_argument('-lip', '--label_index_path', type=str)
    parser.add_argument('-ltp', '--label_taxonomy_path', type=str)
    parser.add_argument('-tip', '--taxonomy_index_path', type=str)
    parser.add_argument('-smp', '--save_model_path', type=str)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1000, help='Minibatch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs (default: 1000)')
    parser.add_argument('--step', type=int, default=2500, help='Epoch to change the learning rate from 1e-4 to 1e-5 (default: 1000)')
    parser.add_argument('--top', type=int, default=10, help='Number of top predictions to consider for metrics (default: 10)')

    args = parser.parse_args()

    # If a config file is provided, load and update arguments
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file {args.config} does not exist.")
            sys.exit(1)
        config_dict = load_config(args.config)
        args = update_args_from_config(args, config_dict)

    logger.info(f'# Training VHSeek paths: embedding_path = {args.embedding_path}\n train_path = {args.train_path}\n validation_path = {args.validation_path}\n label_index_path = {args.label_index_path}\n label_taxonomy_path = {args.label_taxonomy_path}\n taxonomy_index_path = {args.taxonomy_index_path}\n save_model_path = {args.save_model_path}\n')
    logger.info(f'# Training VHSeek parameters: batch_size = {args.batch_size}, epochs = {args.epochs}, step = {args.step}, top = {args.top}')

    # Initialize W&B run
    wandb.init(project="vhseek", 
        entity="maovshao_team",
        config={
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "step": args.step,
        "top": args.top,
    })
    config = wandb.config

    # load label index
    label_index = load_label_index(args.label_index_path)
    logger.info(f'# Label size (index): {len(label_index)}')

    # load host label taxonomy
    label_taxonomy = load_host_label_taxonomy(args.label_taxonomy_path)
    logger.info(f'# Label to taxonomy size: {len(label_taxonomy)}')

    # load taxonomy index
    taxonomy_index = load_label_index(args.taxonomy_index_path)
    logger.info(f'# Taxonomy size (index): {len(taxonomy_index)}')

    esm_embedding = load_embeddings(args.embedding_path)

    # Load datasets
    train_dataset = load_dataset(esm_embedding, label_index, args.train_path)
    validation_dataset = load_dataset(esm_embedding, label_index, args.validation_path)

    # Get dimensions
    input_dim = train_dataset.get_dim()
    output_dim = train_dataset.get_class_num()

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info("Using GPU for training")
    else:
        logger.info("Using CPU for training")

    # Create model
    model = vhseek_model(input_dim, output_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = StepLR(optimizer, step_size=args.step, gamma=0.1)

    # Watch model gradients and parameters with W&B
    wandb.watch(model, log="all")

    best_fmax_val = 0

    # Training loop
    for epoch in range(config.epochs):
        wandb_log = {}
        train_loss = train_ss(model, train_dataloader, device, optimizer)
        wandb_log["train_loss"] = train_loss
        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb_log["learning_rate"] = current_lr

        if epoch % 100 == 0:
            logger.info(f"\n-------------------------------\nEpoch {epoch}\n-------------------------------")
            # Evaluate on validation set
            loss_val, fmax_val, aupr_val, cr_val, ir_val, na_val = eval_ss(model, validation_dataloader, device, label_index, label_taxonomy, taxonomy_index, top=config.top)
            logger.info(f"Validation Loss: {loss_val:.6f}")
            logger.info(f"Validation Fmax: {fmax_val:.3f}")
            logger.info(f"Validation AUPR: {aupr_val:.3f}")
            logger.info(f"Validation Correct: {cr_val:.3f}")
            logger.info(f"Validation InCorrect: {ir_val:.3f}")
            logger.info(f"Validation No answer: {na_val:.3f}")
            wandb_log["validation_loss"] = loss_val
            wandb_log["validation_fmax"] = fmax_val
            wandb_log["validation_aupr"] = aupr_val
            wandb_log["validation_correct"] = cr_val
            wandb_log["validation_incorrect"] = ir_val
            wandb_log["validation_no_answer"] = na_val

            # Check for improvement based on Fmax
            if fmax_val > best_fmax_val:
                best_fmax_val = fmax_val
                # Save the model in ONNX format
                # should be saved to different path
                torch.save(model.state_dict(), args.save_model_path)
                wandb.save(args.save_model_path)
        wandb.log(wandb_log)

    # After training completes
    wandb.finish()  # Mark the W&B run as finished
    logger.info("Training Complete!")

if __name__ == "__main__":
    main()
