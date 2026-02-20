import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List

import torch
from logzero import logger
from modelgenerator.tasks import Embed

# --------------------------------------------------------------------------- #
# FASTA parser
# --------------------------------------------------------------------------- #
def read_fasta(fasta_path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    current: str | None = None
    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].strip()
                seqs.setdefault(current, [])
            else:
                if current is None:
                    raise ValueError("Sequence before header.")
                seqs[current].append(line)
    return {k: "".join(v) for k, v in seqs.items()}

# --------------------------------------------------------------------------- #
# Chunk helper
# --------------------------------------------------------------------------- #
def chunk_sequence(seq: str, max_len: int) -> List[str]:
    """Return non‑overlapping chunks of length ≤ max_len."""
    return [seq[i:i + max_len] for i in range(0, len(seq), max_len)]

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def aido_embedding_generate(fasta: Path,
                              out_pkl: Path,
                              batch_size: int,
                              nogpu: bool,
                              max_len: int,
                              device_ids: List[int]) -> None:
    # Determine whether to use GPU and which devices
    use_cuda = torch.cuda.is_available() and not nogpu
    device_list = device_ids if use_cuda else []
    main_device = torch.device(f"cuda:{device_list[0]}") if use_cuda else torch.device("cpu")
    #logger.info(f"Using device(s): {', '.join(f'cuda:{i}' for i in device_list) if use_cuda else 'cpu'}")

    # 1. Load genomes
    virus_seqs = read_fasta(fasta)
    logger.info(f"Loaded {len(virus_seqs):,} virus genomes from {fasta}")

    # 2. Build chunk lists
    expanded_seqs: List[str] = []
    owners: List[str] = []  # owners[i] = virus_name of chunk i
    for vname, seq in virus_seqs.items():
        chunks = chunk_sequence(seq, max_len)
        expanded_seqs.extend(chunks)
        owners.extend([vname] * len(chunks))
    logger.info(f"Total chunks to embed: {len(expanded_seqs):,}")

    # 3. Load model (use aido_dna_300m backbone)
    logger.info("Loading aido_300M backbone …")
    base_model = Embed.from_config({"model.backbone": "aido_dna_300m"}).eval().to(main_device)
    model = torch.nn.DataParallel(base_model, device_ids=device_list) if use_cuda else base_model

    # 4. Batch inference
    chunk_vecs: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, len(expanded_seqs), batch_size):
            end = min(start + batch_size, len(expanded_seqs))
            seq_batch = expanded_seqs[start:end]
            owner_batch = owners[start:end]

            # Tokenize on CPU to avoid redundant copies inside DataParallel
            batch = base_model.transform({"sequences": seq_batch})
            batch = {k: v.to(main_device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            out = model(batch)  # (B, L, D)

            # Mean-pool inside each chunk (exclude special tokens)
            pooled = []
            for idx, seq in enumerate(seq_batch):
                vec = out[idx, 1:len(seq) + 1, :].mean(0).cpu()
                pooled.append(vec)

            # Collect per owner
            for owner, vec in zip(owner_batch, pooled):
                chunk_vecs.setdefault(owner, []).append(vec)

            logger.info(f"Processed {end}/{len(expanded_seqs)} chunks")

    # 5. Aggregate chunks per virus
    virus_embeddings: Dict[str, torch.Tensor] = {
        v: torch.stack(vecs).mean(0) for v, vecs in chunk_vecs.items()
    }
    logger.info(f"Aggregated embeddings for {len(virus_embeddings)} viruses.")

    # 6. Save to new output path
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as fh:
        pickle.dump(virus_embeddings, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved to {out_pkl}")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--fasta",
        type=Path,
        default="example/virus_genome.fasta",
        help="Input FASTA"
    )
    parser.add_argument(
        "-ve", "--virus_embedding",
        type=Path,
        default="example/embedding/virus_aido_300.pkl",
        help="Output pickle file"
    )
    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=128,
        help="Batch size (reduce if OOM)"
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="Force CPU even if GPU available"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=4000,
        help="Maximum length of each sequence chunk"
    )
    parser.add_argument(
        "--device",
        nargs="+",
        type=int,
        default=[0,1],
        help="List of CUDA device IDs to use"
    )
    args = parser.parse_args()

    time_start = time.time()
    aido_embedding_generate(
        args.fasta,
        args.virus_embedding,
        batch_size=args.batch_size,
        nogpu=args.nogpu,
        max_len=args.max_len,
        device_ids=args.device
    )
    logger.info(f"Done in {time.time() - time_start:.1f}s")

if __name__ == "__main__":
    main()
