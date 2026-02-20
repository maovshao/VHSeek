import argparse
import torch
import time
import pickle
from logzero import logger
from vhseek_util.util import make_parent_dir, get_virus_name, load_embeddings
from esm import FastaBatchedDataset, pretrained
import esm

def esm_embedding_generate(esm_model_path, fasta, embedding, nogpu, onehot, device_ids):
    """
    This function generates embeddings using ESM model if esm_model_path != 'onehot'.
    Otherwise, it calls onehot_embedding_generate for 'onehot' embeddings.
    """

    if esm_model_path is None:
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        logger.info("Loaded default esm2_t33_650M_UR50D model.")
    else:
        esm_model, alphabet = pretrained.load_model_and_alphabet(esm_model_path)
        logger.info(f"Loaded model from {esm_model_path}")

    esm_model.eval()

    use_cuda = torch.cuda.is_available() and not nogpu
    device_list = device_ids if use_cuda else []
    main_device = torch.device(f"cuda:{device_list[0]}") if use_cuda else torch.device("cpu")
    if use_cuda:
        if len(device_list) > 1:
            esm_model = torch.nn.DataParallel(esm_model, device_ids=device_list)
        esm_model = esm_model.to(main_device)
        print(f"Transferred model to GPU(s): {device_list}")

    # unwrap to get attributes
    model_for_attr = esm_model.module if isinstance(esm_model, torch.nn.DataParallel) else esm_model

    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(65536, extra_toks_per_seq=1)
    total_batches = len(batches)

    base_converter = alphabet.get_batch_converter(1022)
    allowed_tokens = set(getattr(alphabet, "standard_toks", []))
    if not allowed_tokens:
        allowed_tokens = set([tok for tok in alphabet.tok_to_idx if len(tok) == 1])

    def clean_sequence(seq: str) -> str:
        seq = seq.strip().upper()
        return "".join(ch if ch in allowed_tokens else "X" for ch in seq)

    def collate(batch):
        # batch: list of (label, sequence) from FastaBatchedDataset
        cleaned = [(lbl, clean_sequence(seq)) for lbl, seq in batch]
        return base_converter(cleaned)

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate, batch_sampler=batches, num_workers=4, pin_memory=use_cuda
    )
    print(f"Read {fasta} with {len(dataset)} sequences", flush=True)
    embedding_dic = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if batch_idx % 50 == 0 or batch_idx + 1 == total_batches:
                print(
                    f"Processing {batch_idx + 1}/{total_batches} batches ({toks.size(0)} sequences)",
                    flush=True,
                )

            if onehot:
                print("Using onehot embedding method.")
                onehot_embedding_dimension = len(alphabet.all_toks)
                onehot_embedding = torch.nn.functional.one_hot(toks, num_classes=onehot_embedding_dimension).float()
                if use_cuda:
                    onehot_embedding = onehot_embedding.to(main_device, non_blocking=True)
                for i, label in enumerate(labels):
                    embedding_dic[label] = onehot_embedding[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                continue

            if use_cuda:
                toks = toks.to(device=main_device, non_blocking=True)

            target_layer = model_for_attr.num_layers
            if batch_idx % 50 == 0 or batch_idx + 1 == total_batches:
                print(f"Get embedding from layer {target_layer}", flush=True)
            # Use keyword argument for tokens to play nicely with DataParallel
            out = esm_model(tokens=toks, repr_layers=[target_layer], return_contacts=False)["representations"][target_layer]

            for i, label in enumerate(labels):
                # get mean embedding
                esm_embedding = out[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                embedding_dic[label] = esm_embedding
        
        with open(embedding, 'wb') as handle:
            pickle.dump(embedding_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # input
    # parser.add_argument('-emp', '--esm_model_path', type=str, default='vhseek_data/model/esm1b/esm1b_t33_650M_UR50S.pt', help="ESM model location")
    # parser.add_argument('-emp', '--esm_model_path', type=str, default='vhseek_data/model/esm2/esm2_t12_35M_UR50D.pt', help="ESM model location")
    #parser.add_argument('-emp', '--esm_model_path', type=str, default='vhseek_data/model/esm2/esm2_t33_650M_UR50D.pt', help="ESM model location")
    parser.add_argument('-emp', '--esm_model_path', type=str, default=None, help="Path to ESM checkpoint; if None, use esm2_t33_650M_UR50D() pretrained weights")
    
    parser.add_argument('-f', '--fasta', type=str, help="Fasta file to generate embedding")

    # output
    parser.add_argument('-pe', '--protein_embedding', type=str, help="Protein embedding result")
    parser.add_argument('-ve', '--virus_embedding', type=str, help="Virus embedding result")

    # parameter
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--onehot", action="store_true", help="Use onehot embedding")
    parser.add_argument(
        "--device",
        nargs="+",
        type=int,
        default=[0,1],
        help="List of CUDA device IDs to use"
    )
    args = parser.parse_args()

    time_start = time.time()
    make_parent_dir(args.protein_embedding)

    # Step 1: Generate protein embedding (using ESM or onehot)
    esm_embedding_generate(args.esm_model_path, args.fasta, args.protein_embedding, args.nogpu, args.onehot, args.device)

    # Step 2: Load protein embeddings
    embeddings_dict = load_embeddings(args.protein_embedding)

    # Total number of virus proteins originally
    total_virus_proteins_original = len(embeddings_dict)

    # Step 3: Map virus names to their corresponding protein IDs
    virus_to_proteins = {}
    total_virus_proteins_collected = 0
    virus_embeddings = {}

    for virusprotein_fullname in embeddings_dict.keys():
        virus_name = get_virus_name(virusprotein_fullname)
        virus_to_proteins.setdefault(virus_name, []).append(virusprotein_fullname)
        total_virus_proteins_collected += 1

    # Step 4: Compute mean embeddings for each virus using PyTorch
    for virus_name, protein_ids in virus_to_proteins.items():
        protein_embeddings = []
        for pid in protein_ids:
            protein_embeddings.append(embeddings_dict.get(pid))
        if protein_embeddings:
            stacked_embeddings = torch.stack(protein_embeddings)
            mean_embedding = torch.mean(stacked_embeddings, dim=0)
            virus_embeddings[virus_name] = mean_embedding
        else:
            logger.warning(f"No embeddings found for proteins of virus '{virus_name}'.")

    logger.info(f"Computed mean embeddings for {len(virus_embeddings)} viruses.")

    # Step 5: Save virus embeddings
    with open(args.virus_embedding, 'wb') as f_emb:
        pickle.dump(virus_embeddings, f_emb, protocol=pickle.HIGHEST_PROTOCOL)

    # Output the counts
    logger.info(f"Total number of virus proteins originally: {total_virus_proteins_original}")
    logger.info(f"Total number of virus proteins collected: {total_virus_proteins_collected}")
    logger.info(f"Total number of viruses with computed embeddings: {len(virus_embeddings)}")
    logger.info(f"Done in {time.time() - time_start:.1f}s")
