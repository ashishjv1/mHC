"""
Data preparation and loading for OpenWebText / FineWeb-Edu.

Tokenizes the full dataset with tiktoken's GPT-2 encoder, saves as
memory-mapped numpy arrays for fast random-access during training.
"""
import os
import numpy as np
import tiktoken
import torch
from tqdm import tqdm


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def prepare_openwebtext(data_dir: str):
    """Download, tokenize, and save OpenWebText as memmap files."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("Preparing OpenWebText dataset...")
    from datasets import load_dataset

    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    split = dataset.train_test_split(test_size=0.005, seed=42, shuffle=True)
    split["val"] = split.pop("test")

    enc = get_tokenizer()

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split.map(
        tokenize,
        remove_columns=["text"],
        num_proc=os.cpu_count(),
        desc="Tokenizing",
    )

    for split_name in ["train", "val"]:
        dset = tokenized[split_name]
        total_len = sum(dset["len"])
        print(f"{split_name}: {total_len:,} tokens")

        path = os.path.join(data_dir, f"{split_name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total_len,))

        idx = 0
        for example in tqdm(dset, desc=f"Writing {split_name}"):
            ids = np.array(example["ids"], dtype=np.uint16)
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()

    print("Data preparation complete.")


def prepare_fineweb_edu(data_dir: str, max_tokens: int = 10_000_000_000):
    """Download and tokenize a slice of FineWeb-Edu."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("Preparing FineWeb-Edu dataset...")
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    enc = get_tokenizer()
    val_tokens = max(max_tokens // 200, 1_000_000)  # 0.5% for val

    all_tokens = []
    total = 0
    for example in tqdm(dataset, desc="Tokenizing FineWeb-Edu"):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        all_tokens.extend(ids)
        total += len(ids)
        if total >= max_tokens:
            break

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    split_idx = len(all_tokens) - val_tokens

    for name, data in [("train", all_tokens[:split_idx]), ("val", all_tokens[split_idx:])]:
        path = os.path.join(data_dir, f"{name}.bin")
        print(f"{name}: {len(data):,} tokens")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(data),))
        arr[:] = data
        arr.flush()

    print("Data preparation complete.")


class DataLoader:
    """Random-access dataloader from memmap token files."""

    def __init__(self, data_dir: str, split: str, batch_size: int,
                 context_len: int, device: str = "cuda"):
        path = os.path.join(data_dir, f"{split}.bin")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.context_len = context_len
        self.device = device

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.context_len, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.context_len].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.context_len].astype(np.int64))
            for i in ix
        ])
        return x.to(self.device), y.to(self.device)
