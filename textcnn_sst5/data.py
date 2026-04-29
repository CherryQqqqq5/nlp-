from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    else:
        ids = ids[:max_len]
    return ids


class SST5Dataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        input_ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        label = int(self.labels[idx])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab: Dict[str, int]
    label_names: List[str]


def create_dataloaders(
    dataset_name: str = "SetFit/sst5",
    max_len: int = 50,
    min_freq: int = 2,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataBundle:
    ds = load_dataset(dataset_name)

    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]
    test_texts = ds["test"]["text"]

    train_labels = ds["train"]["label"]
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]

    vocab = build_vocab(train_texts, min_freq=min_freq)
    label_names = ds["train"].features["label"].names

    train_dataset = SST5Dataset(train_texts, train_labels, vocab, max_len)
    val_dataset = SST5Dataset(val_texts, val_labels, vocab, max_len)
    test_dataset = SST5Dataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        label_names=label_names,
    )


if __name__ == "__main__":
    bundle = create_dataloaders()
    batch = next(iter(bundle.train_loader))
    print("label_names:", bundle.label_names)
    print("vocab_size:", len(bundle.vocab))
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("labels shape:", tuple(batch["labels"].shape))
