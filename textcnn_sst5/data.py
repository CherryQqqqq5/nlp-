import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"[a-z]+|[0-9]+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    # Keep punctuation tokens because TextCNN relies on local n-grams.
    text = text.lower()
    text = text.replace("-lrb-", "(").replace("-rrb-", ")")
    return TOKEN_PATTERN.findall(text)


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


def load_glove_embeddings(
    glove_path: str,
    vocab: Dict[str, int],
    embed_dim: int,
    pad_token: str = "<pad>",
) -> Tuple[torch.Tensor, float, int]:
    embeddings = torch.empty((len(vocab), embed_dim), dtype=torch.float32)
    torch.nn.init.uniform_(embeddings, -0.25, 0.25)

    if pad_token in vocab:
        embeddings[vocab[pad_token]].zero_()

    found = 0
    path = Path(glove_path)
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) <= embed_dim:
                continue
            word = parts[0]
            if word not in vocab:
                continue
            vec = parts[1:]
            if len(vec) != embed_dim:
                continue
            try:
                embeddings[vocab[word]] = torch.tensor([float(x) for x in vec], dtype=torch.float32)
            except ValueError:
                continue
            found += 1

    coverage = found / len(vocab) if vocab else 0.0
    return embeddings, coverage, found


def estimate_oov_ratio(texts: List[str], vocab: Dict[str, int]) -> float:
    total = 0
    oov = 0
    for text in texts:
        toks = tokenize(text)
        total += len(toks)
        for tok in toks:
            if tok not in vocab:
                oov += 1
    return (oov / total) if total else 0.0


class SST5Dataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.vocab = vocab
        self.max_len = max_len
        self.input_ids = [torch.tensor(encode_text(text, self.vocab, self.max_len), dtype=torch.long) for text in texts]
        self.labels = [torch.tensor(int(label), dtype=torch.long) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab: Dict[str, int]
    label_names: List[str]
    stats: Dict[str, float]


def create_dataloaders(
    dataset_name: str = "SetFit/sst5",
    max_len: int = 50,
    min_freq: int = 2,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataBundle:
    if Path(dataset_name).exists():
        ds = load_from_disk(dataset_name)
    else:
        ds = load_dataset(dataset_name)

    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]
    test_texts = ds["test"]["text"]

    train_labels = ds["train"]["label"]
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]

    vocab = build_vocab(train_texts, min_freq=min_freq)
    label_feature = ds["train"].features["label"]
    if hasattr(label_feature, "names"):
        label_names = label_feature.names
    else:
        id_to_text = {}
        if "label_text" in ds["train"].features:
            for row in ds["train"]:
                idx = int(row["label"])
                if idx not in id_to_text:
                    id_to_text[idx] = row["label_text"]
                if len(id_to_text) == len(set(train_labels)):
                    break
        label_names = [id_to_text[i] if i in id_to_text else str(i) for i in sorted(set(train_labels))]

    train_dataset = SST5Dataset(train_texts, train_labels, vocab, max_len)
    val_dataset = SST5Dataset(val_texts, val_labels, vocab, max_len)
    test_dataset = SST5Dataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_lengths = [len(tokenize(t)) for t in train_texts]
    val_lengths = [len(tokenize(t)) for t in val_texts]
    test_lengths = [len(tokenize(t)) for t in test_texts]
    stats = {
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "test_samples": len(test_texts),
        "vocab_size": len(vocab),
        "avg_train_len": (sum(train_lengths) / len(train_lengths)) if train_lengths else 0.0,
        "avg_val_len": (sum(val_lengths) / len(val_lengths)) if val_lengths else 0.0,
        "avg_test_len": (sum(test_lengths) / len(test_lengths)) if test_lengths else 0.0,
        "val_oov_ratio": estimate_oov_ratio(val_texts, vocab),
        "test_oov_ratio": estimate_oov_ratio(test_texts, vocab),
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        label_names=label_names,
        stats=stats,
    )


if __name__ == "__main__":
    bundle = create_dataloaders()
    batch = next(iter(bundle.train_loader))
    print("label_names:", bundle.label_names)
    print("vocab_size:", len(bundle.vocab))
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("labels shape:", tuple(batch["labels"].shape))
