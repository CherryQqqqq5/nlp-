import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from sklearn.metrics import classification_report, confusion_matrix

from data import encode_text
from model import TextCNN
from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--num_error_examples", type=int, default=5)
    return parser.parse_args()


def load_splits(dataset_name: str):
    if Path(dataset_name).exists():
        return load_from_disk(dataset_name)
    return load_dataset(dataset_name)


def build_model(vocab, label_names, artifact_dir):
    config = load_json(f"{artifact_dir}/config.json")
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        num_classes=len(label_names),
        num_filters=config["num_filters"],
        kernel_sizes=(3, 4, 5),
        dropout=config["dropout"],
        pad_idx=vocab["<pad>"],
    )
    state_dict = torch.load(f"{artifact_dir}/best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def batched_predict(model, texts, vocab, max_len: int, batch_size: int):
    preds = []
    confs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            input_ids = [encode_text(t, vocab, max_len=max_len) for t in batch_texts]
            x = torch.tensor(input_ids, dtype=torch.long)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            batch_preds = torch.argmax(probs, dim=1).tolist()
            batch_confs = probs.max(dim=1).values.tolist()
            preds.extend(batch_preds)
            confs.extend(batch_confs)
    return preds, confs


def collect_error_examples(texts, labels, preds, confs, label_names, max_per_pair: int):
    grouped = defaultdict(list)
    for text, y_true, y_pred, conf in zip(texts, labels, preds, confs):
        if y_true == y_pred:
            continue
        key = (int(y_true), int(y_pred))
        if len(grouped[key]) >= max_per_pair:
            continue
        grouped[key].append(
            {
                "text": text,
                "true_id": int(y_true),
                "true_label": label_names[int(y_true)],
                "pred_id": int(y_pred),
                "pred_label": label_names[int(y_pred)],
                "confidence": float(conf),
            }
        )

    items = []
    for (y_true, y_pred), examples in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        items.append(
            {
                "true_label": label_names[y_true],
                "pred_label": label_names[y_pred],
                "count": len(examples),
                "examples": examples,
            }
        )
    return items


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    config = load_json(str(artifact_dir / "config.json"))
    vocab = load_json(str(artifact_dir / "vocab.json"))
    label_names = load_json(str(artifact_dir / "label_names.json"))

    dataset_name = args.dataset_name or config.get("dataset_name", "SetFit/sst5")
    max_len = args.max_len if args.max_len > 0 else int(config.get("max_len", 50))

    ds = load_splits(dataset_name)
    test_texts = ds["test"]["text"]
    test_labels = [int(x) for x in ds["test"]["label"]]

    model = build_model(vocab, label_names, str(artifact_dir))
    preds, confs = batched_predict(model, test_texts, vocab, max_len=max_len, batch_size=args.batch_size)

    report = classification_report(
        test_labels,
        preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        test_labels,
        preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, preds).tolist()

    errors = collect_error_examples(
        test_texts,
        test_labels,
        preds,
        confs,
        label_names,
        max_per_pair=args.num_error_examples,
    )

    (artifact_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    (artifact_dir / "confusion_matrix.json").write_text(
        json.dumps(cm, indent=2, ensure_ascii=False)
    )
    (artifact_dir / "error_examples.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False)
    )

    print(report_text)
    print("Confusion matrix saved:", artifact_dir / "confusion_matrix.json")
    print("Error examples saved:", artifact_dir / "error_examples.json")


if __name__ == "__main__":
    main()
