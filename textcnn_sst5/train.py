import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import create_dataloaders
from model import TextCNN
from utils import compute_accuracy, save_json, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool, max_steps: int = 0):
    model.train() if train_mode else model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    steps = 0

    for batch in tqdm(loader, desc="train" if train_mode else "eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids)
            loss = criterion(logits, labels)
            acc = compute_accuracy(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        return 0.0, 0.0
    return epoch_loss / steps, epoch_acc / steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SetFit/sst5")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_dataloaders(
        dataset_name=args.dataset_name,
        max_len=args.max_len,
        min_freq=args.min_freq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    model = TextCNN(
        vocab_size=len(bundle.vocab),
        embed_dim=args.embed_dim,
        num_classes=len(bundle.label_names),
        num_filters=args.num_filters,
        kernel_sizes=(3, 4, 5),
        dropout=args.dropout,
        pad_idx=bundle.vocab["<pad>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    first_batch = next(iter(bundle.train_loader))
    print("Sanity check batch input shape:", tuple(first_batch["input_ids"].shape))
    with torch.no_grad():
        logits = model(first_batch["input_ids"][:4].to(device))
    print("Sanity check logits shape:", tuple(logits.shape))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            bundle.train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            max_steps=args.max_train_steps,
        )
        val_loss, val_acc = run_epoch(
            model,
            bundle.val_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            max_steps=args.max_eval_steps,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), artifact_dir / "best_model.pt")

    model.load_state_dict(torch.load(artifact_dir / "best_model.pt", map_location=device))
    test_loss, test_acc = run_epoch(
        model,
        bundle.test_loader,
        criterion,
        optimizer,
        device,
        train_mode=False,
        max_steps=args.max_eval_steps,
    )
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")

    save_json(bundle.vocab, str(artifact_dir / "vocab.json"))
    save_json(bundle.label_names, str(artifact_dir / "label_names.json"))
    save_json(vars(args), str(artifact_dir / "config.json"))
    save_json(history, str(artifact_dir / "history.json"))
    save_json(
        {
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        str(artifact_dir / "metrics.json"),
    )

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "curves.png", dpi=200)
    print(f"Artifacts saved to: {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
