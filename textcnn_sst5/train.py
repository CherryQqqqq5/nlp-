import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data import create_dataloaders, load_glove_embeddings
from model import TextCNN
from utils import save_json, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool, max_steps: int = 0, grad_clip: float = 0.0):
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    steps = 0

    for batch in tqdm(loader, desc="train" if train_mode else "eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size
        steps += 1

        if max_steps > 0 and steps >= max_steps:
            break

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


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
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--artifact_dir", type=str, default="artifacts")

    # regularization and training control
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--scheduler_metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--min_delta", type=float, default=1e-4)

    # pretrained embeddings
    parser.add_argument("--glove_path", type=str, default="")
    parser.add_argument("--freeze_embedding", type=str, default="false", choices=["true", "false"])
    return parser.parse_args()


def is_improved(metric_name: str, current: float, best: float, min_delta: float) -> bool:
    if metric_name == "val_loss":
        return current < (best - min_delta)
    return current > (best + min_delta)


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

    pretrained_embeddings = None
    glove_coverage = None
    glove_found = 0
    freeze_embedding = args.freeze_embedding == "true"
    if args.glove_path:
        pretrained_embeddings, glove_coverage, glove_found = load_glove_embeddings(
            glove_path=args.glove_path,
            vocab=bundle.vocab,
            embed_dim=args.embed_dim,
        )
        print(
            f"Loaded GloVe from {args.glove_path} | "
            f"coverage={glove_coverage:.4f} ({glove_found}/{len(bundle.vocab)}) | "
            f"freeze_embedding={freeze_embedding}"
        )

    model = TextCNN(
        vocab_size=len(bundle.vocab),
        embed_dim=args.embed_dim,
        num_classes=len(bundle.label_names),
        num_filters=args.num_filters,
        kernel_sizes=(3, 4, 5),
        dropout=args.dropout,
        pad_idx=bundle.vocab["<pad>"],
        pretrained_embeddings=pretrained_embeddings,
        freeze_embedding=freeze_embedding,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        scheduler_mode = "max" if args.scheduler_metric == "val_acc" else "min"
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # best tracking for checkpoint and early stop
    best_checkpoint_score = float("-inf")
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

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
            grad_clip=args.grad_clip,
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

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # checkpoint follows val_acc for backward compatibility
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), artifact_dir / "best_model.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # early stopping metric can be configured independently
        current_es_score = val_acc if args.early_stop_metric == "val_acc" else val_loss
        if best_checkpoint_score == float("-inf") and args.early_stop_metric == "val_loss":
            best_checkpoint_score = float("inf")

        if is_improved(args.early_stop_metric, current_es_score, best_checkpoint_score, args.min_delta):
            best_checkpoint_score = current_es_score
            bad_epochs = 0
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler_value = val_acc if args.scheduler_metric == "val_acc" else val_loss
            scheduler.step(scheduler_value)

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(metric={args.early_stop_metric}, patience={args.early_stop_patience})"
            )
            break

    model.load_state_dict(torch.load(artifact_dir / "best_model.pt", map_location=device, weights_only=False))
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
    save_json(bundle.stats, str(artifact_dir / "data_stats.json"))
    save_json(history, str(artifact_dir / "history.json"))
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "glove_coverage": glove_coverage,
            "glove_found": glove_found,
        },
        str(artifact_dir / "metrics.json"),
    )

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("epoch")
    plt.ylabel("learning_rate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(artifact_dir / "curves.png", dpi=200)
    print(f"Artifacts saved to: {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
