import argparse

import torch

from data import encode_text
from model import TextCNN
from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--max_len", type=int, default=50)
    return parser.parse_args()


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
    state_dict = torch.load(f"{artifact_dir}/best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_one(model, text, vocab, label_names, max_len=50):
    input_ids = encode_text(text, vocab, max_len=max_len)
    x = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
    return {
        "pred_id": pred_id,
        "pred_label": label_names[pred_id],
        "confidence": float(probs[pred_id].item()),
    }


def main():
    args = parse_args()
    vocab = load_json(f"{args.artifact_dir}/vocab.json")
    label_names = load_json(f"{args.artifact_dir}/label_names.json")
    model = build_model(vocab, label_names, args.artifact_dir)

    if args.text:
        result = predict_one(model, args.text, vocab, label_names, args.max_len)
        print(result)
    else:
        print("Interactive mode. Type a sentence, or press Ctrl+C to exit.")
        while True:
            text = input(">> ").strip()
            if not text:
                continue
            result = predict_one(model, text, vocab, label_names, args.max_len)
            print(result)


if __name__ == "__main__":
    main()
