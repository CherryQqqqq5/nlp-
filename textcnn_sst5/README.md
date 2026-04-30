# TextCNN SST-5 Course Practice

This project implements a reproducible 5-class sentiment classifier on SST-5 with a high-score-oriented experiment workflow (ablation + multi-seed + detailed error analysis).

## Task

Given an English movie review sentence, classify it into:

- 0: very negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very positive

## Structure

- `data.py`: dataset loading, tokenizer, vocabulary, GloVe loader, stats
- `model.py`: TextCNN model with optional pretrained embedding init
- `train.py`: training, validation, early stopping, scheduler, checkpoint, test
- `evaluate.py`: classification report, confusion matrix, error examples
- `predict.py`: single-sentence inference (default max_len from artifact config)
- `utils.py`: seed and JSON utils
- `data/sst5_hf`: offline SST-5 dataset
- `embeddings/`: pretrained embeddings (e.g., GloVe)
- `artifacts_hs/`: high-score sprint experiment outputs
- `artifacts_final/`: previous full-run baseline outputs

## Install

```bash
cd textcnn_sst5
pip install -r requirements.txt
```

## Download GloVe (100d)

```bash
mkdir -p embeddings
cd embeddings
wget -c https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
python - <<PY
from zipfile import ZipFile
with ZipFile("glove.6B.zip", "r") as zf:
    zf.extract("glove.6B.100d.txt", ".")
print("done")
PY
cd ..
```

## Main Train Command (glove-non-static)

```bash
python train.py \
  --dataset_name data/sst5_hf \
  --artifact_dir artifacts_hs/glove_nonstatic \
  --epochs 12 \
  --batch_size 64 \
  --embed_dim 100 \
  --num_filters 100 \
  --max_len 50 \
  --lr 5e-4 \
  --dropout 0.5 \
  --weight_decay 1e-4 \
  --use_scheduler \
  --scheduler_metric val_acc \
  --scheduler_patience 1 \
  --scheduler_factor 0.5 \
  --early_stop_patience 3 \
  --early_stop_metric val_acc \
  --glove_path embeddings/glove.6B.100d.txt \
  --freeze_embedding false \
  --device auto
```

## Detailed Evaluation

```bash
python evaluate.py \
  --artifact_dir artifacts_hs/glove_nonstatic \
  --dataset_name data/sst5_hf
```

Outputs include:

- `classification_report.json`
- `confusion_matrix.json`
- `error_examples.json`

## Predict

```bash
python predict.py \
  --artifact_dir artifacts_hs/glove_nonstatic \
  --text "This movie is absolutely wonderful and touching."
```

## Multi-Seed Repro

```bash
for seed in 13 42 3407; do
  python train.py \
    --dataset_name data/sst5_hf \
    --artifact_dir artifacts_hs/multiseed/glove_nonstatic_seed${seed} \
    --epochs 12 \
    --batch_size 64 \
    --embed_dim 100 \
    --num_filters 100 \
    --max_len 50 \
    --lr 5e-4 \
    --dropout 0.5 \
    --weight_decay 1e-4 \
    --use_scheduler \
    --scheduler_metric val_acc \
    --scheduler_patience 1 \
    --scheduler_factor 0.5 \
    --early_stop_patience 3 \
    --early_stop_metric val_acc \
    --glove_path embeddings/glove.6B.100d.txt \
    --freeze_embedding false \
    --seed ${seed} \
    --device auto

  python evaluate.py \
    --artifact_dir artifacts_hs/multiseed/glove_nonstatic_seed${seed} \
    --dataset_name data/sst5_hf
done
```

## Video Submission

The 1-2 minute demo video is submitted outside this repository.
Recommended flow: project structure -> training/evaluation results -> predict demo.
