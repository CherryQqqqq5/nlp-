# TextCNN SST-5 Final Submission

This repository contains a reproducible 5-class sentiment classification project on SST-5 using TextCNN, with final results stored under `artifacts_submit/`.

## Task

Given one English movie review sentence, predict one of five labels:

- very negative
- negative
- neutral
- positive
- very positive

## Project Structure

- `data.py`: data loading, regex tokenizer, vocab, GloVe loading, data stats
- `model.py`: TextCNN model
- `train.py`: training + validation + checkpoint + full final test evaluation
- `evaluate.py`: full-test classification report / confusion matrix / error examples
- `predict.py`: single sentence inference
- `artifacts_submit/`: final full-run experiments (main submission artifacts)
- `debug_ablation_step_budget/`: step-budget strategy screening artifacts (non-final)

## Environment

```bash
cd textcnn_sst5
pip install -r requirements.txt
```

## Data and Embeddings

- Dataset: `data/sst5_hf` (offline SST-5)
- Embedding: `embeddings/glove.6B.100d.txt`

## Main Final Experiment (full-train + full-eval)

> Final experiment does **not** pass `--max_train_steps` or `--max_eval_steps`.

```bash
python train.py \
  --dataset_name data/sst5_hf \
  --artifact_dir artifacts_submit/glove_nonstatic_seed13_full \
  --epochs 25 \
  --batch_size 256 \
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
  --early_stop_patience 5 \
  --early_stop_metric val_acc \
  --glove_path embeddings/glove.6B.100d.txt \
  --freeze_embedding false \
  --seed 13 \
  --device auto
```

## Evaluate Final Artifact

```bash
python evaluate.py \
  --artifact_dir artifacts_submit/glove_nonstatic_seed13_full \
  --dataset_name data/sst5_hf
```

Generated files:

- `classification_report.json`
- `confusion_matrix.json`
- `error_examples.json`

## Predict (after Train + Evaluate)

```bash
python predict.py \
  --artifact_dir artifacts_submit/glove_nonstatic_seed13_full \
  --text "This movie is absolutely wonderful and touching."
```

## Full Multi-Seed Repro (13 / 42 / 3407)

```bash
for seed in 13 42 3407; do
  out="artifacts_submit/glove_nonstatic_seed${seed}_full"

  python train.py \
    --dataset_name data/sst5_hf \
    --artifact_dir "${out}" \
    --epochs 25 \
    --batch_size 256 \
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
    --early_stop_patience 5 \
    --early_stop_metric val_acc \
    --glove_path embeddings/glove.6B.100d.txt \
    --freeze_embedding false \
    --seed "${seed}" \
    --device auto

  python evaluate.py --artifact_dir "${out}" --dataset_name data/sst5_hf
done
```

Then summarize from full reports (`classification_report.json`) to `artifacts_submit/multiseed_summary.json`.

## Metric Contract

- `train.py` enforces final test as full evaluation (`max_steps=0` fixed at test stage).
- `metrics.json` includes `test_eval_mode`, `test_num_batches`, `test_num_samples`.
- Final reporting should use full-test metrics from `classification_report.json`.
