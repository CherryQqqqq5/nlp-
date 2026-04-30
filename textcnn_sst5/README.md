# TextCNN SST-5 Course Practice

This project implements a reproducible 5-class sentiment classifier on SST-5 using PyTorch + TextCNN.

## Task

Given an English movie review sentence, classify it into:

- 0: very negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very positive

## Structure

- data.py: dataset loading, tokenization, vocabulary, dataloaders
- model.py: TextCNN model
- train.py: training, validation, checkpoint, test, curves
- predict.py: single-sentence prediction
- utils.py: seed and utility helpers
- data/sst5_hf: offline SST-5 dataset copy
- artifacts_final/: final training outputs
- .cursor/agents/: role definitions for collaboration

## Install

```bash
cd textcnn_sst5
pip install -r requirements.txt
```

## Train (main experiment)

```bash
python train.py \
  --dataset_name data/sst5_hf \
  --epochs 15 \
  --batch_size 64 \
  --embed_dim 128 \
  --num_filters 100 \
  --max_len 50 \
  --artifact_dir artifacts_final \
  --device auto
```

## Predict

```bash
python predict.py \
  --artifact_dir artifacts_final \
  --max_len 50 \
  --text "This movie is absolutely wonderful and touching."
```

## Video Submission

The 1-2 minute demo video is submitted outside this GitHub repository.
Recording checklist:

1. Show project structure and task definition.
2. Show training run (or final trained artifacts) and test accuracy.
3. Run predict.py with several sample sentences.
