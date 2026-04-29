# TextCNN SST-5 Course Practice

This project implements a clean and reproducible 5-class sentiment classifier on SST-5 using PyTorch + TextCNN.

## Task

Given an English movie review sentence, classify it into:

- 0: very negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very positive

## Structure

- `data.py`: dataset loading, tokenization, vocabulary, dataloaders
- `model.py`: TextCNN model
- `train.py`: training, validation, checkpoint, test, curves
- `predict.py`: single-sentence prediction
- `utils.py`: seed and utility helpers
- `artifacts/`: generated model and outputs
- `.cursor/agents/`: role definitions for collaboration

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 10 --batch_size 64
```

## Predict

```bash
python predict.py --text "This movie is absolutely wonderful and touching."
```
