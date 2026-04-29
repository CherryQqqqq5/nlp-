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

- data.py: dataset loading, tokenization, vocabulary, dataloaders
- model.py: TextCNN model
- train.py: training, validation, checkpoint, test, curves
- predict.py: single-sentence prediction
- utils.py: seed and utility helpers
- data/sst5_hf: offline SST-5 dataset copy
- artifacts/: generated model and outputs
- .cursor/agents/: role definitions for collaboration

## Install

pip install -r requirements.txt

## Train (offline dataset)

python train.py --dataset_name /cephfs/qiuyn/nlp/textcnn_sst5/data/sst5_hf --epochs 10 --batch_size 8 --embed_dim 16 --num_filters 8 --max_len 10 --max_train_steps 5 --max_eval_steps 2 --artifact_dir artifacts --device cpu

## Predict

python predict.py --artifact_dir artifacts --max_len 10 --text This movie is absolutely wonderful and touching.
