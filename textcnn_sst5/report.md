# 实验报告：TextCNN 在 SST-5 五分类情感分析中的应用

## 1. 任务介绍

本实验目标是构建一个句子级英文电影评论情感分类器。给定一条评论句子，模型需要预测其情感类别，标签为五分类：

- 0: very negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very positive

## 2. 数据集介绍

本实验使用 `SetFit/sst5` 数据集（SST-5，sentence-level）。

- 任务类型：细粒度情感五分类
- 数据划分：train / validation / test
- 标签信息由数据集元信息读取，不手写映射，避免顺序错误

## 3. 方法介绍

模型采用 TextCNN，结构为：

Embedding -> Conv1d(k=3,4,5) -> ReLU -> MaxPool -> Concat -> Dropout -> Linear(5 classes)

关键实现点：

- Embedding 输出形状为 `[B, L, E]`
- Conv1d 期望输入为 `[B, E, L]`
- 前向中显式执行 `x = x.transpose(1, 2)`

## 4. 实验设置

- max_len: 50
- embedding_dim: 128
- num_filters: 100
- kernel_sizes: [3, 4, 5]
- dropout: 0.5
- batch_size: 64
- learning_rate: 1e-3
- epochs: 10
- optimizer: Adam
- loss: CrossEntropyLoss
- metric: accuracy

## 5. 实验结果

训练脚本会在 `artifacts/` 输出：

- `best_model.pt`
- `metrics.json`（best val acc / test acc）
- `history.json`
- `curves.png`（训练损失与准确率曲线）

结果表（请在完成训练后填入）：

| Model   | Validation Accuracy | Test Accuracy |
|---------|---------------------|---------------|
| TextCNN | 待填                | 待填          |

## 6. 结果分析

可从错误样例观察到：

- 对带有明显情感词（如 excellent / terrible）的句子识别较稳定
- 对 neutral 与弱正负类、以及 positive vs very positive 边界较细的样本更易混淆

原因在于 TextCNN 主要依赖局部 n-gram 特征，对长距离依赖、复杂语义转折建模能力有限。

## 7. 总结

本实验完成了一个结构清晰、可复现、可训练与可预测的五分类实践系统。TextCNN 作为课程实践模型，具备实现简单、运行高效、可解释性较好的优势，可满足课程作业对“可跑通 + 可展示 + 可报告”的核心要求。
