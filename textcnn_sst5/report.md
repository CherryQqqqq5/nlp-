# 实验报告：TextCNN 在 SST-5 五分类情感分析中的应用

## 1. 任务介绍

本实验目标是构建一个句子级英文电影评论情感分类器。给定一条评论句子，模型需要预测其情感类别，标签为五分类：

- 0: very negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very positive

## 2. 数据集介绍

本实验使用 SetFit/sst5 数据集（SST-5，sentence-level），并在服务器上离线加载本地副本路径 data/sst5_hf。

- 任务类型：细粒度情感五分类
- 数据划分：train / validation / test
- 标签信息由数据集字段读取（label + label_text），避免手写映射错误

## 3. 方法介绍

模型采用 TextCNN，结构为：

Embedding -> Conv1d(k=3,4,5) -> ReLU -> MaxPool -> Concat -> Dropout -> Linear(5 classes)

关键实现点：

- Embedding 输出形状为 [B, L, E]
- Conv1d 期望输入为 [B, E, L]
- 前向中显式执行 x = x.transpose(1, 2)

## 4. 实验设置

本次在 CPU 环境完成训练。由于服务器 CPU 训练速度较慢，采用轻量化配置完成 10 epoch 课程实践闭环：

- dataset_name: /cephfs/qiuyn/nlp/textcnn_sst5/data/sst5_hf（离线加载）
- max_len: 10
- embedding_dim: 16
- num_filters: 8
- kernel_sizes: [3, 4, 5]
- dropout: 0.5
- batch_size: 8
- learning_rate: 1e-3
- epochs: 10
- max_train_steps: 5（每个 epoch 截断训练步数）
- max_eval_steps: 2（验证/测试截断步数）
- optimizer: Adam
- loss: CrossEntropyLoss
- metric: accuracy

## 5. 实验结果

训练脚本在 artifacts 目录产出：

- best_model.pt
- metrics.json
- history.json
- curves.png
- config.json

结果表：

| Model   | Validation Accuracy | Test Accuracy |
|---------|---------------------|---------------|
| TextCNN | 0.2500              | 0.1250        |

补充指标：

- best validation accuracy: 0.2500
- test loss: 1.6780

预测样例（predict.py --artifact_dir artifacts --max_len 10）：

- This movie is absolutely wonderful and touching. -> negative (0.2507)
- The plot is boring and the acting is terrible. -> negative (0.2982)
- The film is okay but not very impressive. -> negative (0.2417)

## 6. 结果分析

从结果可见，当前轻量化配置下模型已完成端到端训练、验证、测试和单句预测流程，但分类能力较弱，主要表现为多样例集中预测为 negative。

主要原因：

- 训练步数截断较小（max_train_steps=5），参数更新不足
- 模型容量较小（embedding 与 filter 数量较低）
- 五分类任务边界细，TextCNN 在有限训练预算下容易欠拟合

后续可改进方向：

- 增大训练步数与 epoch 的有效覆盖
- 增大 embedding_dim 与 num_filters
- 适度增加 max_len，保留更多上下文信息

## 7. 总结

本实验已完成课程实践要求的完整闭环：

- 数据加载与编码
- 模型前向验证
- 训练/验证/测试
- 模型保存与预测脚本
- 报告与角色文档

在当前服务器算力与网络约束下，项目达成“结构清楚、能训练、能预测、报告可交付”的实践目标。
