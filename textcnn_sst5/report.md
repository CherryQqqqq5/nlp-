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

本次在服务器 CPU 环境完成完整训练（不截断 step），采用标准配置：

- dataset_name: data/sst5_hf（离线加载）
- max_len: 50
- embedding_dim: 128
- num_filters: 100
- kernel_sizes: [3, 4, 5]
- dropout: 0.5
- batch_size: 64
- learning_rate: 1e-3
- epochs: 15
- optimizer: Adam
- loss: CrossEntropyLoss
- metric: accuracy

输出目录：artifacts_final/

## 5. 实验结果

训练脚本在 artifacts_final 目录产出：

- best_model.pt
- metrics.json
- history.json
- curves.png
- config.json

结果表：

| Model   | Best Validation Accuracy | Test Accuracy | Test Loss |
|---------|--------------------------|---------------|-----------|
| TextCNN | 0.3769                   | 0.3511        | 1.4853    |

补充指标（来自 metrics.json 与 history.json）：

- best validation accuracy: 0.3769300636（第 4 个 epoch）
- best validation loss: 1.4584（第 5 个 epoch）
- final epoch(15) train_acc: 0.9171
- final epoch(15) val_acc: 0.3669

## 6. 结果分析

从曲线与指标可见：

- 训练集性能持续提升（train_acc 从 0.2654 升至 0.9171）
- 验证集在第 4-5 轮达到峰值后进入波动并整体退化
- 最终训练-验证存在较大泛化差距（约 0.55），呈现明显过拟合

说明：

- TextCNN 在该设置下具备较强拟合能力，但对 SST-5 细粒度类别边界的泛化有限
- 课程实践目标（能训练、能验证、能测试、能预测、可复现）已完成

可行改进方向：

- 使用 early stopping（按验证集指标在第 4-5 轮附近停止）
- 增加正则化（更高 dropout、权重衰减）
- 引入预训练词向量或更强的上下文编码模型
- 进行类别混淆矩阵分析，针对易混类别优化数据与损失设计

## 7. 总结

本实验完成了 SST-5 五分类 TextCNN 的完整工程闭环：

- 数据离线加载与编码
- 模型训练与最佳模型保存
- 验证/测试评估与曲线可视化
- 单句预测脚本与可复现实验配置

最终测试准确率为 0.3511。结果表明，TextCNN 可作为课程实践的可靠基线，但在细粒度情感任务上需要更强正则化与建模能力以提升泛化表现。
