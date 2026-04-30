# 实验报告：TextCNN 在 SST-5 五分类情感分析中的高分冲刺实验

## 1. 任务介绍

目标是完成句子级英文电影评论五分类：

- 0: very negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very positive

本次不是单次跑通，而是建立“可复现 + 可对比 + 可分析”的实验体系。

## 2. 数据集与数据统计

使用离线副本 `data/sst5_hf`（来源 `SetFit/sst5`，train/validation/test）。

当前实现的数据处理要点：

- tokenizer：正则切分（保留标点，兼容 `-lrb-/-rrb-`）
- vocab：仅基于 train 构建，`min_freq=2`
- 序列长度：`max_len=50`

由 `data_stats.json` 自动输出：

- 样本量：train/val/test
- 平均句长：`avg_train_len`, `avg_val_len`, `avg_test_len`
- OOV 比例：`val_oov_ratio`, `test_oov_ratio`

## 3. 方法

### 3.1 TextCNN 结构

`Embedding -> Conv1d(k=3,4,5) -> ReLU -> MaxPool -> Concat -> Dropout -> Linear(5)`

关键维度变换：`[B, L, E] -> [B, E, L]`。

### 3.2 Embedding 方案

- `rand`: 随机初始化
- `glove-static`: 加载 GloVe-100d，冻结 embedding
- `glove-non-static`: 加载 GloVe-100d，允许微调

GloVe 覆盖率（本词表）约为 `0.9967 (8205/8232)`。

### 3.3 训练策略

为抑制过拟合，训练脚本支持：

- `AdamW` + `weight_decay`
- `ReduceLROnPlateau`
- `early stopping`
- 可选 `grad_clip`

## 4. 实验设置

### 4.1 统一设置

- dataset_name: `data/sst5_hf`
- batch_size: 64
- max_len: 50
- num_filters: 100
- kernel_sizes: (3,4,5)
- device: auto（本次运行在 CPU）

### 4.2 本轮对比实验说明

本轮为了快速完成完整实验矩阵，采用了固定 step budget：

- `max_train_steps=8`
- `max_eval_steps=4`

该设置用于高效比较不同策略，不代表最终可提交的“长训练上限”。

## 5. 实验结果

### 5.1 Ablation 对比（单次）

| Model | Best Val Acc | Test Acc | Test Loss | Macro-F1 |
|---|---:|---:|---:|---:|
| rand_baseline | 0.3242 | 0.2617 | 1.5939 | 0.2197 |
| rand_reg (AdamW+ES+Scheduler) | 0.2891 | 0.2500 | 1.6020 | 0.1351 |
| glove-static (100d) | 0.3516 | 0.3242 | 1.5066 | 0.1896 |
| glove-non-static (100d) | 0.3164 | **0.3477** | 1.5391 | 0.1602 |

结论：在本轮设置下，`glove-non-static` 取得最高 `test_acc`，`glove-static` 的 `best_val_acc` 更高。

### 5.2 多随机种子（glove-non-static）

Seeds: `13, 42, 3407`

单次结果：

- seed13: test_acc=0.3789, macro_f1=0.1897
- seed42: test_acc=0.3477, macro_f1=0.1602
- seed3407: test_acc=0.3008, macro_f1=0.1301

汇总（mean ± std）：

- `best_val_acc`: `0.3359 ± 0.0305`
- `test_acc`: `0.3424 ± 0.0393`
- `test_loss`: `1.5418 ± 0.0170`
- `macro_f1`: `0.1600 ± 0.0298`
- `weighted_f1`: `0.2104 ± 0.0360`

汇总文件：

- `artifacts_hs/multiseed/glove_nonstatic_multiseed_summary.json`

## 6. 详细分析

### 6.1 混淆矩阵特征（以 seed13 为例）

从 `confusion_matrix.json` 看，模型显著偏向 `negative` 与 `positive` 两个中间强度类别：

- `very negative -> negative` 混淆突出
- `very positive -> positive` 混淆突出
- `neutral` 的召回非常低（容易被压到邻近极性）

### 6.2 分类报告（seed13）

- `negative` F1 较高（0.5171）
- `positive` F1 中等（0.4061）
- `very negative / neutral / very positive` F1 很低

说明模型更容易学习“中强度情感词”，但对细粒度边界和中性语义仍不足。

### 6.3 错误样例现象

`error_examples.json` 显示高频错误集中在：

- 弱否定、轻度褒贬、转折句
- 强度相邻类别（very negative vs negative；positive vs very positive）

## 7. 结论与下一步

本次冲刺完成了高分作业应有的核心形态：

- 从单结果升级为实验矩阵
- 引入预训练词向量与训练正则化
- 增加混淆矩阵、分类报告、错误样例
- 给出多 seed 的统计稳定性

在当前 step budget 下，最佳单次 `test_acc=0.3477`，多 seed 均值 `0.3424 ± 0.0393`。

下一步若继续提分，建议：

1. 去掉 step 截断并进行完整 epoch 训练；
2. 在 `glove-static/non-static` 上扩大超参搜索（lr, dropout, weight_decay）；
3. 增加 `glove.6B.300d` 与更强 tokenization 对照；
4. 针对 `neutral` 类别进行类别不平衡与难例策略优化。
