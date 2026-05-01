# 实验报告：TextCNN 在 SST-5 五分类情感分类中的最终提交结果

## 1. 任务与目标

任务：对英文电影评论句子做五分类（very negative / negative / neutral / positive / very positive）。

本次提交的验收目标是：

- 最终结果必须来自 full-train + full-test（不截断）
- 指标口径统一（`metrics.json` 与 `classification_report.json` 对齐）
- 主结论仅引用 `artifacts_submit/` 下结果

## 2. 数据与预处理

- 数据集：`data/sst5_hf`（离线 SST-5）
- 分词：正则 tokenizer（保留标点，统一小写）
- 词表：仅基于 train 构建，`min_freq=2`
- 截断长度：`max_len=50`
- 词向量：`glove.6B.100d`

数据统计文件：`artifacts_submit/glove_nonstatic_seed13_full/data_stats.json`。

## 3. 方法与训练策略

模型：TextCNN (`Embedding -> Conv1d(k=3,4,5) -> MaxPool -> Concat -> Dropout -> Linear`)。

训练策略：

- `AdamW` + `weight_decay`
- `ReduceLROnPlateau`
- `early stopping`
- 可选 `class-weighted CrossEntropyLoss`（对照实验）

关键口径约束（已在代码实现）：

- 最终 test 阶段固定 full-eval（`max_steps=0`）
- `metrics.json` 写入 `test_eval_mode/test_num_batches/test_num_samples`

## 4. 实验设置

统一参数（final 配置）：

- `epochs=25`
- `batch_size=256`
- `embed_dim=100`
- `num_filters=100`
- `lr=5e-4`
- `dropout=0.5`
- `weight_decay=1e-4`
- `early_stop_patience=5`
- `scheduler_metric=val_acc`

设备：CPU（`--device auto` 实际选择 CPU）。

## 5. 结果

### 5.1 表 1：Step-budget 策略筛选（非最终提交口径）

说明：`debug_ablation_step_budget/` 仅用于早期策略筛选，不用于最终结论。

| 用途 | 目录 | 是否作为最终结果 |
|---|---|---|
| 快速筛选 | `debug_ablation_step_budget/` | 否 |
| 最终提交 | `artifacts_submit/` | 是 |

### 5.2 表 2：Final full-run 主结果（最终提交口径）

来源：`artifacts_submit/glove_nonstatic_seed{13,42,3407}_full` 与 `artifacts_submit/multiseed_summary.json`。

| Seed | Best Val Acc | Full Test Acc | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| 13 | 0.4342 | 0.4507 | 0.3914 | 0.4246 |
| 42 | 0.4396 | 0.4538 | 0.4048 | 0.4330 |
| 3407 | 0.4332 | 0.4597 | 0.4102 | 0.4384 |
| mean ± std | - | **0.4548 ± 0.0037** | **0.4021 ± 0.0079** | **0.4320 ± 0.0057** |

指标口径一致性：三组实验 `metrics.json.test_acc` 与 `classification_report.json.accuracy` 差值均为 0。

### 5.3 类别塌缩对照：class-weighted（推荐对照）

来源：

- baseline: `artifacts_submit/glove_nonstatic_seed13_full`
- class-weighted: `artifacts_submit/glove_nonstatic_seed13_full_classweighted`
- 汇总：`artifacts_submit/classweight_ablation_summary.json`

| Setting | Full Test Acc | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| baseline | 0.4507 | 0.3914 | 0.4246 |
| class-weighted | 0.4294 | 0.4200 | 0.4288 |
| delta | -0.0213 | +0.0287 | +0.0042 |

少数类召回变化（class-weighted - baseline）：

- very negative: +0.2509
- neutral: +0.0720
- very positive: +0.1629

结论：class-weighted 明显缓解少数类塌缩并提升 macro-F1，但牺牲了整体 accuracy。

## 6. 结论

本次最终提交满足验收门槛：

1. 最终 test 采用 full-eval，未使用 step 截断；
2. 主结果来自 `artifacts_submit/`，口径统一且可复现；
3. 多 seed full-run 给出稳定均值：`Full Test Acc = 0.4548 ± 0.0037`；
4. class-weighted 对照验证了“accuracy 与少数类鲁棒性”的权衡。

后续可继续尝试：

- GloVe 300d 对照
- 更强文本编码器（如 BiLSTM/Transformer）
- 面向 neutral/very negative 的 targeted augmentation
