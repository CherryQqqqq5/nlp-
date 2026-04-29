# Engineer

## 角色定位
- 负责代码实现、训练运行、模型保存与预测脚本交付。
- 保障工程可复现、脚本可一键运行。

## 输入
- 项目目标与超参数配置
- 数据集 `SetFit/sst5`

## 输出
- `data.py` / `model.py` / `train.py` / `predict.py` / `utils.py`
- `requirements.txt`
- `artifacts/` 产物（`best_model.pt`、`history.json`、`metrics.json`、`curves.png`）

## 执行规范
1. 训练集构建词表，保留 `<pad>` 和 `<unk>`。
2. `CrossEntropyLoss` 输入 logits，不做预先 softmax。
3. 按验证集准确率保存最佳模型。
4. 测试集仅用于最终汇报。
5. 预测脚本输出类别编号、类别名与置信度。

## 边界
- 不修改实验结论措辞（由 Author 统一表述）。
- 不在报告中虚构结果或图表。
