# EEG 睡眠分期项目 Pipeline

## 1. 目标与范围
- 任务：基于 EEG 信号预测睡眠阶段。
- 阶段标签：N1、N2、N3、REM（可选保留 W，训练时支持 4 类或 5 类配置）。
- 核心模型：`EEG -> 1D CNN 特征提取 -> Transformer Encoder -> 分类头`。
- 输出：模型权重、评估指标、混淆矩阵、频段重要性分析。

## 2. 输入输出定义
- 输入数据：Sleep-EDF（EDF 信号 + 对应睡眠分期标注）。
- 基本样本单位：30 秒 epoch。
- 输入张量（单通道示例）：`[batch, channels=1, samples=3000]`（100 Hz 下）。
- 输出：每个 epoch 的类别概率 `p(y|x)` 与预测标签。

## 3. 程序总流程
1. 数据下载与索引构建。
2. EEG 预处理与 epoch 切分。
3. 数据集划分（按被试划分，避免数据泄漏）。
4. 模型训练（CNN + Transformer）。
5. 模型评估与可解释性分析。
6. 保存结果与报告。

## 4. 数据与预处理设计
### 4.1 数据准备
- 数据源：Sleep-EDF（建议先使用 Sleep-EDF-20 作为起步实验）。
- 构建 `metadata.csv`，字段至少包含：
  - `subject_id`
  - `record_id`
  - `eeg_path`
  - `label_path`
  - `split`（train/val/test）

### 4.2 预处理标准
- 通道选择：固定一个主通道（如 Fpz-Cz），后续可扩展多通道。
- 重采样：统一到 100 Hz。
- 滤波：0.3-35 Hz 带通；工频陷波（50/60 Hz，按数据来源设置）。
- 分段：按 30 秒切 epoch。
- 标签映射：`{N1,N2,N3,REM}`（或 `{W,N1,N2,N3,REM}`）。
- 标准化：每条记录 z-score，或每个被试 z-score（二选一并固定）。

### 4.3 质量控制
- 丢弃无效标注 epoch。
- 统计各类别样本数，输出类别分布报告。
- 记录异常样本比例（缺失、饱和、强伪迹）。

## 5. 模型设计
### 5.1 CNN 特征提取器
- 使用 1D 卷积块（Conv1d + BN + GELU + Pooling）。
- 目标：从原始时序中提取局部时频模式。
- 输出：固定长度 embedding 向量（例如 128 或 256 维）。

### 5.2 Transformer Encoder
- 输入：CNN embedding 序列。
- 建议将相邻 epoch 组成上下文窗口（如 `t-2 ... t+2`），提升时序一致性。
- 配置建议：
  - `num_layers=2~4`
  - `num_heads=4~8`
  - `dropout=0.1`

### 5.3 分类头
- `LayerNorm + Linear` 输出类别 logits。
- 损失函数：加权交叉熵（处理类别不平衡）。

## 6. 训练策略
- 优化器：AdamW。
- 学习率：初始 `1e-3`（可从 `3e-4` 到 `1e-3` 网格搜索）。
- 调度器：Cosine Annealing 或 ReduceLROnPlateau。
- 批大小：32/64（按显存调整）。
- 早停：监控验证集 `macro-F1`，`patience=10`。
- 不平衡处理：
  - 类别权重；
  - 可选 WeightedRandomSampler。

## 7. 评估与分析
### 7.1 主评估指标
- Accuracy
- Macro-F1
- Cohen's Kappa
- 各类别 Precision/Recall/F1

### 7.2 可视化输出
- 混淆矩阵热力图。
- 训练曲线（loss/F1）。

### 7.3 频段重要性分析
- 频段：delta(0.5-4), theta(4-8), alpha(8-13), sigma(12-16), beta(13-30)。
- 方法建议：
  - 频段遮挡（band-stop ablation）后观察性能下降；
  - 或使用积分梯度/输入梯度得到频段贡献。
- 输出：各频段的重要性排名与解释结论。

## 8. 工程目录建议
- `Context/`
  - `Descripition.md`
  - `Pipeline.md`
- `Data/`
  - `raw/` 原始 EDF
  - `processed/` 预处理后样本
  - `metadata.csv`
- `Main/`
  - `config.py` 超参数配置
  - `dataset.py` 数据读取与切分
  - `preprocess.py` 预处理流程
  - `model.py` CNN+Transformer 模型
  - `train.py` 训练入口
  - `evaluate.py` 评估与混淆矩阵
  - `interpret.py` 频段重要性分析
- `Result/`
  - `checkpoints/`
  - `metrics.json`
  - `confusion_matrix.png`
  - `band_importance.csv`

## 9. 最小可用版本（MVP）
1. 单通道 EEG + 4 分类（N1/N2/N3/REM）。
2. 完成训练、验证、测试闭环。
3. 输出 `Macro-F1` 与混淆矩阵。
4. 完成一次频段遮挡分析并生成结论表。

## 10. 验收标准
- 能在测试集稳定输出睡眠分期结果。
- 结果文件完整可复现（配置、模型、指标、图表）。
- 提供频段重要性分析，且结论与睡眠生理规律基本一致。
