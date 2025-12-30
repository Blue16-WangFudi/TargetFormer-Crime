# 实施方案（可复现流水线）

## 0) 硬约束（按任务要求执行）

- 仅使用既有 Conda 环境：`conda activate pytorch`（不新建 env）
- 记录并导出依赖：
  - `environment.yml`：`conda env export -n pytorch > environment.yml`
  - `requirements.txt`：`conda run -n pytorch pip freeze > requirements.txt`
- 硬件：RTX 4060 Laptop（8GB），默认开启 AMP，必要时用梯度累积
- 复现实验：固定随机种子、保存每次实验的配置快照、指标与预测数组、绘图原始数据
- 发布 GitHub 时必须排除：数据集、LaTeX 源码、outputs/缓存/模型权重等

## 1) 项目结构与入口

- `src/targetformer_crime/`：核心代码（数据集扫描/预处理/模型/训练/评估/可视化）
- `configs/`：`default.yaml`（审计）、`smoke.yaml`（快速端到端）、`full.yaml`（全量实验）
- `scripts/run.py`：统一入口，注入 `src/` 并设置 Windows/确定性相关环境变量
- `Makefile`：任务式执行
  - `make smoke`：审计→预处理→训练→评估→可视化（小子集）
  - `make audit`：数据集审计与清单
  - `make preprocess`：预处理缓存（可断点续跑）
  - `make train / eval / viz`：全量实验训练、评估与可视化
  - `make paper`：XeLaTeX 编译 `../latex/main.tex`

## 2) 数据集审计（Dataset Audit）

目标：不假设固定目录名，在 `paths.datasets_root` 下自动发现 UCF-Crime（本机为 `../datasets`）。

产物（必须保存）：

- `outputs/dataset_audit/manifest.csv`：每个视频/帧序列的清单（路径、标签、类别、帧数、分辨率、估算时长等）
- `outputs/figures/dataset_audit_*.png`：类别统计、时长直方图、fps/分辨率分布等
- `outputs/figures_data/dataset_audit_*.csv`：绘图底层数据

## 3) 预处理（YOLO + Tracking → Segment Tokens）

目标：将每个视频转换为固定长度的 token 序列，供弱监督 MIL 训练使用。

关键设计：

- 采样 FPS：默认 10（可配）
- 段数：默认 32（与 UCF-Crime 常用协议对齐）
- 每段 Top-K：默认 10（可配）
- token 组成：几何（bbox）、运动（速度/加速度统计）、外观（ROI→ResNet18 embedding）
- 缓存：保存为 `npz`（`tokens/masks/meta`），支持 `resume: true` 断点续跑

多变体缓存（用于基线与消融）：

- `outputs/precomputed`：主设置（tracking + fps10）
- `outputs/precomputed_global`：B1（无 YOLO，全帧特征）
- `outputs/precomputed_no_track`：B3（无 tracking）
- `outputs/precomputed_fps5`：A6（fps5）

## 4) 模型与训练（TargetFormer + MIL）

模型：Transformer Encoder（片段位置 + 目标位置双位置编码）输出每段异常分数。

损失：

- MIL 排序损失：约束异常视频 max score > 正常视频 max score（margin）
- 平滑正则：$\sum_i (s_i-s_{i+1})^2$
- 稀疏正则：异常视频分数平均值（鼓励少量峰值）
- 原型库（可选）：用于行为模式挖掘与解释，并支持消融（A5）

GPU-safe：

- AMP：`amp: true`
- 梯度累积：`grad_accum_steps: 4`（全量默认）
- 批大小：`batch_size: 1`

## 5) 评估与可视化

评估输出：

- 每个 run/seed：`metrics.json`（AUC/AP）、`predictions.npz`（`y_true/y_score/seg_scores` 等）

可视化输出（发表级 & 保存底层数据）：

- `outputs/figures/*.png`：训练曲线、ROC/PR、t-SNE、token 权重热力图等
- `outputs/figures_data/*.csv`：每张图对应的原始数据（曲线点、t-SNE 坐标、权重矩阵等）
- `outputs/qualitative/*.mp4`：3–5 个测试视频的叠加可视化（bbox/track id/异常分数时间线）

## 6) 论文与发布

- 论文：在 `../latex` 下按模板写作并 `XeLaTeX` 编译通过（无错误）
- GitHub：仅提交代码/配置/脚本/测试/README（严格 `.gitignore` 排除数据/latex/outputs）
