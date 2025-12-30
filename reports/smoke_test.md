# 冒烟测试记录（make smoke）

## 1) 目的

在较小数据子集上快速验证整条流水线可用性：数据审计 → 预处理缓存 → 训练 → 评估 → 可视化，并生成至少 1 个指标与 1 张图（含底层数据）。

## 2) 命令与配置

```bash
conda activate pytorch
cd TargetFormer-Crime
conda run -n pytorch make smoke
```

配置文件：`configs/smoke.yaml`

- 审计：`max_videos=8`
- 预处理：`max_videos=4`、`max_frames_per_video=60`、`fps=5`、`num_segments=16`、`K=10`
- 训练：`epochs=1`、`d_model=128`、`layers=2`（快速验证）

## 3) 关键输出路径

- 审计清单：`outputs/dataset_audit_smoke/manifest.csv`
- 预处理缓存：`outputs/precomputed_smoke/`（Train=4，Test=4）
- 冒烟实验目录：`outputs/exp_20251230_081521/`
  - 指标与预测：`outputs/exp_20251230_081521/smoke_targetformer/seed_0/metrics.json`、`predictions.npz`
- 图与数据：
  - `outputs/figures_smoke/`
  - `outputs/figures_data_smoke/`

## 4) 冒烟结果（seed=0，Test=4）

来自 `outputs/exp_20251230_081521/smoke_targetformer/seed_0/metrics.json`：

- AUC(video)=0.6667，AP(video)=0.9167
- AUC(segment)=0.4023，AP(segment)=0.7133

## 5) 产出示例（至少一图）

- 训练曲线：`outputs/figures_smoke/smoke_targetformer_seed_0_loss_curves.png`
- ROC 曲线：`outputs/figures_smoke/smoke_targetformer_seed_0_roc.png`
- PR 曲线：`outputs/figures_smoke/smoke_targetformer_seed_0_pr.png`
- t-SNE：`outputs/figures_smoke/smoke_targetformer_seed_0_tsne.png`
- Token 热力图：`outputs/figures_smoke/smoke_targetformer_seed_0_token_heatmap.png`

对应底层数据在 `outputs/figures_data_smoke/` 中保存为 CSV（训练历史、ROC/PR 点、t-SNE 坐标、token 权重等）。
