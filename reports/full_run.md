# 全量实验记录（configs/full.yaml）

## 1) 命令序列（从预处理到可视化）

```bash
conda activate pytorch
cd TargetFormer-Crime

#（已完成）数据审计
conda run -n pytorch make audit

#（已完成）全量预处理（含多变体缓存）
conda run -n pytorch make preprocess

#（已完成）全量训练：主模型 3 seeds + 基线 + 消融
conda run -n pytorch make train

#（已完成）评估：生成 metrics.json 与 predictions.npz
conda run -n pytorch make eval

#（已完成）可视化：曲线/热力图/t-SNE/定性视频
conda run -n pytorch make viz
```

全量配置文件：`configs/full.yaml`

## 2) 关键输出目录

### 数据审计

- 清单：`outputs/dataset_audit/manifest.csv`
- 图：`outputs/figures/dataset_audit_*.png`
- 图数据：`outputs/figures_data/dataset_audit_*.csv`

### 预处理缓存（可断点续跑）

- 主缓存（tracking+fps10）：`outputs/precomputed/`（Train=1610，Test=290）
- B1 全局特征（无 YOLO）：`outputs/precomputed_global/`（Train=1610，Test=290）
- B3 无 tracking：`outputs/precomputed_no_track/`（Train=1610，Test=290）
- A6 fps5：`outputs/precomputed_fps5/`（Train=1610，Test=290）

### 全量实验目录

本次全量实验输出根：`outputs/exp_20251230_073809/`

- 训练汇总：`outputs/exp_20251230_073809/results_summary.json`
- 结果表：`outputs/exp_20251230_073809/results_table.csv`
- 每个 run/seed 的评估：
  - `outputs/exp_20251230_073809/<run>/seed_*/metrics.json`
  - `outputs/exp_20251230_073809/<run>/seed_*/predictions.npz`

## 3) 全量结果汇总（Video-level best checkpoint）

来自 `outputs/exp_20251230_073809/results_summary.json`（AUC/AP 越大越好）：

### 主模型（3 seeds，mean ± std）

- main_targetformer：AUC=0.8205 ± 0.0056；AP=0.7799 ± 0.0143

### 基线（单种子）

- B1_global_mlp：AUC=0.8776；AP=0.8562
- B2_yolo_gru：AUC=0.8472；AP=0.8105
- B3_yolo_no_track：AUC=0.8079；AP=0.7790

### 消融（单种子）

- A1_motion_only：AUC=0.5518；AP=0.4837
- A2_appearance_only：AUC=0.7954；AP=0.7685
- A3_k_5：AUC=0.7685；AP=0.7457
- A3_k_20：AUC=0.8027；AP=0.7704
- A4_depth_2：AUC=0.8382；AP=0.8151
- A4_depth_6：AUC=0.7495；AP=0.7184
- A5_no_prototypes：AUC=0.7823；AP=0.7408
- A6_fps_5：AUC=0.8035；AP=0.7641

说明：训练阶段内部用 video-level AUC 选择 `checkpoint_best.pt`；评估阶段对每个 run/seed 输出 `auc_video/ap_video` 以及基于弱标签近似的 `auc_segment/ap_segment`。

## 4) 论文级可视化产物（含底层数据）

### 曲线图（每个 run/seed）

- 训练曲线：`outputs/figures/<run>_<seed>_loss_curves.png`
- AUC 曲线：`outputs/figures/<run>_<seed>_auc_curve.png`
- ROC/PR：`outputs/figures/<run>_<seed>_roc.png`、`_pr.png`
- 曲线点数据：`outputs/figures_data/<run>_<seed>_roc.csv`、`_pr.csv`
- 训练历史数据：`outputs/figures_data/<run>_<seed>_train_history.csv`

### 嵌入与注意力（主模型 seed_0）

- t-SNE：`outputs/figures/main_targetformer_seed_0_tsne.png`
- t-SNE 数据：`outputs/figures_data/main_targetformer_seed_0_tsne.csv`
- Token 贡献热力图：`outputs/figures/main_targetformer_seed_0_token_heatmap.png`
- Token 权重数据：`outputs/figures_data/main_targetformer_seed_0_token_weights.csv`

### 定性视频（主模型 seed_0，5 个样例）

- 视频：`outputs/qualitative/main_targetformer_seed_0_*.mp4`
- 时间线数据：`outputs/qualitative_data/main_targetformer_seed_0_*_timeline.csv`

## 5) 可复现性与审计信息

- 依赖导出：`environment.yml`、`requirements.txt`
- 每个 run/seed 的环境与设备信息：`outputs/exp_20251230_073809/<run>/seed_*/run_meta.json`
- 每个 run/seed 的配置快照：`outputs/exp_20251230_073809/<run>/seed_*/config.yaml`

> 注意：`outputs/`、`datasets/`、`../latex/` 均被 `.gitignore` 严格排除，不会推送到 GitHub。
