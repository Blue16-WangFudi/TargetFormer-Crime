# 实验设计（Baselines + Ablations）

## 1) 数据集与划分（自动探测）

数据根目录由 `configs/*.yaml` 的 `paths.datasets_root` 指定（本机为 `../datasets`），程序自动扫描其下的 `Train/` 与 `Test/`。

本次审计结果（`outputs/dataset_audit/manifest.csv`）：

- 总数：1900
- Train：1610
- Test：290
- 标签：Normal 950 / Abnormal 950（由 `NormalVideos` 目录名判定 normal，其余类别为 abnormal）
- 数据形态：已解帧的帧序列（64×64，按文件名 frame_idx 间隔可估算来源 fps）

## 2) 评价指标与协议

由于当前本地数据目录未提供标准的 temporal annotation 文件，采用以下两类指标：

1) **Video-level AUC/AP**：以每个视频的 `max(segment_score)` 作为视频分数，与视频标签计算 ROC-AUC 与 AP。  
2) **Segment-level AUC/AP（弱标签近似）**：将视频标签复制到每个片段，计算 `seg_scores` 的 AUC/AP。该指标主要用于对比不同设置下的“分数分布与分离度”，不等价于真实定位 AUC。

所有评估均保存原始数组以便复现与二次分析：

- `outputs/exp_*/<run>/seed_*/predictions.npz`：`uids/y_true/y_score/seg_scores/y_true_segment`
- `outputs/exp_*/<run>/seed_*/metrics.json`：`auc_video/ap_video/auc_segment/ap_segment`

## 3) 主模型（TargetFormer-Crime）

预处理：YOLO(person) + 稳定关联追踪 → 每视频 N=32 段、每段 Top-K=10 目标 token。  
模型：Transformer Encoder（默认 `d_model=256, nhead=8, layers=4, dropout=0.1`）  
弱监督：MIL ranking + smoothness + sparsity；并使用 Prototype Bank 进行模式挖掘（可消融）。

统计：主模型运行 3 个随机种子（0,1,2），报告 mean ± std。

## 4) 必跑基线（B1–B3）

- **B1：Global segment feature + MIL-MLP（无 YOLO）**  
  使用全帧 ResNet18 embedding 构建 `K=1` token，MLP 得到片段分数。
- **B2：YOLO tokens + GRU（替代 Transformer）**  
  先做 token 聚合得到每段 embedding，再用 GRU 建模序列并输出分数。
- **B3：YOLO tokens + Transformer（无 tracking）**  
  在采样帧上仅做检测，按 frame-wise top-K 构建段内 token，不保证轨迹一致性。

## 5) 必跑消融（A1–A6）

- **A1：motion-only**（去除外观）  
- **A2：appearance-only**（去除运动）  
- **A3：K ∈ {5, 10, 20}**（其中 K=10 为主模型默认；另外跑 K=5 与 K=20）  
- **A4：Transformer depth ∈ {2, 4, 6}**（4 为默认；另外跑 2 与 6）  
- **A5：Prototype Bank on/off**（关闭原型库）  
- **A6：预处理 FPS=5 vs 10**（对比低采样率）

## 6) 训练配置与资源约束（RTX 4060 8GB）

全量配置见 `configs/full.yaml`，关键默认值：

- `batch_size=1`
- `amp=true`
- `grad_accum_steps=4`
- `steps_per_epoch=200`（主模型单独设为 400）
- `epochs=8~10`（主模型/基线 10，部分消融 8）

所有实验都会保存：

- `outputs/exp_*/config.yaml`：实验总配置快照
- `outputs/exp_*/<run>/seed_*/config.yaml`：run/seed 级别配置快照
- `outputs/exp_*/<run>/seed_*/run_meta.json`：python/torch/cuda/cudnn/gpu/driver/git hash
- `outputs/exp_*/results_summary.json` 与 `results_table.csv`：结果汇总表
