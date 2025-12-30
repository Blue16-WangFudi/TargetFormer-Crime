# TargetFormer-Crime

TargetFormer-Crime：面向目标（target-centric）的 UCF-Crime 视频异常行为模式挖掘与弱监督检测研究代码。

## 1. 环境

仅使用既有 Conda 环境：

```bash
conda activate pytorch
```

建议先导出环境信息（可复现记录）：

```bash
make audit
```

## 2. 数据集

默认自动探测数据集根目录（优先 `/datasets`，其次 `../datasets`；均要求包含 `Train/` 与 `Test/`）。
如需自定义路径，使用环境变量：

- `TFC_DATASETS_ROOT`：数据集根目录（包含 `Train/` 与 `Test/`）

## 3. 任务入口（Makefile）

```bash
make smoke       # 端到端冒烟：audit→preprocess→train→eval→viz
make audit       # 数据集审计 + 环境信息记录
make preprocess  # YOLO(+tracking) 目标 token 预计算（可断点续跑）
make train       # 训练（TargetFormer/基线/消融，配置驱动）
make eval        # 评估 + 保存预测数组
make viz         # 生成论文级图与对应数据文件
make paper       # XeLaTeX 编译（../latex）
make clean       # 清理 outputs/*
```

## 4. 复现实验（推荐流程）

1) `make audit`
2) `make preprocess`
3) `make train`
4) `make eval`
5) `make viz`

所有输出均写入 `outputs/`（默认被 `.gitignore` 排除）。
