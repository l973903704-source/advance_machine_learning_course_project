# Advance Machine Learning Course Project

本仓库是高级机器学习课程项目代码，主要围绕 POI（Point of Interest）推荐任务展开，包含数据预处理、低频 POI 过滤实验、STAN 相关模型训练，以及邻居协同增强版本的实验脚本。

英文版运行说明见 [README_EN.md](README_EN.md)。

## 项目结构

```text
.
├── data/
│   ├── NYC.npy
│   └── NYC_POI.npy
├── collab_common.py
├── layers.py
├── load.py
├── models.py
├── run_stan_collab.sh
├── train.py
├── train_contrastive.py
├── train_neighbor.py
├── train_proto.py
├── README.md
└── README_EN.md
```

主要文件说明：

- `load.py`：数据预处理与低频 POI 过滤。
- `models.py`、`layers.py`：模型结构与网络层定义。
- `train.py`：基础训练入口。
- `train_neighbor.py`：邻居协同版本训练入口。
- `train_contrastive.py`：对比学习相关训练入口。
- `train_proto.py`：原型相关训练入口。
- `collab_common.py`：协同增强实验中的公共工具函数。
- `run_stan_collab.sh`：用于服务器 Slurm 环境的实验提交脚本。

## 环境依赖

推荐使用 Python 3.7 及以上版本。

主要依赖：

```bash
pip install numpy joblib tqdm torch matplotlib
```

如果在服务器上运行，建议使用 Conda 创建独立环境，并根据服务器 CUDA 版本安装对应的 PyTorch。

## 数据准备

项目默认需要以下原始数据文件：

```text
data/NYC.npy
data/NYC_POI.npy
```

预处理缓存文件 `data/NYC_data.pkl` 不需要上传到 GitHub，可在本地或服务器重新生成：

```bash
python load.py --dname NYC
```

## 低频 POI 过滤实验

可以通过 `load.py` 生成不同过滤阈值的数据集。例如：

```bash
python load.py \
  --dname NYC \
  --output_dname NYC_min20 \
  --min_loc_visits 20 \
  --min_user_checkins 6
```

生成后会在 `data/` 目录下得到类似文件：

```text
data/NYC_min20.npy
data/NYC_min20_POI.npy
data/NYC_min20_data.pkl
```

也可以将 `--min_loc_visits` 改为 `30`、`50` 等阈值，用于比较不同过滤强度对推荐效果的影响。

## 运行训练

基础训练可参考：

```bash
python train.py --dname NYC
```

邻居协同版本可参考：

```bash
python train_neighbor.py --dname NYC_min20
```

具体参数请结合各训练脚本中的命令行参数设置。

## Slurm 服务器运行

仓库提供了 Slurm 提交脚本：

```text
run_stan_collab.sh
```

提交任务前需要设置项目路径：

```bash
PROJECT_DIR=/path/to/your/project sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

参数含义：

- `neighbor`：使用邻居协同版本。
- `NYC_min20`：数据集名称。
- `0`：使用全部用户。
- `30`：训练 30 个 epoch。

如果服务器的分区、GPU、内存或 Conda 环境不同，请先修改 `run_stan_collab.sh` 中对应配置，或通过环境变量覆盖。

## 输出文件

训练过程中通常会生成：

- `.log`：训练配置、损失、验证集和测试集指标。
- `.err`：Slurm 错误输出和进度条信息。
- `.pth` / `.pt`：模型 checkpoint。
- `data/*_data.pkl`：预处理缓存。

这些文件通常较大，已经在 `.gitignore` 中排除，不建议提交到仓库。

## 备注

本项目用于课程实验与结果复现。更详细的实验流程、Slurm 参数说明和日志解释可查看 [README_EN.md](README_EN.md)。
