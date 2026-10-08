# ISY5002 交通预测与监测

两个互相独立的模块，和一个把它们产物呈现出来的控制台：

| 模块 | 做什么 | 产物 |
|---|---|---|
| `src/forecast/` | 在公开数据集 PEMSD8 上训练、评估并对比时空预测模型，含一个自研模型 | `results/forecast/` |
| `src/detection/` | 对 LTA 相机图跑目标检测，统计车辆 | `results/detection/` |
| `src/demo/` | 控制台，按「模块间只通过产物通信」的约定读上面两者的产物 | 无 |

两个模块**不互相 import**，只通过 `results/` 下的文件通信。

---

## 环境

`requirements.txt` **不含 torch / torchvision**，它们要按 GPU 架构选 CUDA 版本单独装。

```bash
conda create -n isy5002-project python=3.13 -y
conda activate isy5002-project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`third_party/LibCity` 是 submodule（仓库里只有引用，没有源码）：

```bash
git submodule update --init third_party/LibCity
```

---

## 从零跑通

```bash
git clone <url> && cd ISY5002-TrafficPrediction

# 1. 第三方源码
git submodule update --init third_party/LibCity

# 2. 环境，见上一节

# 3. 数据集与权重（LTA 相机图 / PEMSD8 / METR_LA / YOLO 权重）
python scripts/fetch_data.py

# 4. 自检一：预测侧，单模型单目标一个 epoch，约 30 秒
python -m src.forecast.run_benchmark --targets traffic_flow --models FNN --max-epoch 1

# 5. 自检二：检测侧，先试 8 张
python -m src.detection.run_detect --limit 8
```

第 3 步会建立 `third_party/LibCity/raw_data` 指向 `datasets/libcity` 的目录联接
（Windows 用 `mklink /J`，不需要管理员权限）。**联接本身不在任何仓库里，每个新 clone 都要重跑。**

跑通之后分别看两个模块的下一步：`src/detection/README.md`、`docs/FORECAST_RESULTS.md`。

---

## 两个模块各自的入口

### 预测 `src/forecast/`

```bash
# 三个基线对照
python -m src.forecast.run_benchmark --dataset PEMSD8 --models FNN,STGCN,MTGNN

# 自研模型
python -m src.forecast.run_benchmark --dataset PEMSD8 --targets traffic_flow --models MultiHeadSTGCN

# 多种子
python -m src.forecast.run_benchmark --models STGCN --seeds 0,1,2
```

已完成的实验、数字、口径与已知陷阱见 **`docs/FORECAST_RESULTS.md`**。
每次运行的原始记录在 `results/forecast/PEMSD8/<目标>/<模型>/<tag>/run.json`（已入版本控制）。

⚠️ **实测并发上限 2 个训练进程**（9 通道输入下每进程约 5.1 GB，三个并行会因内存耗尽而崩）。

### 检测 `src/detection/`

```bash
python -m src.detection.run_detect --limit 8     # 先试跑
python -m src.detection.run_detect               # 全量
python -m src.detection.run_detect --weights <你微调的模型>/weights/last.pt
```

**接入自己微调的模型看 `src/detection/README.md`**。除了 `--weights`，通常还要写
`src/detection/classes.yaml` 把模型的类名映射到 `car/bus/motorcycle/truck`。
漏了不会报错，只会得到一份空的 CSV。

### 控制台 `src/demo/`

```bash
python -m src.demo.web.server          # 默认 8600 端口
```

---

## 文档在哪

| 文件 | 内容 |
|---|---|
| `docs/FORECAST_RESULTS.md` | 预测部分的结果台账：定型模型、基线对照（含论文值）、消融阶梯、负结果、作废项、口径与噪声底线 |
| `src/detection/README.md` | 检测模块的接入契约：换权重、写类名映射、换配置后清空产物 |
| `docs/LTA/` | LTA 交通图像 API 的接口文档与样本响应（相机坐标的来源） |
| `submission/Project_Proposal_FT6.docx` | 立项时的提案 |

---

## 已知缺口

这些是当前公认的缺口，接手时先看这一节。

**1. 预测部分的模型权重已丢失。**
`results/forecast/**` 下不再有任何训练好的权重文件（清理时误删，且 `src/forecast/` 本身
从不保存模型，权重原本只存在于 LibCity 的 `model_cache/*.tar` 里）。
要接着用定型模型必须重跑：单次约 12 分钟，三个目标 × 5 种子约 3 小时。

**2. demo 的预测侧需要 `.npz`，而它们不在版本控制里。**
`src/demo/data.py` 靠 `evaluate_cache/*_predictions.npz`（约 4.7 GB）画时间序列与图结构。
检测侧没有这个问题（`results/detection/*.csv` 可以自己跑 `run_detect.py` 生成）。
所以新 clone 里 demo 能显示运行元信息，但时间序列页会缺数据。

**3. `.env.example` 已无人使用。**
里面的 `API_KEY` 原本供感知模块取 LTA API key，那个链路已重做，没有代码读它。

**4. 预测部分的多种子不全。**
`traffic_occupancy` 与 `traffic_speed` 各差一个种子（见 `docs/FORECAST_RESULTS.md` 第 1b 节）。
