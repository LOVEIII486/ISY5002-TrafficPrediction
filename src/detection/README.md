# detection —— LTA 相机图的车辆检测

跑 YOLO 识别 LTA 相机图里的车辆，产出 `results/detection/` 下三份 CSV
（`detections.csv` / `frames.csv` / `cameras.csv`）。demo 按「模块之间只通过产物通信」
的约定读它们，**不 import 本模块**——所以 CSV 的列名和取值是这批代码的对外契约，别改。

## 跑通默认链路

```bash
python scripts/fetch_data.py lta                # LTA 相机图 -> datasets/lta/
python scripts/fetch_data.py weights            # COCO 预训练权重 -> yolo/
python -m src.detection.run_detect --limit 8    # 先试 8 张
python -m src.detection.run_detect              # 全量
```

默认权重是 COCO 预训练的 `yolo26m`，开箱即用，不需要配任何东西。

## 换用自己微调的模型

两步。**第二步最容易被漏掉，漏了不会报错，只会得到一份空的 CSV。**

### 1. 指定权重

```bash
python -m src.detection.run_detect --weights <你的实验>/weights/last.pt
```

### 2. 写类名映射 `classes.yaml`

管线只统计 `car / motorcycle / bus / truck` 四类。COCO 预训练权重的类名本来就叫这些，
按名字大小写不敏感就能对上，所以不用配；**自建数据集上微调过的权重几乎一定要配**。

先看你的权重有哪些类：

```bash
python -c "from ultralytics import YOLO; print(YOLO('你的.pt').names)"
```

再把映射写进同目录的 `classes.yaml`——左边是权重里的类名，右边必须是上面四类之一：

```yaml
Truck: truck
Bus: bus
Car: car
Van: car        # 厢式车归入 car，与 COCO 的做法一致
```

不属于车辆的类不用写，会被跳过。**如果一个都映射不上，程序直接报错并列出全部类名**，
不会静默产出空结果。这个检查是特意加的——在自建数据集上微调过的权重，类名几乎不可能
和 COCO 一模一样，而类名对不上时旧版代码会安静地丢掉每一个框。

### 3. 验证

```bash
python -m src.detection.run_detect --limit 8
```

输出第二行会打印识别到的车辆类：

```
[2/3] 权重 last.pt · 车辆类 ['bus', 'car', 'truck'] · device=0 · imgsz=1280 · conf=0.25
```

跑完确认检出目标数不是 0。是 0 就说明映射还没对。

## 换配置后必须清空产物目录

三份 CSV 是一套配置的产物。换权重或换推理参数后 `run_detect.py` 会**拒绝续跑**并提示清空：

```bash
rm results/detection/detections.csv results/detection/frames.csv results/detection/run.json
```

这是有意的：两个模型的结果混进同一份 CSV 后，下游分不出哪一行来自哪个模型。
`run.json` 里记了权重路径、类映射与推理参数，是这套产物的自证。

## 不在这里的东西

数据集的选择与训练代码不在本仓库。本模块只定义**接入契约**：

```
你的权重 + classes.yaml  ->  detections.csv / frames.csv / cameras.csv  ->  demo
```

只要产出的 CSV 遵循列名约定，用哪个数据集、训出什么模型都能接进来。
