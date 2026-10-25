# detection —— LTA 相机图的车辆检测

*中文 · [English](README.en.md)*

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

## 微调流程

四步，全部在本模块里，产物不入版本控制（`datasets/`、`results/` 都被忽略），按同一种子可重建。

```bash
python -m src.detection.sample      # 1. 分层抽样并只下载抽中的图：week1 训练 240 张、week2 验证 80 张
python -m src.detection.prelabel    # 2. yolo26x 以原图分辨率预标注 -> datasets/lta_yolo/（YOLO 格式）
python -m src.detection.train --imgsz 960 --name m960_pseudo        # 3. 微调 yolo26m
python -m src.detection.evaluate --weights yolo/yolo26m.pt \
    results/detection/finetune/m960_pseudo/weights/best.pt --imgsz 960   # 4. 同一把尺子比较
```

- **划分按周切**：week1 训练、week2 验证，时间不重叠，经核对两周之间也没有字节相同的图。这适合评估「已有摄像头的
  未来时段」；两周是同一批 8 个摄像头，不能据此声称泛化到新摄像头。
- **白天 / 夜间按新加坡时间分层**。数据集 README 说文件名中段的 HHMM 是新加坡时间，**实际是 UTC**，`sample.py` 已按 UTC+8 处理。
- **第 2 步产出的是伪标签**（模型猜的）。它既是人工修正的起点（`datasets/lta_yolo/` 可直接导入 CVAT 的
  Ultralytics YOLO Detection 1.0 格式，或 Label Studio / Roboflow），也可以在人工标注完成前先跑通微调。
  **用伪标签评估时，指标只衡量与教师预测的一致程度**：教师漏标的真车会被算成学生的误检。它既不是新加坡场景的
  准确率，也不能证明场景适应的收益，`evaluate.py` 会打印这句提醒。已知问题：教师模型会漏掉大量远处的摩托车。
- `prelabel.py` 发现 `labels/` 下已有标签（可能含人工修正）时拒绝运行，加 `--overwrite` 才会清空重建。
- **评估为什么不用 `model.val()`**：COCO 权重 80 类、编号与本标注集对不上，val() 不会自动重映射真值编号；
  `evaluate.py` 按类名统一映射后自己算 mAP（COCO 101 点插值、贪心匹配，无 crowd / ignore 处理，
  与 ultralytics 或官方 COCO 评估器的数值不必逐位相同）、阈值 0.25 下的 P/R 与计数误差，并拆白天 / 夜间。
- **显存**：yolo26m 在 1280 下 batch 4 约需 19 GB；Apple M4 Pro（24 GB 统一内存）上会换页、每步 45 秒，
  降到 960 后约 11 GB、每步 1 秒。5070（12 GB）上 1280 建议 batch 2–4，或用 960。

### 在公开数据集 MIO-TCD 上评估与微调（人工标注真值）

LTA 图没有人工标注，上面的伪标签实验给不出准确率。有真值的口径用 **MIO-TCD 定位子集**
（北美 8,000 个交通监控摄像头，11 万张带框训练图，720×480 与 342×228 两种尺寸，CC BY-NC-SA 4.0，引用 Luo 等 IEEE TIP 2018）：

```bash
curl -L -o datasets/miotcd/MIO-TCD-Localization.tar https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar   # 3.7 GB，官方直链
python -m src.detection.miotcd        # 4 类映射、场景分组、切 train/dev/test、复核 -> datasets/miotcd_yolo_v4/、joint_yolo_v4/（已存在则拒绝）
python -m src.detection.audit_split   # 人工抽查评估集与训练集的同机位重叠（生成抽查图与待填写的记录表）

# 正式对照实验：统一从 COCO 权重起步，同一尺寸 / 批次 / 增强，固定 30 轮（默认关闭早停）
python -m src.detection.train --data datasets/miotcd_yolo_v4/data.yaml --imgsz 736 --batch 8 --epochs 30 --name m736_mio_v4
python -m src.detection.train --data datasets/joint_yolo_v4/data.yaml  --imgsz 736 --batch 8 --epochs 30 --name m736_joint_v4

# 开发阶段只看 dev；所有决定做完后，三个模型在 test 与 test_moto 上各评估一次，之后不再调参
python -m src.detection.evaluate --data datasets/miotcd_yolo_v4 --split dev  --imgsz 736 --weights yolo/yolo26m.pt \
    results/detection/finetune/m736_mio_v4/weights/best.pt results/detection/finetune/m736_joint_v4/weights/best.pt
python -m src.detection.evaluate --data datasets/miotcd_yolo_v4 --split test --imgsz 736 --weights ...（同上三个）
```

`train.py` 训练前按 data.yaml **实际引用**的清单核对清单摘要与标签内容摘要（联合训练还核对 dev 清单与 LTA 伪标签）；
`evaluate.py` 评估前核对该划分的清单与标签。版本化目录缺少冻结记录直接报错。实现见 `frozen.py`。
冻结范围只覆盖清单与标签，**不含原图内容**：原始 JPG 被替换或符号链接改指向时检查不出来。

**场景分组（v4）**：MIO-TCD 不提供摄像头编号，同一机位贡献大量连续帧（11 万张里 9,627 组字节完全相同）。
四条规则任一成立即并入同一组，整组只进一个划分：文件哈希相同；256 位 dHash 距离 ≤ 16；背景缩略图相关 ≥ 0.95；
**用标注框抠掉车辆后**的背景缩略图相关 ≥ 0.93（0.93 是工程参数，不是「93% 同机位置信度」）。超过 1,500 张的组、
以及摩托车超过 30 张的组一律放进训练池。抽看最大的两个组，各自都是同一机位（同一机位的白天与夜间也被正确并组），
未见误并。构建后的规则复核：不同划分之间无相同文件；train~test 最小 dHash 距离 35，
最大背景相关 0.94、最大抠车背景相关 0.9271。

**残余的同机位重叠（必须在报告中披露）**：分组依据图像内容，不是真实摄像头编号；前景大车、强阴影、光照变化都会让
同机位的帧低于阈值而分进不同组。`audit_split` 以种子 42 从 test 抽 40 张，每张只配训练集中**最相似的一张**做比对
（记录在 `results/detection/finetune/audit/v4_test/audit.csv`，含图片编号、时间戳与判定）。**人工复核**：
9 对同机位，2 对可能同机位，4 对无法判断（夜间测试图配白天训练图），25 对为不同机位。同机位的 9 对里，
两对相隔不足 10 分钟（2 分 18 秒、8 分 27 秒），另一对约 10 分钟（10 分 03 秒）。

这只是本次抽查的观察值，不是真实机位重叠率：每张图只检索了最相似的一张训练图，可能漏掉其他同机位画面；
判定也可能有误。9/40（22.5%）的 Wilson 95% 区间 12%–38% 只反映抽样误差，不涵盖检索遗漏和判定错误。
结论：经过重复文件与近重复过滤后，**仍存在同机位画面与近邻帧**；测试成绩反映的是「已见与未见机位混合」的表现，
**不能声称是对全新摄像头的泛化**。

| 划分 | 图片 | 摩托车框 | 用途 |
|---|---|---|---|
| train | 8000 | 833（15 个组，最大组占 15%） | 训练；摩托车图优先纳入（富集） |
| dev | 1000 | 93（20 个组，最大组占 12%） | 选 checkpoint、比较方案；**分层开发集**：dev 池的摩托车图每组最多 10 张全部纳入、其余随机补足 |
| test | 2000 | 33（17 个组，最大组占 18%） | **冻结**，所有决定做完后只评估一次；测试池内均匀抽样 |
| test_moto | 161 | 173（40 个组，最大组占 9%） | 补充诊断：测试池里剩余的摩托车图（每组最多 10 张），与 test 不重叠但共享场景组 |

各划分每一类涉及的组数与最大组占比见 `dataset.json` 的 `class_diversity`。

**口径与限制（写报告时照此表述）**

- 结果只适用于**筛选后的四类子集**：含 `motorized_vehicle`（官方定义：太小、无法细分车型的机动车）的图整张剔除，
  约 18%，这会系统性去掉含困难小目标的场景。**不是官方 MIO-TCD 基准成绩**，mAP 也不能写成「准确率 xx%」。
- 类别合并（work_van→car，三类卡车→truck）是本项目的定义，不保证与 COCO 的车型边界一致。
- dev / test **不是自然分布**：超大组与摩托车大户组进训练池，其余含摩托车的组按目标 30/20/50（测试/开发/训练）分池、
  其余组按 15/10/75，再在池内抽样；实际配比见 `dataset.json` 的 `pool_rare_share_actual`。
  可以表述为「按稀有类分层、按组隔离后池内抽样」。
- 主结果用 test 完整报告四类 AP，注明摩托车框少；test_moto 只作补充诊断，**不能拿它的 AP 替换 test 的摩托车项重算 mAP**。
  两者共享场景组，不是独立证据；框数也不等于独立样本数。
- `best.pt` 由 ultralytics 在 dev 上的 mAP@0.5:0.95 挑选（不是自写评估器的 mAP@0.5）。两组固定跑满 30 轮，
  但联合训练每轮多约 6% 的样本（8,480 vs 8,000），比较的是「实用方案」，不能完全分离额外 LTA 数据与额外计算量的作用。
- 没有重复实验与置信区间时，零点零几的 mAP 差距不要说成稳定提升；需要区间时按场景组重采样。

**联合训练**：`joint_yolo_v4/train.txt` = MIO-TCD 训练图 + LTA 训练图 × `--joint-lta-repeat`（默认 2），
开发集仍是 MIO-TCD 的 dev 真值。默认值下 LTA 约占训练图片的 6%，但约占标注框的 21%（LTA 画面车多），
不要只按图片占比判断它的影响；加大重复次数不会增加独立信息，反而更多暴露于教师的误检和漏检。

**探索实验与废弃版本（不作为正式结论）**：v1 按编号连续块切分，2,000 张验证图里有 109 张在训练集中有字节相同的副本；
v2 去掉了重复文件，但同机位相隔几秒的帧仍跨入训练与测试；v3 的抠车特征把框坐标按 720×480 缩放，对约 20% 的
342×228 图挖错了位置。各探索实验的输入尺寸与轮数也不统一（LTA 伪标签微调 960 / 30 轮，MIO-TCD 与联合试跑 736 / 15 轮）。
结果保留在 `eval/miotcd_yolo_v1/` 与 `eval/lta_yolo/`，只能作为方向性观察：只用 MIO-TCD 微调后，与 LTA 教师预测的
一致性明显下降；加入 LTA 伪标签的联合微调没有这种下降。正式结论以 v4 为准。

### 正式实验结果（v4）

配置：三个模型都按上面的命令训练 / 评估；两个微调模型从 COCO 权重起步、736 输入、批 8、固定 30 轮，在同一台 Apple M4 Pro
上训练（MIO-TCD 10.1 小时，联合 9.8 小时）；`best.pt` 都是第 30 轮。每个配置只训练一次。

**冻结测试集**（MIO-TCD v4 test，2,000 张、31 个场景组，人工标注真值，只评估一次）：

| 模型 | mAP@0.5（95% 区间） | mAP@0.5:0.95 | 小汽车 | 摩托车* | 公交车 | 货车 | 计数误差 |
|---|---|---|---|---|---|---|---|
| 官方预训练 yolo26m | 0.430（0.337–0.516） | 0.320 | 0.656 | 0.136 | 0.674 | 0.253 | 1.21 |
| 只用 MIO-TCD 微调 | 0.709（0.626–0.817） | 0.561 | 0.883 | 0.570 | 0.920 | 0.463 | 0.65 |
| MIO-TCD + LTA 联合微调 | 0.720（0.645–0.825） | 0.563 | 0.873 | 0.646 | 0.908 | 0.453 | 0.70 |

\* 测试集只有 33 个摩托车框，该列波动大。区间来自 `bootstrap.py`：按场景组重采样 2,000 次的百分位区间，
只反映换一批场景组的抽样波动，不含训练随机性。配对差值：

| 差值 | mAP@0.5 差 | 95% 区间 |
|---|---|---|
| 只用 MIO-TCD − 官方预训练 | +0.279 | 0.194–0.407 |
| 联合微调 − 官方预训练 | +0.291 | 0.213–0.411 |
| 联合微调 − 只用 MIO-TCD | +0.011 | −0.024–0.048（含 0） |

**其他划分**：

| 模型 | dev mAP@0.5 | test_moto 摩托车 AP@0.5 | LTA val 与教师一致性 mAP@0.5（960） |
|---|---|---|---|
| 官方预训练 | 0.370 | 0.305 | 0.555 |
| 只用 MIO-TCD 微调 | 0.750 | 0.830 | 0.174（平均每图少检 9.2 辆） |
| 联合微调 | 0.703 | 0.838 | 0.634 |

**结论**

1. 两个微调模型在测试集上都比官方预训练高约 0.28–0.29 mAP@0.5，区间不含 0，计数误差约减半，摩托车提升最大。
2. 两个微调模型在 MIO-TCD 上没有可区分的差距：dev 上只用 MIO-TCD 高 0.047，test 上联合高 0.011，差值区间含 0。
3. 只用 MIO-TCD 微调的模型在新加坡画面上基本失效；联合微调保住了新加坡场景，与教师预测的一致性还高于官方预训练。
4. 因此 Demo 与检测管线使用 **`m736_joint_v4`**。货车（三类卡车合并而成）是最弱的一类，约 0.45。
5. 试跑阶段（v1）「联合微调两边都不吃亏」的说法被正式实验修正：联合微调在 dev 上确有约 0.05 的代价，只是测试集上
   看不出显著差别。

**部署到 LTA 全量（Demo 用）**：`m736_joint_v4` 以 1280 输入（LTA val 上与教师一致性 1280 为 0.640、960 为 0.634、
736 为 0.599）跑完全部 15,316 张，检出 279,476 辆（小汽车 187,928、摩托车 41,109、货车 35,328、公交车 15,111），
Apple M4 Pro 上 15 分钟。平均每帧 18.2 辆（白天 27.2、夜间 8.9）；1,478 帧零检出，多为夜间。新柔长堤上能检出摩托车了，
但抽看高峰帧仍有不少摩托车漏检，摩托车召回不完整。产物是 `results/detection/` 下的三份 CSV，`run.json` 记录了权重路径。

复现：`python -m src.detection.bootstrap --data datasets/miotcd_yolo_v4 --split test --imgsz 736 --weights ...（三个）`；
报告用图：`python -m src.detection.report_figures`（产物在 `results/detection/finetune/figures/v4/`）。

**`--base` 与第二阶段微调**：`--base` 指向已有微调权重时，ultralytics 只加载权重、重新开始一个训练阶段
（新的优化器状态、学习率计划与 mosaic 阶段），不是断点续训，报告里应写成「以某权重初始化的第二阶段微调」。
正式对照实验统一从 COCO 权重开始。

微调后的权重类名就是这四类，接入检测管线不需要写 `classes.yaml`：

```bash
python -m src.detection.run_detect --weights results/detection/finetune/<name>/weights/best.pt
```

## 不在这里的东西

数据集的选择不在本仓库。本模块只定义**接入契约**：

```
你的权重 + classes.yaml  ->  detections.csv / frames.csv / cameras.csv  ->  demo
```

只要产出的 CSV 遵循列名约定，用哪个数据集、训出什么模型都能接进来。
