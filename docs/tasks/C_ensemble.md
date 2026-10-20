# 任务 C：多模型集成（WBF 加权框融合）

## 为什么要做

课程要求项目覆盖「监督学习、深度学习、**混合 / 集成方法**、智能感知」四项中的至少三项，目前我们缺「集成方法」这一项。
另外我们手上正好有几个各有所长的模型：
- `m736_mio_v4`：只用 MIO-TCD 微调，MIO-TCD 上很强，新加坡画面上很弱；
- `m736_joint_v4`：MIO-TCD + 新加坡联合微调，两边都不错（Demo 用的就是它）；
- `yolo26x`：官方最大的 COCO 预训练模型，没微调，但对新加坡远景的小车更敏感。

你的问题是：**把它们的检测框融合起来，能不能超过最好的单个模型？哪种组合最好？**

## 方法

**WBF（Weighted Boxes Fusion，Solovyev et al., 2021）**：多个模型对同一辆车的框按置信度加权平均成一个框，
而不是像 NMS 那样只留一个。库：`pip install ensemble-boxes`，函数 `ensemble_boxes.weighted_boxes_fusion`。
它要求坐标归一化到 0–1，参数主要是各模型权重 `weights`、合并阈值 `iou_thr`、最低分 `skip_box_thr`。

## 你会拿到

- 代码：仓库 `develop` 分支（按 `src/detection/README.md` 装环境）；
- 权重包 `detection_results_and_weights.zip`（两个微调模型）；`yolo26x.pt` 用 `ultralytics` 自动下载；
- 评估数据包 `detection_eval_data.zip`：MIO-TCD 的 dev / test / test_moto（人工真值）与新加坡 320 张图；
- 任务 A 完成后：新加坡人工测试集 `datasets/lta_human_v1/`。

## 步骤

1. **各模型单独预测并缓存**：每个模型在每个评估集上推理一次（置信度阈值 0.001），把框存成文件，之后调融合参数不用重复推理。
   推理尺寸：MIO-TCD 用 736，新加坡用 1280；`yolo26x` 是 80 类 COCO 模型，用 `run_detect.load_class_map` 映射到四类；
2. **实现融合**：新建 `src/detection/ensemble.py`，读缓存、做 WBF、输出评估器要的格式（见下方「评估接口」）；
3. **调参**（只用调参集）：组合至少比较
   - `joint + mio`、`joint + yolo26x`、三者全部；
   - 权重相等 vs 偏向某个模型；`iou_thr` 0.5 / 0.6；
   每组记录 mAP@0.5、各类 AP、召回率与计数误差；
4. **定下最终组合与参数后**，在最终评估集上各评估一次，与最好的单模型并排；
5. 挑 2–3 张图展示「单模型 vs 集成」的差异；
6. 写结果。

## 评估纪律（必须遵守）

- **MIO-TCD 上：调参只用 dev**；最终配置定下后，在 test 上**只评估一次**，不能再根据它改。
- **新加坡上：调参只用人工测试集的 batch B（order 41–80）**，最终在 **batch A（order 1–40）** 上评估一次。
  不要用伪标签调参：融合 `yolo26x` 后结果会更像伪标签（伪标签就是 `yolo26x` 画的），分数虚高。
  如果任务 A 只完成了 batch A：新加坡上不调参，直接用在 MIO-TCD dev 上选出的配置，在 batch A 上评估一次。
  人工测试集的 `split.csv` 有 `batch` 列（A / B），按它筛图片列表即可。
- 所有尝试过的组合和参数都记下来，报告里如实写试了几组。
- 和单模型比较时，单模型要用**完全相同**的评估集与评估代码。

## 评估接口

不要改 `evaluate.py` 的计算逻辑，直接复用：

```python
import numpy as np
from src.detection import evaluate as ev
from src.detection.run_detect import VEHICLE_CLASSES   # ("car", "motorcycle", "bus", "truck")

truth = ev.load_ground_truth(data_dir, split, images)
predictions = {name: {"cls":  np.array([...], dtype=int),       # 0..3
                      "conf": np.array([...]),
                      "xyxy": np.array([[x1, y1, x2, y2], ...])}  # 原图像素坐标（WBF 输出的 0–1 坐标要乘回宽高）
               for name in images}
sizes = {name: (width, height) for name in images}
metrics = ev.detection_metrics(images, truth, predictions, sizes)
```

`ev.predict(weights, data_dir, split, eval_dir, split_frame, imgsz, examples=0)` 能直接拿到单模型的预测字典，
可以拿来做缓存。置信区间可以复用 `src/detection/bootstrap.py` 的 `per_image_matches` 与 `ap50`；
MIO-TCD 上按 `split.csv` 的 `scene_group` 列做组重采样，和正式结果口径一致。

## 交付物

1. 代码 `src/detection/ensemble.py`，在单独的分支上提交（例如 `feature/ensemble`），提 PR 合到 `develop`；
2. 结果表：各组合在调参集上的指标，最终配置在最终评估集上的指标（含 95% 区间），与最好的单模型并排；
3. 2–3 张对比图；
4. 一段英文结果描述（5–8 句），能直接放进报告，写明这满足了课程的「集成方法」要求；
5. 一段中文小结发群里。

## 时间

1–2 天，只做推理和融合，不需要训练。三个模型在所有评估集上各推理一次约半小时，之后调参都在缓存上跑，很快。

## 提示

- 集成不一定更好：如果某个模型在某个场景很差（比如 `m736_mio_v4` 在新加坡上），硬加进来可能拖后腿，这本身就是有价值的结论；
- 集成的代价是速度变成几个模型之和，要报告；
- 遇到问题先看 `src/detection/README.md`，再找负责人。
