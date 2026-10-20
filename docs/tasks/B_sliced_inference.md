# 任务 B：切片推理（SAHI）提升远处小车与摩托车检测

## 为什么要做

新加坡摄像头是 1920×1080 的远景，远处车辆只有 6–10 像素宽。模型推理时整张图被缩到 1280（训练时 736），
小目标进一步缩小，摩托车大量漏检（抽看高峰画面就能看到一串没框到的骑手）。
**切片推理**不改模型、不重新训练：把大图切成重叠的小块分别检测，再把结果合并回原图，相当于把小目标「放大」看。
这是小目标检测的常用方法，代表实现是 SAHI（Akyon et al., ICIP 2022）。

你的问题是：**在我们的联合微调模型上加切片推理，新加坡画面上的检测能提升多少？代价（速度）是多少？**

## 你会拿到

- 代码：仓库 `develop` 分支（按 `src/detection/README.md` 装环境，`pip install sahi`）；
- 权重包 `detection_results_and_weights.zip`：`m736_joint_v4/weights/best.pt`（主角）与 `m736_mio_v4/weights/best.pt`；
- 评估数据包 `detection_eval_data.zip`：解压到仓库根目录，得到
  - `datasets/miotcd_yolo_v4/`：MIO-TCD 的 dev / test / test_moto（人工真值，冻结）；
  - `datasets/lta_yolo/`：新加坡 320 张图与大模型伪标签；
- 任务 A 完成后：新加坡人工测试集 `datasets/lta_human_v1/`（负责人生成后发你）。

## 步骤

1. **跑通基线**：用 `m736_joint_v4` 整图推理（1280），在 `lta_yolo` 的 val 上算一遍，确认能复现
   README 里的数字（与教师一致性 mAP@0.5 约 0.640）；
2. **实现切片推理**：新建 `src/detection/sliced_inference.py`，用 SAHI 的
   `AutoDetectionModel.from_pretrained(model_type="ultralytics", ...)` + `get_sliced_prediction`。建议的起点：
   切片 640×640、重叠 0.2、同时保留整图预测（`perform_standard_pred=True`）、合并方式 GREEDYNMM。
   输出整理成评估器要的格式（见下方「评估接口」）；
3. **调参**（只用调参集，见「评估纪律」）：最多试 4–6 组，变量建议：切片大小（512 / 640 / 960）、重叠（0.2 / 0.3）、
   是否保留整图预测。记录每组的 mAP@0.5、各类 AP（重点看摩托车）、召回率、每张图耗时；
4. **定下最终配置后**，在最终评估集上各评估一次：新加坡人工测试集 batch A；可选再在 MIO-TCD test 上评估一次作为补充
   （MIO-TCD 是 720×480 的近景，切片收益预计很小，可以说明方法在什么场景有用）；
5. 挑 3–4 张「整图 vs 切片」的对比图（同一帧左右并排，标出新增检出的车），重点选长堤高峰和夜间画面；
6. 写结果。

## 评估纪律（必须遵守）

- **不要用伪标签调参**：切片的目的就是找回大模型漏掉的车，拿大模型的伪标签当答案，找回的车会被算成误检，调参方向就反了。
- **调参集 = 新加坡人工测试集的 batch B（order 41–80）**；**最终评估集 = batch A（order 1–40）**，配置定下来之前不看 batch A 的结果。
  如果任务 A 只完成了 batch A，就不调参，直接用上面建议的起点配置，在 batch A 上评估一次。
  人工测试集的 `split.csv` 有 `batch` 列（A / B），按它筛图片列表即可。
- MIO-TCD test 已经用于正式结论，只允许在最终配置定下后**评估一次**作为补充，不能根据它改配置。
- 所有尝试过的配置都记下来，报告里如实写试了几组。

## 评估接口

不要改 `evaluate.py` 的计算逻辑，直接复用它，保证和正式结果同一把尺子：

```python
import numpy as np
from src.detection import evaluate as ev
from src.detection.run_detect import VEHICLE_CLASSES   # ("car", "motorcycle", "bus", "truck")

truth = ev.load_ground_truth(data_dir, split, images)          # images: 图片文件名列表
predictions = {name: {"cls":  np.array([...], dtype=int),       # 0..3，对应 VEHICLE_CLASSES
                      "conf": np.array([...]),
                      "xyxy": np.array([[x1, y1, x2, y2], ...])}  # 原图像素坐标
               for name in images}
sizes = {name: (width, height) for name in images}
metrics = ev.detection_metrics(images, truth, predictions, sizes)   # mAP50、各类 AP50、P/R、计数误差
```

预测时置信度阈值要低（0.001），否则 PR 曲线不完整、AP 偏低。需要置信区间可以复用 `src/detection/bootstrap.py`
里的 `per_image_matches` 与 `ap50`（按场景组或按图片重采样）。

## 交付物

1. 代码 `src/detection/sliced_inference.py`，在单独的分支上提交（例如 `feature/sahi`），提 PR 合到 `develop`，不要直接推 `develop`；
2. 结果表（CSV 或 JSON）：每组配置在调参集上的指标与耗时，最终配置在最终评估集上的指标，与整图推理并排；
3. 3–4 张对比图；
4. 一段英文结果描述（5–8 句），能直接放进报告：做了什么、提升多少（尤其摩托车与召回率）、速度代价、局限；
5. 一段中文小结发群里。

## 时间

1–2 天。调参只跑推理不训练，Mac 或任何有 GPU 的电脑都可以。

## 提示

- 切片会把一辆车切成两半，在相邻块各检出半辆，合并时去重很关键，注意观察计数是否虚高（计数误差和 `count_bias` 会反映）；
- 切片会让推理变慢好几倍，速度要和整图推理对比报告；
- 遇到问题先看 `src/detection/README.md`，再找负责人。
