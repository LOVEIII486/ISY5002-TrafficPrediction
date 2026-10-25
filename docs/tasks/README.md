# 识别部分组员任务

| 任务 | 内容 | 说明文件 | 状态 |
|---|---|---|---|
| A | 80 张新加坡图人工修正标注 | [A_lta_annotation.md](A_lta_annotation.md) | 已完成：Fu Junchun 标注 63 张（17 张过难未标），冻结为 `datasets/lta_human_v1` |

## 材料

- `lta_annotation_package.zip`：任务 A 全部材料，不需要项目代码；
- 标注规范：[../LTA_ANNOTATION_GUIDE.md](../LTA_ANNOTATION_GUIDE.md)；
- 收回与冻结：`python -m src.detection.annotate collect <导出 JSON> --annotator "<姓名>"`。

图片含车牌，只在组内使用。
