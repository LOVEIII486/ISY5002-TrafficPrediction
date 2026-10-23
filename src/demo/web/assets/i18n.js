"use strict";

/* 工作台的界面文案。语言存 localStorage，默认英文；API 侧的动态文案（数据来源、
   路段名、错误信息）由 server.py 按 ?lang= 生成，不在这里。
   产品页的注意力条目由服务端给 code + 数值，前端在 t() 里拼句子。 */
window.ConsoleI18n = (() => {
  const LANG_KEY = "traffic-console.lang";
  const DEFAULT_LANG = "en";

  const MESSAGES = {
    en: {
      "app.title": "Traffic Monitoring and Forecasting Workbench",
      "app.skipLink": "Skip to main content",
      "app.navLabel": "Main navigation",
      "app.navSection": "WORKSPACE",
      "app.brand.name": "Traffic Workbench",
      "app.brand.tagline": "MONITORING & FORECAST",
      "app.footer": "Traffic Monitoring and Forecasting Workbench",
      "app.poweredBy": "Powered by Tabler",

      "nav.overview": "Overview",
      "nav.monitoring": "Camera monitoring",
      "nav.forecast": "Congestion forecast",

      "header.menu": "Toggle navigation",
      "header.refresh": "Refresh",
      "header.breadcrumb.ops": "TRAFFIC MONITORING",
      "header.breadcrumb.console": "WORKBENCH",
      "header.replayBadge": "HISTORICAL REPLAY",
      "header.replayNote": "PEMSD8 2016-08-19 → 08-31 · LTA 2026-09-13 → 09-29",
      "header.unavailable": "API unavailable",
      "header.verify": "{count} item(s) to verify",

      "views.overview.title": "System overview",
      "views.overview.subtitle": "What needs attention right now.",
      "views.overview.pretitle": "SITUATION",
      "views.monitoring.title": "Camera monitoring",
      "views.monitoring.subtitle": "What the cameras see, and whether their counts can be trusted.",
      "views.monitoring.pretitle": "CAMERA FEEDS",
      "views.forecast.title": "Congestion forecast",
      "views.forecast.subtitle": "Where traffic will congest, when, and how bad.",
      "views.forecast.pretitle": "FORECAST",

      "domain.camera": "CAMERA · LTA SINGAPORE",
      "domain.forecast": "FORECAST · PEMSD8 CALIFORNIA",
      "domain.system": "SYSTEM",

      "severity.severe": "Severe",
      "severity.congested": "Congested",
      "severity.watch": "Watch",
      "severity.normal": "Normal",

      "status.ongoing": "Ongoing",
      "status.new": "New",
      "status.recovering": "Recovering",
      "status.watch": "Watch",
      "status.normal": "Normal",

      "health.ok": "OK",
      "health.watch": "Watch",
      "health.warn": "Warning",

      "thresholds.summary": "Thresholds",
      "thresholds.freeflow":
        "Anchor: the test-period free-flow median is 64.9 mph. The three lines are set at 55 (−15%), 45 (−31%) and 35 (−46%).",
      "thresholds.share":
        "Share of all observations below each line: 6.1% / 3.1% / 1.6%.",
      "thresholds.basis":
        "These are operational definitions chosen for this workbench — not a published standard and not a model output. HCM level of service is density-based, which this dataset cannot support.",
      "thresholds.scope":
        "All 170 detectors have p85 free-flow speeds of 60.3–73.1 mph, so one absolute scale is reasonable; a low-speed arterial would need a relative threshold.",
      "situation.replayMeta": "Replay {time} · not live traffic",
      "situation.cameraMeta": "Latest frame {time} · historical capture",
      "situation.forecast.clear":
        "No detector is forecast below {watch} mph within {minutes} minutes.",
      "situation.forecast.now":
        "{congested} of {total} detectors are currently below {congestedSpeed} mph, {severe} of them below {severeSpeed} mph.",
      "situation.forecast.oneSevere": "1 of {total} detectors is currently below {severeSpeed} mph.",
      "situation.forecast.oneCongested": "1 of {total} detectors is currently below {congestedSpeed} mph.",
      "situation.forecast.noneNow": "No detector is currently below {congestedSpeed} mph.",
      "situation.forecast.forecastMore": "{count} more are forecast within {minutes} minutes.",
      "situation.forecast.new": "{count} newly at risk.",
      "situation.forecast.worst": "Worst: Detector {node}, {speed} mph.",
      "situation.camera.clear": "All {count} camera feeds are reporting normally.",
      "situation.camera.risk":
        "{count} camera feeds have reliability concerns; {ids} should not be trusted without checking.",
      "situation.camera.risk.one":
        "1 camera feed has reliability concerns; {ids} should not be trusted without checking.",
      "situation.system": "{count} display value(s) disagree with the report.",

      "attention.title": "Needs attention",
      "attention.empty":
        "Nothing needs attention. Camera feeds and forecast are within normal limits.",
      "attention.view": "Open",
      "attention.forecast.congestion":
        "Detector {node} · now {now} mph, worst {worst} mph at +{minutes} min",
      "attention.forecast.congestionNow":
        "Detector {node} · now {now} mph, already at its worst in this window",
      "attention.detection.lowDetections":
        "Camera {camera} · {rate}% of frames have no detections",
      "attention.detection.missingFrames":
        "Camera {camera} · only {completeness}% of the expected frames were captured",
      "attention.detection.lowConfidence":
        "Camera {camera} · mean detection confidence {confidence}",
      "attention.system.metricsMismatch":
        "{count} display value(s) disagree with the report",

      "overview.health.detection": "Camera feeds",
      "overview.health.forecast": "Forecast",
      "overview.health.cameras": "Reporting",
      "overview.health.frames": "Frames",
      "overview.health.confidence": "Mean confidence",
      "overview.health.zeroRate": "Frames with no detection",
      "overview.health.window": "Capture window",
      "overview.health.attention": "Feeds needing attention",
      "overview.health.engine": "Engine",
      "overview.health.mape": "Mean error (speed)",
      "overview.health.nodes": "Detectors",
      "overview.health.replay": "Replay time",
      "overview.health.severity": "Forecast severity",
      "overview.card.cameras": "Camera coverage",
      "overview.cameraMapNote":
        "Singapore · colour = feed health · size = completeness. This is not the forecast network.",
      "overview.cameraHover":
        "%{text}<br>no detection %{customdata[0]}% · confidence %{customdata[1]}<extra></extra>",
      "overview.cameraEmpty": "No camera locations available",

      "monitoring.situation.ok":
        "{name} · {pct}% of frames have detections · {frames} frames processed",
      "monitoring.situation.risk":
        "{name} · {pct}% of frames have no detections; counts should not be trusted without checking, especially at night.",
      "monitoring.picker": "Camera feeds",
      "monitoring.pickerNote": "{zero}% zero · {completeness}% complete",
      "monitoring.reliability": "Reliability",
      "monitoring.reliability.frames": "Frames processed",
      "monitoring.reliability.zeroRate": "Frames with no detection",
      "monitoring.reliability.confidence": "Mean confidence",
      "monitoring.reliability.completeness": "Completeness",
      "monitoring.reliability.last": "Last frame",
      "monitoring.reliability.filtered": "Removed by camera filter",
      "monitoring.filter.stats": "{pct} · {road} off-road · {exclude} known false positives",
      "monitoring.filter.toggle": "Camera filter",
      "monitoring.filter.outside": "Outside road region",
      "monitoring.filter.zone": "Detection zone",
      "monitoring.filter.exclude": "Known false-positive spot",
      "monitoring.filter.removed": "Removed box",
      "monitoring.filter.reason.off_road": "off-road",
      "monitoring.filter.reason.exclude": "false positive",
      "monitoring.congestion": "Congestion by weekday and hour",
      "monitoring.congestion.chipTitle": "Congestion level of this frame, judged inside the detection zones",
      "monitoring.congestion.note": "Congested in {share} of frames · worst slot {day} {hour}:00 ({peak})",
      "monitoring.congestion.noteNone": "No congested frames in two weeks · moderate or worse in {busy}",
      "monitoring.congestion.hover": "{day} {hour}:00<br>Congested {congested}%<br>Moderate or worse {busy}%<br>{frames} frames",
      "monitoring.congestion.x": "Hour (SGT)",
      "monitoring.congestion.method":
        "Each frame is rated free-flowing, moderate or congested inside the calibrated detection zones by combining the detector's vehicle count, a background difference and CLIP. Two weeks of data, so each weekday appears twice.",
      "monitoring.congestion.accuracy":
        "On {n} human-labelled test frames the rating agrees {acc} of the time (95% CI {lo}–{hi}, weighted κ {kappa}).",
      "monitoring.congestion.cameraAccuracy": "On this camera's {n} test frames: {acc}.",
      "monitoring.congestion.notEvaluated": "Not yet checked against human labels.",
      "congestion.level.free": "Free-flowing",
      "congestion.level.moderate": "Moderate",
      "congestion.level.congested": "Congested",
      "weekday.0": "Mon", "weekday.1": "Tue", "weekday.2": "Wed", "weekday.3": "Thu",
      "weekday.4": "Fri", "weekday.5": "Sat", "weekday.6": "Sun",
      "monitoring.frame": "Detection frame",
      "monitoring.frameAlt": "Traffic camera frame",
      "monitoring.frameRange": "Select detection frame",
      "monitoring.frameTime": "Time (SGT)",
      "monitoring.emptyFrame": "No detections yet",
      "monitoring.frameNote": "{count} objects · {time}",
      "monitoring.frameNote.one": "{count} object · {time}",
      "monitoring.frameNote.filtered": "{count} objects · {removed} removed by filter · {time}",
      "monitoring.zeroFrame":
        "No objects detected in this frame. This feed has no detections in {rate}% of frames, mostly at night; do not read this as zero traffic.",
      "monitoring.timeline": "Counts by vehicle class",
      "monitoring.timelineNote": "{count} sampled frames · grey bands = no detections",
      "monitoring.timelineX": "Local time (SGT)",
      "monitoring.timelineY": "Vehicles",
      "monitoring.classHover": "%{x}<br>%{y} vehicles<extra></extra>",
      "monitoring.noArtifacts": "No detection artifacts yet. Run: python -m src.detection.run_detect",
      "monitoring.noData": "No data",

      "forecast.replay": "Replay time",
      "forecast.latest": "Latest",
      "forecast.stepBack": "−5 min",
      "forecast.stepForward": "+5 min",
      "forecast.horizon": "Horizon",
      "forecast.horizonOption": "+{minutes} min",
      "forecast.situation.clear":
        "No detector is forecast below {watch} mph within {minutes} minutes.",
      "forecast.situation.risk":
        "{congested} of {total} detectors are forecast below {congestedSpeed} mph within {minutes} minutes; {severe} below {severeSpeed} mph.",
      "forecast.situation.new": "{count} newly at risk.",
      "forecast.situation.watch": "{count} detectors are in the watch band (below {speed} mph).",
      "forecast.situation.worst": "Worst: Detector {node}, {speed} mph.",
      "forecast.situation.simultaneous":
        "At most {count} detectors are below {speed} mph at the same time.",
      "forecast.table.title": "Risk ranking",
      "forecast.table.note": "{count} detectors need attention",
      "forecast.table.showAll": "Show all {count}",
      "forecast.table.showLess": "Show top 10",
      "forecast.table.empty": "No congestion is forecast in this window.",
      "forecast.table.closest": "Closest to the threshold",
      "forecast.table.rank": "#",
      "forecast.table.node": "Detector",
      "forecast.table.now": "Now",
      "forecast.table.worst": "Worst",
      "forecast.table.when": "When",
      "forecast.table.atHorizon": "At +{minutes}",
      "forecast.table.status": "Status",
      "forecast.table.severity": "Severity",
      "forecast.table.nowLabel": "now",
      "forecast.table.minutes": "+{minutes} min",
      "forecast.map.title": "Predicted severity",
      "forecast.map.note":
        "PEMSD8 has no GPS coordinates; this layout is reconstructed from inter-station distances (distance correlation 0.93). Not geographic. Edge and node colour encode predicted severity.",
      "forecast.map.hint": "Scroll to zoom · drag to pan · click a point to inspect · double-click to reset",
      "forecast.map.reset": "Reset view",
      "forecast.map.hover":
        "Detector %{text}<br>now %{customdata[0]} mph · worst %{customdata[1]} mph<extra></extra>",
      "forecast.detail.title": "Detector {node}",
      "forecast.detail.note": "History 60 min → forecast {minutes} min",
      "forecast.detail.curveHistory": "Observed (before replay time)",
      "forecast.detail.curveForecast": "Forecast",
      "forecast.detail.curveTruth": "Actual (replay)",
      "forecast.detail.curveX": "Minutes from replay time",
      "forecast.detail.curveY": "Speed (mph)",
      "forecast.detail.thresholds":
        "Severe < {severe} · Congested < {congested} · Watch < {watch} mph",
      "forecast.detail.bandNote":
        "Shaded band = 10th–90th percentile of this detector's forecast errors on the test period. A typical error range, not a formal prediction interval.",
      "forecast.detail.verdict.severe": "Below {speed} mph within {minutes} min.",
      "forecast.detail.verdict.now": "Currently below {speed} mph.",
      "forecast.detail.verdict.watch": "Expected to drop below {speed} mph within {minutes} min.",
      "forecast.detail.verdict.recovering":
        "Currently congested; expected to recover above {speed} mph by +{minutes} min.",
      "forecast.detail.verdict.clear": "Expected to stay above {speed} mph.",
      "forecast.detail.noNode": "Select a detector to see its forecast.",

      "class.car": "Car",
      "class.motorcycle": "Motorcycle",
      "class.bus": "Bus",
      "class.truck": "Truck",

      "unit.mph": "mph",
      "time.utc": "UTC",
      "time.sgt": "SGT",

      "toast.exported": "Copied current page data ({view}) to clipboard",
      "lang.group": "Language",

      "error.network": "Cannot reach the workbench API",
      "error.notJson": "The API did not return JSON (HTTP {status})",
      "error.http": "HTTP {status}"
    },

    zh: {
      "app.title": "交通监控与预测工作台",
      "app.skipLink": "跳至主内容",
      "app.navLabel": "主导航",
      "app.navSection": "工作区",
      "app.brand.name": "Traffic Workbench",
      "app.brand.tagline": "MONITORING & FORECAST",
      "app.footer": "交通监控与预测工作台",
      "app.poweredBy": "Powered by Tabler",

      "nav.overview": "总览",
      "nav.monitoring": "相机监控",
      "nav.forecast": "拥堵预测",

      "header.menu": "展开导航",
      "header.refresh": "刷新",
      "header.breadcrumb.ops": "TRAFFIC MONITORING",
      "header.breadcrumb.console": "WORKBENCH",
      "header.replayBadge": "历史回放",
      "header.replayNote": "PEMSD8 2016-08-19 → 08-31 · LTA 2026-09-13 → 09-29",
      "header.unavailable": "接口不可用",
      "header.verify": "待核对 {count} 项",

      "views.overview.title": "系统总览",
      "views.overview.subtitle": "现在有什么需要处理。",
      "views.overview.pretitle": "SITUATION",
      "views.monitoring.title": "相机监控",
      "views.monitoring.subtitle": "相机看到了什么，以及计数能不能信。",
      "views.monitoring.pretitle": "CAMERA FEEDS",
      "views.forecast.title": "拥堵预测",
      "views.forecast.subtitle": "哪里会堵、什么时候、多严重。",
      "views.forecast.pretitle": "FORECAST",

      "domain.camera": "相机 · 新加坡 LTA",
      "domain.forecast": "预测 · 加州 PEMSD8",
      "domain.system": "系统",

      "severity.severe": "严重",
      "severity.congested": "拥堵",
      "severity.watch": "关注",
      "severity.normal": "正常",

      "status.ongoing": "持续",
      "status.new": "新增",
      "status.recovering": "恢复中",
      "status.watch": "关注",
      "status.normal": "正常",

      "health.ok": "正常",
      "health.watch": "关注",
      "health.warn": "警告",

      "thresholds.summary": "阈值说明",
      "thresholds.freeflow":
        "锚点：测试期自由流中位数 64.9 mph。三条线取 55（−15%）、45（−31%）、35（−46%）。",
      "thresholds.share": "全部观测中低于各线的占比：6.1% / 3.1% / 1.6%。",
      "thresholds.basis":
        "这是工作台自定的运营口径，不是公开标准，也不是模型输出。HCM 的服务水平按密度划分，本数据集没有密度和车道数。",
      "thresholds.scope":
        "170 个检测点的 p85 自由流为 60.3–73.1 mph，量级一致，所以用一套绝对阈值；若混入低自由流路段，应改为相对阈值。",
      "situation.replayMeta": "回放时刻 {time} · 非实时交通",
      "situation.cameraMeta": "最近一帧 {time} · 历史采集",
      "situation.forecast.clear": "未来 {minutes} 分钟没有检测点预计低于 {watch} mph。",
      "situation.forecast.now":
        "{total} 个检测点中，{congested} 个当前低于 {congestedSpeed} mph，其中 {severe} 个低于 {severeSpeed} mph。",
      "situation.forecast.oneSevere": "{total} 个检测点中，1 个当前低于 {severeSpeed} mph。",
      "situation.forecast.oneCongested": "{total} 个检测点中，1 个当前低于 {congestedSpeed} mph。",
      "situation.forecast.noneNow": "当前没有检测点低于 {congestedSpeed} mph。",
      "situation.forecast.forecastMore": "另有 {count} 个预计在 {minutes} 分钟内进入该区间。",
      "situation.forecast.new": "{count} 个新增风险。",
      "situation.forecast.worst": "最严重：检测点 {node}，{speed} mph。",
      "situation.camera.clear": "{count} 路相机全部正常。",
      "situation.camera.risk":
        "{count} 路相机存在可靠性问题，{ids} 的计数不宜直接采信。",
      "situation.camera.risk.one":
        "1 路相机存在可靠性问题，{ids} 的计数不宜直接采信。",
      "situation.system": "{count} 项展示值与报告不一致。",

      "attention.title": "需要关注",
      "attention.empty": "没有需要处理的事项。相机与预测均在正常范围。",
      "attention.view": "查看",
      "attention.forecast.congestion":
        "检测点 {node} · 当前 {now} mph，最差 {worst} mph（+{minutes} 分钟）",
      "attention.forecast.congestionNow": "检测点 {node} · 当前 {now} mph，已是窗口内最差",
      "attention.detection.lowDetections": "相机 {camera} · {rate}% 的帧无检出",
      "attention.detection.missingFrames": "相机 {camera} · 只采集到预期帧数的 {completeness}%",
      "attention.detection.lowConfidence": "相机 {camera} · 平均检出置信度 {confidence}",
      "attention.system.metricsMismatch": "{count} 项展示值与报告不一致",

      "overview.health.detection": "相机数据",
      "overview.health.forecast": "预测",
      "overview.health.cameras": "上报",
      "overview.health.frames": "帧数",
      "overview.health.confidence": "平均置信度",
      "overview.health.zeroRate": "零检出帧",
      "overview.health.window": "采集区间",
      "overview.health.attention": "需关注相机",
      "overview.health.engine": "引擎",
      "overview.health.mape": "平均误差（速度）",
      "overview.health.nodes": "检测点",
      "overview.health.replay": "回放时刻",
      "overview.health.severity": "预测严重度",
      "overview.card.cameras": "相机分布",
      "overview.cameraMapNote": "新加坡 · 颜色=可靠性 · 大小=完整度。这不是预测路网。",
      "overview.cameraHover":
        "%{text}<br>零检出 %{customdata[0]}% · 置信度 %{customdata[1]}<extra></extra>",
      "overview.cameraEmpty": "暂无相机坐标",

      "monitoring.situation.ok": "{name} · {pct}% 的帧有检出 · 已处理 {frames} 帧",
      "monitoring.situation.risk":
        "{name} · {pct}% 的帧无检出，计数不宜直接采信，夜间尤其明显。",
      "monitoring.picker": "相机列表",
      "monitoring.pickerNote": "零检出 {zero}% · 完整度 {completeness}%",
      "monitoring.reliability": "可靠性",
      "monitoring.reliability.frames": "已处理帧数",
      "monitoring.reliability.zeroRate": "零检出帧",
      "monitoring.reliability.confidence": "平均置信度",
      "monitoring.reliability.completeness": "完整度",
      "monitoring.reliability.last": "最近一帧",
      "monitoring.reliability.filtered": "机位过滤删除",
      "monitoring.filter.stats": "{pct}（道路外 {road} · 错误集 {exclude}）",
      "monitoring.filter.toggle": "机位过滤",
      "monitoring.filter.outside": "道路范围外",
      "monitoring.filter.zone": "检测区",
      "monitoring.filter.exclude": "错误集位置",
      "monitoring.filter.removed": "被删除的框",
      "monitoring.filter.reason.off_road": "道路外",
      "monitoring.filter.reason.exclude": "错误集",
      "monitoring.congestion": "拥挤规律（星期 × 小时）",
      "monitoring.congestion.chipTitle": "这一帧的拥挤程度，只看检测区",
      "monitoring.congestion.note": "{share} 的帧为拥挤 · 最堵时段 {day} {hour}:00（{peak}）",
      "monitoring.congestion.noteNone": "两周内没有拥挤帧 · 一般及以上占 {busy}",
      "monitoring.congestion.hover": "{day} {hour}:00<br>拥挤 {congested}%<br>一般及以上 {busy}%<br>{frames} 帧",
      "monitoring.congestion.x": "小时（SGT）",
      "monitoring.congestion.method":
        "每一帧只看标定的检测区，由检测模型车辆数、背景差分和 CLIP 三种方法组合判为畅通、一般或拥挤。数据只有两周，每个星期几只出现两次。",
      "monitoring.congestion.accuracy":
        "在 {n} 张人工标注的考卷上，与人工判断一致 {acc}（95% 区间 {lo}–{hi}，加权 κ {kappa}）。",
      "monitoring.congestion.cameraAccuracy": "这个机位的 {n} 张考卷：{acc}。",
      "monitoring.congestion.notEvaluated": "尚未用人工标注检验。",
      "congestion.level.free": "畅通",
      "congestion.level.moderate": "一般",
      "congestion.level.congested": "拥挤",
      "weekday.0": "周一", "weekday.1": "周二", "weekday.2": "周三", "weekday.3": "周四",
      "weekday.4": "周五", "weekday.5": "周六", "weekday.6": "周日",
      "monitoring.frame": "检测画面",
      "monitoring.frameAlt": "交通相机画面",
      "monitoring.frameRange": "选择检测画面",
      "monitoring.frameTime": "时间（SGT）",
      "monitoring.emptyFrame": "尚无检测结果",
      "monitoring.frameNote": "{count} 个目标 · {time}",
      "monitoring.frameNote.one": "{count} 个目标 · {time}",
      "monitoring.frameNote.filtered": "{count} 个目标 · 过滤删除 {removed} 个 · {time}",
      "monitoring.zeroFrame":
        "本帧未检出目标。该路 {rate}% 的帧无检出（多在夜间），不能读成「没有车」。",
      "monitoring.timeline": "分车型计数",
      "monitoring.timelineNote": "抽样 {count} 帧 · 灰色区间为零检出",
      "monitoring.timelineX": "当地时间（SGT）",
      "monitoring.timelineY": "车辆数",
      "monitoring.classHover": "%{x}<br>%{y} 辆<extra></extra>",
      "monitoring.noArtifacts": "尚无检测产物：先跑 python -m src.detection.run_detect",
      "monitoring.noData": "暂无数据",

      "forecast.replay": "回放时间",
      "forecast.latest": "最新",
      "forecast.stepBack": "−5 分钟",
      "forecast.stepForward": "+5 分钟",
      "forecast.horizon": "预见期",
      "forecast.horizonOption": "+{minutes} 分钟",
      "forecast.situation.clear": "未来 {minutes} 分钟没有检测点预计低于 {watch} mph。",
      "forecast.situation.risk":
        "{total} 个检测点中，{congested} 个预计在 {minutes} 分钟内低于 {congestedSpeed} mph，{severe} 个低于 {severeSpeed} mph。",
      "forecast.situation.new": "{count} 个新增风险。",
      "forecast.situation.watch": "{count} 个检测点处于关注区间（低于 {speed} mph）。",
      "forecast.situation.worst": "最严重：检测点 {node}，{speed} mph。",
      "forecast.situation.simultaneous": "同一时刻最多 {count} 个检测点低于 {speed} mph。",
      "forecast.table.title": "风险排行",
      "forecast.table.note": "{count} 个检测点需要关注",
      "forecast.table.showAll": "展开全部 {count} 项",
      "forecast.table.showLess": "只看前 10 项",
      "forecast.table.empty": "本时段无拥堵预测。",
      "forecast.table.closest": "最接近阈值的点位",
      "forecast.table.rank": "#",
      "forecast.table.node": "检测点",
      "forecast.table.now": "当前",
      "forecast.table.worst": "最差",
      "forecast.table.when": "何时",
      "forecast.table.atHorizon": "+{minutes} 分",
      "forecast.table.status": "状态",
      "forecast.table.severity": "严重度",
      "forecast.table.nowLabel": "当前",
      "forecast.table.minutes": "+{minutes} 分钟",
      "forecast.map.title": "预测严重度",
      "forecast.map.note":
        "PEMSD8 不带经纬度；这个布局由站间里程还原（距离相关系数 0.93）。不是地理坐标。连线与节点颜色表示预测严重度。",
      "forecast.map.hint": "滚轮缩放 · 拖动平移 · 点击点位查看 · 双击复位",
      "forecast.map.reset": "复位视图",
      "forecast.map.hover":
        "检测点 %{text}<br>当前 %{customdata[0]} mph · 最差 %{customdata[1]} mph<extra></extra>",
      "forecast.detail.title": "检测点 {node}",
      "forecast.detail.note": "历史 60 分钟 → 预测 {minutes} 分钟",
      "forecast.detail.curveHistory": "历史实测",
      "forecast.detail.curveForecast": "系统预测",
      "forecast.detail.curveTruth": "事后实测（回放）",
      "forecast.detail.curveX": "相对回放时刻（分钟）",
      "forecast.detail.curveY": "速度（mph）",
      "forecast.detail.thresholds": "严重 < {severe} · 拥堵 < {congested} · 关注 < {watch} mph",
      "forecast.detail.bandNote":
        "阴影 = 该检测点在测试期上的预测误差 10–90 分位。是典型误差范围，不是正式预测区间。",
      "forecast.detail.verdict.severe": "{minutes} 分钟内降至 {speed} mph 以下。",
      "forecast.detail.verdict.now": "当前已低于 {speed} mph。",
      "forecast.detail.verdict.watch": "预计 {minutes} 分钟内降至 {speed} mph 以下。",
      "forecast.detail.verdict.recovering":
        "当前拥堵，预计 {minutes} 分钟内恢复到 {speed} mph 以上。",
      "forecast.detail.verdict.clear": "预计维持在 {speed} mph 以上。",
      "forecast.detail.noNode": "选择一个检测点查看预测。",

      "class.car": "轿车",
      "class.motorcycle": "摩托车",
      "class.bus": "公交车",
      "class.truck": "卡车",

      "unit.mph": "mph",
      "time.utc": "UTC",
      "time.sgt": "SGT",

      "toast.exported": "已复制当前页数据（{view}）到剪贴板",
      "lang.group": "语言",

      "error.network": "无法连接工作台接口",
      "error.notJson": "接口返回的不是 JSON（HTTP {status}）",
      "error.http": "HTTP {status}"
    }
  };

  let current = DEFAULT_LANG;

  function normalize(value) {
    const text = String(value || "").toLowerCase();
    return text === "zh" || text === "zh-cn" || text === "cn" ? "zh" : "en";
  }

  function readStored() {
    try {
      return normalize(localStorage.getItem(LANG_KEY));
    } catch {
      return DEFAULT_LANG;
    }
  }

  function initialLang() {
    /* ?lang= 只用于分享与无头截图，不写回 localStorage，免得改掉用户的偏好。 */
    try {
      const override = new URLSearchParams(location.search).get("lang");
      if (override) return normalize(override);
    } catch {
      /* 非浏览器环境（Node 测试）没有 location */
    }
    return readStored();
  }

  const getLang = () => current;

  function setLang(value) {
    current = normalize(value);
    try {
      localStorage.setItem(LANG_KEY, current);
    } catch {
      /* 隐私模式下写不进去不影响本次会话 */
    }
    return current;
  }

  function lookup(table, key) {
    return Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;
  }

  /* 只替换调用方真的传了的变量：Plotly 模板里有 %{x} 这种花括号，不能一律替换。 */
  function t(key, values) {
    const table = MESSAGES[current] || MESSAGES.en;
    const template = lookup(table, key) ?? lookup(MESSAGES.en, key) ?? key;
    if (!values) return template;
    return template.replace(/\{(\w+)\}/g, (match, name) =>
      Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : match
    );
  }

  function classLabel(name) {
    const key = `class.${name}`;
    return lookup(MESSAGES[current] || MESSAGES.en, key) ? t(key) : name;
  }

  function applyStaticI18n() {
    document.documentElement.lang = current === "zh" ? "zh-CN" : "en";
    document.querySelectorAll("[data-i18n]").forEach(node => {
      node.textContent = t(node.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-aria-label]").forEach(node => {
      node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel));
    });
    document.querySelectorAll("[data-i18n-alt]").forEach(node => {
      node.setAttribute("alt", t(node.dataset.i18nAlt));
    });
  }

  current = initialLang();

  return { getLang, setLang, t, applyStaticI18n, classLabel, normalize };
})();
