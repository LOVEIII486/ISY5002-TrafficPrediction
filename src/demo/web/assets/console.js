"use strict";

const $ = id => document.getElementById(id);
const api = async (path, params = {}) => {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`/api${path}${query ? "?" + query : ""}`);
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`接口返回的不是 JSON（HTTP ${response.status}）`);
  }
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
};

/* 配色取自 dataviz 参考调色板的深色列。series 的顺序本身就是色盲安全机制
   （相邻色对在 CVD 下 ΔE ≥ 8），不要再重排。改动后必须重跑 validate_palette.js。 */
const P = {
  surface: "#fcfcfb", plane: "#f9f9f7", ink: "#0b0b0b", ink2: "#52514e",
  muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
  /* 单色蓝阶。浅底上「低值靠近背景、高值压深」，顺序是浅→深（与深色模式相反）。 */
  seqBlue: ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"],
};
const FONT = 'system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
const CLASS_COLOR = { car: P.series[0], motorcycle: P.series[1], bus: P.series[2], truck: P.series[3] };
const TARGET_LABEL = { traffic_flow: "流量", traffic_occupancy: "占有率", traffic_speed: "速度" };

const colorscale = ramp => ramp.map((color, index) => [index / (ramp.length - 1), color]);

/* 浅色底上 aqua(#1baf7a, 2.74:1) 与 yellow(#eda100, 2.11:1) 对比度低于 3:1，
   dataviz 的 relief 规则要求配可见的直接标注，不能只靠图例。

   标注放在曲线末端，但夜间几条线都贴近 0，末点会挤成一片，所以按末值排序给垂直错位。
   用 layout.annotations 而不是 trace 的 text —— trace 的 textposition 没法逐条错开。 */
function endLabels(xValues, series) {
  const last = xValues[xValues.length - 1];
  const ranked = series
    .filter(item => item.values.length)
    .slice()
    .sort((a, b) => a.values[a.values.length - 1] - b.values[b.values.length - 1]);
  return ranked.map((item, index) => ({
    x: last,
    y: item.values[item.values.length - 1],
    text: item.name,
    showarrow: false,
    xanchor: "left",
    xshift: 6,
    yshift: (index - (ranked.length - 1) / 2) * 13,
    font: { size: 11, color: item.color },
  }));
}

function axis(extra) {
  return Object.assign({
    gridcolor: P.grid, zerolinecolor: P.grid, linecolor: P.axis,
    tickfont: { color: P.muted, size: 11 },
    title: { font: { color: P.ink2, size: 12 } },
    showline: false,
  }, extra || {});
}

function baseLayout(extra) {
  return Object.assign({
    paper_bgcolor: P.surface,
    plot_bgcolor: P.surface,
    font: { family: FONT, size: 13, color: P.ink2 },
    colorway: P.series.slice(),
    xaxis: axis(), yaxis: axis(),
    /* 图例放底部：放顶部会和标题抢同一行；y 要压得比 x 轴标题更低，否则底边相撞。 */
    legend: {
      bgcolor: "rgba(0,0,0,0)", font: { color: P.ink2, size: 12 },
      orientation: "h", yanchor: "top", y: -0.26, xanchor: "left", x: 0,
    },
    margin: { l: 54, r: 18, t: 16, b: 104 },
    hoverlabel: { bgcolor: P.plane, bordercolor: P.axis, font: { color: P.ink, size: 12 } },
  }, extra || {});
}

const CONFIG = { displayModeBar: false, responsive: true };
const draw = (id, traces, extra) => Plotly.react($(id), traces, baseLayout(extra), CONFIG);

function toast(message, isError) {
  const element = $("toast");
  element.textContent = message;
  element.classList.toggle("error", Boolean(isError));
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 5200);
}

/* tone 只有 'lower-better' 时涨才记坏。检出量这类指标涨跌无所谓，传 'neutral' 免得上错色。 */
function statCard(label, value, unit, delta, tone = "lower-better") {
  let cls = "flat", arrow = "";
  if (delta && tone === "lower-better") {
    if (delta.startsWith("+")) { cls = "bad"; arrow = "▲ "; }
    else if (delta.startsWith("-")) { cls = "good"; arrow = "▼ "; }
  }
  const deltaHtml = delta ? `<span class="stat-delta ${cls}">${arrow}${delta}</span>` : "";
  return `<article class="card"><span class="page-pretitle">${label}</span>
    <span class="stat-value">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</span>${deltaHtml}</article>`;
}

const fmtTime = value => String(value || "").replace("T", " ").slice(0, 16);

/* 两个数据域来源不同，各自页面上都要写明，否则「同一个系统」的观感本身就是误导。 */
function renderSources(elementId, overview, keys, note) {
  const sources = overview.sources || {};
  const lines = keys.filter(key => sources[key]).map(key => {
    const item = sources[key];
    const links = (item.links || [])
      .map(link => `<a href="${link.url}" target="_blank" rel="noopener noreferrer">${link.label}</a>`)
      .join("　");
    return `<b>${item.name}</b>　${item.detail}${links ? `　${links}` : ""}`;
  });
  if (note) lines.push(`<span class="domain-note">${note}</span>`);
  $(elementId).innerHTML = lines.join("<br>");
}

const VIEWS = {
  overview: ["总览", "路网监测与交通预测的整体运行状况。", "SYSTEM OVERVIEW", loadOverview],
  monitoring: ["实时监控", "交通相机的车辆识别与分车型计数。", "LIVE MONITORING", loadMonitoring],
  forecast: ["交通预测", "未来 60 分钟的路网负荷预测与准确度。", "TRAFFIC FORECAST", loadForecast],
};
let activeView = "overview";
let cameraSeries = null;

function showView(view) {
  if (!Object.hasOwn(VIEWS, view)) view = "overview";
  activeView = view;
  const [title, subtitle, pretitle] = VIEWS[view];
  document.querySelectorAll(".view").forEach(node => node.classList.toggle("active", node.id === `view-${view}`));
  document.querySelectorAll(".workspace-nav [data-view]").forEach(link => {
    const on = link.dataset.view === view;
    link.classList.toggle("active", on);
    if (on) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  $("page-pretitle").textContent = pretitle;
  $("page-title").textContent = title;
  $("page-subtitle").textContent = subtitle;
  document.title = `${title} · 交通监控与预测系统`;
  closeMenu();
  VIEWS[view][3]().catch(error => toast(error.message, true));
}

/* ---------- 总览 ---------- */

async function loadOverview() {
  const overview = await api("/overview");
  const done = overview.frames_done || 0;
  const total = overview.frames_total || 0;
  $("overview-stats").innerHTML = [
    statCard("监测点位", overview.cameras, "个", "", "neutral"),
    statCard("已处理画面", done.toLocaleString(), total ? ` / ${total.toLocaleString()}` : "", "", "neutral"),
    statCard("累计检出车辆", (overview.detected_vehicles || 0).toLocaleString(), "辆", "", "neutral"),
    statCard("预测节点", overview.nodes || 0, "个", "", "neutral"),
  ].join("");
  $("overview-map-note").textContent = `${overview.cameras} 个点位 · 圆点按累计检出量`;
  renderSources("overview-sources", overview, ["detection", "forecast"]);
  renderServices(overview.services || []);

  const { cameras } = await api("/detection/cameras");
  renderPointMap("chart-overview-map", cameras, null);
  await renderOverviewRhythm();
  await renderShotGrid();
}

function renderServices(services) {
  $("service-list").innerHTML = services.map(item => `
    <div class="service-row">
      <span class="status" data-state="${item.state}">${item.name}</span>
      <div class="service-detail">${item.detail}<small>${item.hint}</small></div>
    </div>`).join("");
}

/* 点位分布图在总览和监控两处都用，只差尺寸和是否高亮选中项。 */
function renderPointMap(elementId, cameras, picked) {
  const located = cameras.filter(camera => camera.lat !== null && camera.lon !== null);
  if (!located.length) {
    draw(elementId, [], {
      xaxis: { visible: false }, yaxis: { visible: false }, showlegend: false,
      annotations: [{ text: "暂无点位数据", showarrow: false, font: { size: 13, color: P.muted } }],
    });
    return;
  }
  const totals = located.map(camera => camera.total);
  const max = Math.max(...totals, 1);
  const traces = [{
    type: "scattermap", lat: located.map(c => c.lat), lon: located.map(c => c.lon),
    mode: "markers",
    marker: {
      size: totals.map(total => total / max * 20 + 10),
      color: totals, colorscale: colorscale(P.seqBlue), showscale: false, opacity: 0.92,
    },
    text: located.map(c => `${c.id} · ${c.name}`), customdata: totals,
    hovertemplate: "%{text}<br>累计检出 %{customdata} 辆<extra></extra>", showlegend: false,
  }];
  const selected = located.find(camera => camera.id === picked);
  if (selected) {
    traces.push({
      type: "scattermap", lat: [selected.lat], lon: [selected.lon], mode: "markers",
      marker: { size: 22, color: "rgba(0,0,0,0)", line: { width: 3, color: P.ink } },
      hoverinfo: "skip", showlegend: false,
    });
  }
  /* 视野按点位实际范围定，不写死跳新加坡中心——Tuas 的两台在岛最西端，会被裁掉。 */
  const lats = located.map(camera => camera.lat);
  const lons = located.map(camera => camera.lon);
  draw(elementId, traces, {
    map: {
      style: "open-street-map",
      center: {
        lat: (Math.min(...lats) + Math.max(...lats)) / 2,
        lon: (Math.min(...lons) + Math.max(...lons)) / 2,
      },
      zoom: 10,
    },
    margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
  });
}

async function renderOverviewRhythm() {
  const data = await api("/detection/rhythm");
  const traces = Object.entries(data.classes).map(([name, values]) => ({
    type: "scatter", mode: "lines", name,
    x: data.hours, y: values, line: { width: 2, color: CLASS_COLOR[name] },
    hovertemplate: `%{x} 时<br>%{y:.1f} 辆/帧<extra>${name}</extra>`,
  }));
  draw("chart-overview-rhythm", traces, {
    xaxis: { title: { text: "当地时间（小时）" }, dtick: 3 },
    yaxis: { title: { text: "每帧平均车辆数" } },
    margin: { l: 54, r: 76, t: 16, b: 104 },
    annotations: endLabels(data.hours, Object.entries(data.classes).map(([name, values], index) => ({
      name, color: P.series[index % P.series.length], values,
    }))),
  });
}

async function renderShotGrid() {
  const { frames } = await api("/detection/latest");
  $("latest-note").textContent = frames.length ? `共 ${frames.length} 路 · 每路最近一次有检出` : "暂无画面";
  $("shot-grid").innerHTML = frames.map(item => `
    <figure class="shot">
      <img src="${item.url}" alt="点位 ${item.camera}" loading="lazy">
      <figcaption><b>${item.camera}</b><span>${fmtTime(item.captured_at)} · ${item.detections} 辆</span></figcaption>
    </figure>`).join("");
}

/* ---------- 实时监控 ---------- */

const FRAME_WIDTH = 1920;
const FRAME_HEIGHT = 1080;

async function loadMonitoring() {
  const [{ cameras, coverage }, overview] = await Promise.all([
    api("/detection/cameras"), api("/overview"),
  ]);
  $("monitoring-stats").innerHTML = [
    statCard("监测点位", cameras.length, "个", "", "neutral"),
    statCard("累计检出车辆", (overview.detected_vehicles || 0).toLocaleString(), "辆", "", "neutral"),
    statCard("平均置信度", (overview.confidence || 0).toFixed(2), "", "", "neutral"),
    statCard("已处理画面", (coverage.frames_done || 0).toLocaleString(),
      ` / ${(coverage.frames_total || 0).toLocaleString()}`, "", "neutral"),
  ].join("");

  const select = $("camera-select");
  if (!cameras.length) {
    select.innerHTML = "";
    $("camera-note").textContent = "尚无检测产物：先跑 python -m src.detection.run_detect";
    return;
  }
  if (select.options.length !== cameras.length) {
    select.innerHTML = cameras.map(camera =>
      `<option value="${camera.id}">${camera.id} · ${camera.name}</option>`).join("");
  }
  const camera = select.value || cameras[0].id;
  const picked = cameras.find(item => item.id === camera);

  $("map-note").textContent = `${cameras.length} 个点位 · 圆点按累计检出量`;
  renderSources("monitoring-sources", overview, ["detection"]);
  $("camera-note").textContent =
    `${picked.name} · 已处理 ${picked.frames} 帧 · 累计检出 ${picked.total} 辆`;

  renderPointMap("chart-camera-map", cameras, camera);
  cameraSeries = await api("/detection/series", { camera });
  renderCameraSeries(cameraSeries);
  await showFrame((cameraSeries.images || []).length - 1);
}

async function showFrame(index) {
  const images = (cameraSeries && cameraSeries.images) || [];
  if (!images.length) return;
  const range = $("frame-range");
  range.max = images.length - 1;
  const position = Math.max(0, Math.min(Number(index), images.length - 1));
  range.value = position;
  const detail = await api("/detection/frame", { image: images[position] });
  renderFrame(detail, cameraSeries.timestamps[position]);
}

/* 检测框用 SVG 叠在原图上，坐标是归一化的，乘回画面尺寸即可。
   夜间一帧可能有二十几个框，全标文字会糊成一片，所以只给高置信度的框标类名，
   其余靠下方色例辨认——这样身份始终不只由颜色承载。 */
const LABEL_ABOVE = 0.5;

function renderFrame(detail, timestamp) {
  $("frame-image").src = detail.url;
  $("camera-frame").classList.add("has-frame");
  $("frame-overlay").innerHTML = detail.boxes.map(box => {
    const color = CLASS_COLOR[box.cls] || P.muted;
    const x = box.x1 * FRAME_WIDTH;
    const y = box.y1 * FRAME_HEIGHT;
    const width = (box.x2 - box.x1) * FRAME_WIDTH;
    const height = (box.y2 - box.y1) * FRAME_HEIGHT;
    const rect = `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${width.toFixed(1)}" ` +
      `height="${height.toFixed(1)}" stroke="${color}" rx="3"/>`;
    if (box.conf < LABEL_ABOVE) return rect;
    return rect + `<text x="${x.toFixed(1)}" y="${(y - 8).toFixed(1)}" fill="${color}" font-size="30">` +
      `${box.cls} ${box.conf.toFixed(2)}</text>`;
  }).join("");

  const seen = new Set(detail.boxes.map(box => box.cls));
  $("frame-legend").innerHTML = Object.keys(CLASS_COLOR)
    .map(name => `<span class="${seen.has(name) ? "" : "muted"}">` +
      `<i style="background:${CLASS_COLOR[name]}"></i>${name}</span>`).join("");
  $("frame-note").textContent = `${detail.boxes.length} 个目标 · ${fmtTime(timestamp)}`;
}

function renderCameraSeries(series) {
  const pack = series.current || { steps: [], classes: {} };
  const traces = (series.class_names || []).map(name => ({
    type: "scatter", mode: "lines", name,
    x: pack.steps, y: pack.classes[name] || [],
    line: { width: 2, color: CLASS_COLOR[name] },
    hovertemplate: `%{x}<br>%{y} 辆<extra>${name}</extra>`,
  }));
  $("series-note").textContent = pack.steps.length ? `抽样 ${pack.steps.length} 帧` : "暂无数据";
  draw("chart-camera-classes", traces, {
    xaxis: { title: { text: "帧序号（抽样）" } }, yaxis: { title: { text: "车辆数" } },
    margin: { l: 54, r: 76, t: 16, b: 104 },
    annotations: endLabels(pack.steps, (series.class_names || []).map((name, index) => ({
      name, color: P.series[index % P.series.length], values: pack.classes[name] || [],
    }))),
  });
}

/* ---------- 交通预测 ---------- */

/* 这一页面向值班人员，不是面向训练模型的人：部署哪个模型是既成事实而不是选项，
   准确度只说百分比误差，不给 MAE / RMSE / 参数量。 */
const graphCache = {};
const HISTORY_POINTS = 12;

async function loadForecast() {
  const overview = await api("/overview");
  const targetSelect = $("target-select");
  if (!targetSelect.options.length) {
    targetSelect.innerHTML = overview.targets
      .map(name => `<option value="${name}">${TARGET_LABEL[name] || name}</option>`).join("");
  }
  const target = targetSelect.value;
  const graph = await ensureGraph(target);
  const select = $("node-select");
  if (!select.options.length) {
    select.innerHTML = graph.nodes
      .map(node => `<option value="${node.id}">监测点 ${String(node.id).padStart(3, "0")}</option>`)
      .join("");
    select.value = "42";
  }
  renderSources("forecast-sources", overview, ["forecast"], overview.domain_note);
  await refreshForecast();
}

async function ensureGraph(target) {
  if (!graphCache[target]) graphCache[target] = await api("/forecast/graph", { target });
  return graphCache[target];
}

async function refreshForecast() {
  const target = $("target-select").value;
  const node = Number($("node-select").value);
  const graph = await ensureGraph(target);
  const engine = await api("/forecast/engine", { target });
  const sample = graph.samples[graph.samples.length - 1];
  const series = await api("/forecast/series", { node, target, models: engine.model, sample });

  const label = TARGET_LABEL[target] || "读数";
  const current = series.history[series.history.length - 1];
  const forecast = series.models[engine.model] || [];
  $("forecast-note").textContent = `预测引擎 ${engine.model} · 每 5 分钟输出一步`;
  $("forecast-stats").innerHTML = [
    statCard(`当前${label}`, current.toFixed(1), "", "", "neutral"),
    statCard("预见期", "60", "分钟", "", "neutral"),
    statCard("预计峰值", forecast.length ? Math.max(...forecast).toFixed(1) : "—", "", "", "neutral"),
    statCard("预测平均误差", engine.mape_avg.toFixed(1), "%", "", "neutral"),
  ].join("");

  $("curve-title").textContent = `监测点 ${String(node).padStart(3, "0")} · ${label}`;
  $("curve-note").textContent = `历史 ${HISTORY_POINTS} 点 → 未来 12 步`;
  renderGraph(node, graph, label);
  renderCurve(series, engine.model, label);
  renderAccuracy(engine);
}

function renderGraph(node, graph, label) {
  const raw = graph.load[graph.load.length - 1];
  /* 按本帧各自归一化：真实观测是原始量（流量几十到几百），按 0–1 着色会全部顶到最亮。 */
  const low = Math.min(...raw);
  const high = Math.max(...raw);
  const span = high - low;
  const norm = raw.map(value => (span > 0 ? (value - low) / span : 0.5));

  const edgeX = [], edgeY = [];
  graph.edges.forEach(([src, dst]) => {
    edgeX.push(graph.nodes[src].x, graph.nodes[dst].x, null);
    edgeY.push(graph.nodes[src].y, graph.nodes[dst].y, null);
  });
  const traces = [
    /* 连线用比轴线更亮的灰：轴线色在卡片底色上几乎看不见，而这里连线本身就是要看的结构。 */
    { type: "scatter", mode: "lines", x: edgeX, y: edgeY, line: { color: "#a8a69d", width: 1.4 }, hoverinfo: "skip", showlegend: false },
    {
      type: "scatter", mode: "markers",
      x: graph.nodes.map(n => n.x), y: graph.nodes.map(n => n.y),
      text: graph.nodes.map(n => n.id), customdata: raw,
      marker: {
        size: 8, color: norm, colorscale: colorscale(P.seqBlue), cmin: 0, cmax: 1,
        colorbar: { title: { text: "繁忙程度", font: { size: 11 } }, thickness: 8, len: 0.55, outlinewidth: 0, tickfont: { size: 10 } },
        line: { width: 0.5, color: P.surface },
      },
      hovertemplate: `监测点 %{text}<br>${label} %{customdata:.1f}<extra></extra>`, showlegend: false,
    },
    {
      type: "scatter", mode: "markers",
      x: [graph.nodes[node].x], y: [graph.nodes[node].y],
      marker: { size: 17, color: "rgba(0,0,0,0)", line: { width: 2, color: P.ink } },
      hoverinfo: "skip", showlegend: false,
    },
  ];
  draw("chart-graph", traces, {
    xaxis: { visible: false, range: [-1.25, 1.25], fixedrange: true },
    yaxis: { visible: false, range: [-1.25, 1.25], fixedrange: true, scaleanchor: "x" },
    margin: { l: 8, r: 8, t: 8, b: 8 }, showlegend: false,
  });
}

/* 横轴用「相对现在的分钟数」而不是步序号：值班的人关心的是「多久之后堵」，
   不是第几个预测步。 */
function renderCurve(series, model, label) {
  const historyX = series.history.map((_, index) => (index - historyPointCount(series)) * 5);
  const futureX = series.truth.map((_, index) => (index + 1) * 5);
  const traces = [
    {
      type: "scatter", mode: "lines", name: "历史实测", showlegend: false,
      x: historyX, y: series.history, line: { width: 2, color: P.muted },
      hovertemplate: "%{x} 分钟前<br>%{y:.1f}<extra>历史实测</extra>",
    },
    {
      type: "scatter", mode: "lines", name: "系统预测",
      x: futureX, y: series.models[model] || [], line: { width: 2.5, color: P.series[0] },
      hovertemplate: "现在 +%{x} 分钟<br>预测 %{y:.1f}<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", name: "事后实测",
      x: futureX, y: series.truth, line: { width: 2, color: P.ink, dash: "dash" },
      hovertemplate: "现在 +%{x} 分钟<br>实测 %{y:.1f}<extra></extra>",
    },
  ];
  draw("chart-curve", traces, {
    shapes: [
      { type: "rect", xref: "x", yref: "paper", x0: 0, x1: 60, y0: 0, y1: 1,
        fillcolor: P.grid, opacity: 0.4, line: { width: 0 }, layer: "below" },
      { type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: P.axis, width: 1 } },
    ],
    annotations: [{
      x: 0, y: 1.02, yref: "paper", text: "现在", showarrow: false,
      font: { size: 11, color: P.muted }, xanchor: "center",
    }],
    xaxis: { title: { text: "相对现在（分钟）" }, dtick: 15 },
    yaxis: { title: { text: label } },
    hovermode: "x unified",
  });
}

const historyPointCount = series => series.history.length;

/* 单序列、零基线：只回答「预见期越长误差越大」这一件事。 */
function renderAccuracy(engine) {
  const values = engine.horizons.map(item => item.mape);
  draw("chart-accuracy", [{
    type: "bar", width: 0.45,
    x: engine.horizons.map(item => item.label), y: values,
    marker: { color: P.series[0] },
    text: values.map(value => `${value.toFixed(1)}%`),
    textposition: "outside", textfont: { size: 12 },
    hovertemplate: "%{x}<br>平均误差 %{y:.1f}%<extra></extra>",
  }], {
    showlegend: false,
    xaxis: { title: { text: "预见期" } },
    yaxis: { title: { text: "平均误差（%）" }, range: [0, Math.max(...values) * 1.35] },
  });
}

/* ---------- 交互 ---------- */

function closeMenu() {
  $("app-sidebar").classList.remove("is-open");
  document.querySelector("[data-menu]").setAttribute("aria-expanded", "false");
}

document.addEventListener("click", async event => {
  const menu = event.target.closest("[data-menu]");
  if (menu) {
    const sidebar = $("app-sidebar");
    sidebar.classList.toggle("is-open");
    menu.setAttribute("aria-expanded", String(sidebar.classList.contains("is-open")));
    return;
  }
  const link = event.target.closest("[data-view]");
  if (link) {
    event.preventDefault();
    location.hash = `#${link.dataset.view}`;
    return;
  }
  const button = event.target.closest("[data-command]");
  if (!button) return;
  const command = button.dataset.command;
  try {
    if (command === "reloadMonitoring") await loadMonitoring();
    else if (command === "reloadForecast") await loadForecast();
    else if (command === "refreshActiveView") await VIEWS[activeView][3]();
    else if (command === "exportView") await exportView();
  } catch (error) {
    toast(error.message, true);
  }
});

async function exportView() {
  const payload = { view: activeView, exported_at: new Date().toISOString() };
  if (activeView === "overview") {
    payload.overview = await api("/overview");
    payload.cameras = (await api("/detection/cameras")).cameras;
  } else if (activeView === "monitoring") {
    payload.cameras = (await api("/detection/cameras")).cameras;
    payload.series = await api("/detection/series", { camera: $("camera-select").value });
  } else {
    payload.engine = await api("/forecast/engine", { target: $("target-select").value });
    payload.series = await api("/forecast/series", {
      node: Number($("node-select").value),
      target: $("target-select").value,
      models: payload.engine.model,
    });
  }
  await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
  toast(`已复制当前页数据（${VIEWS[activeView][0]}）到剪贴板`);
}

$("frame-range").addEventListener("input", () => {
  if (!cameraSeries) return;
  showFrame(Number($("frame-range").value)).catch(error => toast(error.message, true));
});

$("target-select").addEventListener("change", () => {
  if (activeView === "forecast") loadForecast().catch(error => toast(error.message, true));
});

window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

(async function start() {
  $("sidebar-endpoint").textContent = location.host;
  try {
    const overview = await api("/overview");
    $("dataset-badge").textContent = `${overview.cameras} 点位 · ${overview.engine.model}`;
    $("dataset-badge").dataset.state = "on";
    /* 服务端会拿展示值和报告记录对一遍；对不上必须让人看见，不能当没事发生。 */
    if (overview.warnings.length) {
      $("sidebar-status").textContent = `待核对 ${overview.warnings.length} 项`;
      $("sidebar-status").dataset.state = "warn";
      toast(overview.warnings.join("；"), true);
    }
  } catch (error) {
    $("dataset-badge").textContent = "接口不可用";
    $("dataset-badge").dataset.state = "err";
    toast(error.message, true);
  }
  showView(location.hash.slice(1) || "overview");
})();
