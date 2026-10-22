"use strict";

const $ = id => document.getElementById(id);
const { getLang, setLang, t, applyStaticI18n, classLabel } = window.ConsoleI18n;

const api = async (path, params = {}) => {
  const requestLang = getLang();
  const query = new URLSearchParams({ ...params, lang: requestLang }).toString();
  let response;
  try {
    response = await fetch(`/api${path}${query ? "?" + query : ""}`);
  } catch {
    throw new Error(t("error.network"));
  }
  const text = await response.text();
  /* 慢响应可能在切到另一种语言之后才回来；丢掉它，免得把旧语言的文字盖回去。 */
  if (requestLang !== getLang()) {
    const stale = new Error("stale language response");
    stale.stale = true;
    throw stale;
  }
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(t("error.notJson", { status: response.status }));
  }
  if (!response.ok) throw new Error(data.error || t("error.http", { status: response.status }));
  return data;
};

function report(error) {
  if (!error || error.stale) return;
  toast(error.message, true);
}

function toast(message, isError) {
  const element = $("toast");
  element.textContent = message;
  element.classList.toggle("error", Boolean(isError));
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 5200);
}

/* 配色取自 dataviz 参考调色板的深色列。series 的顺序本身就是色盲安全机制
   （相邻色对在 CVD 下 ΔE ≥ 8），不要再重排。改动后必须重跑 validate_palette.js。 */
const P = {
  surface: "#fcfcfb", plane: "#f9f9f7", ink: "#0b0b0b", ink2: "#52514e",
  muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
};
const FONT = 'system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
const CLASS_COLOR = { car: P.series[0], motorcycle: P.series[1], bus: P.series[2], truck: P.series[3] };

/* 严重度同时用颜色和形状编码：只靠颜色的话，色觉障碍或黑白打印就读不出来。 */
const SEVERITY = {
  severe: { color: "#c62f2f", symbol: "diamond", size: 14 },
  congested: { color: "#e05a22", symbol: "triangle-up", size: 12 },
  watch: { color: "#d99a00", symbol: "circle", size: 10 },
  normal: { color: "#8f8e86", symbol: "circle", size: 7 },
};
const HEALTH_COLOR = { ok: "#1baf7a", watch: "#eda100", warn: "#d03b3b" };
const THRESHOLD_COLOR = { severe: "#d03b3b", congested: "#eb6834", watch: "#eda100" };

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
    legend: {
      bgcolor: "rgba(0,0,0,0)", font: { color: P.ink2, size: 12 },
      orientation: "h", yanchor: "top", y: -0.26, xanchor: "left", x: 0,
    },
    margin: { l: 54, r: 18, t: 16, b: 104 },
    hoverlabel: { bgcolor: P.plane, bordercolor: P.axis, font: { color: P.ink, size: 12 } },
  }, extra || {});
}

const CONFIG = { displayModeBar: false, responsive: true };
/* 网络图有 170 个节点，必须能缩放；其余图表保持无工具栏。 */
const MAP_CONFIG = {
  /* 工具栏只在悬停时出现：常驻会压住图例。缩放主要靠滚轮与双击复位。 */
  displayModeBar: "hover", displaylogo: false, responsive: true, scrollZoom: true,
  doubleClick: "reset", modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"],
};
const draw = (id, traces, extra, config = CONFIG) =>
  Plotly.react($(id), traces, baseLayout(extra), config);

/* ---------- 时间与格式 ---------- */

const pad = value => String(value).padStart(2, "0");

function fmtUtc(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} `
    + `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())} UTC`;
}

const SGT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Singapore", year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", hour12: false,
});

function fmtSgt(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return `${SGT.format(date).replace(",", "")} SGT`;
}

const pct = value => `${(Number(value) * 100).toFixed(1)}%`;
/* 模板里已经写了 % 的地方用这个，避免出现 35.8%%。 */
const numPct = value => (Number(value) * 100).toFixed(1);

/* ---------- 状态 ---------- */

const state = {
  view: "overview",
  sample: null,
  horizon: 60,
  node: null,
  camera: null,
  showAll: false,
  range: null,
  risk: null,
  graph: null,
  mapRange: null,
  cameras: [],
};

const VIEWS = {
  overview: {
    title: "views.overview.title",
    subtitle: "views.overview.subtitle",
    pretitle: "views.overview.pretitle",
    load: loadOverview,
  },
  monitoring: {
    title: "views.monitoring.title",
    subtitle: "views.monitoring.subtitle",
    pretitle: "views.monitoring.pretitle",
    load: loadMonitoring,
  },
  forecast: {
    title: "views.forecast.title",
    subtitle: "views.forecast.subtitle",
    pretitle: "views.forecast.pretitle",
    load: loadForecast,
  },
};

/* ---------- 通用组件 ---------- */

function severityChip(code) {
  return `<span class="chip" data-severity="${code}">${t(`severity.${code}`)}</span>`;
}

function statusChip(code) {
  return `<span class="chip chip-quiet" data-status="${code}">${t(`status.${code}`)}</span>`;
}

function healthChip(code) {
  return `<span class="chip" data-health="${code}">${t(`health.${code}`)}</span>`;
}

function domainChip(domain) {
  return `<span class="chip chip-domain" data-domain="${domain}">${t(`domain.${domain}`)}</span>`;
}

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

function situationCard(tone, headline, meta, explain = "") {
  return `<section class="situation" data-tone="${tone}">
    <p class="situation-headline">${headline}</p>
    ${meta ? `<div class="situation-meta">${meta}</div>` : ""}
    ${explain}
  </section>`;
}

/* 严重度阈值是可展开的说明：口径、数据锚点、实测占比、边界都说清楚。 */
function thresholdsExplain() {
  return `<details class="explain">
    <summary>${t("thresholds.summary")}</summary>
    <div class="explain-body">
      <p>${t("thresholds.freeflow")}</p>
      <ul>
        <li>${t("thresholds.share")}</li>
        <li>${t("thresholds.basis")}</li>
        <li>${t("thresholds.scope")}</li>
      </ul>
    </div>
  </details>`;
}

function metric(label, value) {
  return `<div class="health-metric"><dt>${label}</dt><dd>${value}</dd></div>`;
}

function healthState(health) {
  return health === "warn" ? "err" : health === "watch" ? "warn" : "on";
}

/* ---------- 总览 ---------- */

async function loadOverview() {
  const [overview, cameras] = await Promise.all([
    api("/overview"), api("/detection/cameras"),
  ]);
  renderOverviewSituation(overview);
  renderAttention(overview.attention || []);
  renderDomainHealth(overview.health);
  renderSources("overview-sources", overview, ["detection", "forecast"], overview.domain_note);
  renderCameraMap("chart-camera-map", cameras.cameras || []);
}

/* 预测页与总览共用这一句话：当前有多少在阈值下，未来还有多少会进来。
   不要把「当前已拥堵」说成「预计拥堵」——样本里就有当前 34 mph、一小时内恢复的点位。 */
function forecastSituation(summary, thresholds, minutes) {
  const parts = [];
  let tone = "ok";
  const nowCongested = summary.now_counts.below_congested;
  const nowSevere = summary.now_counts.below_severe;
  const newRisks = summary.status_counts.new;
  const watchCount = summary.severity_counts.watch;

  if (nowCongested === 1) {
    parts.push(t(nowSevere === 1
      ? "situation.forecast.oneSevere"
      : "situation.forecast.oneCongested", {
      total: summary.total,
      congestedSpeed: thresholds.congested,
      severeSpeed: thresholds.severe,
    }));
    tone = nowSevere > 0 ? "bad" : "warn";
  } else if (nowCongested > 1) {
    parts.push(t("situation.forecast.now", {
      congested: nowCongested,
      total: summary.total,
      severe: nowSevere,
      congestedSpeed: thresholds.congested,
      severeSpeed: thresholds.severe,
    }));
    tone = nowSevere > 0 ? "bad" : "warn";
  } else if (newRisks > 0) {
    parts.push(t("situation.forecast.noneNow", { congestedSpeed: thresholds.congested }));
    tone = "warn";
  } else if (watchCount > 0) {
    parts.push(t("forecast.situation.watch", { count: watchCount, speed: thresholds.watch }));
    tone = "warn";
  } else {
    parts.push(t("situation.forecast.clear", { watch: thresholds.watch, minutes }));
  }

  if (newRisks > 0) {
    parts.push(t("situation.forecast.forecastMore", { count: newRisks, minutes }));
    if (tone === "ok") tone = "warn";
  }
  if (nowCongested > 0 || newRisks > 0) {
    parts.push(t("situation.forecast.worst", {
      node: summary.worst_node, speed: Number(summary.worst_speed).toFixed(1),
    }));
  }
  return { parts, tone };
}

function renderOverviewSituation(overview) {
  const summary = overview.forecast.summary;
  const thresholds = overview.forecast.thresholds;
  const detection = overview.health.detection;
  const parts = [];
  let tone = "ok";

  const composition = forecastSituation(summary, thresholds, 60);
  parts.push(...composition.parts);
  tone = composition.tone;

  if (detection.attention > 0) {
    const ids = detection.attention_ids || [];
    const key = detection.attention === 1
      ? "situation.camera.risk.one" : "situation.camera.risk";
    parts.push(t(key, { count: detection.attention, ids: ids.join(", ") }));
    if (tone === "ok") tone = "warn";
  } else {
    parts.push(t("situation.camera.clear", { count: detection.cameras }));
  }

  if ((overview.warnings || []).length) {
    parts.push(t("situation.system", { count: overview.warnings.length }));
    tone = "warn";
  }

  $("overview-situation").innerHTML = situationCard(
    tone,
    parts.join(" "),
    t("situation.replayMeta", { time: fmtUtc(overview.time.iso) }),
    thresholdsExplain()
  );
}

function attentionText(item) {
  if (item.code === "attention.forecast.congestion") {
    return item.values.minutes > 0
      ? t("attention.forecast.congestion", {
        node: item.node,
        now: Number(item.values.now).toFixed(1),
        worst: Number(item.values.worst).toFixed(1),
        minutes: item.values.minutes,
      })
      : t("attention.forecast.congestionNow", {
        node: item.node, now: Number(item.values.now).toFixed(1),
      });
  }
  if (item.code === "attention.detection.lowDetections") {
    return t(item.code, { camera: item.camera, rate: numPct(item.values.zeroRate) });
  }
  if (item.code === "attention.detection.missingFrames") {
    return t(item.code, { camera: item.camera, completeness: numPct(item.values.completeness) });
  }
  if (item.code === "attention.detection.lowConfidence") {
    return t(item.code, {
      camera: item.camera, confidence: Number(item.values.confidence).toFixed(2),
    });
  }
  return t(item.code, { count: item.values.count });
}

function renderAttention(items) {
  const container = $("overview-attention");
  if (!items.length) {
    container.innerHTML = `<p class="empty-state">${t("attention.empty")}</p>`;
    $("overview-attention-note").textContent = "";
    return;
  }
  container.innerHTML = items.map(item => {
    const chip = item.domain === "forecast" ? severityChip(item.severity) : healthChip(item.severity);
    return `<div class="attention-row" data-severity="${item.severity}">
      <div class="attention-chips">${domainChip(item.domain)}${chip}</div>
      <p class="attention-text">${attentionText(item)}</p>
      <button class="btn btn-ghost-secondary btn-sm" type="button"
        data-goto-view="${item.view}" data-goto-node="${item.node ?? ""}"
        data-goto-camera="${item.camera ?? ""}" data-goto-sample="${item.sample ?? ""}">
        ${t("attention.view")}
      </button>
    </div>`;
  }).join("");
  $("overview-attention-note").textContent = "";
}

function renderDomainHealth(health) {
  const detection = health.detection;
  const forecast = health.forecast;
  $("overview-health").innerHTML = `
    <section class="card panel health-card">
      <div class="card-head"><h4>${t("overview.health.detection")}</h4>${domainChip("camera")}</div>
      <dl class="health-metrics">
        ${metric(t("overview.health.cameras"), `${detection.reporting} / ${detection.cameras}`)}
        ${metric(t("overview.health.frames"), Number(detection.frames).toLocaleString())}
        ${metric(t("overview.health.zeroRate"), pct(detection.zero_rate))}
        ${metric(t("overview.health.confidence"), Number(detection.confidence).toFixed(2))}
        ${metric(t("overview.health.attention"), detection.attention)}
      </dl>
    </section>
    <section class="card panel health-card">
      <div class="card-head"><h4>${t("overview.health.forecast")}</h4>${domainChip("forecast")}</div>
      <dl class="health-metrics">
        ${metric(t("overview.health.engine"), forecast.engine)}
        ${metric(t("overview.health.mape"), `${Number(forecast.mape_avg).toFixed(1)}%`)}
        ${metric(t("overview.health.nodes"), forecast.nodes)}
        ${metric(t("overview.health.severity"),
          `${t("severity.severe")} ${forecast.severity_counts.severe} · `
          + `${t("severity.congested")} ${forecast.severity_counts.congested} · `
          + `${t("severity.watch")} ${forecast.severity_counts.watch}`)}
        ${metric(t("overview.health.replay"), fmtUtc(forecast.time.iso))}
      </dl>
    </section>`;
}

function renderCameraMap(elementId, cameras) {
  const located = cameras.filter(camera => camera.lat !== null && camera.lon !== null);
  if (!located.length) {
    draw(elementId, [], {
      xaxis: { visible: false }, yaxis: { visible: false }, showlegend: false,
      annotations: [{
        text: t("overview.cameraEmpty"), showarrow: false,
        font: { size: 13, color: P.muted },
      }],
    });
    return;
  }

  const groups = { ok: [], watch: [], warn: [] };
  located.forEach(camera => (groups[camera.health] || groups.ok).push(camera));
  const traces = Object.entries(groups)
    .filter(([, items]) => items.length)
    .map(([health, items]) => ({
      type: "scattermap", lat: items.map(camera => camera.lat), lon: items.map(camera => camera.lon),
      mode: "markers", name: `${t(`health.${health}`)} (${items.length})`,
      marker: {
        size: items.map(camera => 10 + 12 * Number(camera.completeness ?? 1)),
        color: HEALTH_COLOR[health],
        line: { width: 1, color: P.surface },
      },
      text: items.map(camera => `${camera.id} · ${camera.name}`),
      customdata: items.map(camera => [numPct(camera.zero_rate), camera.confidence ?? "—"]),
      hovertemplate: t("overview.cameraHover"), showlegend: true,
    }));

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
    margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: true,
    legend: { orientation: "h", yanchor: "top", y: 0.99, xanchor: "left", x: 0.01 },
  });
}

/* ---------- 相机监控 ---------- */

let cameraSeries = null;

async function loadMonitoring() {
  const [payload, overview] = await Promise.all([
    api("/detection/cameras"), api("/overview"),
  ]);
  const cameras = payload.cameras || [];
  state.cameras = cameras;

  if (!cameras.length) {
    $("monitoring-situation").innerHTML = situationCard("warn", t("monitoring.noArtifacts"), "");
    $("camera-picker").innerHTML = "";
    $("monitoring-reliability").innerHTML = "";
    return;
  }

  if (!cameras.some(camera => camera.id === state.camera)) state.camera = cameras[0].id;
  const picked = cameras.find(camera => camera.id === state.camera);

  renderCameraPicker(cameras, picked.id);
  renderMonitoringSituation(picked);
  renderReliability(picked);
  renderSources("monitoring-sources", overview, ["detection"]);

  cameraSeries = await api("/detection/series", { camera: picked.id });
  const timeInput = $("frame-time");
  if (picked.first) timeInput.min = sgtInputValue(picked.first);
  if (picked.last) timeInput.max = sgtInputValue(picked.last);
  renderCameraTimeline(cameraSeries);
  await showLatestFrame();
  writeRoute();
}

function renderMonitoringSituation(camera) {
  const unreliable = camera.health === "warn" || camera.health === "watch";
  const headline = unreliable
    ? t("monitoring.situation.risk", {
      name: `${camera.id} · ${camera.name}`, pct: numPct(camera.zero_rate),
    })
    : t("monitoring.situation.ok", {
      name: `${camera.id} · ${camera.name}`,
      pct: (100 - Number(camera.zero_rate) * 100).toFixed(1),
      frames: Number(camera.frames).toLocaleString(),
    });
  $("monitoring-situation").innerHTML = situationCard(
    unreliable ? "warn" : "ok",
    headline,
    t("situation.cameraMeta", { time: fmtSgt(camera.last) })
  );
}

function renderCameraPicker(cameras, selected) {
  $("camera-picker").innerHTML = cameras.map(camera => `
    <button type="button" class="camera-item" data-camera="${camera.id}"
      aria-pressed="${camera.id === selected}">
      <span class="status" data-state="${healthState(camera.health)}"></span>
      <span class="camera-item-name"><b>${camera.id}</b><small>${camera.name}</small></span>
      <span class="camera-item-stats">${t("monitoring.pickerNote", {
        zero: numPct(camera.zero_rate), completeness: numPct(camera.completeness),
      })}</span>
    </button>`).join("");
}

function renderReliability(camera) {
  $("monitoring-reliability").innerHTML = [
    metric(t("monitoring.reliability.frames"), Number(camera.frames).toLocaleString()),
    metric(t("monitoring.reliability.zeroRate"), pct(camera.zero_rate)),
    metric(t("monitoring.reliability.confidence"),
      camera.confidence === null ? "—" : Number(camera.confidence).toFixed(3)),
    metric(t("monitoring.reliability.completeness"), pct(camera.completeness)),
    metric(t("monitoring.reliability.last"), fmtSgt(camera.last)),
  ].join("");
  $("reliability-note").innerHTML = healthChip(camera.health);
}

async function showLatestFrame() {
  const latest = cameraSeries && cameraSeries.latest;
  if (!latest) return;
  /* 必须先设 max：只调 showLatestFrame 时 max 还是 0，滑条就动不了。 */
  const range = $("frame-range");
  range.max = Math.max(0, (cameraSeries.images || []).length - 1);
  range.value = range.max;
  const detail = await api("/detection/frame", { image: latest.image });
  renderFrame(detail, latest.timestamp);
  setFrameTime(latest.timestamp);
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
  setFrameTime(cameraSeries.timestamps[position]);
}

/* datetime-local 不带时区；相机页统一按 SGT 解释输入与显示。 */
function sgtInputValue(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return new Date(date.getTime() + 8 * 3600 * 1000).toISOString().slice(0, 16);
}

function sgtInputToIso(value) {
  return value ? `${value}:00+08:00` : null;
}

function setFrameTime(iso) {
  const input = $("frame-time");
  if (input && iso) input.value = sgtInputValue(iso);
}

async function showFrameAtTime(value) {
  if (!state.camera) return;
  const detail = await api("/detection/frame", {
    camera: state.camera, time: sgtInputToIso(value),
  });
  renderFrame(detail, detail.captured_at);
  const target = new Date(detail.captured_at).getTime();
  const stamps = (cameraSeries.timestamps || []).map(stamp => new Date(stamp).getTime());
  if (stamps.length) {
    let nearest = 0;
    stamps.forEach((stamp, index) => {
      if (Math.abs(stamp - target) < Math.abs(stamps[nearest] - target)) nearest = index;
    });
    $("frame-range").value = nearest;
  }
  setFrameTime(detail.captured_at);
}

/* 检测框用 SVG 叠在原图上，坐标是归一化的，乘回画面尺寸即可。
   夜间一帧可能有二十几个框，全标文字会糊成一片，所以只给高置信度的框标类名，
   其余靠下方色例辨认——这样身份始终不只由颜色承载。 */
const LABEL_ABOVE = 0.5;
const FRAME_WIDTH = 1920;
const FRAME_HEIGHT = 1080;

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
      `${classLabel(box.cls)} ${box.conf.toFixed(2)}</text>`;
  }).join("");

  const seen = new Set(detail.boxes.map(box => box.cls));
  $("frame-legend").innerHTML = Object.keys(CLASS_COLOR)
    .map(name => `<span class="${seen.has(name) ? "" : "muted"}">` +
      `<i style="background:${CLASS_COLOR[name]}"></i>${classLabel(name)}</span>`).join("");
  const frameKey = detail.boxes.length === 1
    ? "monitoring.frameNote.one" : "monitoring.frameNote";
  $("frame-note").textContent = t(frameKey, {
    count: detail.boxes.length, time: fmtSgt(timestamp),
  });

  const picked = state.cameras.find(camera => camera.id === state.camera);
  const zeroNote = $("zero-frame-note");
  if (detail.zero && picked) {
    zeroNote.hidden = false;
    zeroNote.textContent = t("monitoring.zeroFrame", { rate: numPct(picked.zero_rate) });
  } else {
    zeroNote.hidden = true;
  }
}

function renderCameraTimeline(series) {
  const classNames = series.class_names || [];
  const timestamps = series.timestamps || [];
  const traces = classNames.map(name => ({
    type: "scatter", mode: "lines", name: classLabel(name),
    x: timestamps, y: (series.current.classes[name] || []),
    line: { width: 1.6, color: CLASS_COLOR[name] },
    hovertemplate: `${t("monitoring.classHover")}<extra>${classLabel(name)}</extra>`,
  }));

  /* 零检出区间用浅灰底标出：夜间那些缺口是漏检，不是真的没车。 */
  const shapes = [];
  let start = null;
  (series.zero || []).forEach((isZero, index) => {
    if (isZero && start === null) start = index;
    const last = index === timestamps.length - 1;
    if (start !== null && (!isZero || last)) {
      const end = isZero && last ? index : index - 1;
      shapes.push({
        type: "rect", xref: "x", yref: "paper",
        x0: timestamps[start], x1: timestamps[end], y0: 0, y1: 1,
        fillcolor: "#0b0b0b", opacity: 0.05, line: { width: 0 }, layer: "below",
      });
      start = null;
    }
  });

  $("timeline-note").textContent = timestamps.length
    ? t("monitoring.timelineNote", { count: timestamps.length })
    : t("monitoring.noData");

  draw("chart-camera-classes", traces, {
    shapes,
    xaxis: { title: { text: t("monitoring.timelineX") }, type: "date", tickformat: "%d %H:%M" },
    yaxis: { title: { text: t("monitoring.timelineY") } },
    margin: { l: 54, r: 18, t: 16, b: 104 },
  });
}

/* ---------- 未来 60 分钟 ---------- */

async function ensureReplayRange() {
  if (!state.range) state.range = await api("/forecast/replay");
  return state.range;
}

async function loadForecast() {
  const range = await ensureReplayRange();
  const slider = $("replay-range");
  slider.min = range.first;
  slider.max = range.last;
  if (state.sample === null || state.sample < range.first || state.sample > range.last) {
    state.sample = range.last;
  }
  slider.value = state.sample;

  const [overview, risk] = await Promise.all([
    api("/overview"),
    api("/forecast/risk", { sample: state.sample, horizon: state.horizon }),
  ]);
  state.sample = risk.sample;
  state.risk = risk;
  state.showAll = false;

  renderReplayControl(risk);
  renderForecastSituation(risk);
  renderSources("forecast-sources", overview, ["forecast"], overview.domain_note);

  const nonNormal = risk.nodes.filter(row => row.severity !== "normal");
  const fallback = nonNormal.length ? nonNormal[0].node : risk.nodes[0].node;
  if (state.node === null || !risk.nodes.some(row => row.node === state.node)) {
    state.node = fallback;
  }

  renderRiskTable(risk);
  const graph = await api("/forecast/graph", {
    sample: state.sample, horizon: state.horizon,
  });
  state.graph = graph;
  renderSeverityMap(graph, state.node);
  await renderRiskCurve(state.node);
  writeRoute();
}

function renderReplayControl(risk) {
  $("replay-range").value = risk.sample;
  $("replay-label").textContent = risk.time.valid
    ? `${fmtUtc(risk.time.iso)} · sample ${risk.sample}`
    : `sample ${risk.sample}`;
  document.querySelectorAll("#horizon-group [data-horizon]").forEach(button => {
    const minutes = Number(button.dataset.horizon);
    button.setAttribute("aria-pressed", String(minutes === state.horizon));
    button.textContent = t("forecast.horizonOption", { minutes });
  });
}

function renderForecastSituation(risk) {
  const thresholds = risk.thresholds;
  const composition = forecastSituation(risk.summary, thresholds, risk.horizon.minutes);
  const parts = composition.parts;
  const tone = composition.tone;

  $("forecast-situation").innerHTML = situationCard(
    tone,
    parts.join(" "),
    `${t("situation.replayMeta", { time: fmtUtc(risk.time.iso) })} · `
    + t("forecast.detail.thresholds", thresholds),
    thresholdsExplain()
  );
}

function whenText(row) {
  return row.worst_at_min > 0
    ? t("forecast.table.minutes", { minutes: row.worst_at_min })
    : t("forecast.table.nowLabel");
}

function renderRiskTable(risk) {
  const flagged = risk.nodes.filter(row => row.severity !== "normal" || row.status !== "normal");
  const empty = flagged.length === 0;
  const rows = empty ? risk.nodes.slice(0, 5) : flagged;
  const shown = state.showAll ? rows : rows.slice(0, 10);

  $("risk-table-note").textContent = empty
    ? ""
    : t("forecast.table.note", { count: flagged.length });

  const header = `<thead><tr>
    <th>${t("forecast.table.rank")}</th>
    <th>${t("forecast.table.node")}</th>
    <th class="num">${t("forecast.table.now")}</th>
    <th class="num">${t("forecast.table.worst")}</th>
    <th>${t("forecast.table.when")}</th>
    <th class="num">${t("forecast.table.atHorizon", { minutes: risk.horizon.minutes })}</th>
    <th>${t("forecast.table.status")}</th>
    <th>${t("forecast.table.severity")}</th>
  </tr></thead>`;

  const body = shown.map(row => `<tr data-node="${row.node}" tabindex="0"
      aria-selected="${row.node === state.node}">
      <td>${row.rank}</td>
      <td><b>${row.node}</b></td>
      <td class="num">${Number(row.now).toFixed(1)}</td>
      <td class="num">${Number(row.worst).toFixed(1)}</td>
      <td>${whenText(row)}</td>
      <td class="num">${Number(row.at_horizon).toFixed(1)}</td>
      <td>${statusChip(row.status)}</td>
      <td>${severityChip(row.severity)}</td>
    </tr>`).join("");

  const table = `<table class="table risk-table">${header}<tbody>${body}</tbody></table>`;
  const toggle = rows.length > 10
    ? `<button class="btn btn-ghost-secondary btn-sm" type="button" data-command="toggleAllRisk">
        ${state.showAll
          ? t("forecast.table.showLess")
          : t("forecast.table.showAll", { count: rows.length })}
      </button>`
    : "";

  const intro = empty
    ? `<p class="empty-state">${t("forecast.table.empty")}</p>
       <p class="table-subtitle">${t("forecast.table.closest")}</p>`
    : "";
  $("risk-table").innerHTML = intro + `<div class="risk-table-wrap">${table}</div>` + toggle;
}

/* 严重度不只落在点上，也落到连线上：红线就是拥堵走廊，读起来像交通图。
   节点改成白底描边、热点加一层很轻的光晕，避免 170 个实心点糊成一团。 */
const SEVERITY_ORDER = { normal: 0, watch: 1, congested: 2, severe: 3 };
const EDGE_STYLE = {
  normal: { color: "#c9c8c0", width: 1.2 },
  watch: { color: "#d99a00", width: 1.9 },
  congested: { color: "#e05a22", width: 2.5 },
  severe: { color: "#c62f2f", width: 2.9 },
};
const NODE_STYLE = {
  severe: { size: 12, symbol: "diamond", fill: "#ffffff", line: "#c62f2f", width: 2.2 },
  congested: { size: 11, symbol: "triangle-up", fill: "#ffffff", line: "#e05a22", width: 2.2 },
  watch: { size: 9, symbol: "circle", fill: "#ffffff", line: "#d99a00", width: 2 },
  normal: { size: 5.5, symbol: "circle", fill: "#8f8e86", line: P.surface, width: 1 },
};

function renderSeverityMap(graph, selected) {
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  const edgeSeverity = ([src, dst]) => {
    const a = byId.get(src);
    const b = byId.get(dst);
    return SEVERITY_ORDER[a.severity] >= SEVERITY_ORDER[b.severity] ? a.severity : b.severity;
  };

  const traces = [];

  const edgesBySeverity = { normal: [], watch: [], congested: [], severe: [] };
  graph.edges.forEach(edge => edgesBySeverity[edgeSeverity(edge)].push(edge));
  Object.keys(EDGE_STYLE).forEach(severity => {
    const items = edgesBySeverity[severity];
    if (!items.length) return;
    const x = [];
    const y = [];
    items.forEach(([src, dst]) => {
      x.push(byId.get(src).x, byId.get(dst).x, null);
      y.push(byId.get(src).y, byId.get(dst).y, null);
    });
    traces.push({
      type: "scatter", mode: "lines", x, y,
      line: { ...EDGE_STYLE[severity], shape: "linear" },
      hoverinfo: "skip", showlegend: false,
    });
  });

  /* 两层低透明度大点叠出光晕；只在严重/拥堵点位上，普通点保持干净。 */
  [["severe", 46, 0.07, 28, 0.10], ["congested", 38, 0.06, 24, 0.09]].forEach(
    ([severity, outer, outerAlpha, inner, innerAlpha]) => {
      const items = graph.nodes.filter(node => node.severity === severity);
      if (!items.length) return;
      [[outer, outerAlpha], [inner, innerAlpha]].forEach(([size, opacity]) => {
        traces.push({
          type: "scatter", mode: "markers",
          x: items.map(node => node.x), y: items.map(node => node.y),
          marker: { size, color: SEVERITY[severity].color, opacity, line: { width: 0 } },
          hoverinfo: "skip", showlegend: false,
        });
      });
    }
  );

  /* 绘制顺序 normal→severe（热点压在最上面），图例顺序 severe→normal 用 legendrank 控制。 */
  const legendRank = { severe: 1, congested: 2, watch: 3, normal: 4 };
  ["normal", "watch", "congested", "severe"].forEach(severity => {
    const items = graph.nodes.filter(node => node.severity === severity);
    if (!items.length) return;
    const style = NODE_STYLE[severity];
    traces.push({
      type: "scatter", mode: "markers",
      name: `${t(`severity.${severity}`)} (${items.length})`,
      legendrank: legendRank[severity],
      x: items.map(node => node.x), y: items.map(node => node.y),
      text: items.map(node => node.id),
      /* 第三项是节点号，供 plotly_click 回查；前两项仍供悬浮框使用。 */
      customdata: items.map(node => [
        Number(node.now).toFixed(1), Number(node.worst).toFixed(1), node.id,
      ]),
      marker: {
        size: style.size, symbol: style.symbol, color: style.fill,
        line: { width: style.width, color: style.line },
      },
      hovertemplate: t("forecast.map.hover"), showlegend: true,
    });
  });

  const node = graph.nodes[selected];
  if (node) {
    traces.push({
      type: "scatter", mode: "markers", x: [node.x], y: [node.y],
      marker: { size: 30, color: "rgba(0,0,0,0)", line: { width: 3, color: P.ink } },
      hoverinfo: "skip", showlegend: false,
    });
  }

  /* 视野按实际节点范围取，别再用固定的 ±1.25 把网络缩在中间。 */
  const xs = graph.nodes.map(item => item.x);
  const ys = graph.nodes.map(item => item.y);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const padding = span * 0.06;
  const range = {
    x: [Math.min(...xs) - padding, Math.max(...xs) + padding],
    y: [Math.min(...ys) - padding, Math.max(...ys) + padding],
  };
  state.mapRange = range;

  const plotted = draw("chart-severity-map", traces, {
    /* 固定的 uirevision：切换样本、预见期或选中点位时，用户已放大的视野不被重置。 */
    uirevision: "severity-map",
    dragmode: "pan",
    hovermode: "closest",
    xaxis: { visible: false, range: range.x },
    yaxis: { visible: false, range: range.y, scaleanchor: "x" },
    annotations: node ? [{
      x: node.x, y: node.y, text: `${node.id}`, showarrow: false, yshift: 19,
      font: { size: 12, color: P.ink }, bgcolor: "rgba(252,252,251,0.92)", borderpad: 2,
    }] : [],
    margin: { l: 8, r: 8, t: 8, b: 8 }, showlegend: true,
    legend: { orientation: "h", yanchor: "top", y: 0.99, xanchor: "left", x: 0.01 },
  }, MAP_CONFIG);

  /* Plotly 的点击事件挂在图上；第一次渲染完成后再绑，避免 div.on 还不存在。 */
  plotted.then(bindMapClicks).catch(() => {});
}

let mapClickBound = false;

function bindMapClicks() {
  const plot = $("chart-severity-map");
  if (mapClickBound || !plot || typeof plot.on !== "function") return;
  plot.on("plotly_click", event => {
    const point = event.points && event.points[0];
    const custom = point && point.customdata;
    /* 连线、光晕、选中圈没有 customdata，点它们不触发选中。 */
    if (!custom || custom[2] === undefined) return;
    const node = Number(custom[2]);
    if (Number.isFinite(node)) selectNode(node).catch(report);
  });
  mapClickBound = true;
}

function firstAt(values, minutes, test) {
  const index = values.findIndex(test);
  return index < 0 ? null : minutes[index];
}

async function renderRiskCurve(node) {
  const detail = await api("/forecast/detail", {
    node, sample: state.sample, horizon: state.horizon,
  });
  const thresholds = detail.thresholds;
  const historyX = detail.history.minutes;
  const futureX = detail.forecast.minutes;

  const traces = [
    {
      type: "scatter", mode: "lines", x: futureX, y: detail.band.high,
      line: { width: 0 }, hoverinfo: "skip", showlegend: false,
    },
    {
      type: "scatter", mode: "lines", x: futureX, y: detail.band.low,
      fill: "tonexty", fillcolor: "rgba(42,120,214,0.14)",
      line: { width: 0 }, hoverinfo: "skip", showlegend: false,
    },
    {
      type: "scatter", mode: "lines", name: t("forecast.detail.curveHistory"), showlegend: false,
      x: historyX, y: detail.history.values, line: { width: 2, color: P.muted },
      hovertemplate: "%{x} min<br>%{y:.1f}<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", name: t("forecast.detail.curveForecast"),
      x: futureX, y: detail.forecast.values, line: { width: 2.5, color: P.series[0] },
      hovertemplate: "+%{x} min<br>%{y:.1f}<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", name: t("forecast.detail.curveTruth"),
      x: futureX, y: detail.truth, line: { width: 2, color: P.ink, dash: "dash" },
      hovertemplate: "+%{x} min<br>%{y:.1f}<extra></extra>",
    },
  ];

  const shapes = [
    {
      type: "rect", xref: "x", yref: "paper", x0: 0, x1: detail.horizon.minutes,
      y0: 0, y1: 1, fillcolor: P.grid, opacity: 0.35, line: { width: 0 }, layer: "below",
    },
    {
      type: "line", xref: "paper", yref: "paper", x0: 0.5, x1: 0.5, y0: 0, y1: 1,
      line: { color: P.axis, width: 1, dash: "dot" },
    },
  ];
  const annotations = [];
  Object.entries(thresholds).forEach(([name, value]) => {
    shapes.push({
      type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: value, y1: value,
      line: { color: THRESHOLD_COLOR[name], width: 1, dash: "dot" },
    });
    annotations.push({
      xref: "paper", x: 1, y: value, xanchor: "right", yanchor: "bottom",
      text: Number(value).toFixed(0), showarrow: false,
      font: { size: 10, color: THRESHOLD_COLOR[name] },
    });
  });

  draw("chart-risk-curve", traces, {
    shapes, annotations,
    xaxis: { title: { text: t("forecast.detail.curveX") }, dtick: 15 },
    yaxis: { title: { text: t("forecast.detail.curveY") }, range: [0, 85] },
    margin: { l: 54, r: 24, t: 16, b: 104 },
  });

  const risk = detail.risk;
  $("curve-title").textContent = t("forecast.detail.title", { node: detail.node });
  $("curve-note").textContent = t("forecast.detail.note", {
    minutes: detail.horizon.minutes,
  });

  let verdict;
  if (risk.status === "recovering") {
    const at = firstAt(detail.forecast.values, futureX, value => value >= thresholds.congested);
    verdict = t("forecast.detail.verdict.recovering", {
      speed: thresholds.congested, minutes: at ?? detail.horizon.minutes,
    });
  } else if (risk.severity === "severe") {
    verdict = risk.severe_at_min > 0
      ? t("forecast.detail.verdict.severe", {
        speed: thresholds.severe, minutes: risk.severe_at_min,
      })
      : t("forecast.detail.verdict.now", { speed: thresholds.severe });
  } else if (risk.severity === "congested") {
    verdict = risk.onset_min > 0
      ? t("forecast.detail.verdict.severe", {
        speed: thresholds.congested, minutes: risk.onset_min,
      })
      : t("forecast.detail.verdict.now", { speed: thresholds.congested });
  } else if (risk.severity === "watch") {
    const at = firstAt(detail.forecast.values, futureX, value => value < thresholds.watch);
    verdict = t("forecast.detail.verdict.watch", {
      speed: thresholds.watch, minutes: at ?? detail.horizon.minutes,
    });
  } else {
    verdict = t("forecast.detail.verdict.clear", { speed: thresholds.watch });
  }
  $("curve-verdict").textContent = verdict;
  $("band-note").textContent = t("forecast.detail.bandNote");
  syncRiskSelection();
}

function syncRiskSelection() {
  document.querySelectorAll("#risk-table tr[data-node]").forEach(row => {
    row.setAttribute("aria-selected", String(Number(row.dataset.node) === state.node));
  });
}

async function selectNode(node) {
  state.node = Number(node);
  writeRoute();
  syncRiskSelection();
  if (state.graph) renderSeverityMap(state.graph, state.node);
  await renderRiskCurve(state.node);
}

/* ---------- 路由与交互 ---------- */

function parseRoute() {
  const raw = location.hash.slice(1) || "overview";
  const [name, query] = raw.split("?");
  return {
    view: Object.hasOwn(VIEWS, name) ? name : "overview",
    params: new URLSearchParams(query || ""),
  };
}

function applyRoute(route) {
  state.view = route.view;
  if (route.params.has("sample")) state.sample = Number(route.params.get("sample"));
  if (route.params.has("node")) state.node = Number(route.params.get("node"));
  if (route.params.has("horizon")) {
    const horizon = Number(route.params.get("horizon"));
    if ([15, 30, 60].includes(horizon)) state.horizon = horizon;
  }
  if (route.params.has("camera")) state.camera = route.params.get("camera");
}

function routeFor(view) {
  const params = new URLSearchParams();
  if (view === "forecast") {
    if (state.sample !== null) params.set("sample", state.sample);
    if (state.node !== null) params.set("node", state.node);
    params.set("horizon", state.horizon);
  } else if (view === "monitoring" && state.camera) {
    params.set("camera", state.camera);
  }
  const query = params.toString();
  return `#${view}${query ? "?" + query : ""}`;
}

function writeRoute() {
  const next = routeFor(state.view);
  if (location.hash !== next) history.replaceState(null, "", next);
}

function showView(view) {
  if (!Object.hasOwn(VIEWS, view)) view = "overview";
  state.view = view;
  const meta = VIEWS[view];
  const title = t(meta.title);

  document.querySelectorAll(".view").forEach(node => {
    node.classList.toggle("active", node.id === `view-${view}`);
  });
  document.querySelectorAll(".workspace-nav [data-view]").forEach(link => {
    const on = link.dataset.view === view;
    link.classList.toggle("active", on);
    if (on) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });

  $("page-pretitle").textContent = t(meta.pretitle);
  $("page-title").textContent = title;
  document.title = `${title} · ${t("app.title")}`;
  closeMenu();
  return meta.load().catch(report);
}

function closeMenu() {
  $("app-sidebar").classList.remove("is-open");
  document.querySelector("[data-menu]").setAttribute("aria-expanded", "false");
}

function syncLangButtons() {
  document.querySelectorAll("[data-lang-choice]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.langChoice === getLang()));
  });
}

async function switchLanguage(next) {
  if (next === getLang()) return;
  setLang(next);
  applyStaticI18n();
  syncLangButtons();
  $("toast").hidden = true;
  await Promise.allSettled([refreshChrome(), showView(state.view)]);
}

async function refreshChrome() {
  $("sidebar-endpoint").textContent = location.host;
  $("replay-badge").title = t("header.replayNote");
  try {
    const overview = await api("/overview");
    if ((overview.warnings || []).length) {
      $("sidebar-status").textContent = t("header.verify", { count: overview.warnings.length });
      $("sidebar-status").dataset.state = "warn";
      toast(overview.warnings.join("；"), true);
    } else {
      $("sidebar-status").textContent = t("health.ok");
      $("sidebar-status").dataset.state = "on";
    }
  } catch (error) {
    $("sidebar-status").textContent = t("header.unavailable");
    $("sidebar-status").dataset.state = "err";
    report(error);
  } finally {
    syncLangButtons();
  }
}

let replayTimer = null;

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
    const next = routeFor(link.dataset.view);
    if (location.hash === next) showView(link.dataset.view);
    else location.hash = next;
    return;
  }

  const langButton = event.target.closest("[data-lang-choice]");
  if (langButton) {
    await switchLanguage(langButton.dataset.langChoice);
    return;
  }

  const cameraButton = event.target.closest("[data-camera]");
  if (cameraButton) {
    state.camera = cameraButton.dataset.camera;
    await loadMonitoring().catch(report);
    return;
  }

  const horizonButton = event.target.closest("[data-horizon]");
  if (horizonButton) {
    const horizon = Number(horizonButton.dataset.horizon);
    if (horizon !== state.horizon) {
      state.horizon = horizon;
      await loadForecast().catch(report);
    }
    return;
  }

  const goto = event.target.closest("[data-goto-view]");
  if (goto) {
    if (goto.dataset.gotoView === "forecast") {
      if (goto.dataset.gotoNode) state.node = Number(goto.dataset.gotoNode);
      if (goto.dataset.gotoSample) state.sample = Number(goto.dataset.gotoSample);
    } else if (goto.dataset.gotoView === "monitoring" && goto.dataset.gotoCamera) {
      state.camera = goto.dataset.gotoCamera;
    }
    location.hash = routeFor(goto.dataset.gotoView);
    return;
  }

  const riskRow = event.target.closest("#risk-table tr[data-node]");
  if (riskRow) {
    await selectNode(riskRow.dataset.node).catch(report);
    return;
  }

  const button = event.target.closest("[data-command]");
  if (!button) return;
  const command = button.dataset.command;
  try {
    if (command === "refreshActiveView") await VIEWS[state.view].load();
    else if (command === "replayBack") await stepReplay(-1);
    else if (command === "replayForward") await stepReplay(1);
    else if (command === "replayLatest") await jumpReplay(await ensureReplayRange());
    else if (command === "resetMap") {
      const range = state.mapRange || { x: [-1.25, 1.25], y: [-1.25, 1.25] };
      Plotly.relayout("chart-severity-map", {
        "xaxis.range": range.x, "yaxis.range": range.y,
      });
    } else if (command === "toggleAllRisk") {
      state.showAll = !state.showAll;
      renderRiskTable(state.risk);
    }
  } catch (error) {
    report(error);
  }
});

async function stepReplay(offset) {
  const range = await ensureReplayRange();
  const next = Math.max(range.first, Math.min(range.last, (state.sample ?? range.last) + offset));
  state.sample = next;
  await loadForecast();
}

async function jumpReplay(range) {
  state.sample = range.last;
  await loadForecast();
}

document.addEventListener("keydown", async event => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("#risk-table tr[data-node]");
  if (!row) return;
  event.preventDefault();
  await selectNode(row.dataset.node).catch(report);
});

$("replay-range").addEventListener("input", event => {
  state.sample = Number(event.target.value);
  clearTimeout(replayTimer);
  replayTimer = setTimeout(() => loadForecast().catch(report), 220);
});

$("replay-range").addEventListener("change", () => {
  clearTimeout(replayTimer);
  loadForecast().catch(report);
});

$("frame-range").addEventListener("input", () => {
  if (!cameraSeries) return;
  showFrame(Number($("frame-range").value)).catch(report);
});

$("frame-time").addEventListener("change", () => {
  const value = $("frame-time").value;
  if (value) showFrameAtTime(value).catch(report);
});

window.addEventListener("hashchange", () => {
  const route = parseRoute();
  applyRoute(route);
  showView(route.view);
});

(async function start() {
  const route = parseRoute();
  applyRoute(route);
  try {
    applyStaticI18n();
    syncLangButtons();
    showView(route.view);
    await refreshChrome();
  } finally {
    document.documentElement.classList.remove("lang-booting");
  }
})();
