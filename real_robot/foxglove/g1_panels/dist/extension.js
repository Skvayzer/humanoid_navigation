"use strict";
// G1 whole-body panels for Foxglove (plain JavaScript, no build step; see ../README.md).

const OPS = [
  ["stand", "#d08000", "IDLE -> STAND (remote L1+A)"],
  ["balance", "#1f9d3a", "STAND -> policy, goal = here (L1+X)"],
  ["go", "#1f9d3a", "walk to the goal (R1+A)"],
  ["stop", "#d08000", "goal = here (R1+B)"],
  ["damp", "#e02020", "DAMP (L2+B on the remote is faster)"],
  ["reset", "#707070", "DAMP -> IDLE (L1+Y)"],
];
const OBSTACLES = [["virtual", "#e0701a"], ["real", "#606060"], ["both", "#2f6fdf"]];
const TOPICS = { operator: "/g1_wb/operator", obstacles: "/g1_wb/obstacle_mode", scene: "/g1_wb/scene", speed: "/g1_wb/speed" };

function theme(scheme) {
  const dark = scheme !== "light";
  return {
    bg: dark ? "#15151a" : "#fcfcfb", fg: dark ? "#e8e7e2" : "#2b2b29", muted: dark ? "#9a998f" : "#76756e",
    track: dark ? "#3a3a40" : "#d9d8d2", button: dark ? "#26262c" : "#ffffff", border: dark ? "#3a3a40" : "#d9d8d2",
  };
}

function el(tag, style, text) {
  const e = document.createElement(tag);
  if (style) Object.assign(e.style, style);
  if (text != null) e.textContent = text;
  return e;
}

function variable(renderState, name) {
  const v = renderState.variables;
  if (!v) return undefined;
  return typeof v.get === "function" ? v.get(name) : v[name];
}

// ---------------------------------------------------------------------------------------------------- ESDF height
function initEsdf(context) {
  const state = Object.assign({ variable: "esdf_z", min: 0.02, max: 1.5, step: 0.04 }, context.initialState || {});
  const root = context.panelElement;
  Object.assign(root.style, { display: "flex", alignItems: "center", gap: "10px", padding: "4px 10px", height: "100%",
    boxSizing: "border-box", fontFamily: "system-ui, sans-serif", fontSize: "12px", userSelect: "none", outline: "none" });
  root.tabIndex = 0;
  const label = el("div", { whiteSpace: "nowrap", minWidth: "118px", fontVariantNumeric: "tabular-nums" });
  const slider = el("input", { flex: "1", minWidth: "80px", cursor: "ns-resize" });
  Object.assign(slider, { type: "range", min: String(state.min), max: String(state.max), step: "0.005" });
  const hint = el("div", { whiteSpace: "nowrap" }, "wheel / ↑↓: one layer");
  root.append(label, slider, hint);
  let value = 0.74, local = 0, colors = theme("dark");

  const show = () => { label.textContent = "ESDF slice z = " + value.toFixed(2) + " m"; slider.value = String(value); };
  const set = (v) => {
    value = Math.min(state.max, Math.max(state.min, v));
    local = Date.now(); show();
    context.setVariable(state.variable, Number(value.toFixed(3)));
  };
  slider.addEventListener("input", () => set(parseFloat(slider.value)));          // every move, not on release
  root.addEventListener("wheel", (e) => { e.preventDefault(); set(value + (e.deltaY < 0 ? 1 : -1) * state.step); }, { passive: false });
  root.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); set(value + state.step); }
    if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); set(value - state.step); }
    if (e.key === "PageUp") { e.preventDefault(); set(value + 5 * state.step); }
    if (e.key === "PageDown") { e.preventDefault(); set(value - 5 * state.step); }
  });
  context.watch("variables");
  context.watch("colorScheme");
  context.onRender = (renderState, done) => {
    colors = theme(renderState.colorScheme);
    root.style.background = colors.bg; root.style.color = colors.fg; hint.style.color = colors.muted;
    const v = Number(variable(renderState, state.variable));
    if (Number.isFinite(v) && Date.now() - local > 400 && Math.abs(v - value) > 1e-6) { value = v; }  // another panel moved it
    show(); done();
  };
  show();
  return () => {};
}

// ---------------------------------------------------------------------------------------------------- G1 controls
const SVG = "http://www.w3.org/2000/svg";
function svg(tag, attrs) { const e = document.createElementNS(SVG, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }

// floor obstacles (stepped over / walked round) vs overhead ones (ducked under): the bottom of the box decides
function boxColor(b) { return b.c[2] - b.h[2] > 0.3 ? "#3d7be0" : b.h[2] < 0.15 ? "#e0a020" : "#9a9a9a"; }

function scenePreview(scene, w, h) {
  const box = svg("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h });
  box.style.overflow = "hidden";                                     // the start -> goal strip, +-1.1 m across; walls clipped
  const x0 = Math.min(0, scene.goal[0]) - 0.4, x1 = Math.max(0, scene.goal[0]) + 0.4;
  const y0 = Math.min(-1.1, scene.goal[1] - 0.3), y1 = Math.max(1.1, scene.goal[1] + 0.3);
  const k = Math.min(w / (x1 - x0), h / (y1 - y0));
  const ox = (w - k * (x1 - x0)) / 2, oy = (h - k * (y1 - y0)) / 2;
  const X = (x) => ox + k * (x - x0), Y = (y) => h - (oy + k * (y - y0));     // +x right, +y up (robot's left)
  for (const b of scene.boxes) {
    const g = svg("rect", { x: -k * b.h[0], y: -k * b.h[1], width: Math.max(2 * k * b.h[0], 1.5), height: Math.max(2 * k * b.h[1], 1.5),
      fill: boxColor(b), transform: "translate(" + X(b.c[0]) + "," + Y(b.c[1]) + ") rotate(" + (-(b.yaw || 0) * 180 / Math.PI) + ")" });
    box.append(g);
  }
  box.append(svg("line", { x1: X(0), y1: Y(0), x2: X(scene.goal[0]), y2: Y(scene.goal[1]), stroke: "#888", "stroke-width": 1, "stroke-dasharray": "2 2" }));
  box.append(svg("circle", { cx: X(0), cy: Y(0), r: 3, fill: "#1f9d3a" }));
  box.append(svg("circle", { cx: X(scene.goal[0]), cy: Y(scene.goal[1]), r: 3, fill: "#e02020" }));
  return box;
}

function initControls(context) {
  const state = Object.assign({ scenes: [], speedMin: 0.1, speedMax: 0.6, speedStep: 0.05 }, context.initialState || {});
  const scenes = state.scenes.map((s) => (typeof s === "string" ? { name: s, boxes: null } : s));
  const root = context.panelElement;
  Object.assign(root.style, { display: "flex", flexDirection: "column", gap: "6px", padding: "6px 8px", height: "100%",
    boxSizing: "border-box", overflowY: "auto", fontFamily: "system-ui, sans-serif", fontSize: "12px" });
  // the String type as the connection names it: "std_msgs/msg/String" (foxglove_bridge, ROS 2) or "std_msgs/String"
  // (rosbridge): taken from the executor's /g1_wb/status
  const advertised = new Map();
  let stringType = "std_msgs/String";
  const publish = (topic, data) => {
    if (!context.publish) return;
    if (advertised.get(topic) !== stringType) {
      if (advertised.has(topic) && context.unadvertise) context.unadvertise(topic);
      context.advertise && context.advertise(topic, stringType); advertised.set(topic, stringType);
    }
    context.publish(topic, { data: String(data) });
  };
  let colors = theme("dark");
  const button = (text, color, tip, onClick, extra) => {
    const b = el("button", Object.assign({ flex: "1 1 auto", minWidth: "58px", padding: "7px 6px", borderRadius: "4px", cursor: "pointer",
      fontSize: "12px", fontWeight: "600", border: "none", background: color, color: "#ffffff" }, extra || {}), text);
    b.title = tip || "";
    b.addEventListener("click", onClick);
    b.addEventListener("mousedown", () => { b.style.filter = "brightness(0.8)"; });
    ["mouseup", "mouseleave"].forEach((ev) => b.addEventListener(ev, () => { b.style.filter = ""; }));
    return b;
  };
  const muted = [];
  const row = (title, items) => {
    const r = el("div", { display: "flex", alignItems: "center", gap: "4px", flexWrap: "wrap" });
    const t = el("div", { width: "64px", flex: "0 0 64px", fontSize: "11px" }, title);
    muted.push(t); r.append(t, ...items); return r;
  };

  const status = el("div", { fontVariantNumeric: "tabular-nums", minHeight: "16px" }, "no status yet");
  const headWarn = el("div", { display: "none", padding: "3px 6px", borderRadius: "4px", background: "#d08000", color: "#ffffff",
    fontWeight: "600" });                               // the executor's head/LiDAR mount watchdog
  root.append(status, headWarn);

  // operator + obstacles on the left, one big DAMP on the right
  const top = el("div", { display: "flex", gap: "8px", alignItems: "stretch" });
  const left = el("div", { display: "flex", flexDirection: "column", gap: "6px", flex: "1 1 auto", minWidth: "0" });
  left.append(row("operator", OPS.filter(([n]) => n !== "damp").map(([n, c, tip]) => button(n, c, tip, () => publish(TOPICS.operator, n)))));
  const obstacleButtons = {};
  left.append(row("obstacles", OBSTACLES.map(([n, c]) => (obstacleButtons[n] = button(n, c, "what the fields are made of (only while standing)",
    () => publish(TOPICS.obstacles, n))))));
  const damp = button("DAMP", "#e02020", "DAMP from any state (L2+B on the remote is faster)", () => publish(TOPICS.operator, "damp"),
    { flex: "0 0 84px", width: "84px", minHeight: "84px", fontSize: "18px", fontWeight: "800", letterSpacing: "1px", borderRadius: "8px",
      boxShadow: "0 0 0 2px #7a0000 inset" });
  top.append(left, damp);
  root.append(top);

  // operator speed: published while dragging (at most every 150 ms) and on release
  const speedLabel = el("div", { width: "64px", flex: "0 0 64px", fontSize: "11px" }, "speed");
  muted.push(speedLabel);
  const speed = el("input", { flex: "1", minWidth: "80px" });
  Object.assign(speed, { type: "range", min: String(state.speedMin), max: String(state.speedMax), step: String(state.speedStep), value: "0.15" });
  const speedValue = el("div", { width: "86px", fontVariantNumeric: "tabular-nums" });
  let lastSent = 0, pending = null, speedTouched = 0;
  const sendSpeed = () => { pending = null; lastSent = Date.now(); publish(TOPICS.speed, Number(speed.value).toFixed(2)); };
  speed.addEventListener("input", () => {
    speedTouched = Date.now(); speedValue.textContent = Number(speed.value).toFixed(2) + " m/s";
    if (Date.now() - lastSent > 150) sendSpeed(); else if (!pending) pending = setTimeout(sendSpeed, 150);
  });
  speed.addEventListener("change", sendSpeed);
  const speedRow = el("div", { display: "flex", alignItems: "center", gap: "6px" });
  speedRow.append(speedLabel, speed, speedValue);
  root.append(speedRow);

  // scenes: a drop-down (top-view previews: green start, red goal; amber = step over, blue = duck under, grey = walls);
  // picking one lays it out in front of the robot (the executor accepts it only while standing)
  const sceneItems = {};
  let dropOpen = false;
  const sceneRow = (sc, compact) => {
    const r = el("div", { display: "flex", alignItems: "center", gap: "8px", minWidth: "0", flex: "1 1 auto" });
    if (sc.boxes) r.append(scenePreview(sc, compact ? 72 : 96, compact ? 26 : 34));
    const txt = el("div", { display: "flex", flexDirection: "column", minWidth: "0" });
    txt.append(el("div", { fontWeight: "600", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, sc.name));
    if (sc.note) { const n = el("div", { fontSize: "11px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, sc.note); n.dataset.muted = "1"; txt.append(n); }
    r.append(txt); return r;
  };
  const drop = el("div", { position: "relative" });
  const head = el("div", { display: "flex", alignItems: "center", gap: "6px", padding: "4px 8px", borderRadius: "4px", cursor: "pointer",
    border: "1px solid", userSelect: "none" });
  const headBody = el("div", { flex: "1 1 auto", minWidth: "0", display: "flex" });
  const caret = el("div", { fontSize: "14px" }, "▾");
  head.append(el("div", { width: "56px", flex: "0 0 56px", fontSize: "11px" }, "scene"), headBody, caret);
  head.firstChild.dataset.muted = "1";
  const menu = el("div", { position: "absolute", left: "0", right: "0", top: "100%", marginTop: "2px", zIndex: "10", display: "none",
    flexDirection: "column", gap: "1px", padding: "3px", borderRadius: "4px", border: "1px solid", maxHeight: "60vh", overflowY: "auto",
    boxShadow: "0 4px 14px rgba(0,0,0,.45)" });
  const setOpen = (v) => { dropOpen = v; menu.style.display = v ? "flex" : "none"; caret.textContent = v ? "▴" : "▾"; };
  head.addEventListener("click", () => setOpen(!dropOpen));
  for (const sc of scenes) {
    const item = el("div", { display: "flex", padding: "3px 6px", borderRadius: "3px", cursor: "pointer" });
    item.title = sc.note || sc.name;
    item.append(sceneRow(sc, false));
    item.addEventListener("click", () => { setOpen(false); publish(TOPICS.scene, sc.name); });
    item.addEventListener("mouseenter", () => { item.dataset.hover = "1"; paint(); });
    item.addEventListener("mouseleave", () => { delete item.dataset.hover; paint(); });
    sceneItems[sc.name] = item; menu.append(item);
  }
  document.addEventListener("mousedown", (e) => { if (dropOpen && !drop.contains(e.target)) setOpen(false); });
  let shownScene;
  const showHead = (name) => {
    if (name === shownScene) return; shownScene = name;
    const sc = scenes.find((x) => x.name === name);
    headBody.replaceChildren(sc ? sceneRow(sc, true) : el("div", {}, name || "(none)"));
  };
  if (scenes.length) { drop.append(head, menu); root.append(drop); showHead(null); }

  let current = { scene: null, obstacles: null };
  const paint = () => {
    root.style.background = colors.bg; root.style.color = colors.fg;
    if (scenes.length) showHead(current.scene);
    for (const n of muted) n.style.color = colors.muted;
    root.querySelectorAll("[data-muted]").forEach((n) => { n.style.color = colors.muted; });
    if (scenes.length) {
      showHead(current.scene);
      Object.assign(head.style, { background: colors.button, borderColor: colors.border });
      Object.assign(menu.style, { background: colors.bg, borderColor: colors.border });
      for (const [name, item] of Object.entries(sceneItems)) {
        item.style.background = item.dataset.hover ? colors.track : name === current.scene ? colors.button : "transparent";
        item.style.boxShadow = name === current.scene ? "inset 3px 0 0 #e0701a" : "none";
      }
    }
    for (const [name, b] of Object.entries(obstacleButtons)) b.style.outline = name === current.obstacles ? "2px solid " + colors.fg : "none";
  };

  context.subscribe([{ topic: "/g1_wb/status" }]);
  context.watch("currentFrame");
  context.watch("colorScheme");
  context.watch("topics");
  context.onRender = (renderState, done) => {
    colors = theme(renderState.colorScheme);
    const st = (renderState.topics || []).find((t) => t.name === "/g1_wb/status");
    if (st && st.schemaName) stringType = st.schemaName.includes("/msg/") ? "std_msgs/msg/String" : "std_msgs/String";
    const frame = renderState.currentFrame || [];
    const last = frame.length ? frame[frame.length - 1] : null;
    if (last) {
      try {
        const s = JSON.parse(last.message.data);
        current = { scene: s.scene, obstacles: s.obstacles };
        const latch = s.latch == null ? "-" : s.latch > 50 ? "walking" : s.latch > 0 ? "stopping" : "standing";
        const range = s.speed_range ? s.speed_range[0] + "-" + s.speed_range[1] : null;
        status.textContent = "mode " + s.mode + "  ·  " + latch +
          "  ·  speed " + (s.speed == null ? "field (fixed)" : Number(s.speed).toFixed(2) + " m/s" + (range ? " (trained " + range + ")" : "")) +
          (s.reason ? "  ·  " + s.reason : "");
        headWarn.textContent = s.head_warning ? "⚠ " + s.head_warning : "";
        headWarn.style.display = s.head_warning ? "block" : "none";
        speed.disabled = s.speed == null;
        if (s.speed != null && Date.now() - speedTouched > 1500) {       // follow the executor unless the operator is dragging
          speed.value = String(s.speed); speedValue.textContent = Number(s.speed).toFixed(2) + " m/s";
        }
        if (s.speed == null) speedValue.textContent = "n/a";
      } catch (e) { /* not JSON yet */ }
    }
    paint(); done();
  };
  paint();
  return () => { if (pending) clearTimeout(pending); };
}

module.exports.activate = function activate(extensionContext) {
  extensionContext.registerPanel({ name: "ESDF height", initPanel: initEsdf });
  extensionContext.registerPanel({ name: "G1 controls", initPanel: initControls });
};
