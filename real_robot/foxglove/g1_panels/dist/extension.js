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
function initControls(context) {
  const state = Object.assign({ scenes: [], speedMin: 0.1, speedMax: 0.6, speedStep: 0.05 }, context.initialState || {});
  const root = context.panelElement;
  Object.assign(root.style, { display: "flex", flexDirection: "column", gap: "6px", padding: "6px 8px", height: "100%",
    boxSizing: "border-box", overflowY: "auto", fontFamily: "system-ui, sans-serif", fontSize: "12px" });
  const advertised = new Set();
  const publish = (topic, data) => {
    if (!context.publish) return;
    if (!advertised.has(topic)) { context.advertise && context.advertise(topic, "std_msgs/String"); advertised.add(topic); }
    context.publish(topic, { data: String(data) });
  };
  let colors = theme("dark");
  const buttons = [];
  const button = (text, color, tip, onClick) => {
    const b = el("button", { flex: "1 1 auto", minWidth: "58px", padding: "6px 6px", borderRadius: "4px", cursor: "pointer",
      fontSize: "12px", fontWeight: "600", border: "1px solid", borderLeft: "4px solid " + color }, text);
    b.title = tip || "";
    b.addEventListener("click", onClick);
    buttons.push(b);
    return b;
  };
  const row = (title, items) => {
    const r = el("div", { display: "flex", alignItems: "center", gap: "4px", flexWrap: "wrap" });
    const t = el("div", { width: "64px", flex: "0 0 64px", fontSize: "11px" }, title);
    t.dataset.muted = "1"; r.append(t, ...items); return r;
  };

  const status = el("div", { fontVariantNumeric: "tabular-nums", minHeight: "16px" }, "no status yet");
  root.append(status);
  root.append(row("operator", OPS.map(([n, c, tip]) => button(n, c, tip, () => publish(TOPICS.operator, n)))));
  root.append(row("obstacles", OBSTACLES.map(([n, c]) => button(n, c, "what the fields are made of (only while standing)",
    () => publish(TOPICS.obstacles, n)))));

  // operator speed: published while dragging (at most every 150 ms) and on release
  const speedLabel = el("div", { width: "64px", flex: "0 0 64px", fontSize: "11px" }, "speed");
  speedLabel.dataset.muted = "1";
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

  if (state.scenes.length) {
    root.append(row("scenes", state.scenes.map((s) => button(s, "#e0701a", "lay this scene out in front of the robot (only while standing)",
      () => publish(TOPICS.scene, s)))));
  }

  context.subscribe([{ topic: "/g1_wb/status" }]);
  context.watch("currentFrame");
  context.watch("colorScheme");
  context.onRender = (renderState, done) => {
    colors = theme(renderState.colorScheme);
    root.style.background = colors.bg; root.style.color = colors.fg;
    for (const b of buttons) Object.assign(b.style, { background: colors.button, color: colors.fg, borderColor: colors.border });
    root.querySelectorAll("[data-muted]").forEach((n) => { n.style.color = colors.muted; });
    const frame = renderState.currentFrame || [];
    const last = frame.length ? frame[frame.length - 1] : null;
    if (last) {
      try {
        const s = JSON.parse(last.message.data);
        const latch = s.latch == null ? "-" : s.latch > 50 ? "walking" : s.latch > 0 ? "stopping" : "standing";
        const range = s.speed_range ? s.speed_range[0] + "-" + s.speed_range[1] : null;
        status.textContent = "mode " + s.mode + "  ·  " + latch +
          "  ·  speed " + (s.speed == null ? "field (fixed)" : Number(s.speed).toFixed(2) + " m/s" + (range ? " (trained " + range + ")" : "")) +
          (s.reason ? "  ·  " + s.reason : "");
        speed.disabled = s.speed == null;
        if (s.speed != null && Date.now() - speedTouched > 1500) {       // follow the executor unless the operator is dragging
          speed.value = String(s.speed); speedValue.textContent = Number(s.speed).toFixed(2) + " m/s";
        }
        if (s.speed == null) speedValue.textContent = "n/a";
      } catch (e) { /* not JSON yet */ }
    }
    done();
  };
  return () => { if (pending) clearTimeout(pending); };
}

module.exports.activate = function activate(extensionContext) {
  extensionContext.registerPanel({ name: "ESDF height", initPanel: initEsdf });
  extensionContext.registerPanel({ name: "G1 controls", initPanel: initControls });
};
