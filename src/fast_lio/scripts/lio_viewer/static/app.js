// LIO map debugger: three.js front end of scripts/lio_viewer/server.py (see its docstring).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const STATUS_NAMES = ['matched', 'no neighbours', 'far neighbours', 'plane fail', 'residual fail'];
const COL = {
  matchedVoxel: 0x4ade80, matchedNN: 0x3b82f6, noNbr: 0x9ca3af, far: 0x8b5cf6, planeFail: 0xf97316, resFail: 0xef4444,
  voxValid: 0x10b981, voxFew: 0x6b7280, voxThick: 0xd97706, voxLine: 0xdb2777,
  raw: 0x4b5563, prior: 0x9ca3af, start: 0xfacc15, posterior: 0x22c55e, reference: 0xffffff, nnLink: 0x60a5fa, selected: 0xffffff,
};
const VOX_CLASS_COL = [COL.voxValid, COL.voxFew, COL.voxThick, COL.voxLine];
const VOX_CLASS_NAMES = ['one surface', 'not resolved', 'too thick', 'not planar'];
const POINT_STRIDE = 32;
let VOXEL_STRIDE = 33, PATCH_STRIDE = 7;   // as the session's replies give them

const S = {
  info: null, scan: null, scanData: null, map: null, poses: null, update: null, reference: null,
  iter: 0, selected: -1, busy: false, pendingRun: false, firstView: true, hoverVoxel: -1,
};

// ------------------------------------------------------------------------------------------ transport
async function getJSON(url, body) {
  const r = await fetch(url, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
async function getFramed(url, body) {
  const r = await fetch(url, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  const buf = await r.arrayBuffer();
  const n = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { head, payload: buf.slice(4 + n) };
}

function setStatus(text, cls = '') {
  const el = $('status');
  el.textContent = text;
  el.className = cls;
}
function showError(e) {
  console.error(e);
  setStatus('Error: ' + (e.message || e), 'error');
}

// ------------------------------------------------------------------------------------------ maths
const toVec = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const toQuat = (q) => new THREE.Quaternion(q[0], q[1], q[2], q[3]);
// lidar -> world for a state {pos, quat}: T(pos) R(quat) T(t_il) R(q_il)
function lidarToWorld(state) {
  const ext = S.scan.extrinsic;
  const m = new THREE.Matrix4().compose(toVec(state.pos), toQuat(state.quat), new THREE.Vector3(1, 1, 1));
  const me = new THREE.Matrix4().compose(toVec(ext.t), toQuat(ext.quat), new THREE.Vector3(1, 1, 1));
  return m.multiply(me);
}
const posDelta = (a, b) => toVec(a.pos).distanceTo(toVec(b.pos));
const rotDelta = (a, b) => toQuat(a.quat).angleTo(toQuat(b.quat));
const cm = (m) => (m * 100).toFixed(2);
const deg = (r) => (r * 180 / Math.PI).toFixed(3);
// Height cut: everything above the lidar + zcut m is hidden (ceilings hide indoor scenes from above)
function zMax() { return $('zcut').checked && S.scan ? S.scan.lidar_pos[2] + (parseFloat($('zcutv').value) || 0) : Infinity; }
function rampColor(t) { // 0..1 -> blue..green..yellow..red
  t = Math.min(1, Math.max(0, t));
  return new THREE.Color().setHSL((1 - t) * 0.66, 0.85, 0.5);
}

// ------------------------------------------------------------------------------------------ scene
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 3000);
camera.up.set(0, 0, 1);
camera.position.set(-8, -8, 6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
scene.add(new THREE.AmbientLight(0xffffff, 1));
const grid = new THREE.GridHelper(200, 200, 0x2a3140, 0x1c2129);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const layers = {};
function setLayer(name, obj) {
  const old = layers[name];
  if (old) {
    scene.remove(old);
    old.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose()); else o.material.dispose(); } });
  }
  layers[name] = obj || null;
  if (obj) scene.add(obj);
}
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);
resize();
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

function recenter(top = false) {
  if (!S.scan) return;
  const c = toVec(S.scan.lidar_pos);
  controls.target.copy(c);
  if (top) camera.position.set(c.x, c.y - 0.01, c.z + 25);
  else camera.position.set(c.x - 5, c.y - 7, c.z + 7);
  controls.update();
}

// ------------------------------------------------------------------------------------------ map layers
function voxelClass(k) {
  // 0 one surface, 1 not resolved (matched through the pooled neighbourhood), 2 too thick, 3 not planar
  const v = S.map.vox, o = k * VOXEL_STRIDE, p = S.scan.params.voxel_map;
  if (S.map.valid[k] > 0) return 0;
  if (!v[o + 32]) return 1;   // not resolved: matched through the pooled neighbourhood only
  if (v[o + 18] > p.pooled_thickness * p.pooled_thickness) return 2;
  return 3;
}
function buildMapLayers() {
  if (!S.map) return;
  const m = S.map;
  // ikd-Tree points, coloured by height
  {
    const n = m.nIkd, zc = zMax();
    const zs = [];
    for (let i = 0; i < n; i++) zs.push(m.ikd[i * 4 + 2]);
    zs.sort((a, b) => a - b);
    const z0 = zs[Math.floor(zs.length * 0.02)] ?? 0, z1 = zs[Math.floor(zs.length * 0.98)] ?? 1;
    const pos = [], col = [], c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const z = m.ikd[i * 4 + 2];
      if (z > zc) continue;
      pos.push(m.ikd[i * 4], m.ikd[i * 4 + 1], z);
      c.setHSL(0.66 - 0.6 * Math.min(1, Math.max(0, (z - z0) / (z1 - z0 || 1))), 0.5, 0.42);
      col.push(c.r, c.g, c.b);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 3));
    setLayer('ikd', new THREE.Points(g, new THREE.PointsMaterial({ size: 0.04, vertexColors: true, sizeAttenuation: true })));
  }
  buildCells();
  buildVoxelLayers();
  applyVisibility();
}
function buildCells() {
  if (!S.map) return;
  const m = S.map, s = S.scan.params.filter_size_map, r = parseFloat($('cellRadius').value) || 6;
  const c = toVec(S.scan.lidar_pos), zc = zMax();
  const keys = new Set(), cells = [];
  for (let i = 0; i < m.nIkd; i++) {
    const x = m.ikd[i * 4], y = m.ikd[i * 4 + 1], z = m.ikd[i * 4 + 2];
    if (z > zc || (x - c.x) ** 2 + (y - c.y) ** 2 + (z - c.z) ** 2 > r * r) continue;
    const ix = Math.floor(x / s), iy = Math.floor(y / s), iz = Math.floor(z / s);
    const k = `${ix},${iy},${iz}`;
    if (!keys.has(k)) { keys.add(k); cells.push([ix, iy, iz]); }
  }
  const E = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const pos = new Float32Array(cells.length * 24 * 3);
  let o = 0;
  for (const [ix, iy, iz] of cells) {
    const corner = (q) => [(ix + (q & 1)) * s, (iy + ((q >> 1) & 1)) * s, (iz + ((q >> 2) & 1)) * s];
    for (const [a, b] of E) { const A = corner(a), B = corner(b); pos.set(A, o); pos.set(B, o + 3); o += 6; }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  setLayer('cells', new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.35 })));
  layers.cells.userData.count = cells.length;
}
function buildVoxelLayers() {
  if (!S.map) return;
  const m = S.map, v = m.vox, size = S.scan.params.voxel_map.size;
  const show = [$('C_valid').checked, $('C_few').checked, $('C_thick').checked, $('C_line').checked];
  const opacity = parseFloat($('voxOpacity').value);
  const cls = new Int8Array(m.nVox), zc = zMax(), shown = new Uint8Array(m.nVox);
  let nEll = 0, nPl = 0;
  for (let k = 0; k < m.nVox; k++) {
    cls[k] = voxelClass(k);
    shown[k] = show[cls[k]] && v[k * VOXEL_STRIDE + 7] <= zc ? 1 : 0;
    if (shown[k]) { nEll++; if (cls[k] === 0) nPl++; }
  }
  m.cls = cls;
  // distributions: 2 sigma ellipsoids
  const ell = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 6), new THREE.MeshBasicMaterial({ transparent: true, opacity, depthWrite: false }), Math.max(nEll, 1));
  const pl = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.75 * size, 0.75 * size), new THREE.MeshBasicMaterial({ transparent: true, opacity: Math.min(1, opacity + 0.25), depthWrite: false, side: THREE.DoubleSide }), Math.max(nPl, 1));
  const ellIdx = new Int32Array(nEll), plIdx = new Int32Array(nPl);
  const M = new THREE.Matrix4(), q = new THREE.Quaternion(), e0 = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), pos = new THREE.Vector3(), sc = new THREE.Vector3();
  const rot = new THREE.Matrix4(), up = new THREE.Vector3(0, 0, 1), nrm = new THREE.Vector3(), color = new THREE.Color();
  let ie = 0, ip = 0;
  const cubePos = [];
  const E = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (let k = 0; k < m.nVox; k++) {
    if (!shown[k]) continue;
    const o = k * VOXEL_STRIDE;
    pos.set(v[o + 5], v[o + 6], v[o + 7]);
    e0.set(v[o + 21], v[o + 22], v[o + 23]); e1.set(v[o + 24], v[o + 25], v[o + 26]); e2.set(v[o + 27], v[o + 28], v[o + 29]);
    if (e0.clone().cross(e1).dot(e2) < 0) e2.negate();
    rot.makeBasis(e0, e1, e2);
    q.setFromRotationMatrix(rot);
    sc.set(Math.max(0.01, 2 * Math.sqrt(Math.max(0, v[o + 18]))), Math.max(0.01, 2 * Math.sqrt(Math.max(0, v[o + 19]))), Math.max(0.01, 2 * Math.sqrt(Math.max(0, v[o + 20]))));
    // the ellipsoid axes are the eigenvectors: x along e0 (normal, smallest), y e1, z e2
    M.compose(pos, q, sc);
    ell.setMatrixAt(ie, M);
    color.setHex(VOX_CLASS_COL[cls[k]]);
    if (v[o + 31] > 0 && cls[k] !== 0) color.multiplyScalar(0.8);
    ell.setColorAt(ie, color);
    ellIdx[ie++] = k;
    if (cls[k] === 0) {
      nrm.set(v[o + 14], v[o + 15], v[o + 16]);
      q.setFromUnitVectors(up, nrm);
      M.compose(pos, q, new THREE.Vector3(1, 1, 1));
      pl.setMatrixAt(ip, M);
      pl.setColorAt(ip, color.setHex(COL.voxValid));
      plIdx[ip++] = k;
    }
    if (cubePos.length < 60000 * 24 * 3) {
      const ix = v[o], iy = v[o + 1], iz = v[o + 2];
      const corner = (c) => [(ix + (c & 1)) * size, (iy + ((c >> 1) & 1)) * size, (iz + ((c >> 2) & 1)) * size];
      for (const [a, b] of E) cubePos.push(...corner(a), ...corner(b));
    }
  }
  ell.count = nEll; pl.count = nPl;
  ell.instanceMatrix.needsUpdate = true; if (ell.instanceColor) ell.instanceColor.needsUpdate = true;
  pl.instanceMatrix.needsUpdate = true; if (pl.instanceColor) pl.instanceColor.needsUpdate = true;
  ell.userData.voxIdx = ellIdx; pl.userData.voxIdx = plIdx;
  setLayer('ell', ell);
  setLayer('planes', pl);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cubePos), 3));
  setLayer('cubes', new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.5 })));
  applyVisibility();
  updateLegend();
}

// ------------------------------------------------------------------------------------------ scan layers
function iterCount() { return S.update ? S.update.iters : 0; }
function recordOf(i, it) { // per-point record of iteration it (clamped to the last recorded one)
  const u = S.update, k = Math.min(it, u.iters - 1);
  return (k * u.points + i) * POINT_STRIDE;
}
function stateAt(it) {
  const u = S.update;
  return it >= u.iters ? u.head.final : u.head.iterations[it];
}
// world point i at the shown iteration: the recorded one, or the lidar point moved by the final state
function worldPoint(i, out) {
  const u = S.update, d = S.scanData.down;
  if (S.iter < u.iters) { const o = recordOf(i, S.iter); out.set(u.rec[o + 4], u.rec[o + 5], u.rec[o + 6]); }
  else out.set(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]).applyMatrix4(S.finalMatrix);
  return out;
}
function pointColor(i, out) {
  const u = S.update, o = recordOf(i, S.iter), r = u.rec, mode = $('colorMode').value;
  if (mode === 'residual') {
    const st = r[o];
    if (st !== 0 && st !== 4) return out.setHex(0x4b5563);
    return out.copy(rampColor(Math.abs(residualAt(i)) / 0.1));
  }
  if (mode === 'patch') {
    const n = S.scanData.patches[i * PATCH_STRIDE];
    return n <= 1 ? out.setHex(0x6b7280) : out.copy(rampColor(Math.min(1, (n - 2) / 20)));
  }
  const st = r[o];
  if (st === 0) return out.setHex(r[o + 1] > 0 ? COL.matchedVoxel : COL.matchedNN);
  return out.setHex([0, COL.noNbr, COL.far, COL.planeFail, COL.resFail][st]);
}
const _p = new THREE.Vector3();
function residualAt(i) { // signed distance to the point's plane (5-NN) or distance to its distribution's mean
  const u = S.update, o = recordOf(i, S.iter), r = u.rec;
  if (S.iter < u.iters) return r[o + 11];
  worldPoint(i, _p);
  if (r[o + 1] > 0) return Math.hypot(_p.x - r[o + 12], _p.y - r[o + 13], _p.z - r[o + 14]);
  return r[o + 7] * _p.x + r[o + 8] * _p.y + r[o + 9] * _p.z + r[o + 10];
}
function buildScanLayers() {
  if (!S.scan || !S.scanData) return;
  const sd = S.scanData, pt = parseFloat($('ptSize').value);
  updateScanPose();
}
function buildRawLayer() { // the raw scan at the shown pose, in world coordinates (so the height cut applies)
  const sd = S.scanData, pt = parseFloat($('ptSize').value), zc = zMax();
  const M = S.update ? lidarToWorld(stateAt(S.iter)) : lidarToWorld(S.scan.prior);
  const p = new THREE.Vector3(), pos = [];
  for (let i = 0; i < sd.nRaw; i++) {
    p.set(sd.raw[i * 4], sd.raw[i * 4 + 1], sd.raw[i * 4 + 2]).applyMatrix4(M);
    if (p.z <= zc) pos.push(p.x, p.y, p.z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  setLayer('raw', new THREE.Points(g, new THREE.PointsMaterial({ size: 0.02 * pt, color: COL.raw, sizeAttenuation: true })));
}
function updateScanPose() { // everything that depends on the shown iteration
  if (!S.scan || !S.scanData) return;
  buildRawLayer();
  if (!S.update) { applyVisibility(); return; }
  const u = S.update, sd = S.scanData, n = u.points, pt = parseFloat($('ptSize').value), zc = zMax();
  S.finalMatrix = lidarToWorld(u.head.final);
  // down points (those above the cut are left out; visibleIdx maps the drawn index back to the point)
  const pos = [], col = [], c = new THREE.Color(), visibleIdx = [];
  const resPos = [], resCol = [];
  const p = new THREE.Vector3();
  S.visible = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    worldPoint(i, p);
    if (p.z > zc) continue;
    S.visible[i] = 1;
    visibleIdx.push(i);
    pos.push(p.x, p.y, p.z);
    pointColor(i, c);
    col.push(c.r, c.g, c.b);
    const o = recordOf(i, S.iter), st = u.rec[o];
    if (st === 0 || st === 4) {
      if (u.rec[o + 1] > 0) resPos.push(p.x, p.y, p.z, u.rec[o + 12], u.rec[o + 13], u.rec[o + 14]);
      else {
        const d = residualAt(i);
        resPos.push(p.x, p.y, p.z, p.x - d * u.rec[o + 7], p.y - d * u.rec[o + 8], p.z - d * u.rec[o + 9]);
      }
      c.setHex(st === 4 ? COL.resFail : u.rec[o + 1] > 0 ? COL.matchedVoxel : COL.matchedNN);
      resCol.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 3));
  setLayer('down', new THREE.Points(g, new THREE.PointsMaterial({ size: 0.09 * pt, vertexColors: true, sizeAttenuation: true })));
  layers.down.userData.visibleIdx = visibleIdx;
  const gr = new THREE.BufferGeometry();
  gr.setAttribute('position', new THREE.BufferAttribute(new Float32Array(resPos), 3));
  gr.setAttribute('color', new THREE.BufferAttribute(new Float32Array(resCol), 3));
  setLayer('res', new THREE.LineSegments(gr, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })));
  buildNNLayer();
  buildFrames();
  buildSelection();
  applyVisibility();
}
function buildNNLayer() {
  const u = S.update, mode = $('nnMode').value;
  const pos = [], p = new THREE.Vector3();
  const addLinks = (i) => {
    const o = recordOf(i, S.iter), nn = u.rec[o + 2];
    worldPoint(i, p);
    for (let k = 0; k < nn; k++) pos.push(p.x, p.y, p.z, u.rec[o + 16 + 3 * k], u.rec[o + 17 + 3 * k], u.rec[o + 18 + 3 * k]);
  };
  if (mode === 'all') for (let i = 0; i < u.points; i++) if (!S.visible || S.visible[i]) addLinks(i);
  else if (mode === 'selected' && S.selected >= 0) addLinks(S.selected);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  setLayer('nn', new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: COL.nnLink, transparent: true, opacity: mode === 'all' ? 0.35 : 0.9 })));
}
function triad(state, size, sphereColor, sphereR) {
  const grp = new THREE.Group();
  const o = toVec(state.pos), R = new THREE.Matrix4().makeRotationFromQuaternion(toQuat(state.quat));
  const axes = [[1, 0, 0, 0xef4444], [0, 1, 0, 0x22c55e], [0, 0, 1, 0x3b82f6]];
  const pos = [], col = [];
  for (const [x, y, z, hex] of axes) {
    const d = new THREE.Vector3(x, y, z).applyMatrix4(R).multiplyScalar(size).add(o);
    const c = new THREE.Color(hex);
    pos.push(o.x, o.y, o.z, d.x, d.y, d.z); col.push(c.r, c.g, c.b, c.r, c.g, c.b);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 3));
  grp.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true })));
  const sph = new THREE.Mesh(new THREE.SphereGeometry(sphereR, 12, 8), new THREE.MeshBasicMaterial({ color: sphereColor }));
  sph.position.copy(o);
  grp.add(sph);
  return grp;
}
function buildFrames() {
  const grp = new THREE.Group();
  if (S.scan) grp.add(triad(S.scan.prior, 0.35, COL.prior, 0.035));
  if (S.update) {
    grp.add(triad(S.update.head.start, 0.35, COL.start, 0.045));
    grp.add(triad(S.update.head.final, 0.55, COL.posterior, 0.045));
  }
  if (S.reference && S.reference !== S.update) grp.add(triad(S.reference.head.final, 0.55, COL.reference, 0.03));
  setLayer('frames', grp);
}
function buildSelection() {
  const grp = new THREE.Group();
  if (S.update && S.selected >= 0 && S.selected < S.update.points) {
    const i = S.selected, u = S.update, o = recordOf(i, S.iter), r = u.rec, p = worldPoint(i, new THREE.Vector3());
    const mk = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color: COL.selected, wireframe: true }));
    mk.position.copy(p);
    grp.add(mk);
    const st = r[o];
    if (st === 0 || st === 4 || st === 3) {
      const n = new THREE.Vector3(r[o + 7], r[o + 8], r[o + 9]);
      if (n.lengthSq() > 0.5) {
        const d = residualAt(i), foot = p.clone().addScaledVector(n, -d);
        const quad = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), new THREE.MeshBasicMaterial({ color: st === 0 ? (r[o + 1] > 0 ? COL.matchedVoxel : COL.matchedNN) : st === 4 ? COL.resFail : COL.planeFail, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }));
        quad.position.copy(foot);
        quad.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n.clone().normalize());
        grp.add(quad);
      }
    }
    const nn = r[o + 2];
    if (nn > 0) {
      const pos = new Float32Array(nn * 3);
      for (let k = 0; k < nn; k++) { pos[k * 3] = r[o + 16 + 3 * k]; pos[k * 3 + 1] = r[o + 17 + 3 * k]; pos[k * 3 + 2] = r[o + 18 + 3 * k]; }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      grp.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.12, color: 0xffffff, sizeAttenuation: true })));
    }
    if (r[o + 1] > 0 && S.scan) {
      const s = S.scan.params.voxel_map.size;
      const box = new THREE.Mesh(new THREE.BoxGeometry(s, s, s), new THREE.MeshBasicMaterial({ color: COL.matchedVoxel, wireframe: true }));
      box.position.set((Math.floor(r[o + 12] / s) + 0.5) * s, (Math.floor(r[o + 13] / s) + 0.5) * s, (Math.floor(r[o + 14] / s) + 0.5) * s);
      grp.add(box);
      const mean = new THREE.Mesh(new THREE.SphereGeometry(0.03, 10, 6), new THREE.MeshBasicMaterial({ color: COL.matchedVoxel }));
      mean.position.set(r[o + 12], r[o + 13], r[o + 14]);
      grp.add(mean);
    }
  }
  setLayer('sel', grp);
  updateSelectedInfo();
}
function applyVisibility() {
  const vis = { ikd: 'L_ikd', cells: 'L_cells', ell: 'L_ell', planes: 'L_planes', cubes: 'L_cubes', raw: 'L_raw', down: 'L_down', res: 'L_res', frames: 'L_frames' };
  for (const [k, id] of Object.entries(vis)) if (layers[k]) layers[k].visible = $(id).checked;
  if (layers.nn) layers.nn.visible = $('nnMode').value !== 'none';
}

// ------------------------------------------------------------------------------------------ UI text
function updateLegend() {
  const sw = (hex, t) => `<div><span class="sw" style="background:#${hex.toString(16).padStart(6, '0')}"></span>${t}</div>`;
  let h = '<div class="sec">scan points</div>' + sw(COL.matchedVoxel, 'matched: voxel distribution') + sw(COL.matchedNN, 'matched: 5-NN plane (ikd-Tree)') +
    sw(COL.noNbr, 'no neighbours') + sw(COL.far, 'far neighbours') + sw(COL.planeFail, '5-point plane fit failed') + sw(COL.resFail, 'residual too large');
  if (S.map && S.map.nVox > 0) {
    const cnt = [0, 0, 0, 0];
    if (S.map.cls) for (let k = 0; k < S.map.nVox; k++) cnt[S.map.cls[k]]++;
    h += '<div class="sec">voxel map</div>' + VOX_CLASS_NAMES.map((n, i) => sw(VOX_CLASS_COL[i], `${n} (${cnt[i]})`)).join('');
  }
  h += '<div class="sec">frames</div>' + sw(COL.prior, 'IMU prior') + sw(COL.start, 'start (prior + noise)') + sw(COL.posterior, 'posterior') + sw(COL.reference, 'posterior without noise');
  $('legend').innerHTML = h;
}
function updateScanUI() {
  const s = S.scan;
  $('scanCount').textContent = `/ ${S.info.n_scans}`;
  $('scanIndex').max = S.info.n_scans;
  if (!s) { $('scanInfo').textContent = 'no scan'; return; }
  $('scanIndex').value = s.index;
  const P = s.P6, sd = (k) => Math.sqrt(Math.max(0, P[k * 6 + k]));
  $('scanInfo').textContent =
    `t = ${s.stamp.toFixed(3)} s  (scan ${s.index}, ${s.poses} poses so far)\n` +
    `raw ${s.raw_points}  down ${s.down_points}${s.patches ? ` (${s.patches} patches)` : ''}  ikd ${s.ikd_size}  voxels ${s.voxels}\n` +
    `prior pos ${s.prior.pos.map((v) => v.toFixed(3)).join(' ')}\n` +
    `prior σ: pos ${cm(sd(0))} ${cm(sd(1))} ${cm(sd(2))} cm, rot ${deg(sd(3))} ${deg(sd(4))} ${deg(sd(5))} °\n` +
    `map: ${s.params.map_type}, scan voxel ${s.params.filter_size_surf} m, ` +
    `map voxel ${s.params.map_type === 'voxel' ? s.params.voxel_map.size : s.params.filter_size_map} m, ${s.params.max_iteration} iterations`;
}
function updateNoiseInfo() {
  if (!S.update) { $('noiseInfo').textContent = ''; return; }
  const u = S.update.head, ref = S.reference ? S.reference.head : null;
  let t = `start → posterior: ${cm(posDelta(u.start, u.final))} cm, ${deg(rotDelta(u.start, u.final))} °\n`;
  t += `prior → posterior: ${cm(posDelta(S.scan.prior, u.final))} cm, ${deg(rotDelta(S.scan.prior, u.final))} °\n`;
  if (ref && ref !== u) t += `posterior − unperturbed posterior: ${cm(posDelta(ref.final, u.final))} cm, ${deg(rotDelta(ref.final, u.final))} °`;
  else if (ref) t += `prior → unperturbed posterior: ${cm(posDelta(S.scan.prior, ref.final))} cm, ${deg(rotDelta(S.scan.prior, ref.final))} °`;
  $('noiseInfo').textContent = t;
}
function updateIterUI() {
  const sl = $('iter');
  if (!S.update) { sl.max = 0; $('iterLabel').textContent = '-'; $('iterTable').innerHTML = ''; return; }
  const u = S.update.head, n = S.update.iters;
  sl.max = n;
  sl.value = S.iter;
  $('iterLabel').textContent = S.iter < n ? `iteration ${S.iter}${u.iterations[S.iter].researched ? ' (map search)' : ' (planes kept)'}` : 'final state';
  let h = '<table><tr><th>#</th><th>search</th><th>matched</th><th>vox</th><th>5-NN</th><th>none</th><th>far</th><th>pl.fail</th><th>res.fail</th><th>res cm</th><th>Δ cm</th><th>Δ °</th></tr>';
  let prev = u.start;
  u.iterations.forEach((it, k) => {
    const c = it.counts;
    h += `<tr${k === S.iter ? ' style="color:#fff;font-weight:600"' : ''}><td>${k}</td><td>${it.researched ? '●' : ''}</td><td>${it.matched}</td><td>${it.dist}</td><td>${it.matched - it.dist}</td><td>${c[1]}</td><td>${c[2]}</td><td>${c[3]}</td><td>${c[4]}</td><td>${cm(it.res_mean)}</td><td>${cm(posDelta(prev, it))}</td><td>${deg(rotDelta(prev, it))}</td></tr>`;
    prev = it;
  });
  h += `<tr${S.iter === n ? ' style="color:#fff;font-weight:600"' : ''}><td>final</td><td colspan="9" class="muted">after the last step</td><td>${cm(posDelta(prev, u.final))}</td><td>${deg(rotDelta(prev, u.final))}</td></tr></table>`;
  $('iterTable').innerHTML = h;
  const last = u.iterations[u.iterations.length - 1];
  $('iterInfo').textContent = last ? `last iteration: ${last.effct} residual rows from ${last.matched} points (${(100 * last.dist / Math.max(1, last.matched)).toFixed(0)} % voxel distributions, ${last.pooled} of them pooled)` : '';
}
function updateSelectedInfo() {
  if (!S.update || S.selected < 0) { $('selInfo').textContent = 'none'; return; }
  const i = S.selected, u = S.update, r = u.rec, sd = S.scanData, pa = sd.patches;
  const lines = [`point ${i}  lidar (${sd.down[i * 4].toFixed(3)}, ${sd.down[i * 4 + 1].toFixed(3)}, ${sd.down[i * 4 + 2].toFixed(3)})  range ${Math.hypot(sd.down[i * 4], sd.down[i * 4 + 1], sd.down[i * 4 + 2]).toFixed(2)} m`];
  const n = pa[i * PATCH_STRIDE];
  if (n > 1) lines.push(`patch of ${n} points`);
  for (let it = 0; it < u.iters; it++) {
    const o = (it * u.points + i) * POINT_STRIDE, st = r[o];
    let t = `it ${it}: ${STATUS_NAMES[st]}`;
    if ((st === 0 || st === 4) && r[o + 1] > 0)
      t += ` ${r[o + 15] ? 'pooled neighbourhood' : 'voxel'} mean (${r[o + 12].toFixed(2)},${r[o + 13].toFixed(2)},${r[o + 14].toFixed(2)}) |p - mean|=${cm(r[o + 11])} cm d²=${r[o + 31].toFixed(1)}`;
    else if (st === 0 || st === 4 || st === 3) t += ` 5-NN n=(${r[o + 7].toFixed(2)},${r[o + 8].toFixed(2)},${r[o + 9].toFixed(2)}) d=${cm(r[o + 11])} cm`;
    if (r[o + 2] > 0) {
      const p = new THREE.Vector3(r[o + 4], r[o + 5], r[o + 6]);
      const ds = [];
      for (let k = 0; k < r[o + 2]; k++) ds.push(p.distanceTo(new THREE.Vector3(r[o + 16 + 3 * k], r[o + 17 + 3 * k], r[o + 18 + 3 * k])));
      t += `  nn ${ds.map((x) => x.toFixed(2)).join(' ')} m`;
    }
    lines.push(t);
  }
  const o = recordOf(i, S.iter);
  if (S.iter >= u.iters) lines.push(r[o + 1] > 0 ? `final: |p - mean|=${cm(residualAt(i))} cm` : `final: d=${cm(residualAt(i))} cm to the last plane`);
  if (r[o + 1] > 0 && S.map) {
    const s = S.scan.params.voxel_map.size;
    const k = findVoxel(Math.floor(r[o + 12] / s), Math.floor(r[o + 13] / s), Math.floor(r[o + 14] / s));
    if (k >= 0) lines.push('matched ' + voxelText(k));
  }
  $('selInfo').textContent = lines.join('\n');
}
function findVoxel(ix, iy, iz) {
  if (!S.map.index) {
    S.map.index = new Map();
    for (let k = 0; k < S.map.nVox; k++) { const o = k * VOXEL_STRIDE; S.map.index.set(`${S.map.vox[o]},${S.map.vox[o + 1]},${S.map.vox[o + 2]}`, k); }
  }
  const k = S.map.index.get(`${ix},${iy},${iz}`);
  return k === undefined ? -1 : k;
}
function voxelText(k) {
  const v = S.map.vox, o = k * VOXEL_STRIDE, cls = S.map.cls ? S.map.cls[k] : voxelClass(k);
  const l0 = v[o + 18], l1 = v[o + 19], l2 = v[o + 20];
  return `voxel (${v[o]},${v[o + 1]},${v[o + 2]}): ${VOX_CLASS_NAMES[cls]}${v[o + 31] ? ', frozen' : ''}\n` +
    `${v[o + 3]} points, weight ${v[o + 4].toFixed(3)}\nσ ⊥ ${cm(Math.sqrt(Math.max(0, l0)))} cm, σ in plane ${cm(Math.sqrt(Math.max(0, l1)))} / ${cm(Math.sqrt(Math.max(0, l2)))} cm, λ0/λ1 ${(l1 > 0 ? l0 / l1 : 0).toFixed(3)}\n` +
    `normal (${v[o + 21].toFixed(2)}, ${v[o + 22].toFixed(2)}, ${v[o + 23].toFixed(2)})  mean (${v[o + 5].toFixed(2)}, ${v[o + 6].toFixed(2)}, ${v[o + 7].toFixed(2)})`;
}
function fillParamsUI() {
  const p = S.scan.params;
  const vm = p.voxel_map;
  $('P_vm_sigma').value = vm.point_sigma; $('P_vm_freeze').value = vm.freeze_sigma; $('P_vm_shape').value = vm.shape_ratio;
  $('P_vm_neff').value = vm.min_n_eff; $('P_vm_thick').value = vm.pooled_thickness; $('P_vm_plan').value = vm.pooled_planarity;
  $('P_surf').value = p.filter_size_surf; $('P_sp_min').value = p.scan_patch.min_points;
  $('P_iter').value = p.max_iteration; $('P_ext').checked = p.extrinsic_est_en;
}
function paramChanges() {
  const p = S.scan.params, out = [];
  const num = (id, cur, key) => { const v = parseFloat($(id).value); if (Number.isFinite(v) && Math.abs(v - cur) > 1e-12) out.push([key, v]); };
  const bool = (id, cur, key) => { const v = $(id).checked; if (v !== cur) out.push([key, v ? 1 : 0]); };
  const vm = p.voxel_map;
  num('P_vm_sigma', vm.point_sigma, 'voxel_map.point_sigma'); num('P_vm_freeze', vm.freeze_sigma, 'voxel_map.freeze_sigma');
  num('P_vm_shape', vm.shape_ratio, 'voxel_map.shape_ratio'); num('P_vm_neff', vm.min_n_eff, 'voxel_map.min_n_eff');
  num('P_vm_thick', vm.pooled_thickness, 'voxel_map.pooled_thickness'); num('P_vm_plan', vm.pooled_planarity, 'voxel_map.pooled_planarity');
  num('P_surf', p.filter_size_surf, 'filter_size_surf'); num('P_sp_min', p.scan_patch.min_points, 'scan_patch.min_points');
  num('P_iter', p.max_iteration, 'max_iteration'); bool('P_ext', p.extrinsic_est_en, 'extrinsic_est_en');
  return out;
}

// ------------------------------------------------------------------------------------------ minimap
function drawMinimap() {
  const cv = $('minimap'), ctx = cv.getContext('2d');
  ctx.fillStyle = '#0f1216'; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!S.info) return;
  const pts = [];
  for (const [idx, x, y] of S.info.trajectory) pts.push([x, y, idx, 0]);
  if (S.poses) for (let k = 0; k < S.poses.length / 9; k++) pts.push([S.poses[k * 9 + 2], S.poses[k * 9 + 3], S.poses[k * 9], 1]);
  if (S.scan) pts.push([S.scan.lidar_pos[0], S.scan.lidar_pos[1], S.scan.index, 2]);
  if (pts.length === 0) { ctx.fillStyle = '#9aa3b2'; ctx.fillText('no trajectory yet', 8, 16); return; }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const pad = 10, sc = Math.min((cv.width - 2 * pad) / Math.max(1e-3, x1 - x0), (cv.height - 2 * pad) / Math.max(1e-3, y1 - y0));
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const X = (x) => cv.width / 2 + (x - cx) * sc, Y = (y) => cv.height / 2 - (y - cy) * sc;
  S.minimapMap = { X, Y, pts };
  const cols = ['#4b5563', '#22d3ee', '#ef4444'];
  for (const kind of [0, 1]) {
    ctx.fillStyle = cols[kind];
    for (const [x, y, , k] of pts) if (k === kind) ctx.fillRect(X(x) - 1, Y(y) - 1, 2, 2);
  }
  for (const [x, y, , k] of pts) if (k === 2) { ctx.fillStyle = cols[2]; ctx.beginPath(); ctx.arc(X(x), Y(y), 4, 0, 2 * Math.PI); ctx.fill(); }
  ctx.fillStyle = '#9aa3b2'; ctx.font = '10px sans-serif';
  ctx.fillText(`${(x1 - x0).toFixed(0)} × ${(y1 - y0).toFixed(0)} m`, 6, cv.height - 6);
}
$('minimap').addEventListener('click', (ev) => {
  if (!S.minimapMap) return;
  const rect = ev.target.getBoundingClientRect();
  const mx = (ev.clientX - rect.left) * ($('minimap').width / rect.width), my = (ev.clientY - rect.top) * ($('minimap').height / rect.height);
  let best = -1, bd = 1e9;
  for (const [x, y, idx, k] of S.minimapMap.pts) {
    if (k === 2) continue;
    const d = (S.minimapMap.X(x) - mx) ** 2 + (S.minimapMap.Y(y) - my) ** 2;
    if (d < bd) { bd = d; best = idx; }
  }
  if (best > 0 && bd < 400) gotoScan(best);
});

// ------------------------------------------------------------------------------------------ actions
function perturb() {
  const r = Math.PI / 180;
  return [parseFloat($('nxv').value) / 100, parseFloat($('nyv').value) / 100, parseFloat($('nzv').value) / 100,
    parseFloat($('nrv').value) * r, parseFloat($('npv').value) * r, parseFloat($('nwv').value) * r].map((v) => (Number.isFinite(v) ? v : 0));
}
async function fetchUpdate(p) {
  const { head, payload } = await getFramed('/api/update', { perturb: p });
  return { head, rec: new Float32Array(payload), iters: head.iterations.length, points: head.points };
}
async function runUpdate() {
  if (!S.scan) return;
  if (S.busy) { S.pendingRun = true; return; }
  S.busy = true;
  try {
    setStatus('running the update…', 'busy');
    const p = perturb(), zero = p.every((v) => v === 0);
    if (!S.reference) S.reference = await fetchUpdate([0, 0, 0, 0, 0, 0]);
    S.update = zero ? S.reference : await fetchUpdate(p);
    S.iter = S.update.iters;
    updateScanPose();
    updateIterUI();
    updateNoiseInfo();
    setStatus(`scan ${S.scan.index}: update done (${S.update.iters} iterations)`);
  } catch (e) { showError(e); } finally {
    S.busy = false;
    if (S.pendingRun) { S.pendingRun = false; runUpdate(); }
  }
}
async function loadMap() {
  const radius = parseFloat($('mapRadius').value) || 30;
  setStatus(`loading the map within ${radius} m…`, 'busy');
  const { head, payload } = await getFramed(`/api/map?radius=${radius}`);
  const f = new Float32Array(payload);
  const ikd = f.subarray(0, head.ikd_points * 4), vox = f.subarray(head.ikd_points * 4);
  VOXEL_STRIDE = head.voxel_stride;
  const valid = new Float32Array(head.voxels);
  for (let k = 0; k < head.voxels; k++) valid[k] = vox[k * VOXEL_STRIDE + 30];
  S.map = { head, ikd, vox, nIkd: head.ikd_points, nVox: head.voxels, valid, cls: null, index: null };
}
async function loadScanData() {
  const { head, payload } = await getFramed('/api/scan');
  const f = new Float32Array(payload);
  const nRaw = head.raw_points, nDown = head.down_points;
  PATCH_STRIDE = head.patch_stride;
  S.scanData = { head, nRaw, nDown, raw: f.subarray(0, nRaw * 4), down: f.subarray(nRaw * 4, nRaw * 4 + nDown * 4), patches: f.subarray(nRaw * 4 + nDown * 4) };
}
async function loadPoses() {
  const { payload } = await getFramed('/api/poses');
  S.poses = new Float64Array(payload);
}
async function applyStatus(status) {
  S.scan = status.scan;
  S.reference = S.update = null;
  S.selected = -1;
  if (!S.scan) {
    updateScanUI();
    if (status.ended) setStatus(`the record ended at scan ${status.ended.scans}; choose an earlier scan`, 'error');
    return;
  }
  updateScanUI();
  fillParamsUI();
  await loadScanData();
  await loadMap();
  await loadPoses();
  buildMapLayers();
  buildScanLayers();
  drawMinimap();
  await runUpdate();
  if (S.firstView) { recenter(); S.firstView = false; }
}
async function gotoScan(k) {
  if (S.busy) return;
  S.busy = true;
  try {
    const back = S.scan && k <= S.scan.index;
    setStatus(back ? `restarting the replay and running to scan ${k}…` : `running the replay to scan ${k}…`, 'busy');
    const status = await getJSON('/api/goto', { scan: k });
    S.busy = false;
    await applyStatus(status);
  } catch (e) { showError(e); } finally { S.busy = false; }
}
async function applyParams() {
  if (!S.scan || S.busy) return;
  const changes = paramChanges();
  if (changes.length === 0) { setStatus('no parameter changed'); return; }
  S.busy = true;
  try {
    let voxelChanged = false, scanChanged = false;
    for (const [key, value] of changes) {
      setStatus(`setting ${key} = ${value}…`, 'busy');
      const r = await getJSON('/api/set', { key, value });
      S.scan.params = r.params;
      S.scan.down_points = r.down_points;
      S.scan.patches = r.patches;
      if (key.startsWith('voxel_map.')) voxelChanged = true;
      if (key.startsWith('scan_patch.') || key === 'filter_size_surf') scanChanged = true;
    }
    if (voxelChanged && S.map) { await loadMap(); buildVoxelLayers(); }
    if (scanChanged) { await loadScanData(); buildScanLayers(); }
    S.selected = -1;
    S.reference = S.update = null;
    updateScanUI();
    fillParamsUI();
  } catch (e) { showError(e); S.busy = false; return; }
  S.busy = false;
  await runUpdate();
}

// ------------------------------------------------------------------------------------------ picking
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let hoverPending = false;
function setPointer(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
}
renderer.domElement.addEventListener('pointermove', (ev) => {
  if (hoverPending) return;
  hoverPending = true;
  requestAnimationFrame(() => {
    hoverPending = false;
    const tip = $('tip');
    if (!S.map || (!layers.ell && !layers.planes)) { tip.style.display = 'none'; return; }
    setPointer(ev);
    raycaster.setFromCamera(pointer, camera);
    const targets = [layers.ell, layers.planes].filter((o) => o && o.visible && o.count > 0);
    const hits = raycaster.intersectObjects(targets, false);
    if (hits.length > 0 && hits[0].instanceId !== undefined) {
      const k = hits[0].object.userData.voxIdx[hits[0].instanceId];
      tip.textContent = voxelText(k);
      tip.style.display = 'block';
      tip.style.left = `${ev.clientX + 14}px`;
      tip.style.top = `${ev.clientY + 14}px`;
    } else tip.style.display = 'none';
  });
});
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; });
renderer.domElement.addEventListener('pointerup', (ev) => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4 || ev.button !== 0) return;
  if (!S.update || !layers.down) return;
  setPointer(ev);
  raycaster.setFromCamera(pointer, camera);
  raycaster.params.Points.threshold = 0.1 * parseFloat($('ptSize').value);
  const hits = raycaster.intersectObject(layers.down, false);
  S.selected = hits.length > 0 ? layers.down.userData.visibleIdx[hits[0].index] : -1;
  buildNNLayer();
  buildSelection();
  applyVisibility();
});

// ------------------------------------------------------------------------------------------ events
function bindSlider(range, num) {
  $(range).addEventListener('input', () => { $(num).value = $(range).value; if ($('autoRun').checked) runUpdate(); });
  $(num).addEventListener('change', () => { $(range).value = $(num).value; if ($('autoRun').checked) runUpdate(); });
}
[['nx', 'nxv'], ['ny', 'nyv'], ['nz', 'nzv'], ['nr', 'nrv'], ['np', 'npv'], ['nw', 'nwv']].forEach(([a, b]) => bindSlider(a, b));
$('run').addEventListener('click', runUpdate);
$('resetNoise').addEventListener('click', () => {
  for (const id of ['nx', 'ny', 'nz', 'nr', 'np', 'nw']) { $(id).value = 0; $(id + 'v').value = 0; }
  runUpdate();
});
$('iter').addEventListener('input', () => { S.iter = parseInt($('iter').value, 10); updateScanPose(); updateIterUI(); });
$('go').addEventListener('click', () => gotoScan(parseInt($('scanIndex').value, 10) || 1));
$('scanIndex').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') gotoScan(parseInt($('scanIndex').value, 10) || 1); });
$('prev').addEventListener('click', () => S.scan && gotoScan(Math.max(1, S.scan.index - 1)));
$('next').addEventListener('click', () => S.scan && gotoScan(S.scan.index + 1));
for (const id of ['L_ikd', 'L_cells', 'L_ell', 'L_planes', 'L_cubes', 'L_raw', 'L_down', 'L_res', 'L_frames']) $(id).addEventListener('change', applyVisibility);
$('nnMode').addEventListener('change', () => { if (S.update) { buildNNLayer(); applyVisibility(); } });
$('colorMode').addEventListener('change', () => S.update && updateScanPose());
$('ptSize').addEventListener('input', () => S.scanData && updateScanPose());
let zcutTimer = null;
function zcutChanged() { clearTimeout(zcutTimer); zcutTimer = setTimeout(() => { if (S.map) buildMapLayers(); if (S.scanData) updateScanPose(); }, 150); }
$('zcut').addEventListener('change', zcutChanged);
$('zcutv').addEventListener('input', zcutChanged);
$('cellRadius').addEventListener('change', () => { buildCells(); applyVisibility(); });
for (const id of ['C_valid', 'C_few', 'C_thick', 'C_line']) $(id).addEventListener('change', buildVoxelLayers);
$('voxOpacity').addEventListener('input', () => {
  const o = parseFloat($('voxOpacity').value);
  if (layers.ell) layers.ell.material.opacity = o;
  if (layers.planes) layers.planes.material.opacity = Math.min(1, o + 0.25);
});
$('reloadMap').addEventListener('click', async () => { if (!S.scan || S.busy) return; S.busy = true; try { await loadMap(); buildMapLayers(); setStatus('map reloaded'); } catch (e) { showError(e); } finally { S.busy = false; } });
$('recenter').addEventListener('click', () => recenter(false));
$('topView').addEventListener('click', () => recenter(true));
$('applyParams').addEventListener('click', applyParams);
$('revertParams').addEventListener('click', () => S.scan && fillParamsUI());
window.addEventListener('keydown', (ev) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (ev.key === 'ArrowLeft' && S.update) { S.iter = Math.max(0, S.iter - 1); updateScanPose(); updateIterUI(); }
  if (ev.key === 'ArrowRight' && S.update) { S.iter = Math.min(S.update.iters, S.iter + 1); updateScanPose(); updateIterUI(); }
  if (ev.key === 'n') $('next').click();
  if (ev.key === 'r') $('resetNoise').click();
});

// Console / scripting access: lio.S (state), lio.camera, lio.controls, lio.look(dx, dy, dz) puts the camera at the
// lidar + offset looking at the lidar, lio.gotoScan(k), lio.runUpdate()
window.lio = { S, camera, controls, layers, gotoScan, runUpdate, project(i) { const v = worldPoint(i, new THREE.Vector3()).project(camera); const r = renderer.domElement.getBoundingClientRect(); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height, v.z]; }, look(dx, dy, dz) { const c = toVec(S.scan.lidar_pos); controls.target.copy(c); camera.position.set(c.x + dx, c.y + dy, c.z + dz); controls.update(); } };

// ------------------------------------------------------------------------------------------ start
(async function start() {
  try {
    S.info = await getJSON('/api/info');
    $('dsname').textContent = `${S.info.dataset.name || ''} · ${S.info.config.split('/').pop()}${S.info.params.length ? ' · ' + S.info.params.join(' ') : ''}`;
    updateLegend();
    drawMinimap();
    if (S.info.status.scan) await applyStatus(S.info.status);
    else if (S.info.status.ended) setStatus(`the record ended at scan ${S.info.status.ended.scans}; choose an earlier scan`, 'error');
    else { updateScanUI(); setStatus('no scan loaded: enter a scan index and press Go'); }
  } catch (e) { showError(e); }
})();
