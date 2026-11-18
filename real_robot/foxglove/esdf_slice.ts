// The policy's ESDF at the height of the "esdf_z" slider: cuts the executor's whole window (/g1_wb/esdf_cube, 2 Hz)
// into one layer. /tf (~55 Hz, subscribed by the 3D panel anyway) only wakes the script, so the slice follows the
// slider at once. The cube is "esdf-zlib-rowdelta" (each row's byte differences, zlib; ~3.4x smaller than raw) or,
// in older recordings, raw "mono8". Source of the layouts' esdf_slice script (make_layout_v3*.py).
import { Input } from "./types";
import { PointCloud } from "@foxglove/schemas";

export const inputs = ["/g1_wb/esdf_cube", "/tf"];
export const output = "/studio_script/esdf_slice";

type Cube = { q: Uint8Array; nx: number; ny: number; nz: number; o: number[]; dx: number; lo: number; hi: number; seq: number; stamp: any };
let cube: Cube | undefined;
let shown = "";
let seq = 0;
const cache = new Map<number, Uint8Array>();
let cacheSeq = -1;

function b64(s: string): Uint8Array {
  const A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const clean = s.replace(/[^A-Za-z0-9+/]/g, "");
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let acc = 0, bits = 0, p = 0;
  for (let i = 0; i < clean.length; i++) {
    acc = (acc << 6) | A.indexOf(clean[i]!);
    bits += 6;
    if (bits >= 8) { bits -= 8; out[p++] = (acc >> bits) & 255; }
  }
  return out;
}

function bytes(d: any): Uint8Array {
  if (d instanceof Uint8Array) return d;
  if (d && d.buffer instanceof ArrayBuffer && typeof d.length === "number") return new Uint8Array(d.buffer, d.byteOffset ?? 0, d.length);
  if (typeof d === "string") return b64(d);
  return Uint8Array.from(d as number[]);
}

// ---- zlib inflate (RFC 1950/1951), synchronous: user scripts cannot import libraries ----
type Tree = { table: Uint16Array; trans: Uint16Array };
const tree = (): Tree => ({ table: new Uint16Array(16), trans: new Uint16Array(288) });
const LEN_BITS = new Uint8Array(30), LEN_BASE = new Uint16Array(30), DIST_BITS = new Uint8Array(30), DIST_BASE = new Uint16Array(30);
function bitsBase(bits: Uint8Array, base: Uint16Array, delta: number, first: number): void {
  for (let i = 0; i < delta; i++) bits[i] = 0;
  for (let i = 0; i < 30 - delta; i++) bits[i + delta] = Math.floor(i / delta);
  for (let sum = first, i = 0; i < 30; i++) { base[i] = sum; sum += 1 << bits[i]!; }
}
bitsBase(LEN_BITS, LEN_BASE, 4, 3); LEN_BITS[28] = 0; LEN_BASE[28] = 258;
bitsBase(DIST_BITS, DIST_BASE, 2, 1);
const CLC_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function build(t: Tree, lengths: Uint8Array, off: number, num: number): void {
  t.table.fill(0);
  for (let i = 0; i < num; i++) t.table[lengths[off + i]!]!++;
  t.table[0] = 0;
  const offs = new Uint16Array(16);
  for (let sum = 0, i = 0; i < 16; i++) { offs[i] = sum; sum += t.table[i]!; }
  for (let i = 0; i < num; i++) { const l = lengths[off + i]!; if (l) t.trans[offs[l]!++] = i; }
}
const FIXED_L = tree(), FIXED_D = tree();
{
  const l = new Uint8Array(288);
  for (let i = 0; i < 288; i++) l[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
  build(FIXED_L, l, 0, 288);
  build(FIXED_D, new Uint8Array(30).fill(5), 0, 30);
}

function inflate(src: Uint8Array, outLen: number): Uint8Array {
  const out = new Uint8Array(outLen);
  let pos = 2, tag = 0, nbits = 0, op = 0;                     // skip the 2-byte zlib header
  const bit = (): number => {
    if (nbits === 0) { tag = src[pos++]!; nbits = 8; }
    const b = tag & 1; tag >>>= 1; nbits--; return b;
  };
  const read = (num: number, base: number): number => {
    let v = 0;
    for (let i = 0; i < num; i++) v |= bit() << i;
    return v + base;
  };
  const sym = (t: Tree): number => {
    let sum = 0, cur = 0, len = 0;
    do { cur = 2 * cur + bit(); len++; sum += t.table[len]!; cur -= t.table[len]!; } while (cur >= 0);
    return t.trans[sum + cur]!;
  };
  const lt = tree(), dt = tree(), lengths = new Uint8Array(288 + 32);
  let last = 0;
  while (!last) {
    last = bit();
    const type = read(2, 0);
    if (type === 0) {                                          // stored
      nbits = 0;
      const len = src[pos]! | (src[pos + 1]! << 8); pos += 4;
      out.set(src.subarray(pos, pos + len), op); pos += len; op += len;
      continue;
    }
    let L = FIXED_L, D = FIXED_D;
    if (type === 2) {                                          // dynamic Huffman trees
      const hlit = read(5, 257), hdist = read(5, 1), hclen = read(4, 4);
      lengths.fill(0, 0, 19);
      for (let i = 0; i < hclen; i++) lengths[CLC_ORDER[i]!] = read(3, 0);
      const ct = tree(); build(ct, lengths, 0, 19);
      let num = 0;
      while (num < hlit + hdist) {
        const s = sym(ct);
        if (s === 16) { const prev = lengths[num - 1]!; for (let n = read(2, 3); n; n--) lengths[num++] = prev; }
        else if (s === 17) { for (let n = read(3, 3); n; n--) lengths[num++] = 0; }
        else if (s === 18) { for (let n = read(7, 11); n; n--) lengths[num++] = 0; }
        else lengths[num++] = s;
      }
      build(lt, lengths, 0, hlit); build(dt, lengths, hlit, hdist); L = lt; D = dt;
    }
    for (;;) {
      let s = sym(L);
      if (s < 256) { out[op++] = s; continue; }
      if (s === 256) break;
      s -= 257;
      const len = read(LEN_BITS[s]!, LEN_BASE[s]!);
      const ds = sym(D);
      let from = op - read(DIST_BITS[ds]!, DIST_BASE[ds]!);
      for (let i = 0; i < len; i++) out[op++] = out[from++]!;
    }
  }
  return out;
}

function decodeCube(m: any, nx: number, ny: number, nz: number): Uint8Array {
  const data = bytes(m.data);
  if (m.encoding !== "esdf-zlib-rowdelta") return data;           // raw mono8 (older recordings)
  const q = inflate(data, nx * ny * nz);
  for (let r = 0; r < nx * nz; r++) {                               // undo the per-row differences
    const b = r * ny;
    for (let j = 1; j < ny; j++) q[b + j] = (q[b + j]! + q[b + j - 1]!) & 255;
  }
  return q;
}
// ---- end inflate ----

export default function script(event: Input<"/g1_wb/esdf_cube"> | Input<"/tf">, globalVars: any): PointCloud | undefined {
  if (event.topic === "/g1_wb/esdf_cube") {
    const m: any = event.message;
    const f = String(m.header.frame_id).split(" ");
    if (f[0] !== "esdf") return undefined;
    const nz = Number(f[5]), nx = m.height / nz, ny = m.width;
    cube = { q: decodeCube(m, nx, ny, nz), nx, ny, nz, o: [Number(f[1]), Number(f[2]), Number(f[3])], dx: Number(f[4]),
             lo: Number(f[6]), hi: Number(f[7]), seq: ++seq, stamp: { sec: event.receiveTime.sec, nsec: event.receiveTime.nsec } };
  }
  if (!cube) return undefined;
  const z = Number(globalVars?.esdf_z ?? 0.75);
  const k = Math.max(0, Math.min(cube.nz - 1, Math.round((z - cube.o[2]!) / cube.dx)));
  const key = `${k} ${cube.seq}`;
  if (key === shown) return undefined;
  shown = key;
  const { nx, ny, dx, o, lo, hi } = cube;
  if (cacheSeq !== cube.seq) { cache.clear(); cacheSeq = cube.seq; }
  let data = cache.get(k);
  if (!data) {
    const n = nx * ny, base = k * n, scale = (hi - lo) / 255;
    data = new Uint8Array(16 * n);
    const v = new DataView(data.buffer);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const p = i * ny + j, b = 16 * p;
        v.setFloat32(b, o[0]! + i * dx, true);
        v.setFloat32(b + 4, o[1]! + j * dx, true);
        v.setFloat32(b + 8, o[2]! + k * dx, true);
        v.setFloat32(b + 12, lo + cube.q[base + p]! * scale, true);
      }
    }
    cache.set(k, data);
  }
  return {
    timestamp: cube.stamp, frame_id: "map",
    pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    point_stride: 16,
    fields: [{ name: "x", offset: 0, type: 7 }, { name: "y", offset: 4, type: 7 }, { name: "z", offset: 8, type: 7 }, { name: "sdf", offset: 12, type: 7 }],
    data,
  };
}
