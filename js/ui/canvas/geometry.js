// Pure geometry helpers for the canvas renderer — no DOM access, all testable via node:test.

// Node vocabulary sizes (design tokens doc: "markov state = circle r 26; tree decision root =
// square; chance = circle; terminal = short vertical end-bar; sub-model attachment = stadium
// double-stroke"). Exported where a consumer (Task 10's hit-testing/Add tool) plausibly needs the
// same radius used here.
export const NODE_R = 26;          // markov state circle, and tree chance-node circle
export const ROOT_HALF = 26;       // tree decision root: square, side = ROOT_HALF*2 (52) — same
                                   // footprint as the circle so root/chance read as one family
export const TERMINAL_HALF = 15;   // tree terminal: vertical end-bar, half-length
export const STADIUM_W = 120;      // tree sub-model attachment: stadium (pill) width
export const STADIUM_H = 32;       // stadium height
export const STADIUM_INSET = 3;    // inner stroke inset, for the "double stroke" look
export const HALO_GAP = 4;         // selection halo: extra offset beyond the node's own edge
export const SELF_LOOP_SPREAD = Math.PI / 6;  // self-loop anchors sit +-30deg either side of top (-90deg)
export const SELF_LOOP_HEIGHT = 2;            // bezier control-point offset above the anchors, as a
                                              // multiple of the node radius
export const BASE_W = 900;         // default viewBox width at zoom 1
export const BASE_H = 640;         // default viewBox height at zoom 1
export const GRID = 12;            // matches css/canvas.css's 12px dot grid
export const HIT_SLACK = 6;        // extra forgiveness around every shape

// edgePath(from, to, r) -> 'd' attribute string for a straight edge from one node's rim to the
// other's, i.e. the raw center-to-center line trimmed by `r` at BOTH ends (so it starts/ends on
// the node circles rather than at their centers, leaving room for an arrowhead marker drawn via
// marker-end on the <path> the caller builds from this string). from/to are [x, y] pairs.
export function edgePath(from, to, r) {
  const [x1, y1] = from;
  const [x2, y2] = to;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len === 0) return `M ${x1} ${y1} L ${x2} ${y2}`; // degenerate: coincident points
  const ux = dx / len;
  const uy = dy / len;
  const sx = x1 + ux * r;
  const sy = y1 + uy * r;
  const ex = x2 - ux * r;
  const ey = y2 - uy * r;
  return `M ${sx} ${sy} L ${ex} ${ey}`;
}

// selfLoopPath(xy, r) -> 'd' attribute string for a small loop that leaves the node's circle just
// above-left of top, arcs up and over, and re-enters just above-right of top — both anchor points
// lie exactly on the circle (distance r from the center), so the loop reads as attached to the
// node rather than floating near it. xy is the node's [x, y] center.
export function selfLoopPath(xy, r) {
  const [cx, cy] = xy;
  const a0 = -Math.PI / 2 - SELF_LOOP_SPREAD;
  const a1 = -Math.PI / 2 + SELF_LOOP_SPREAD;
  const x0 = cx + r * Math.cos(a0);
  const y0 = cy + r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy + r * Math.sin(a1);
  const h = r * SELF_LOOP_HEIGHT;
  return `M ${x0} ${y0} C ${x0} ${y0 - h} ${x1} ${y1 - h} ${x1} ${y1}`;
}

// Label anchor for a self-loop: above the loop's own apex, so it never overlaps the arc.
export function selfLoopLabelPos(xy, r) {
  const [cx, cy] = xy;
  return [cx, cy - r - r * SELF_LOOP_HEIGHT - 6];
}

// edgeLabelPos(from, to) -> [x, y]: the midpoint of the two points, offset 10px perpendicular to
// the line — always to the same side (rotate the edge vector -90deg, i.e. (dy, -dx) normalized),
// so a label sits consistently "above" a left-to-right edge rather than flipping sides
// unpredictably depending on edge direction.
export function edgeLabelPos(from, to) {
  const [x1, y1] = from;
  const [x2, y2] = to;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const px = dy / len;
  const py = -dx / len;
  return [mx + px * 10, my + py * 10];
}

// NEW: Shape descriptors for hit-testing. `hit` objects live on every nodeIndex entry (Task 4
// builds them): {shape:'circle', r} | {shape:'rect', w, h} | {shape:'stadium', w, h}.

export function hitShapeFor(kind) {
  if (kind === 'state' || kind === 'chance') return { shape: 'circle', r: NODE_R };
  if (kind === 'root') return { shape: 'rect', w: 2 * ROOT_HALF, h: 2 * ROOT_HALF };
  if (kind === 'submodel') return { shape: 'stadium', w: STADIUM_W, h: STADIUM_H };
  if (kind === 'terminal') return { shape: 'rect', w: 16, h: 30 };
  throw new Error(`hitShapeFor: unrecognized node kind '${kind}'`);
}

// A stadium is a rect of (w - h) by h with a half-circle cap of radius h/2 at each end. Clamp the
// point's x into the straight section, then do one circle test against that clamped centre — this
// is the standard point-to-capsule distance, and it correctly rejects the bounding box's corners.
function insideStadium(dx, dy, w, h, slack) {
  const r = h / 2;
  const half = Math.max(0, w / 2 - r);
  const cx = Math.max(-half, Math.min(half, dx));
  return Math.hypot(dx - cx, dy) <= r + slack;
}

// `point` must be a [x, y] array, matching every other coordinate pair in this module (edgePath,
// selfLoopPath, ...) — NOT an {x, y} object. This is deliberately strict rather than duck-typed:
// point[0]/point[1] on an {x, y} object are silently `undefined`, every distance below becomes
// NaN, every `NaN <= ...` comparison is false, and pickNode would return null unconditionally —
// a hit-test that always misses, with no error anywhere. Per this repo's standing rule ("errors
// surfaced, never swallowed"), that failure mode must throw instead of silently discarding every
// click (task-4 review: this exact shape mismatch reached production once already).
export function isInside(point, xy, hit, slack = HIT_SLACK) {
  if (!Array.isArray(point) || point.length !== 2 || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) {
    throw new Error(`isInside: point must be a [x, y] array of two finite numbers, got ${JSON.stringify(point)}`);
  }
  const dx = point[0] - xy[0];
  const dy = point[1] - xy[1];
  if (hit.shape === 'circle') return Math.hypot(dx, dy) <= hit.r + slack;
  if (hit.shape === 'stadium') return insideStadium(dx, dy, hit.w, hit.h, slack);
  return Math.abs(dx) <= hit.w / 2 + slack && Math.abs(dy) <= hit.h / 2 + slack;
}

// Topmost-wins: nodeIndex is built in render order, so the LAST match is the one drawn on top —
// the one the user believes they are pointing at.
export function pickNode(point, nodeIndex, slack = HIT_SLACK) {
  for (let i = nodeIndex.length - 1; i >= 0; i -= 1) {
    if (isInside(point, nodeIndex[i].xy, nodeIndex[i].hit, slack)) return nodeIndex[i];
  }
  return null;
}

// The trailing `+ 0` is load-bearing, not decoration: Math.round(-5 / 12) is -0, and -0 * 12 stays
// -0. A -0 reaching setLayout serializes into the YAML as `-0`, which makes two otherwise identical
// models compare unequal — and node:assert/strict's deepEqual distinguishes it from 0, so the test
// below catches it. Adding 0 collapses -0 to 0 and leaves every other value alone.
export function snapToGrid(xy, grid = GRID) {
  return [Math.round(xy[0] / grid) * grid + 0, Math.round(xy[1] / grid) * grid + 0];
}

export function fitBox(positions, pad = 60) {
  if (!positions.length) return { x: 0, y: 0, w: BASE_W, h: BASE_H };
  const xs = positions.map((p) => p[0]);
  const ys = positions.map((p) => p[1]);
  const minX = Math.min(...xs); const maxX = Math.max(...xs);
  const minY = Math.min(...ys); const maxY = Math.max(...ys);
  return { x: minX - pad, y: minY - pad, w: (maxX - minX) + pad * 2, h: (maxY - minY) + pad * 2 };
}
