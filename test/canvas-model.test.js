import test from 'node:test';
import assert from 'node:assert/strict';
import { edgePath, selfLoopPath, edgeLabelPos, scopedStore, NODE_R } from '../js/ui/canvas.js';
import { createStore } from '../js/ui/store.js';
import { addState } from '../js/ui/ops.js';

// This file covers ONLY the DOM-free geometry helpers canvas.js exports (per the task-9 brief:
// "Test: test/canvas-model.test.js (geometry helpers only)"), plus the scopedStore wrapper (also
// pure — no DOM touched by any of these).

test('NODE_R matches the design token (markov circle radius 26)', () => {
  assert.equal(NODE_R, 26);
});

// --- edgePath: line trimmed to node radii on each side ---

test('edgePath: horizontal pair trimmed by r on each side', () => {
  const d = edgePath([0, 0], [100, 0], 26);
  assert.equal(d, 'M 26 0 L 74 0');
});

test('edgePath: vertical pair trimmed by r on each side', () => {
  const d = edgePath([50, 0], [50, 100], 26);
  assert.equal(d, 'M 50 26 L 50 74');
});

test('edgePath: diagonal pair trimmed along the line direction (3-4-5 triangle, r=10)', () => {
  // (0,0) -> (30,40): length 50, unit vector (0.6, 0.8); trimmed 10 in from each end.
  const d = edgePath([0, 0], [30, 40], 10);
  assert.equal(d, 'M 6 8 L 24 32');
});

test('edgePath: coincident points do not throw or produce NaN (degenerate line)', () => {
  const d = edgePath([5, 5], [5, 5], 26);
  assert.equal(d, 'M 5 5 L 5 5');
});

// --- selfLoopPath: arc above the node, starting and ending exactly on the circle ---

// selfLoopPath returns "M x0 y0 C cx0 cy0 cx1 cy1 ex ey" — a single SVG cubic-Bezier command.
// Pull out all 8 numbers in that fixed order rather than hand-rolling ad hoc capture groups
// (fragile — a mismatched group count silently misaligns which number lands in which variable).
function parseSelfLoop(d) {
  const nums = d.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g).map(Number);
  assert.equal(nums.length, 8, `expected 8 numbers in an "M x y C cx1 cy1 cx2 cy2 ex ey" path, got: ${d}`);
  const [x0, y0, cx0, cy0, cx1, cy1, ex, ey] = nums;
  return { x0, y0, cx0, cy0, cx1, cy1, ex, ey };
}

test('selfLoopPath: both anchor points sit exactly on the circle at the documented spread angle', () => {
  const cx = 100;
  const cy = 100;
  const r = 26;
  const spread = Math.PI / 6; // 30 degrees either side of top (-90deg) — matches the implementation
  const a0 = -Math.PI / 2 - spread;
  const a1 = -Math.PI / 2 + spread;
  const ex0 = cx + r * Math.cos(a0);
  const ey0 = cy + r * Math.sin(a0);
  const ex1 = cx + r * Math.cos(a1);
  const ey1 = cy + r * Math.sin(a1);

  const { x0, y0, ex, ey } = parseSelfLoop(selfLoopPath([cx, cy], r));

  assert.ok(Math.abs(x0 - ex0) < 1e-9, 'start x matches the spread-angle point on the circle');
  assert.ok(Math.abs(y0 - ey0) < 1e-9, 'start y matches the spread-angle point on the circle');
  assert.ok(Math.abs(ex - ex1) < 1e-9, 'end x matches the spread-angle point on the circle');
  assert.ok(Math.abs(ey - ey1) < 1e-9, 'end y matches the spread-angle point on the circle');
});

test('selfLoopPath: start and end are each exactly r from the center (on the circle)', () => {
  const cx = 40;
  const cy = 60;
  const r = 26;
  const { x0, y0, ex, ey } = parseSelfLoop(selfLoopPath([cx, cy], r));
  assert.ok(Math.abs(Math.hypot(x0 - cx, y0 - cy) - r) < 1e-9, 'start point distance from center is r');
  assert.ok(Math.abs(Math.hypot(ex - cx, ey - cy) - r) < 1e-9, 'end point distance from center is r');
});

test('selfLoopPath: both anchors (and the loop between them) sit above the node center', () => {
  const cx = 0;
  const cy = 0;
  const r = 26;
  const { y0, cy0, cy1, ey } = parseSelfLoop(selfLoopPath([cx, cy], r));
  assert.ok(y0 < cy, 'start anchor is above the node center');
  assert.ok(ey < cy, 'end anchor is above the node center');
  assert.ok(cy0 < y0 && cy1 < ey, 'the bezier control points arc further above the anchors, not below');
});

test('selfLoopPath: a bigger radius produces a proportionally bigger loop', () => {
  const { x0: x0a, y0: y0a } = parseSelfLoop(selfLoopPath([0, 0], 10));
  const { x0: x0b, y0: y0b } = parseSelfLoop(selfLoopPath([0, 0], 20));
  assert.ok(Math.abs(Math.hypot(x0a, y0a) - 10) < 1e-9);
  assert.ok(Math.abs(Math.hypot(x0b, y0b) - 20) < 1e-9);
  // the anchor scales linearly with r, since it's r*cos/sin of a fixed angle
  assert.ok(Math.abs(x0b - x0a * 2) < 1e-9);
  assert.ok(Math.abs(y0b - y0a * 2) < 1e-9);
});

// --- edgeLabelPos: midpoint offset perpendicular by 10px ---

test('edgeLabelPos: horizontal pair — midpoint offset 10px perpendicular (above the line)', () => {
  assert.deepEqual(edgeLabelPos([0, 0], [100, 0]), [50, -10]);
});

test('edgeLabelPos: vertical pair — midpoint offset 10px perpendicular (to the right of the line)', () => {
  assert.deepEqual(edgeLabelPos([0, 0], [0, 100]), [10, 50]);
});

test('edgeLabelPos: reversing the edge direction flips which side the label lands on', () => {
  const a = edgeLabelPos([0, 0], [100, 0]);
  const b = edgeLabelPos([100, 0], [0, 0]);
  assert.deepEqual(a, [50, -10]);
  assert.deepEqual(b, [50, 10]);
});

test('edgeLabelPos: offset is always exactly 10px from the true midpoint, for any direction', () => {
  const from = [10, 20];
  const to = [130, 90];
  const pos = edgeLabelPos(from, to);
  const mx = (from[0] + to[0]) / 2;
  const my = (from[1] + to[1]) / 2;
  const dist = Math.hypot(pos[0] - mx, pos[1] - my);
  assert.ok(Math.abs(dist - 10) < 1e-9);
});

// --- scopedStore: pure store-wrapping logic (not required by the brief's "three helpers", but
// DOM-free and cheap to verify given constraints.md's "pure logic lives in DOM-free modules with
// tests") ---

const SUB_MARKOV_TEXT = `
econeval: 1
type: markov
name: top
settings:
  cycles: 5
states:
  a: {cost: 0, utility: 1}
transitions:
  a: {a: 1}
models:
  sub:
    type: markov
    settings: {cycles: 5}
    states:
      x: {cost: 0, utility: 1}
    transitions:
      x: {x: 1}
`;

test('scopedStore.get() unwraps to the named sub-model', () => {
  const store = createStore(SUB_MARKOV_TEXT);
  const scoped = scopedStore(store, 'sub');
  assert.equal(scoped.get().model.name, 'sub');
  assert.deepEqual(scoped.get().model.states.map((s) => s.name), ['x']);
});

test('scopedStore.applyOp maps the op over model.models[name] and commits through the real store', () => {
  const store = createStore(SUB_MARKOV_TEXT);
  const scoped = scopedStore(store, 'sub');
  scoped.applyOp((m) => addState(m, 'y'));
  assert.deepEqual(store.get().model.states.map((s) => s.name), ['a']); // top-level untouched
  assert.deepEqual(scoped.get().model.states.map((s) => s.name), ['x', 'y']);
  assert.deepEqual(store.get().model.models.sub.states.map((s) => s.name), ['x', 'y']);
});

test('scopedStore.applyOp: a throwing op leaves the store untouched (same guarantee as the base store)', () => {
  const store = createStore(SUB_MARKOV_TEXT);
  const before = store.get();
  const scoped = scopedStore(store, 'sub');
  assert.throws(() => scoped.applyOp((m) => addState(m, 'x'))); // 'x' already exists -> throws
  assert.deepEqual(store.get(), before);
});

test('scopedStore: composes to chain through nested sub-model names (scopedStore(scopedStore(s,"a"),"b"))', () => {
  const text = `
econeval: 1
type: markov
name: top
settings:
  cycles: 5
states:
  a: {cost: 0, utility: 1}
transitions:
  a: {a: 1}
models:
  outer:
    type: markov
    settings: {cycles: 5}
    states:
      p: {cost: 0, utility: 1}
    transitions:
      p: {p: 1}
    models:
      inner:
        type: markov
        settings: {cycles: 5}
        states:
          q: {cost: 0, utility: 1}
        transitions:
          q: {q: 1}
`;
  const store = createStore(text);
  const nested = scopedStore(scopedStore(store, 'outer'), 'inner');
  assert.equal(nested.get().model.name, 'inner');

  nested.applyOp((m) => addState(m, 'r'));
  assert.deepEqual(nested.get().model.states.map((s) => s.name), ['q', 'r']);
  assert.deepEqual(
    store.get().model.models.outer.models.inner.states.map((s) => s.name),
    ['q', 'r'],
  );
  assert.deepEqual(store.get().model.states.map((s) => s.name), ['a']); // untouched
  assert.deepEqual(store.get().model.models.outer.states.map((s) => s.name), ['p']); // untouched
});

test('scopedStore: select/undo/redo/markSaved/subscribe pass straight through to the base store', () => {
  const store = createStore(SUB_MARKOV_TEXT);
  const scoped = scopedStore(store, 'sub');

  scoped.select({ kind: 'state', id: 'x' });
  assert.deepEqual(store.get().selection, { kind: 'state', id: 'x' });
  assert.deepEqual(scoped.get().selection, { kind: 'state', id: 'x' });

  let notified = 0;
  const unsub = scoped.subscribe(() => { notified += 1; });
  scoped.applyOp((m) => addState(m, 'y'));
  assert.equal(notified, 1);
  unsub();

  assert.equal(scoped.get().canUndo, true);
  scoped.undo();
  assert.deepEqual(store.get().model.models.sub.states.map((s) => s.name), ['x']);
  scoped.redo();
  assert.deepEqual(store.get().model.models.sub.states.map((s) => s.name), ['x', 'y']);

  assert.equal(scoped.get().dirty, true);
  scoped.markSaved();
  assert.equal(store.get().dirty, false);
});
