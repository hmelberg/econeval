import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { layoutFor, autoMarkov, autoTree } from '../js/ui/layouts.js';

// 4 states, declaration order well/sick/recovered/dead — n=4 keeps radius pinned at the
// max(140, ...) floor (46*4/pi =~ 58.6), so every ring coordinate lands on an exact integer
// with zero rounding error, which lets the ring-distance assertion be exact rather than fuzzy.
const M_MARKOV = () => parseModel(`
econeval: 1
type: markov
name: m
settings:
  cycles: 10
states:
  well: {cost: 0, utility: 1}
  sick: {cost: 100, utility: 0.5}
  recovered: {cost: 0, utility: 0.9}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: 0.9, sick: 0.1}
  sick: {sick: 0.5, recovered: 0.3, dead: 0.2}
  recovered: {recovered: 1}
  dead: {dead: 1}
`);

// Same markov model but with an explicit layout entry for 'well' — for the layoutFor merge test.
const M_MARKOV_LAYOUT = () => parseModel(`
econeval: 1
type: markov
name: m
settings:
  cycles: 10
states:
  well: {cost: 0, utility: 1}
  sick: {cost: 100, utility: 0.5}
  recovered: {cost: 0, utility: 0.9}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: 0.9, sick: 0.1}
  sick: {sick: 0.5, recovered: 0.3, dead: 0.2}
  recovered: {recovered: 1}
  dead: {dead: 1}
layout:
  well: [10, 20]
`);

// The brief's surgery example: root 'Treatment?' -> Surgery/Medication -> Success/Failure leaves.
const M_TREE = () => parseModel(`
econeval: 1
type: tree
name: Surgery vs medication
params:
  p_success_surg:
    value: 0.9
    dist: beta(90, 10)
tree:
  Treatment?:
    Surgery:
      cost: 5000
      Success: {p: p_success_surg, utility: 0.95}
      Failure: {p: rest, utility: 0.40, cost: 2000}
    Medication:
      cost: 800
      Success: {p: 0.60, utility: 0.90}
      Failure: {p: rest, utility: 0.50}
`);

// --- autoMarkov ---

test('autoMarkov: 4 states land exactly on the ring, first state at top', () => {
  const m = M_MARKOV();
  const pos = autoMarkov(m);
  const cx = 360, cy = 280, n = 4;
  const r = Math.max(140, 46 * n / Math.PI);
  const names = m.states.map((s) => s.name);

  assert.equal(names.length, 4);
  names.forEach((name, i) => {
    const angle = -Math.PI / 2 + i * (2 * Math.PI / n);
    const ex = Math.round(cx + r * Math.cos(angle));
    const ey = Math.round(cy + r * Math.sin(angle));
    assert.deepEqual(pos[name], [ex, ey], `state ${name} at index ${i}`);

    const dx = pos[name][0] - cx;
    const dy = pos[name][1] - cy;
    assert.ok(Math.abs(Math.sqrt(dx * dx + dy * dy) - r) < 1e-6, `state ${name} within 1e-6 of radius`);
  });

  // first state at angle -90 deg == straight up from center == same x as center, smaller y
  assert.deepEqual(pos[names[0]], [360, 140]);
});

test('autoMarkov is deterministic (two calls deepEqual)', () => {
  const m = M_MARKOV();
  assert.deepEqual(autoMarkov(m), autoMarkov(m));
});

// --- autoTree ---

test('autoTree: surgery example leaves at y 60,134,208,282 in depth-first declaration order', () => {
  const m = M_TREE();
  const pos = autoTree(m);
  assert.deepEqual(pos['Treatment?/Surgery/Success'], [430, 60]);
  assert.deepEqual(pos['Treatment?/Surgery/Failure'], [430, 134]);
  assert.deepEqual(pos['Treatment?/Medication/Success'], [430, 208]);
  assert.deepEqual(pos['Treatment?/Medication/Failure'], [430, 282]);
});

test('autoTree: internal node y is the rounded mean of its children y', () => {
  const m = M_TREE();
  const pos = autoTree(m);
  assert.deepEqual(pos['Treatment?/Surgery'], [260, 97]); // mean(60,134) = 97
  assert.deepEqual(pos['Treatment?/Medication'], [260, 245]); // mean(208,282) = 245
});

test('autoTree: root included, x = 90 + depth*170 (root 90, depth-1 260, leaves 430)', () => {
  const m = M_TREE();
  const pos = autoTree(m);
  assert.deepEqual(pos['Treatment?'], [90, 171]); // mean(97, 245) = 171
  assert.ok('Treatment?/Surgery' in pos);
  assert.equal(pos['Treatment?/Surgery'][0], 260);
  assert.equal(pos['Treatment?/Surgery/Success'][0], 430);
});

test('autoTree is deterministic (two calls deepEqual)', () => {
  const m = M_TREE();
  assert.deepEqual(autoTree(m), autoTree(m));
});

// --- layoutFor: merge ---

test('layoutFor: explicit layout entries win, auto-fills the rest (markov)', () => {
  const m = M_MARKOV_LAYOUT();
  const pos = layoutFor(m);
  assert.deepEqual(pos.well, [10, 20]); // explicit wins, NOT the ring position
  const auto = autoMarkov(m);
  assert.deepEqual(pos.sick, auto.sick);
  assert.deepEqual(pos.recovered, auto.recovered);
  assert.deepEqual(pos.dead, auto.dead);
});

test('layoutFor: explicit layout entries win, auto-fills the rest (tree)', () => {
  const explicit = { 'Treatment?/Surgery': [999, 999] };
  const m = { ...M_TREE(), layout: explicit };
  const pos = layoutFor(m);
  assert.deepEqual(pos['Treatment?/Surgery'], [999, 999]);
  const auto = autoTree(m);
  assert.deepEqual(pos['Treatment?/Medication'], auto['Treatment?/Medication']);
  assert.deepEqual(pos['Treatment?/Surgery/Success'], auto['Treatment?/Surgery/Success']);
});

test('layoutFor is deterministic (two calls deepEqual)', () => {
  const m = M_MARKOV_LAYOUT();
  assert.deepEqual(layoutFor(m), layoutFor(m));
  const t = M_TREE();
  assert.deepEqual(layoutFor(t), layoutFor(t));
});

test('layoutFor works on a model with no explicit layout at all (layout: null)', () => {
  const m = M_MARKOV();
  assert.equal(m.layout, null);
  const pos = layoutFor(m);
  assert.deepEqual(pos, autoMarkov(m));
});
