import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseModel } from '../js/core/model.js';
import { oneWay } from '../js/analysis/dsa.js';
import {
  availability, gate, runBase, traceFor, runTornado, runPsa, psaDerived,
} from '../js/ui/analyses.js';

const load = (f) => parseModel(readFileSync(new URL(`../examples/${f}`, import.meta.url), 'utf8'));

test('hiv availability: tornado ok with p_AB listed (after low/high added), psa ok, trace true', () => {
  const m = load('hiv.yaml');
  // hiv.yaml has no low/high on any param out of the box — add it to p_AB only, mirroring how
  // check.test.js hand-augments the same fixture, so tornado.params ends up exactly ['p_AB'].
  m.params.get('p_AB').low = 0.15;
  m.params.get('p_AB').high = 0.25;

  const a = availability(m);
  assert.equal(a.tornado.ok, true);
  assert.deepEqual(a.tornado.params, ['p_AB']);
  assert.equal(a.tornado.reason, undefined);
  assert.equal(a.psa.ok, true);       // every p_ param carries a dist: in hiv.yaml
  assert.equal(a.trace, true);        // markov
});

test('availability: not-ok cases carry a reason', () => {
  // Single-strategy tree, no low/high, no dist anywhere -> both tornado and psa unavailable.
  const m = parseModel(`
econeval: 1
type: tree
name: solo
tree:
  Root:
    Only:
      Leaf: {p: 1, utility: 1}
`);
  const a = availability(m);
  assert.equal(a.tornado.ok, false);
  assert.equal(typeof a.tornado.reason, 'string');
  assert.deepEqual(a.tornado.params, []);
  assert.equal(a.psa.ok, false);
  assert.equal(typeof a.psa.reason, 'string');
  assert.equal(a.trace, false);       // tree, not markov
});

test('hiv runBase: combo ICER within 1e-2 of the phase-1 golden value 5976.80', () => {
  const m = load('hiv.yaml');
  const { results, ceaRows, strategies } = runBase(m, { wtp: 30000 });

  assert.deepEqual(strategies, ['mono', 'combo']);
  assert.ok(results.strategies.mono);
  assert.ok(results.strategies.combo);

  const combo = ceaRows.find((r) => r.strategy === 'combo');
  assert.ok(Math.abs(combo.icer - 5976.80) < 1e-2, `combo ICER ${combo.icer}`);
});

test('traceFor: hiv combo trace has 20 cycles and the model\'s own state list', () => {
  const m = load('hiv.yaml');
  const { results } = runBase(m, {});
  const t = traceFor(results, 'combo');

  assert.equal(t.cycles, 20);
  assert.deepEqual(t.states, ['A', 'B', 'C', 'death']);
  assert.equal(t.series.A.length, 20);
  assert.equal(t.series.B.length, 20);
  assert.equal(t.series.C.length, 20);
  assert.equal(t.series.death.length, 20);
  // series[state][i] = occupancy at trace entry i (t = i+1) — spot-check against the raw trace.
  const rawTrace = results.strategies.combo.trace;
  assert.equal(t.series.A[0], rawTrace[0].occupancy.A);
  assert.equal(t.series.death[19], rawTrace[19].occupancy.death);
});

test('surgery: trace unavailable (tree, not markov); ceaRows Surgery ICER 28387.10 +/- 1e-2', () => {
  const m = load('surgery.yaml');
  const a = availability(m);
  assert.equal(a.trace, false);

  const { ceaRows, strategies } = runBase(m, {});
  assert.deepEqual(strategies, ['Surgery', 'Medication']);
  const surgery = ceaRows.find((r) => r.strategy === 'Surgery');
  assert.ok(Math.abs(surgery.icer - 28387.10) < 1e-2, `Surgery ICER ${surgery.icer}`);
});

test('gate: a broken model (row-sum error) returns ok:false with the E_ROWSUM finding', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: broken
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
  b: {utility: 0}
transitions:
  a: {a: 0.5, b: 0.3}
  b: {b: 1}
`);
  const g = gate(m);
  assert.equal(g.ok, false);
  assert.ok(g.errors.some((f) => f.code === 'E_ROWSUM'));
});

test('gate: a valid model (hiv) returns ok:true', () => {
  const m = load('hiv.yaml');
  assert.deepEqual(gate(m), { ok: true });
});

test('runTornado on hiv (a=mono, b=combo, wtp=30000) matches an independent dsa.oneWay call, one bar (p_AB)', () => {
  const m = load('hiv.yaml');
  m.params.get('p_AB').low = 0.15;
  m.params.get('p_AB').high = 0.25;

  const bars = runTornado(m, { a: 'mono', b: 'combo', wtp: 30000 });
  assert.equal(bars.length, 1);
  assert.equal(bars[0].param, 'p_AB');

  const expected = oneWay(m, { a: 'mono', b: 'combo', wtp: 30000 });
  assert.deepEqual(bars, expected);
});

test('psaDerived on a tiny seeded psa run: wtps length 26 starting at 0, ceac curves sum to 1 per wtp, evpi >= 0', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: p
settings: {cycles: 3, start: well, correction: none, psa: {n: 60, seed: 7}}
params:
  p_die:
    dist: beta(10, 90)
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: p_die}
  dead: {dead: 1}
strategies:
  usual: {}
  better: {p_die: 0.05}
`);
  const psaResult = runPsa(m, { n: 60, seed: 7 });
  const { wtps, ceacCurves, evpi, plane } = psaDerived(psaResult, { comparator: 'usual', wtpMax: 50000 });

  assert.equal(wtps.length, 26);
  assert.equal(wtps[0], 0);
  assert.equal(wtps[25], 50000);

  for (let w = 0; w < wtps.length; w++) {
    const sum = psaResult.strategies.reduce((s, name) => s + ceacCurves[name][w], 0);
    assert.ok(Math.abs(sum - 1) < 1e-9, `ceac sum at wtp[${w}]=${wtps[w]} is ${sum}`);
  }
  for (const v of evpi) assert.ok(v >= -1e-9, `evpi ${v} is negative`);

  assert.ok(Array.isArray(plane.better));
  assert.equal(plane.better.length, psaResult.samples.length);
});
