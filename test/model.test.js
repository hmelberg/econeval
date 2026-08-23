import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, ModelError } from '../js/core/model.js';

const HIV = `
econeval: 1
type: markov
name: HIV combination therapy
settings:
  cycles: 20
  cycle: 1 year
  discount: {cost: 0.06, effect: 0}
  correction: none
  psa: {n: 1000, seed: 42}
params:
  p_AB:
    value: 0.202
    low: 0.15
    high: 0.25
    dist: beta(202, 798)
  rr: 1
  c_drug: 2278
states:
  A: {cost: 2756 + c_drug, utility: 0.85}
  B: {cost: 3052 + c_drug, utility: 0.71}
  death: {cost: 0, utility: 0}
transitions:
  A: {A: rest, B: p_AB * rr, death: 0.01}
  B: {B: rest, death: 0.15}
  death: {death: 1}
strategies:
  mono: {}
  combo:
    c_drug: 5343
    rr: lognormal(-0.675, 0.173)
`;

test('parses and normalizes the markov example', () => {
  const m = parseModel(HIV);
  assert.equal(m.type, 'markov');
  assert.equal(m.settings.cycles, 20);
  assert.equal(m.settings.cycleYears, 1);
  assert.equal(m.settings.discount.cost, 0.06);
  assert.equal(m.settings.psa.seed, 42);
  assert.deepEqual(m.params.get('p_AB'), { value: 0.202, low: 0.15, high: 0.25, dist: 'beta(202, 798)' });
  assert.deepEqual(m.params.get('rr'), { value: 1 });
  assert.equal(m.states.length, 3);
  assert.deepEqual(m.states[0], { name: 'A', payoffs: { cost: '2756 + c_drug', utility: 0.85 } });
  assert.deepEqual(m.transitions.A, { type: 'p', to: { A: { p: 'rest' }, B: { p: 'p_AB * rr' }, death: { p: 0.01 } } });
  assert.deepEqual(m.strategies.combo.overrides, { c_drug: 5343, rr: 'lognormal(-0.675, 0.173)' });
  assert.equal(m.settings.start, null);
});

test('defaults: correction, discount, psa, implicit base strategy, start string', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: mini
settings: {cycles: 3, start: well}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);
  assert.equal(m.settings.correction, 'half-cycle');
  assert.deepEqual(m.settings.discount, { cost: 0, effect: 0 });
  assert.equal(m.settings.psa.n, 1000);
  assert.deepEqual(m.settings.start, { well: 1 });
  assert.deepEqual(Object.keys(m.strategies), ['base']);
});

test('multinomial rows and transition rewards normalize', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: forms
settings: {cycles: 2}
states:
  a: {utility: 1}
  b: {utility: 0}
transitions:
  a: {multinomial: {a: 9, b: 1}}
  b: {b: {p: 1, cost: 500}}
`);
  assert.deepEqual(m.transitions.a, { type: 'multinomial', counts: { a: 9, b: 1 } });
  assert.deepEqual(m.transitions.b.to.b, { p: 1, cost: 500 });
});

test('cycle units', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: u
settings: {cycles: 12, cycle: 1 month}
states: {a: {utility: 1}}
transitions: {a: {a: 1}}
`);
  assert.ok(Math.abs(m.settings.cycleYears - 1/12) < 1e-12);
});

test('flow-comma mistake gets a targeted hint', () => {
  const bad = `
econeval: 1
type: markov
name: x
settings: {cycles: 1}
params:
  p: {value: 1, dist: beta(1, 2)}
states: {a: {utility: 1}}
transitions: {a: {a: 1}}
`;
  try { parseModel(bad); assert.fail('should throw'); }
  catch (err) {
    assert.ok(err instanceof ModelError);
    assert.match(err.hint ?? '', /quote|block style/i);
  }
});

test('unknown param keys are rejected (strict normalization)', () => {
  assert.throws(() => parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1}
params:
  p:
    value: 1
    typo_field: 3
states: {a: {utility: 1}}
transitions: {a: {a: 1}}
`), /typo_field/);
});

test('structural errors: missing name/cycles/states, unknown type', () => {
  assert.throws(() => parseModel('econeval: 1\ntype: nope\nname: x'), /type/);
  assert.throws(() => parseModel('econeval: 1\ntype: markov\nname: x\nstates: {a: {utility: 1}}\ntransitions: {a: {a: 1}}'), /cycles/);
  assert.throws(() => parseModel('econeval: 1\ntype: markov\nname: x\nsettings: {cycles: 1}'), /states/);
});
