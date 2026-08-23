import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { makeEnv } from '../js/engine/resolve.js';
import { runMarkov } from '../js/engine/markov.js';
import { rng } from '../js/core/dist.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

const WELLDEAD = (extra = '') => parseModel(`
econeval: 1
type: markov
name: wd
settings: {cycles: 3, start: well, correction: none${extra}}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);

// m1 well=.9, m2=.81, m3=.729  (sum .9+.81+.729 = 2.439)
test('correction none, no discount', () => {
  const m = WELLDEAD();
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(r.totals.cost, 243.9);
  close(r.totals.qaly, 0.8 * 2.439);
  close(r.trace[0].occupancy.well, 0.9);
  close(r.trace[2].occupancy.dead, 0.271);
});

// basis sums: (1+.9)/2 + (.9+.81)/2 + (.81+.729)/2 = 2.5745
test('half-cycle correction', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: wd
settings: {cycles: 3, start: well, correction: half-cycle}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(r.totals.cost, 257.45);
  close(r.totals.qaly, 0.8 * 2.5745);
});

test('discounting, end-of-cycle timing under none', () => {
  const m = WELLDEAD();
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0.05, effect: 0 } });
  close(r.totals.cost, 100 * (0.9/1.05 + 0.81/1.05**2 + 0.729/1.05**3));
  close(r.totals.qaly, 0.8 * 2.439);           // effect rate 0 -> undiscounted
});

test('delayYears shifts every discount exponent', () => {
  const m = WELLDEAD();
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0.05, effect: 0 }, delayYears: 2 });
  close(r.totals.cost, 100 * (0.9/1.05**3 + 0.81/1.05**4 + 0.729/1.05**5));
});

test('transition rewards accrue on flow', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: rw
settings: {cycles: 3, start: well, correction: none}
states:
  well: {utility: 1}
  dead: {utility: 0}
transitions:
  well: {well: rest, dead: {p: 0.1, cost: 1000}}
  dead: {dead: 1}
`);
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(r.totals.cost, 1000 * (0.1 + 0.09 + 0.081));   // flows into death
});

test('multinomial row: deterministic = normalized counts; sampled differs but is seeded', () => {
  const src = `
econeval: 1
type: markov
name: mn
settings: {cycles: 3, start: well, correction: none}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {multinomial: {well: 9, dead: 1}}
  dead: {dead: 1}
`;
  const m = parseModel(src);
  const det = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(det.totals.cost, 243.9);
  const s1 = runMarkov(m, makeEnv(m, { mode: 'sample', rand: rng(4) }), { discount: { cost: 0, effect: 0 } });
  const s2 = runMarkov(m, makeEnv(m, { mode: 'sample', rand: rng(4) }), { discount: { cost: 0, effect: 0 } });
  close(s1.totals.cost, s2.totals.cost);
  assert.notEqual(s1.totals.cost, det.totals.cost);
});

test('extras tracked; c_-prefixed extras discounted at cost rate', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: ex
settings: {cycles: 3, start: well, correction: none}
states:
  well: {cost: 100, utility: 0.8, c_drug: 50, visits: 1}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0.05, effect: 0 } });
  close(r.totals.extras.c_drug, 50 * (0.9/1.05 + 0.81/1.05**2 + 0.729/1.05**3));
  close(r.totals.extras.visits, 2.439);
});

// Regression: a typo'd transition target (a state name not declared in `states:`) used to be
// silently dropped — advance() would credit it once, then every later cycle only sums over the
// declared state list, so the mass just vanishes with no error. Must fail loud instead.
test('typo\'d transition target throws instead of silently losing mass', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: typo
settings: {cycles: 3, start: well, correction: none}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, daed: 0.1}
  dead: {dead: 1}
`);
  assert.throws(() => runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } }), /daed/);
});

test('bad row sum raises with cycle context', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: bad
settings: {cycles: 2, start: a}
states: {a: {utility: 1}, b: {utility: 1}}
transitions:
  a: {a: 0.7, b: 0.6}
  b: {b: 1}
`);
  assert.throws(() => runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } }), /sum|cycle/i);
});

// Controller ruling (amends the brief): correction: 'life-table' is preserved as a literal by
// the parser, but the engine must treat it exactly like 'half-cycle' (both = cycle-average
// basis, per constraints.md). Same WELLDEAD fixture, only the correction value differs.
test('life-table correction is an alias for half-cycle (controller ruling)', () => {
  // WELLDEAD's template hardcodes `correction: none${extra}` — build both models directly
  // instead of fighting that template, so the correction value is unambiguous.
  const mHalf = parseModel(`
econeval: 1
type: markov
name: wd
settings: {cycles: 3, start: well, correction: half-cycle}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);
  const mLife = parseModel(`
econeval: 1
type: markov
name: wd
settings: {cycles: 3, start: well, correction: life-table}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);
  const rHalf = runMarkov(mHalf, makeEnv(mHalf, {}), { discount: { cost: 0.05, effect: 0.03 } });
  const rLife = runMarkov(mLife, makeEnv(mLife, {}), { discount: { cost: 0.05, effect: 0.03 } });
  close(rLife.totals.cost, rHalf.totals.cost);
  close(rLife.totals.qaly, rHalf.totals.qaly);
  // also pin against the brief's own hand-computed half-cycle totals (undiscounted case)
  const rHalfNoDisc = runMarkov(mHalf, makeEnv(mHalf, {}), { discount: { cost: 0, effect: 0 } });
  const rLifeNoDisc = runMarkov(mLife, makeEnv(mLife, {}), { discount: { cost: 0, effect: 0 } });
  close(rLifeNoDisc.totals.cost, 257.45);
  close(rHalfNoDisc.totals.cost, 257.45);
});
