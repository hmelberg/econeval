import test from 'node:test';
import assert from 'node:assert/strict';
import { psa, ceac, evpi, cePlane } from '../js/analysis/psa.js';
import { parseModel } from '../js/core/model.js';

test('psa is seeded-reproducible and shares draws across strategies', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: p
settings: {cycles: 3, start: well, correction: none, psa: {n: 50, seed: 7}}
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
  discount10: {}
`);
  const r1 = psa(m, {});
  const r2 = psa(m, {});
  assert.equal(r1.samples.length, 50);
  assert.deepEqual(r1.samples[0], r2.samples[0]);
  // identical strategies + shared draw => identical outcomes per iteration
  for (const s of r1.samples) assert.equal(s.cost.usual, s.cost.discount10);
});

test('evpi and ceac on hand-built samples', () => {
  const res = { strategies: ['a', 'b'], samples: [
    { cost: { a: 0, b: 0 }, qaly: { a: 1.0, b: 0.4 } },
    { cost: { a: 0, b: 0 }, qaly: { a: 0.0, b: 0.4 } },
  ]};
  // wtp 10: NMB a = {10, 0}, b = {4, 4}; mean a=5 > b=4; E[max]=(10+4)/2=7 -> EVPI 2
  const e = evpi(res, [10]);
  assert.ok(Math.abs(e[0] - 2) < 1e-12);
  const c = ceac(res, [10]);
  assert.deepEqual(c.curves.a, [0.5]);
  assert.deepEqual(c.curves.b, [0.5]);
});

test('cePlane increments vs comparator', () => {
  const res = { strategies: ['a', 'b'], samples: [
    { cost: { a: 100, b: 300 }, qaly: { a: 1, b: 1.5 } },
  ]};
  const p = cePlane(res, { comparator: 'a' });
  assert.deepEqual(p.b, [{ dcost: 200, dqaly: 0.5 }]);
});

test('correlations: copula induces rank correlation, marginals survive', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: corr
settings:
  cycles: 1
  start: a
  correction: none
  psa:
    n: 1500
    seed: 3
    correlations:
      - {a: x, b: y, r: 0.95}
params:
  x:
    dist: normal(0, 1)
  y:
    dist: normal(0, 1)
  # params must be used or the model ignores them; feed them into a payoff
states:
  a: {cost: x + y, utility: 1}
transitions: {a: {a: 1}}
`);
  const r = psa(m, {});
  // cost per iteration = x+y (1 cycle, occupancy 1). Var(x+y) = 2 + 2*0.95 = 3.9 under corr;
  // would be 2 if independent. Check sample variance is near 3.9.
  const costs = r.samples.map(s => s.cost.base);
  const meanC = costs.reduce((s, v) => s + v, 0) / costs.length;
  const varC = costs.reduce((s, v) => s + (v - meanC) ** 2, 0) / (costs.length - 1);
  assert.ok(varC > 3.2, `variance ${varC} suggests correlation not applied`);
  assert.ok(Math.abs(meanC) < 0.2);
});
