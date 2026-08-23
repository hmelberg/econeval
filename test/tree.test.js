import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { makeEnv } from '../js/engine/resolve.js';
import { runTree } from '../js/engine/tree.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

const SURGERY = parseModel(`
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

test('rollback expected values', () => {
  const r = runTree(SURGERY, makeEnv(SURGERY, {}), { discount: { cost: 0, effect: 0 } });
  // Surgery: cost 5000 + .1*2000 = 5200 ; EU = .9*.95 + .1*.40 = 0.895
  close(r.strategies.Surgery.cost, 5200);
  close(r.strategies.Surgery.qaly, 0.895);
  // Medication: 800 ; EU = .6*.9 + .4*.5 = 0.74
  close(r.strategies.Medication.cost, 800);
  close(r.strategies.Medication.qaly, 0.74);
});

test('extras accumulate weighted by path probability', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: ex
tree:
  Root:
    OnlyOption:
      relapses: 1
      Good: {p: 0.7, utility: 1}
      Bad: {p: rest, utility: 0, relapses: 2}
`);
  const r = runTree(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(r.strategies.OnlyOption.extras.relapses, 1 + 0.3 * 2);
});

test('sibling probabilities must sum to 1', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: bad
tree:
  Root:
    A:
      X: {p: 0.7, utility: 1}
      Y: {p: 0.7, utility: 0}
`);
  assert.throws(() => runTree(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } }), /sum/i);
});

test('model terminal without attach throws', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: sub
models:
  s:
    type: markov
    settings: {cycles: 1}
    states: {a: {utility: 1}}
    transitions: {a: {a: 1}}
tree:
  Root:
    A:
      Leaf: {p: 1, model: s}
`);
  assert.throws(() => runTree(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } }), /attach|sub-model/i);
});
