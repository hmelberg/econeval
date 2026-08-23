import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';
import * as ops from '../js/ui/ops.js';

const M = () => parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 3, start: well}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);

test('addState invents a free name and a rest self-row', () => {
  const m = M();
  const m2 = ops.addState(m);
  assert.ok(m2.states.some(s => s.name === 'state1'));
  assert.deepEqual(m2.transitions.state1, { type: 'p', to: { state1: { p: 'rest' } } });
  // CONTROLLER RULING: original model must be unchanged (ops are pure).
  assert.equal(m.states.length, 2);
  assert.ok(!('state1' in m.transitions));
});

test('renameState rewrites rows, targets, start, layout', () => {
  let m = M(); m = ops.setLayout(m, 'well', [40, 40]);
  const m2 = ops.renameState(m, 'well', 'healthy');
  assert.ok(m2.states.some(s => s.name === 'healthy'));
  assert.ok(m2.transitions.healthy && !m2.transitions.well);
  assert.equal(m2.transitions.healthy.to.healthy.p, 'rest');
  assert.deepEqual(m2.settings.start, { healthy: 1 });
  assert.deepEqual(m2.layout.healthy, [40, 40]);
  assert.throws(() => ops.renameState(m2, 'healthy', 'dead'), /exists/);
});

test('renameState same-name is a no-op (returns an equal model, does not throw)', () => {
  const m = M();
  const m2 = ops.renameState(m, 'well', 'well');
  assert.deepEqual(m2, m);
});

test('deleteState scrubs every reference', () => {
  const m2 = ops.deleteState(M(), 'dead');
  assert.ok(!m2.transitions.dead);
  assert.ok(!('dead' in m2.transitions.well.to));
  assert.ok(!m2.states.some(s => s.name === 'dead'));
});

test('addTransition rest-default rule', () => {
  let m = ops.addState(M());                       // state1 row = {state1: rest}
  const a = ops.addTransition(m, 'state1', 'dead'); // row already has rest -> p 0
  assert.equal(a.transitions.state1.to.dead.p, 0);
  let noRest = ops.setTransitionAttr(M(), 'well', 'well', 'p', 0.9);
  const b = ops.addTransition(noRest, 'dead', 'well'); // dead row {dead:1} has no rest -> 'rest'
  assert.equal(b.transitions.dead.to.well.p, 'rest');
});

test('payoff and transition attr set/remove', () => {
  const m2 = ops.setStatePayoff(M(), 'well', 'c_drug', '2278');
  assert.equal(m2.states.find(s => s.name === 'well').payoffs.c_drug, '2278');
  const m3 = ops.setStatePayoff(m2, 'well', 'c_drug', null);
  assert.ok(!('c_drug' in m3.states.find(s => s.name === 'well').payoffs));
  const m4 = ops.setTransitionAttr(M(), 'well', 'dead', 'cost', 500);
  assert.equal(m4.transitions.well.to.dead.cost, 500);
});

test('setStatePayoff throws naming the key for reserved extras (source/notes); cost/utility stay allowed', () => {
  assert.throws(() => ops.setStatePayoff(M(), 'well', 'source', 'x'), /source/);
  assert.throws(() => ops.setStatePayoff(M(), 'well', 'notes', 'x'), /notes/);
  assert.doesNotThrow(() => ops.setStatePayoff(M(), 'well', 'cost', 999));
  assert.doesNotThrow(() => ops.setStatePayoff(M(), 'well', 'utility', 0.5));
});

test('every op round-trips through serialize/parse', () => {
  let m = ops.addState(M());
  m = ops.addTransition(m, 'state1', 'dead');
  m = ops.setLayout(m, 'state1', [120, 80]);
  assert.deepEqual(parseModel(serializeModel(m)), m);
});
