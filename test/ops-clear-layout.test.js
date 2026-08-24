import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';
import * as ops from '../js/ui/ops.js';

const TREE = () => parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Win: {p: rest, utility: 1}
    B:
      Win: {p: rest, utility: 2}
layout:
  Root: [0, 0]
  Root/A: [10, 10]
  Root/A/Win: [20, 20]
  Root/B: [10, 60]
`);

const MARKOV = () => parseModel(`
econeval: 1
type: markov
name: m
settings:
  cycles: 1
states:
  well: {cost: 1, utility: 1}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
layout:
  well: [10, 10]
  dead: [90, 10]
`);

test('clearLayout() with no key drops the whole layout as null', () => {
  const m2 = ops.clearLayout(TREE());
  assert.equal(m2.layout, null);
});

test('clearLayout(key) on a tree drops the node AND its subtree', () => {
  const m2 = ops.clearLayout(TREE(), 'Root/A');
  assert.ok(!('Root/A' in m2.layout));
  assert.ok(!('Root/A/Win' in m2.layout));
  assert.deepEqual(m2.layout['Root/B'], [10, 60]);
  assert.deepEqual(m2.layout.Root, [0, 0]);
});

test('clearLayout(key) on a markov model drops exactly that key', () => {
  const m2 = ops.clearLayout(MARKOV(), 'well');
  assert.ok(!('well' in m2.layout));
  assert.deepEqual(m2.layout.dead, [90, 10]);
});

test('clearing the last remaining key normalizes layout to null', () => {
  let m = ops.clearLayout(MARKOV(), 'well');
  m = ops.clearLayout(m, 'dead');
  assert.equal(m.layout, null);
});

test('clearing a key that is not there is a no-op, not an error', () => {
  const m2 = ops.clearLayout(MARKOV(), 'ghost');
  assert.deepEqual(m2.layout, MARKOV().layout);
});

test('a cleared model round-trips through serialize/parse', () => {
  const m2 = ops.clearLayout(TREE());
  assert.deepEqual(parseModel(serializeModel(m2)), m2);
});

test('does not mutate its input', () => {
  const m = TREE();
  ops.clearLayout(m, 'Root/A');
  assert.deepEqual(m.layout['Root/A'], [10, 10]);
});
