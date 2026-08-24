import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';
import * as ops from '../js/ui/ops.js';

// Root children A, B are strategies (no p). A's children include a 'rest' sibling; B's do not.
// Both A and B have a child named 'Win' — that pair is the name-collision case.
const M = () => parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Win: {p: 0.5, utility: 10}
      Lose: {p: rest, utility: 0}
    B:
      Win: {p: 1, utility: 5}
layout:
  Root: [0, 0]
  Root/A: [10, 10]
  Root/A/Win: [20, 20]
  Root/A/Lose: [20, 40]
  Root/B: [10, 60]
  Root/B/Win: [20, 60]
`);

test('re-parents a node and keeps an existing p untouched', () => {
  const m2 = ops.moveNode(M(), ['Root', 'A', 'Lose'], ['Root', 'B']);
  assert.equal(ops.nodeAt(m2, ['Root', 'B', 'Lose']).p, 'rest');
  assert.deepEqual(ops.nodeAt(m2, ['Root', 'A']).children.map((c) => c.name), ['Win']);
  assert.deepEqual(ops.nodeAt(m2, ['Root', 'B']).children.map((c) => c.name), ['Win', 'Lose']);
});

test('re-parenting moves the whole subtree layout, not just the node', () => {
  const m2 = ops.moveNode(M(), ['Root', 'B'], ['Root', 'A']);
  assert.deepEqual(m2.layout['Root/A/B'], [10, 60]);
  assert.deepEqual(m2.layout['Root/A/B/Win'], [20, 60]);
  assert.ok(!('Root/B' in m2.layout));
  assert.ok(!('Root/B/Win' in m2.layout));
});

test('promoting to a root child removes p (strategies are unconditional)', () => {
  const m2 = ops.moveNode(M(), ['Root', 'A', 'Win'], ['Root']);
  assert.equal(ops.nodeAt(m2, ['Root', 'Win']).p, undefined);
  assert.deepEqual(m2.layout['Root/Win'], [20, 20]);
});

test('demoting a strategy gives it 0 when a destination sibling already has rest', () => {
  // A's children: Win (0.5), Lose (rest) -> B lands beside a 'rest' sibling
  const m2 = ops.moveNode(M(), ['Root', 'B'], ['Root', 'A']);
  assert.equal(ops.nodeAt(m2, ['Root', 'A', 'B']).p, 0);
});

test("demoting a strategy gives it 'rest' when no destination sibling has one", () => {
  // B's children: Win (1) -> no 'rest' present
  const m2 = ops.moveNode(M(), ['Root', 'A'], ['Root', 'B']);
  assert.equal(ops.nodeAt(m2, ['Root', 'B', 'A']).p, 'rest');
});

test('rejects moving the root', () => {
  assert.throws(() => ops.moveNode(M(), ['Root'], ['Root', 'A']), /root/i);
});

test('rejects dropping a node onto itself', () => {
  assert.throws(() => ops.moveNode(M(), ['Root', 'A'], ['Root', 'A']), /itself/i);
});

test('rejects dropping a node into its own subtree', () => {
  assert.throws(() => ops.moveNode(M(), ['Root', 'A'], ['Root', 'A', 'Win']), /descendant/i);
});

test('rejects a sibling name collision instead of renaming silently', () => {
  assert.throws(() => ops.moveNode(M(), ['Root', 'A', 'Win'], ['Root', 'B']), /already exists/i);
});

test('rejects a non-tree model and an unknown path', () => {
  const markov = parseModel('econeval: 1\ntype: markov\nname: m\nsettings:\n  cycles: 1\nstates:\n  a: {cost: 0}\ntransitions:\n  a: {a: 1}\n');
  assert.throws(() => ops.moveNode(markov, ['Root'], ['Root']), /tree/);
  assert.throws(() => ops.moveNode(M(), ['Root', 'Nope'], ['Root', 'B']), /Nope/);
});

test('does not mutate its input', () => {
  const m = M();
  ops.moveNode(m, ['Root', 'A', 'Lose'], ['Root', 'B']);
  assert.deepEqual(ops.nodeAt(m, ['Root', 'A']).children.map((c) => c.name), ['Win', 'Lose']);
});

test('the result round-trips through serialize/parse', () => {
  const m2 = ops.moveNode(M(), ['Root', 'B'], ['Root', 'A']);
  assert.deepEqual(parseModel(serializeModel(m2)), m2);
});

test('dropping a node onto its own parent is a no-op', () => {
  const m = M();
  const m2 = ops.moveNode(m, ['Root', 'A', 'Win'], ['Root', 'A']);
  assert.deepEqual(ops.nodeAt(m2, ['Root', 'A']).children.map((c) => c.name), ['Win', 'Lose']);
  assert.deepEqual(m2.layout, m.layout);
});

test('a distinct numeric p value is preserved when moving', () => {
  const m1 = parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Preserve: {p: 0.5, utility: 10}
    B:
      Keep: {p: rest, utility: 5}
`);
  const m2 = ops.moveNode(m1, ['Root', 'A', 'Preserve'], ['Root', 'B']);
  assert.equal(ops.nodeAt(m2, ['Root', 'B', 'Preserve']).p, 0.5);
});
