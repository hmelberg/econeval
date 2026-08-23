import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';
import { check } from '../js/analysis/check.js';
import * as ops from '../js/ui/ops.js';

// Root children (A, B) are strategies — no 'p'. A's children have a 'rest' sibling (Lose); B's
// child does not. Win's cost references param p_a, for the renameParam-orphan test.
const M = () => parseModel(`
econeval: 1
type: tree
name: t
params:
  p_a:
    value: 0.5
tree:
  Root:
    A:
      Win: {p: 0.5, utility: 10, cost: p_a}
      Lose: {p: rest, utility: 0}
    B:
      Win2: {p: 1, utility: 5}
layout:
  Root: [0, 0]
  Root/A: [10, 10]
  Root/A/Win: [20, 20]
  Root/A/Lose: [20, 40]
  Root/B: [10, 60]
`);

// --- nodeAt ---

test('nodeAt finds nodes by root-inclusive path', () => {
  const m = M();
  assert.equal(ops.nodeAt(m, ['Root']).name, 'Root');
  assert.equal(ops.nodeAt(m, ['Root', 'A', 'Win']).name, 'Win');
  assert.throws(() => ops.nodeAt(m, ['Root', 'Nope']), /Nope/);
  assert.throws(() => ops.nodeAt(m, ['Wrong']), /Wrong/);
});

// --- addChild: default-p rule (three cases) ---

test('addChild: siblings have a rest -> new child p is 0', () => {
  const m2 = ops.addChild(M(), ['Root', 'A'], 'New');
  const child = m2.tree.children[0].children.find((c) => c.name === 'New');
  assert.equal(child.p, 0);
  assert.deepEqual(child.payoffs, { utility: 0 });
  assert.deepEqual(child.children, []);
});

test('addChild: no rest among siblings -> new child p is rest', () => {
  const m2 = ops.addChild(M(), ['Root', 'B'], 'New2');
  const child = m2.tree.children[1].children.find((c) => c.name === 'New2');
  assert.equal(child.p, 'rest');
});

test('addChild: root child (strategy) gets NO p at all', () => {
  const m2 = ops.addChild(M(), ['Root'], 'C');
  const child = m2.tree.children.find((c) => c.name === 'C');
  assert.ok(!('p' in child));
});

test('addChild invents a free default name among siblings (branch1, branch2, ...)', () => {
  let m = ops.addChild(M(), ['Root'], 'branch1');
  m = ops.addChild(m, ['Root']); // should skip taken 'branch1'
  assert.ok(m.tree.children.some((c) => c.name === 'branch2'));
});

test('addChild rejects an empty or colliding explicit name', () => {
  assert.throws(() => ops.addChild(M(), ['Root'], ''), /empty/);
  assert.throws(() => ops.addChild(M(), ['Root'], 'A'), /exists/);
});

// --- renameNode: sibling-unique + layout subtree cascade ---

test('renameNode cascades layout keys for the node and its whole subtree', () => {
  const m2 = ops.renameNode(M(), ['Root', 'A'], 'AA');
  assert.equal(m2.tree.children[0].name, 'AA');
  assert.deepEqual(m2.layout['Root/AA'], [10, 10]);
  assert.deepEqual(m2.layout['Root/AA/Win'], [20, 20]);
  assert.deepEqual(m2.layout['Root/AA/Lose'], [20, 40]);
  assert.ok(!('Root/A' in m2.layout));
  assert.ok(!('Root/A/Win' in m2.layout));
  // untouched sibling subtree
  assert.deepEqual(m2.layout['Root/B'], [10, 60]);
});

test('renameNode rejects a name colliding with a sibling', () => {
  assert.throws(() => ops.renameNode(M(), ['Root', 'A'], 'B'), /exists/);
});

test('renameNode same-name is a no-op (returns an equal model, does not throw)', () => {
  const m = M();
  const m2 = ops.renameNode(m, ['Root', 'A'], 'A');
  assert.deepEqual(m2, m);
});

// --- deleteNode: not-root + scrubs subtree layout ---

test('deleteNode removes the node and scrubs its subtree layout, leaves siblings alone', () => {
  const m2 = ops.deleteNode(M(), ['Root', 'A']);
  assert.ok(!m2.tree.children.some((c) => c.name === 'A'));
  assert.ok(!('Root/A' in m2.layout));
  assert.ok(!('Root/A/Win' in m2.layout));
  assert.ok(!('Root/A/Lose' in m2.layout));
  assert.deepEqual(m2.layout['Root/B'], [10, 60]);
  assert.deepEqual(m2.layout['Root'], [0, 0]);
});

test('deleteNode refuses to delete the root', () => {
  assert.throws(() => ops.deleteNode(M(), ['Root']), /root/);
});

// --- setNodeAttr ---

test('setNodeAttr sets and null-removes p/delay/model/notes/source', () => {
  let m = ops.setNodeAttr(M(), ['Root', 'A', 'Win'], 'notes', 'check this');
  assert.equal(ops.nodeAt(m, ['Root', 'A', 'Win']).notes, 'check this');
  m = ops.setNodeAttr(m, ['Root', 'A', 'Win'], 'notes', null);
  assert.ok(!('notes' in ops.nodeAt(m, ['Root', 'A', 'Win'])));

  m = ops.setNodeAttr(M(), ['Root', 'A', 'Win'], 'delay', '1 month');
  assert.ok(Math.abs(ops.nodeAt(m, ['Root', 'A', 'Win']).delay - 1 / 12) < 1e-9);

  m = ops.setNodeAttr(M(), ['Root', 'A', 'Win'], 'model', 'sub1');
  assert.equal(ops.nodeAt(m, ['Root', 'A', 'Win']).model, 'sub1');
});

test('setNodeAttr throws when setting p on a root child', () => {
  assert.throws(() => ops.setNodeAttr(M(), ['Root', 'A'], 'p', 0.5), /root/);
});

test('setNodeAttr rejects an unknown key', () => {
  assert.throws(() => ops.setNodeAttr(M(), ['Root', 'A', 'Win'], 'bogus', 1), /bogus/);
});

// --- setNodePayoff ---

test('setNodePayoff sets and null-removes a payoff', () => {
  let m = ops.setNodePayoff(M(), ['Root', 'A', 'Win'], 'cost', 42);
  assert.equal(ops.nodeAt(m, ['Root', 'A', 'Win']).payoffs.cost, 42);
  m = ops.setNodePayoff(m, ['Root', 'A', 'Win'], 'utility', null);
  assert.ok(!('utility' in ops.nodeAt(m, ['Root', 'A', 'Win']).payoffs));
});

// --- setWith ---

test('setWith adds/removes params; empty with is removed entirely', () => {
  let m = ops.setWith(M(), ['Root', 'A', 'Win'], 'p_prog', 0.2);
  assert.deepEqual(ops.nodeAt(m, ['Root', 'A', 'Win']).with, { p_prog: 0.2 });
  m = ops.setWith(m, ['Root', 'A', 'Win'], 'p_prog', null);
  assert.ok(!('with' in ops.nodeAt(m, ['Root', 'A', 'Win'])));
});

// --- params CRUD ---

test('addParam invents a free default name and default spec', () => {
  const m2 = ops.addParam(M());
  assert.deepEqual(m2.params.get('param1'), { value: 0 });
});

test('addParam with explicit name/spec; rejects empty or colliding name', () => {
  const m2 = ops.addParam(M(), 'p_b', { value: 1, dist: 'beta(1,1)' });
  assert.deepEqual(m2.params.get('p_b'), { value: 1, dist: 'beta(1,1)' });
  assert.throws(() => ops.addParam(M(), ''), /empty/);
  assert.throws(() => ops.addParam(M(), 'p_a'), /exists/);
});

test('setParam sets/removes fields; removing value with no dist throws', () => {
  let m = ops.setParam(M(), 'p_a', 'low', 0.1);
  assert.equal(m.params.get('p_a').low, 0.1);
  m = ops.setParam(m, 'p_a', 'low', null);
  assert.ok(!('low' in m.params.get('p_a')));
  assert.throws(() => ops.setParam(M(), 'p_a', 'value', null), /dist/);
  const withDist = ops.setParam(M(), 'p_a', 'dist', 'beta(1,1)');
  const noValue = ops.setParam(withDist, 'p_a', 'value', null);
  assert.ok(!('value' in noValue.params.get('p_a')));
});

test('deleteParam removes the param', () => {
  const m2 = ops.deleteParam(M(), 'p_a');
  assert.ok(!m2.params.has('p_a'));
});

test('renameParam does NOT rewrite expressions; check() flags the orphan', () => {
  const m2 = ops.renameParam(M(), 'p_a', 'p_b');
  assert.ok(!m2.params.has('p_a'));
  assert.ok(m2.params.has('p_b'));
  // the expression string in the tree still names the OLD param — untouched, on purpose.
  const win = ops.nodeAt(m2, ['Root', 'A', 'Win']);
  assert.equal(win.payoffs.cost, 'p_a');
  const findings = check(m2);
  assert.ok(findings.some((f) => f.code === 'E_UNKNOWN_NAME' && /p_a/.test(f.message)));
});

test('renameParam throws on missing oldName and on a colliding newName', () => {
  assert.throws(() => ops.renameParam(M(), 'nope', 'x'), /not found/);
  const m = ops.addParam(M(), 'p_b');
  assert.throws(() => ops.renameParam(m, 'p_b', 'p_a'), /exists/);
});

test('renameParam same-name is a no-op (returns an equal model, does not throw)', () => {
  const m = M();
  const m2 = ops.renameParam(m, 'p_a', 'p_a');
  assert.deepEqual(m2, m);
});

// --- setSetting ---

test('setSetting sets a nested keyPath (discount.cost)', () => {
  const m2 = ops.setSetting(M(), 'discount.cost', 0.035);
  assert.equal(m2.settings.discount.cost, 0.035);
});

test("setSetting('cycle', ...) re-parses into cycleYears via the shared unit table", () => {
  const m2 = ops.setSetting(M(), 'cycle', '1 month');
  assert.ok(Math.abs(m2.settings.cycleYears - 1 / 12) < 1e-9);
});

test('setSetting sets psa.n', () => {
  const m2 = ops.setSetting(M(), 'psa.n', 500);
  assert.equal(m2.settings.psa.n, 500);
});

// --- everything round-trips ---

test('a sequence of tree/param/setting ops round-trips through serialize/parse', () => {
  let m = M();
  m = ops.addChild(m, ['Root'], 'C');
  m = ops.addChild(m, ['Root', 'C'], 'Win3');
  m = ops.setNodeAttr(m, ['Root', 'C', 'Win3'], 'p', 0.5);
  m = ops.setNodePayoff(m, ['Root', 'C', 'Win3'], 'utility', 7);
  m = ops.setWith(m, ['Root', 'C', 'Win3'], 'p_a', 0.9);
  m = ops.renameNode(m, ['Root', 'C'], 'CC');
  m = ops.addParam(m, 'p_c', { value: 2, low: 1, high: 3 });
  m = ops.setSetting(m, 'discount.cost', 0.03);
  m = ops.setSetting(m, 'cycle', '6 months');
  assert.deepEqual(parseModel(serializeModel(m)), m);
});
