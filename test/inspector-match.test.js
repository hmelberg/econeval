// Tests for the DOM-free helpers exported from js/ui/inspector.js: the check.js-path
// construction (scopePrefix, nodePathToCheckPath) and the findings/rendered-field matching
// (splitFindings, countByLevel). The DOM-heavy half of inspector.js (createInspector itself) is
// exercised by manual verification only, per constraints.md ("Pure logic ... lives in DOM-free
// modules with tests; DOM modules ... hold no business logic" — importing js/ui/inspector.js here
// is safe: it also imports js/ui/canvas/index.js and js/ui/panels.js, both of which are already proven
// Node-import-safe by test/canvas-model.test.js and test/panels.test.js — nothing at module-load
// time touches `document`).

import test from 'node:test';
import assert from 'node:assert/strict';
import { scopePrefix, nodePathToCheckPath } from '../js/ui/outline/build.js';
import { splitFindings, countByLevel } from '../js/ui/inspector.js';
import { parseModel } from '../js/core/model.js';
import { check } from '../js/analysis/check.js';

// --- scopePrefix ---

test('scopePrefix: no modelPath (top-level) -> empty string', () => {
  assert.equal(scopePrefix([]), '');
  assert.equal(scopePrefix(undefined), '');
});

test('scopePrefix: one level matches check.js\'s "models.<name>." prefix', () => {
  assert.equal(scopePrefix(['sub']), 'models.sub.');
});

test('scopePrefix: nested chain matches check.js\'s recursive prefix building', () => {
  assert.equal(scopePrefix(['a', 'b']), 'models.a.models.b.');
});

// --- nodePathToCheckPath ---

test('nodePathToCheckPath: the root itself -> bare "tree" (root name is never in the path)', () => {
  assert.equal(nodePathToCheckPath(['Root']), 'tree');
});

test('nodePathToCheckPath: a root child -> "tree.<childName>"', () => {
  assert.equal(nodePathToCheckPath(['Root', 'Surgery']), 'tree.Surgery');
});

test('nodePathToCheckPath: a grandchild -> dot-joined, still omitting the root name', () => {
  assert.equal(nodePathToCheckPath(['Root', 'Surgery', 'Success']), 'tree.Surgery.Success');
});

// --- nodePathToCheckPath against a REAL check() run: confirms the convention actually matches
// check.js's own path construction, not just self-consistency of the helper. ---

const TREE = `
econeval: 1
type: tree
name: t

tree:
  Root:
    Surgery:
      Success: {p: 0.9, utility: 1}
      Fail: {p: 0.05, utility: 0}
    Medication:
      utility: 1
`;

test('nodePathToCheckPath matches check.js\'s actual E_TREE_PSUM path for a broken sibling group', () => {
  const model = parseModel(TREE);
  const findings = check(model);
  const psum = findings.find((f) => f.code === 'E_TREE_PSUM');
  assert.ok(psum, 'expected an E_TREE_PSUM finding (Success+Fail sum to 0.95, not 1)');
  // Surgery is the PARENT of the broken sibling group -> its own bare check-path.
  assert.equal(psum.path, nodePathToCheckPath(['Root', 'Surgery']));
});

const PARAM_MODEL = `
econeval: 1
type: markov
name: m
settings: {cycles: 3, start: well}
params:
  bad: {value: 1, dist: "unknown_fn(1, 2)"}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`;

test('scopePrefix\'d param path matches a real top-level check() finding path', () => {
  const model = parseModel(PARAM_MODEL);
  const findings = check(model);
  const f = findings.find((x) => x.path === `${scopePrefix([])}params.bad.dist`);
  assert.ok(f, `expected a finding at params.bad.dist, got paths: ${findings.map((x) => x.path).join(', ')}`);
  assert.equal(f.level, 'error');
});

// --- countByLevel ---

test('countByLevel: tallies errors and warnings separately, ignores nothing else', () => {
  const findings = [
    { level: 'error', code: 'A', path: 'x', message: 'm1' },
    { level: 'warning', code: 'B', path: 'y', message: 'm2' },
    { level: 'error', code: 'C', path: 'z', message: 'm3' },
  ];
  assert.deepEqual(countByLevel(findings), { errors: 2, warnings: 1 });
});

test('countByLevel: empty list -> zero/zero', () => {
  assert.deepEqual(countByLevel([]), { errors: 0, warnings: 0 });
});

// --- splitFindings ---

test('splitFindings: a finding whose path is in renderedPaths goes into inline, keyed by path', () => {
  const findings = [
    { level: 'error', code: 'A', path: 'states.well.cost', message: 'm1' },
    { level: 'warning', code: 'B', path: 'params', message: 'm2' },
  ];
  const { inline, rest } = splitFindings(findings, new Set(['states.well.cost']));
  assert.deepEqual(inline.get('states.well.cost'), [findings[0]]);
  assert.deepEqual(rest, [findings[1]]);
});

test('splitFindings: multiple findings on the same rendered path stack under that one key', () => {
  const findings = [
    { level: 'error', code: 'A', path: 'p', message: 'm1' },
    { level: 'error', code: 'B', path: 'p', message: 'm2' },
  ];
  const { inline, rest } = splitFindings(findings, new Set(['p']));
  assert.deepEqual(inline.get('p'), findings);
  assert.deepEqual(rest, []);
});

test('splitFindings: no rendered paths -> everything falls to rest, in original order', () => {
  const findings = [
    { level: 'error', code: 'A', path: 'a', message: 'm1' },
    { level: 'warning', code: 'B', path: 'b', message: 'm2' },
  ];
  const { inline, rest } = splitFindings(findings, new Set());
  assert.equal(inline.size, 0);
  assert.deepEqual(rest, findings);
});
