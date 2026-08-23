import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { check } from '../js/analysis/check.js';

// The valid HIV markov fixture from Task 5's test/model.test.js — must yield zero ERROR findings
// (warnings are allowed: this fixture has no `settings.start`, so reachability is skipped, and
// p_AB does carry a dist so W_NO_DIST does not fire either — but the assertion only requires zero
// errors regardless).
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

test('valid HIV fixture yields zero errors (warnings allowed)', () => {
  const m = parseModel(HIV);
  const findings = check(m);
  assert.deepEqual(findings.filter((f) => f.level === 'error'), []);
});

test('E_ROWSUM — markov row does not sum to 1', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
  b: {utility: 0}
transitions:
  a: {a: 0.5, b: 0.3}
  b: {b: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_ROWSUM'));
});

test('E_PRANGE — explicit probability outside [0,1]', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
  b: {utility: 0}
transitions:
  a: {a: rest, b: 1.5}
  b: {b: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_PRANGE'));
});

test('E_TWO_RESTS — two rest targets in one row', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
  b: {utility: 0}
  c: {utility: 0}
transitions:
  a: {a: rest, b: rest, c: 0.1}
  b: {b: 1}
  c: {c: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_TWO_RESTS'));
});

test('E_NO_ROW — a state has no transitions row', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
  b: {utility: 0}
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_NO_ROW'));
});

test('E_UNKNOWN_STATE — transition target is not a defined state', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
transitions:
  a: {a: rest, ghost: 0.1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_UNKNOWN_STATE'));
});

test('E_UNKNOWN_NAME — expression references an undefined name', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1, cost: mystery_param}
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_UNKNOWN_NAME'));
});

test('E_UNKNOWN_NAME — referencing age without settings.age gets a targeted message', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1, cost: age}
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  const f = findings.find((fd) => fd.code === 'E_UNKNOWN_NAME');
  assert.ok(f);
  assert.match(f.message, /settings\.age/);
});

test('E_EXPR — unknown function call', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a:
    utility: 1
    cost: bta(1, 2)
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_EXPR' && f.level === 'error'));
});

test('E_EXPR — distribution call with wrong arity', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
params:
  p:
    dist: beta(5)
states:
  a: {utility: 1, cost: p}
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_EXPR' && f.level === 'error'));
});

test('E_PARAM_CYCLE — two params reference each other', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
params:
  p: {value: q}
  q: {value: p}
states:
  a: {utility: 1}
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_PARAM_CYCLE'));
});

test('E_SUBMODEL_MISSING — tree node references an unknown sub-model', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: x
tree:
  Root:
    Only:
      Leaf: {p: 1, model: ghost}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_SUBMODEL_MISSING'));
});

test('E_SUBMODEL_CYCLE — two tree sub-models reference each other', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: x
models:
  a:
    type: tree
    tree:
      R: {X: {p: 1, model: b}}
  b:
    type: tree
    tree:
      R: {X: {p: 1, model: a}}
tree:
  Root:
    Only:
      Leaf: {p: 1, model: a}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_SUBMODEL_CYCLE'));
});

test('E_DECISION_BELOW_ROOT — non-root node has children without p', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: x
tree:
  Root:
    Strat:
      BranchA: {utility: 1}
      BranchB: {utility: 0}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_DECISION_BELOW_ROOT'));
});

test('E_TREE_PSUM — sibling p sum != 1', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: x
tree:
  Root:
    Strat:
      BranchA: {p: 0.3, utility: 1}
      BranchB: {p: 0.3, utility: 0}
`);
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_TREE_PSUM'));
});

test('E_TABLE_SHAPE — malformed table (defensive: unreachable via parseModel, model.js already ' +
  'rejects this shape at parse time, so the fixture is constructed by mutating a valid parsed ' +
  'model directly — check() must still catch a hand-built/mutated Model, not just parser output)', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states: {a: {utility: 1}}
transitions: {a: {a: 1}}
`);
  m.tables = { mort: { age: [40, 50, 60], rate: [0.1, 0.2] } };
  const findings = check(m);
  assert.ok(findings.some((f) => f.code === 'E_TABLE_SHAPE'));
});

test('W_UNREACHABLE — a state is unreachable from start', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
states:
  a: {utility: 1}
  b: {utility: 0}
  ghost: {utility: 0}
transitions:
  a: {a: rest, b: 0.1}
  b: {b: 1}
  ghost: {ghost: 1}
`);
  const findings = check(m);
  const f = findings.find((fd) => fd.code === 'W_UNREACHABLE');
  assert.ok(f);
  assert.equal(f.level, 'warning');
});

test('W_NO_DIST — no param carries a dist', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1, start: a}
params:
  p: 0.1
states:
  a: {utility: 1, cost: p}
transitions:
  a: {a: 1}
`);
  const findings = check(m);
  const f = findings.find((fd) => fd.code === 'W_NO_DIST');
  assert.ok(f);
  assert.equal(f.level, 'warning');
});

test('check never throws on a malformed-but-parsed model (many simultaneous issues)', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: x
settings: {cycles: 1}
params:
  p: {value: q}
  q: {value: p}
states:
  a: {utility: 1, cost: mystery}
  b: {utility: 1}
transitions:
  a: {a: rest, b: rest, ghost: 3}
`);
  assert.doesNotThrow(() => check(m));
  const findings = check(m);
  assert.ok(findings.length > 0);
});
