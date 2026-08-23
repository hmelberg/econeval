// Tests for js/ui/validation-path.js's pure resolveFindingPath(path, model) -- the Validation
// tab's (Task 6, js/ui/results.js) path -> canvas-selection resolver. Mirrors
// test/inspector-match.test.js's own style: real parseModel() fixtures, cross-checked against a
// genuine check() run where useful so the resolver is proven against check.js's ACTUAL path
// vocabulary, not just self-consistent with its own assumptions about that vocabulary.

import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveFindingPath } from '../js/ui/validation-path.js';
import { parseModel } from '../js/core/model.js';
import { check } from '../js/analysis/check.js';
import { createStore } from '../js/ui/store.js';

// Round-trip proof (review fix): store.select(sel) alone never validates -- store.js only
// reconciles an invalid selection back to {kind:null,id:null} inside commit() (setText/applyOp),
// never inside select() itself. Forcing a no-op applyOp((m) => m) after selecting exercises the
// SAME reconciliation path (isSelectionValid) a real canvas gesture's own commit would, without
// actually changing the model -- so "the selection survives" is real proof it validates, not just
// that select() accepted whatever object it was handed.
function assertSelectionRoundTrips(text, sel) {
  const store = createStore(text);
  store.select(sel);
  store.applyOp((m) => m);
  assert.deepEqual(store.get().selection, sel, 'selection did not survive a reconciling commit — resolver produced an unresolvable id');
}

const MARKOV = `
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
`;

// Same fixture examples/surgery.yaml uses -- root name "Treatment?", root child "Surgery" --
// chosen deliberately so the check-path convention lines up with the brief's own example.
const TREE = `
econeval: 1
type: tree
name: t
tree:
  Treatment?:
    Surgery:
      cost: 5000
      Success: {p: 0.9, utility: 0.95}
      Failure: {p: rest, utility: 0.40}
    Medication:
      cost: 800
      Success: {p: 0.6, utility: 0.9}
      Failure: {p: rest, utility: 0.5}
`;

// Same "chronic" sub-model fixture test/run.test.js uses -- a tree top model attaching a markov
// sub-model named "chronic" with a "well" state, via `models:`.
const SUBMODEL = `
econeval: 1
type: tree
name: attach
models:
  chronic:
    type: markov
    settings: {cycles: 3, start: well, correction: none}
    params: {p_die: 0.1}
    states:
      well: {cost: 100, utility: 0.8}
      dead: {cost: 0, utility: 0}
    transitions:
      well: {well: rest, dead: p_die}
      dead: {dead: 1}
tree:
  Root:
    Treat:
      cost: 50
      Cure: {p: 0.5, model: chronic}
      Fail: {p: rest, model: chronic, with: {p_die: 0.5}}
`;

// --- states.<name> ---

test('states.well -> state well, top-level modelPath', () => {
  const model = parseModel(MARKOV);
  assert.deepEqual(resolveFindingPath('states.well', model), { kind: 'state', id: 'well', modelPath: [] });
});

test('states.<name>.<field> (a payoff eval error nested under a state) still resolves to the state', () => {
  const model = parseModel(MARKOV);
  assert.deepEqual(resolveFindingPath('states.well.cost', model), { kind: 'state', id: 'well', modelPath: [] });
});

// --- transitions.<from>... ---

test('transitions.well.to.dead -> state well (the row\'s SOURCE state, never a distinct edge)', () => {
  const model = parseModel(MARKOV);
  assert.deepEqual(resolveFindingPath('transitions.well.to.dead', model), { kind: 'state', id: 'well', modelPath: [] });
});

test('transitions.<from> (a bare row-level finding, e.g. E_ROWSUM/E_TWO_RESTS) resolves the same way', () => {
  const model = parseModel(MARKOV);
  assert.deepEqual(resolveFindingPath('transitions.well', model), { kind: 'state', id: 'well', modelPath: [] });
});

test('transitions.<from> against a REAL check() E_ROWSUM finding path resolves correctly', () => {
  const model = parseModel(`
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
  const findings = check(model);
  const f = findings.find((x) => x.code === 'E_ROWSUM');
  assert.ok(f, 'expected an E_ROWSUM finding');
  assert.deepEqual(resolveFindingPath(f.path, model), { kind: 'state', id: 'a', modelPath: [] });
});

// --- tree.<...> ---

test('tree.Surgery (root "Treatment?", omitted per check.js convention) -> node path [Treatment?, Surgery]', () => {
  const model = parseModel(TREE);
  assert.deepEqual(
    resolveFindingPath('tree.Surgery', model),
    { kind: 'node', id: ['Treatment?', 'Surgery'], modelPath: [] },
  );
});

test('tree.Surgery.Success (a grandchild) -> the full node path, root name re-attached', () => {
  const model = parseModel(TREE);
  assert.deepEqual(
    resolveFindingPath('tree.Surgery.Success', model),
    { kind: 'node', id: ['Treatment?', 'Surgery', 'Success'], modelPath: [] },
  );
});

test('bare "tree" -> the root node itself', () => {
  const model = parseModel(TREE);
  assert.deepEqual(resolveFindingPath('tree', model), { kind: 'node', id: ['Treatment?'], modelPath: [] });
});

test('tree.<name that does not exist> -> null (best-effort, never guesses)', () => {
  const model = parseModel(TREE);
  assert.equal(resolveFindingPath('tree.NoSuchNode', model), null);
});

// --- tree.<...>: a trailing FIELD segment (not a child) trims to the nearest ancestor node ---
// (review fix -- real check() paths run past the node name into payoff keys / `with.start` /
// etc., and the resolver must not bail out to a plain row just because the LAST segment isn't
// another child name.)

// Same TREE fixture, but Surgery's own `cost` payoff references an undefined name -> a real
// E_UNKNOWN_NAME finding at path 'tree.Surgery.cost' (walkTreeNode's own path-building: a node's
// OWN payoffs are checked at `${path}.${payoffKey}`, where `path` is the node's own bare
// check-path -- 'cost' here is never another child, it's a trailing field on Surgery itself).
const TREE_BAD_PAYOFF = `
econeval: 1
type: tree
name: t
tree:
  Treatment?:
    Surgery:
      cost: undefined_param_xyz
      Success: {p: 0.9, utility: 0.95}
      Failure: {p: rest, utility: 0.40}
    Medication:
      cost: 800
      Success: {p: 0.6, utility: 0.9}
      Failure: {p: rest, utility: 0.5}
`;

test('tree.Surgery.cost (a trailing payoff-field segment) trims to the Surgery node, not null', () => {
  const model = parseModel(TREE_BAD_PAYOFF);
  const expected = { kind: 'node', id: ['Treatment?', 'Surgery'], modelPath: [] };
  assert.deepEqual(resolveFindingPath('tree.Surgery.cost', model), expected);
});

test('tree.Surgery.cost against a REAL check() E_UNKNOWN_NAME finding path resolves correctly', () => {
  const model = parseModel(TREE_BAD_PAYOFF);
  const findings = check(model);
  const f = findings.find((x) => x.code === 'E_UNKNOWN_NAME' && x.path === 'tree.Surgery.cost');
  assert.ok(f, `expected an E_UNKNOWN_NAME finding at tree.Surgery.cost, got: ${findings.map((x) => `${x.code}@${x.path}`).join(', ')}`);
  assert.deepEqual(resolveFindingPath(f.path, model), { kind: 'node', id: ['Treatment?', 'Surgery'], modelPath: [] });
});

test('tree.Surgery.cost: the resolved selection round-trips through the store (isSelectionValid)', () => {
  const resolved = resolveFindingPath('tree.Surgery.cost', parseModel(TREE_BAD_PAYOFF));
  assertSelectionRoundTrips(TREE_BAD_PAYOFF, resolved);
});

// A "chemo-style" fixture (mirrors test/run.test.js's own attach fixture): Treat -> Cure attaches
// the markov sub-model 'chronic' with a `with: {start: <bad state>}` override -> a real
// E_UNKNOWN_STATE finding at path 'tree.Treat.Cure.with.start' (checkAttachments' own walk: the
// finding path is the ATTACHING node's own bare path (`tree.Treat.Cure`) plus `.with.start` --
// 'with'/'start' are never children of Cure, they're fields on it).
const CHEMO = `
econeval: 1
type: tree
name: chemo
models:
  chronic:
    type: markov
    settings: {cycles: 3, start: well, correction: none}
    states:
      well: {cost: 100, utility: 0.8}
      dead: {cost: 0, utility: 0}
    transitions:
      well: {well: rest, dead: 0.1}
      dead: {dead: 1}
tree:
  Root:
    Treat:
      cost: 50
      Cure: {p: 0.5, model: chronic, with: {start: nosuchstate}}
      Fail: {p: rest, model: chronic}
`;

test('tree.Treat.Cure.with.start (two trailing field segments) trims to the Cure node', () => {
  const model = parseModel(CHEMO);
  const expected = { kind: 'node', id: ['Root', 'Treat', 'Cure'], modelPath: [] };
  assert.deepEqual(resolveFindingPath('tree.Treat.Cure.with.start', model), expected);
});

test('tree.Treat.Cure.with.start against a REAL check() E_UNKNOWN_STATE finding path resolves correctly', () => {
  const model = parseModel(CHEMO);
  const findings = check(model);
  const f = findings.find((x) => x.code === 'E_UNKNOWN_STATE' && x.path === 'tree.Treat.Cure.with.start');
  assert.ok(f, `expected an E_UNKNOWN_STATE finding at tree.Treat.Cure.with.start, got: ${findings.map((x) => `${x.code}@${x.path}`).join(', ')}`);
  assert.deepEqual(resolveFindingPath(f.path, model), { kind: 'node', id: ['Root', 'Treat', 'Cure'], modelPath: [] });
});

test('tree.Treat.Cure.with.start: the resolved selection round-trips through the store', () => {
  const resolved = resolveFindingPath('tree.Treat.Cure.with.start', parseModel(CHEMO));
  assertSelectionRoundTrips(CHEMO, resolved);
});

test("tree.Nonexistent.cost -> null (the FIRST content segment never matches a child -- a broken/typo'd reference, never trimmed back to the root)", () => {
  const model = parseModel(TREE_BAD_PAYOFF);
  assert.equal(resolveFindingPath('tree.Nonexistent.cost', model), null);
});

// --- models.<name>.tree...: the same trailing-field trimming applies inside a nested sub-model ---

// A markov top model with a TREE sub-model 'branch' (models: content validation recurses
// regardless of the top model's own type) whose own root child 'Left' has a broken payoff ->
// E_UNKNOWN_NAME at 'models.branch.tree.Left.cost'.
const NESTED_TREE_SUBMODEL = `
econeval: 1
type: tree
name: nested
models:
  branch:
    type: tree
    tree:
      Root2:
        Left: {p: 0.5, cost: bad_expr_name}
        Right: {p: rest}
tree:
  Root: {}
`;

test("models.branch.tree.Left.cost (trailing field, inside a nested TREE sub-model) trims to the Left node, modelPath ['branch']", () => {
  const model = parseModel(NESTED_TREE_SUBMODEL);
  const expected = { kind: 'node', id: ['Root2', 'Left'], modelPath: ['branch'] };
  assert.deepEqual(resolveFindingPath('models.branch.tree.Left.cost', model), expected);
});

test('models.branch.tree.Left.cost against a REAL check() finding path resolves correctly', () => {
  const model = parseModel(NESTED_TREE_SUBMODEL);
  const findings = check(model);
  const f = findings.find((x) => x.path === 'models.branch.tree.Left.cost');
  assert.ok(f, `expected a finding at models.branch.tree.Left.cost, got: ${findings.map((x) => `${x.code}@${x.path}`).join(', ')}`);
  assert.deepEqual(resolveFindingPath(f.path, model), { kind: 'node', id: ['Root2', 'Left'], modelPath: ['branch'] });
});

test('models.branch.tree.Left.cost: the resolved selection round-trips through the store', () => {
  const resolved = resolveFindingPath('models.branch.tree.Left.cost', parseModel(NESTED_TREE_SUBMODEL));
  assertSelectionRoundTrips(NESTED_TREE_SUBMODEL, resolved);
});

// --- models.<name>... (sub-model-scoped findings) ---

test("models.chronic.states.well -> state well, modelPath ['chronic']", () => {
  const model = parseModel(SUBMODEL);
  assert.deepEqual(
    resolveFindingPath('models.chronic.states.well', model),
    { kind: 'state', id: 'well', modelPath: ['chronic'] },
  );
});

test('models.<unknown submodel>.states.well -> null (ambiguous/unresolvable, plain row)', () => {
  const model = parseModel(SUBMODEL);
  assert.equal(resolveFindingPath('models.nope.states.well', model), null);
});

test('a sub-model-scoped finding that does not name a state/node -> null', () => {
  const model = parseModel(SUBMODEL);
  assert.equal(resolveFindingPath('models.chronic.params.p_die.value', model), null);
});

// --- never click-through ---

test('params.x -> null', () => {
  const model = parseModel(MARKOV);
  assert.equal(resolveFindingPath('params.x', model), null);
});

test('settings.cycles -> null', () => {
  const model = parseModel(MARKOV);
  assert.equal(resolveFindingPath('settings.cycles', model), null);
});

test('(model) internal-error sentinel path -> null', () => {
  const model = parseModel(MARKOV);
  assert.equal(resolveFindingPath('(model)', model), null);
});

// --- defensive: never throws on odd input ---

test('non-string path -> null', () => {
  const model = parseModel(MARKOV);
  assert.equal(resolveFindingPath(undefined, model), null);
  assert.equal(resolveFindingPath(null, model), null);
});

test('empty string path -> null', () => {
  const model = parseModel(MARKOV);
  assert.equal(resolveFindingPath('', model), null);
});

test('null model -> null, never throws', () => {
  assert.equal(resolveFindingPath('states.well', null), null);
});

test('states.<name> against a TREE model (type mismatch) -> null', () => {
  const model = parseModel(TREE);
  assert.equal(resolveFindingPath('states.well', model), null);
});

test('tree.<...> against a MARKOV model (type mismatch) -> null', () => {
  const model = parseModel(MARKOV);
  assert.equal(resolveFindingPath('tree.Surgery', model), null);
});
