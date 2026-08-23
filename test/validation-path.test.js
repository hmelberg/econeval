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
