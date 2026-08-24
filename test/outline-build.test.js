import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import {
  buildOutline, filterRows, attachFindings, rowForSelection, collapseFilter,
  ancestorIds, addAfterIndex,
} from '../js/ui/outline/build.js';

const MARKOV = () => parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 3}
params:
  c_well: {value: 100}
states:
  well: {cost: c_well, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);

const TREE = () => parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Win: {p: rest, utility: 10}
`);

const byId = (rows, id) => rows.find((r) => r.id === id);

test('markov: states are depth 1, their outgoing transitions depth 2', () => {
  const rows = buildOutline(MARKOV());
  assert.equal(byId(rows, 'group:structure').depth, 0);
  const well = byId(rows, 'state:well');
  assert.equal(well.depth, 1);
  assert.equal(well.parentId, 'group:structure');
  assert.deepEqual(well.sel, { kind: 'state', id: 'well', modelPath: [] });

  const edge = byId(rows, 'edge:well>dead');
  assert.equal(edge.depth, 2);
  assert.equal(edge.parentId, 'state:well');
  assert.equal(edge.label, '→ dead');
  assert.equal(edge.detail, '0.1');
  assert.deepEqual(edge.sel, { kind: 'edge', id: { from: 'well', to: 'dead' }, modelPath: [] });
});

test("markov: a 'rest' transition shows rest verbatim", () => {
  assert.equal(byId(buildOutline(MARKOV()), 'edge:well>well').detail, 'rest');
});

test('tree: nodes nest by depth with root-inclusive path ids', () => {
  const rows = buildOutline(TREE());
  assert.equal(byId(rows, 'node:Root').depth, 1);
  assert.equal(byId(rows, 'node:Root/A').depth, 2);
  const win = byId(rows, 'node:Root/A/Win');
  assert.equal(win.depth, 3);
  assert.equal(win.parentId, 'node:Root/A');
  assert.deepEqual(win.sel, { kind: 'node', id: ['Root', 'A', 'Win'], modelPath: [] });
});

test('parameters and settings get their own groups', () => {
  const rows = buildOutline(MARKOV());
  const p = byId(rows, 'param:c_well');
  assert.equal(p.parentId, 'group:parameters');
  assert.deepEqual(p.checkPaths, ['params.c_well']);
  assert.ok(byId(rows, 'group:settings'));
});

test('modelPath scopes both the selection and the check paths', () => {
  const rows = buildOutline(MARKOV(), ['post']);
  const well = byId(rows, 'state:well');
  assert.deepEqual(well.sel.modelPath, ['post']);
  assert.ok(well.checkPaths.every((p) => p.startsWith('models.post.')));
});

// --- scoped structure + top-level params/settings/sub-models (final-review Finding 1) ---
//
// The three-argument form is what lets ONE outline mix a sub-model's structure with the document
// root's parameters/settings/sub-model registry, which is exactly what spec §3's "Scope" paragraph
// requires (STRUCTURE follows the canvas scope; PARAMETERS and SETTINGS stay top-level in v1).

const WITH_SUB = () => parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 3}
params:
  c_well: {value: 100}
states:
  well: {cost: c_well, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
models:
  post:
    type: markov
    settings: {cycles: 2}
    params:
      c_sub: {value: 5}
    states:
      healthy: {cost: 0, utility: 1}
      gone: {cost: 0, utility: 0}
    transitions:
      healthy: {healthy: rest, gone: 0.05}
      gone: {gone: 1}
`);

test('scoped build: structure comes from the sub-model, params/settings/sub-models from the top', () => {
  const top = WITH_SUB();
  const rows = buildOutline(top.models.post, ['post'], top);

  // Structure: the SUB-MODEL's states, scoped for both selection and check paths.
  assert.ok(byId(rows, 'state:healthy'), 'the sub-model\'s own states are the structure rows');
  assert.equal(byId(rows, 'state:well'), undefined, 'the top-level states are NOT listed while scoped in');
  assert.deepEqual(byId(rows, 'state:healthy').sel, { kind: 'state', id: 'healthy', modelPath: ['post'] });
  assert.deepEqual(byId(rows, 'state:healthy').checkPaths, [
    'models.post.states.healthy', 'models.post.transitions.healthy',
  ]);
  assert.deepEqual(byId(rows, 'edge:healthy>gone').sel, {
    kind: 'edge', id: { from: 'healthy', to: 'gone' }, modelPath: ['post'],
  });

  // Parameters: the TOP-LEVEL model's, unprefixed — the sub-model's own params are YAML-only in v1.
  assert.ok(byId(rows, 'param:c_well'), 'top-level params stay listed while scoped in');
  assert.equal(byId(rows, 'param:c_sub'), undefined, 'a sub-model\'s own params are not listed');
  assert.deepEqual(byId(rows, 'param:c_well').checkPaths, ['params.c_well']);

  // Settings + sub-model registry: top-level and unprefixed too.
  assert.deepEqual(byId(rows, 'group:settings').checkPaths, ['settings']);
  assert.deepEqual(byId(rows, 'submodel:post').checkPaths, ['models.post']);
});

test('scoped build: rowForSelection matches a selection made inside the sub-model', () => {
  const top = WITH_SUB();
  const rows = buildOutline(top.models.post, ['post'], top);

  // The shape scoped-store.js's select() stamps for a canvas click inside the sub-model.
  assert.equal(rowForSelection(rows, { kind: 'state', id: 'healthy', modelPath: ['post'] }).id, 'state:healthy');
  assert.equal(
    rowForSelection(rows, { kind: 'edge', id: { from: 'healthy', to: 'gone' }, modelPath: ['post'] }).id,
    'edge:healthy>gone',
  );
  // A top-level selection does NOT match a scoped row (and vice versa) — the guard that made the
  // unscoped outline reject every sub-model click before this fix.
  assert.equal(rowForSelection(rows, { kind: 'state', id: 'healthy', modelPath: [] }), null);
  // Params are top-level in both, so a param selection still resolves while scoped in.
  assert.equal(rowForSelection(rows, { kind: 'param', id: 'c_well', modelPath: [] }).id, 'param:c_well');
});

test('scoped build: a finding inside the sub-model lands on its structure row, not the sub-model row', () => {
  const top = WITH_SUB();
  const rows = buildOutline(top.models.post, ['post'], top);
  const { byRow, counts, residual } = attachFindings(rows, [
    { level: 'error', code: 'E_EXPR', path: 'models.post.transitions.healthy.gone', message: 'bad p' },
    { level: 'error', code: 'E_SUB', path: 'models.post', message: 'sub-model itself' },
  ]);
  assert.deepEqual(byRow.get('edge:healthy>gone').map((f) => f.code), ['E_EXPR']);
  assert.deepEqual(byRow.get('submodel:post').map((f) => f.code), ['E_SUB']);
  assert.deepEqual(counts.get('group:structure'), { errors: 1, warnings: 0 });
  assert.deepEqual(residual, []);
});

test('scoped build: a tree sub-model scopes its node paths too', () => {
  const top = parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A: {model: sub}
models:
  sub:
    type: tree
    tree:
      SubRoot:
        Win: {p: rest, utility: 10}
`);
  const rows = buildOutline(top.models.sub, ['sub'], top);
  const win = byId(rows, 'node:SubRoot/Win');
  assert.deepEqual(win.sel, { kind: 'node', id: ['SubRoot', 'Win'], modelPath: ['sub'] });
  assert.deepEqual(win.checkPaths, ['models.sub.tree.Win']);
  assert.equal(rowForSelection(rows, { kind: 'node', id: ['SubRoot', 'Win'], modelPath: ['sub'] }).id, 'node:SubRoot/Win');
});

test('filter keeps matches and their ancestors, drops the rest', () => {
  const rows = buildOutline(MARKOV());
  const out = filterRows(rows, 'dead');
  const ids = out.map((r) => r.id);
  assert.ok(ids.includes('state:dead'));
  assert.ok(ids.includes('edge:well>dead'));
  assert.ok(ids.includes('state:well'), 'the matched edge keeps its parent state');
  assert.ok(ids.includes('group:structure'), 'the group header survives');
  assert.ok(!ids.includes('param:c_well'));
  assert.ok(!ids.includes('group:parameters'), 'a group with no surviving descendant is dropped');
});

test('filter is case-insensitive and matches detail text too', () => {
  const rows = buildOutline(MARKOV());
  assert.ok(filterRows(rows, 'DEAD').some((r) => r.id === 'state:dead'));
  assert.ok(filterRows(rows, '0.1').some((r) => r.id === 'edge:well>dead'));
});

test('an empty filter returns every row unchanged', () => {
  const rows = buildOutline(MARKOV());
  assert.deepEqual(filterRows(rows, ''), rows);
});

test('findings land on the most specific row that owns their path', () => {
  const rows = buildOutline(MARKOV());
  const findings = [
    { level: 'error', code: 'E_ROWSUM', path: 'transitions.well', message: 'row sums to 1.2' },
    { level: 'error', code: 'E_EXPR', path: 'transitions.well.dead', message: 'bad p' },
    { level: 'warning', code: 'W_X', path: 'states.well.cost', message: 'check cost' },
    { level: 'error', code: 'E_NOWHERE', path: 'meta.author', message: 'orphan' },
  ];
  const { byRow, counts, residual } = attachFindings(rows, findings);

  // 'transitions.well' is owned by the state row; 'transitions.well.dead' is longer, so it belongs
  // to the edge row rather than rolling up into the state's own bucket.
  assert.deepEqual(byRow.get('state:well').map((f) => f.code), ['E_ROWSUM', 'W_X']);
  assert.deepEqual(byRow.get('edge:well>dead').map((f) => f.code), ['E_EXPR']);
  assert.deepEqual(residual.map((f) => f.code), ['E_NOWHERE']);
});

test('counts roll descendants up into their ancestors', () => {
  const rows = buildOutline(MARKOV());
  const { counts } = attachFindings(rows, [
    { level: 'error', code: 'E_EXPR', path: 'transitions.well.dead', message: 'bad p' },
    { level: 'warning', code: 'W_X', path: 'states.well.cost', message: 'check cost' },
  ]);
  assert.deepEqual(counts.get('edge:well>dead'), { errors: 1, warnings: 0 });
  assert.deepEqual(counts.get('state:well'), { errors: 1, warnings: 1 });
  assert.deepEqual(counts.get('group:structure'), { errors: 1, warnings: 1 });
  assert.equal(counts.get('state:dead'), undefined);
});

test('settings group receives settings-scoped findings', () => {
  const rows = buildOutline(MARKOV());
  const settingsGroup = byId(rows, 'group:settings');
  assert.deepEqual(settingsGroup.checkPaths, ['settings']);

  const findings = [
    { level: 'error', code: 'E_UNKNOWN_STATE', path: 'settings.start', message: 'unknown state' },
  ];
  const { byRow, residual } = attachFindings(rows, findings);

  assert.deepEqual(byRow.get('group:settings').map((f) => f.code), ['E_UNKNOWN_STATE']);
  assert.deepEqual(residual, []);
});

test('markov: multinomial transitions show count/total', () => {
  const model = parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 3}
states:
  well: {cost: 0, utility: 1}
  dead: {cost: 0, utility: 0}
transitions:
  well:
    multinomial:
      dead: 5
      well: 15
  dead: {dead: 1}
`);
  const rows = buildOutline(model);
  const edgeWellDead = byId(rows, 'edge:well>dead');
  assert.equal(edgeWellDead.label, '→ dead');
  assert.equal(edgeWellDead.detail, '5/20');
  assert.deepEqual(edgeWellDead.sel, { kind: 'edge', id: { from: 'well', to: 'dead' }, modelPath: [] });
  const edgeWellWell = byId(rows, 'edge:well>well');
  assert.equal(edgeWellWell.detail, '15/20');
});

test('submodels group appears when models are populated', () => {
  const model = parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 3}
states:
  well: {cost: 0, utility: 1}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
models:
  post:
    type: markov
    settings: {cycles: 2}
    states:
      healthy: {cost: 0, utility: 1}
      dead: {cost: 0, utility: 0}
    transitions:
      healthy: {healthy: rest, dead: 0.05}
      dead: {dead: 1}
`);
  const rows = buildOutline(model);
  const submodelsGroup = byId(rows, 'group:submodels');
  assert.ok(submodelsGroup, 'submodels group should exist');
  assert.equal(submodelsGroup.depth, 0);
  assert.deepEqual(submodelsGroup.checkPaths, []);

  const postSubmodel = byId(rows, 'submodel:post');
  assert.ok(postSubmodel);
  assert.equal(postSubmodel.depth, 1);
  assert.equal(postSubmodel.parentId, 'group:submodels');
  assert.deepEqual(postSubmodel.checkPaths, ['models.post']);
});

test('submodels group is omitted when models is empty or absent', () => {
  const rows = buildOutline(MARKOV());
  const submodelsGroup = byId(rows, 'group:submodels');
  assert.equal(submodelsGroup, undefined, 'submodels group should not exist when models is empty');
});

test('null p-value shows as empty string, matching render.js pLabelText convention', () => {
  const model = MARKOV();
  // Directly mutate the parsed model's transition entry to test the null case without requiring invalid YAML
  model.transitions.well.to.dead.p = null;
  const rows = buildOutline(model);
  const edge = byId(rows, 'edge:well>dead');
  assert.equal(edge.detail, '', 'null p-value should show as empty string');
});

// --- rowForSelection (Task 10) ---

test('rowForSelection: matches a state selection to its row', () => {
  const rows = buildOutline(MARKOV());
  const row = rowForSelection(rows, { kind: 'state', id: 'well', modelPath: [] });
  assert.equal(row.id, 'state:well');
});

test('rowForSelection: matches an edge selection by {from, to}', () => {
  const rows = buildOutline(MARKOV());
  const row = rowForSelection(rows, { kind: 'edge', id: { from: 'well', to: 'dead' }, modelPath: [] });
  assert.equal(row.id, 'edge:well>dead');
});

test('rowForSelection: matches a tree node selection by its full path array', () => {
  const rows = buildOutline(TREE());
  const row = rowForSelection(rows, { kind: 'node', id: ['Root', 'A'], modelPath: [] });
  assert.equal(row.id, 'node:Root/A');
});

test('rowForSelection: a param selection falls back to matching a param row by name (params carry sel:null)', () => {
  const rows = buildOutline(MARKOV());
  const row = rowForSelection(rows, { kind: 'param', id: 'c_well', modelPath: [] });
  assert.equal(row.id, 'param:c_well');
});

test('rowForSelection: no selection, or a selection matching nothing, returns null', () => {
  const rows = buildOutline(MARKOV());
  assert.equal(rowForSelection(rows, null), null);
  assert.equal(rowForSelection(rows, { kind: null, id: null }), null);
  assert.equal(rowForSelection(rows, { kind: 'state', id: 'nope', modelPath: [] }), null);
  assert.equal(rowForSelection(rows, { kind: 'param', id: 'c_well', modelPath: ['post'] }), null);
});

// --- collapseFilter (Task 10) ---

test('collapseFilter: a collapsed group drops its descendants but keeps the group row itself', () => {
  const rows = buildOutline(MARKOV());
  const visible = collapseFilter(rows, new Set(['group:structure']));
  assert.ok(visible.some((r) => r.id === 'group:structure'));
  assert.ok(!visible.some((r) => r.id === 'state:well'));
  assert.ok(!visible.some((r) => r.id === 'edge:well>dead'));
  // an unrelated group's rows are untouched
  assert.ok(visible.some((r) => r.id === 'group:parameters'));
  assert.ok(visible.some((r) => r.id === 'param:c_well'));
});

test('collapseFilter: collapsing a state drops its own edges without touching sibling states', () => {
  const rows = buildOutline(MARKOV());
  const visible = collapseFilter(rows, new Set(['state:well']));
  assert.ok(visible.some((r) => r.id === 'state:well'));
  assert.ok(!visible.some((r) => r.id === 'edge:well>dead'));
  assert.ok(visible.some((r) => r.id === 'state:dead'));
});

test('collapseFilter: an empty collapsed set returns every row unchanged', () => {
  const rows = buildOutline(MARKOV());
  assert.deepEqual(collapseFilter(rows, new Set()), rows);
});

// --- ancestorIds (Task 10 review, Finding 2: extracted from inspector.js's revealSelection) ---

test('ancestorIds: an edge row -> [its state, group:structure], immediate parent first', () => {
  const rows = buildOutline(MARKOV());
  const edge = byId(rows, 'edge:well>dead');
  assert.deepEqual(ancestorIds(rows, edge), ['state:well', 'group:structure']);
});

test('ancestorIds: a root-level row (no parent) -> []', () => {
  const rows = buildOutline(MARKOV());
  const group = byId(rows, 'group:structure');
  assert.deepEqual(ancestorIds(rows, group), []);
});

test('ancestorIds: a param row -> [group:parameters]', () => {
  const rows = buildOutline(MARKOV());
  const param = byId(rows, 'param:c_well');
  assert.deepEqual(ancestorIds(rows, param), ['group:parameters']);
});

test('ancestorIds: a deeply nested tree node walks every ancestor up to the group header', () => {
  const rows = buildOutline(TREE());
  const win = byId(rows, 'node:Root/A/Win');
  assert.deepEqual(ancestorIds(rows, win), ['node:Root/A', 'node:Root', 'group:structure']);
});

// --- addAfterIndex (Task 10 review, Finding 2: extracted from inspector.js's render()) ---

test('addAfterIndex: anchors after the LAST visible child of the group', () => {
  const rows = buildOutline(MARKOV());
  const idx = addAfterIndex(rows, 'group:parameters', new Set());
  assert.equal(rows[idx].id, 'param:c_well');
});

test('addAfterIndex: falls back to the group header itself when it has no visible children', () => {
  const noParams = parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 1}
states:
  well: {cost: 0, utility: 1}
transitions:
  well: {well: 1}
`);
  const rows = buildOutline(noParams);
  const idx = addAfterIndex(rows, 'group:parameters', new Set());
  assert.equal(rows[idx].id, 'group:parameters');
});

test('addAfterIndex: a collapsed group returns -1 (its children are hidden, nothing to anchor to)', () => {
  const rows = buildOutline(MARKOV());
  assert.equal(addAfterIndex(rows, 'group:parameters', new Set(['group:parameters'])), -1);
});

test('addAfterIndex: a group filtered out of visibleRows entirely also returns -1', () => {
  const rows = buildOutline(MARKOV());
  const withoutParamsGroup = rows.filter((r) => r.id !== 'group:parameters' && r.parentId !== 'group:parameters');
  assert.equal(addAfterIndex(withoutParamsGroup, 'group:parameters', new Set()), -1);
});
