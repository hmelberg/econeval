import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { buildOutline, filterRows, attachFindings } from '../js/ui/outline/build.js';

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
