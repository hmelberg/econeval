import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';

const CHEMO = `
econeval: 1
type: tree
name: Chemo vs surgery
params:
  p_cure_chemo:
    value: 0.40
    dist: beta(40, 60)
models:
  survival:
    type: markov
    settings: {cycles: 40}
    params:
      p_prog: 0.10
    states:
      well: {cost: 500, utility: 0.90}
      prog: {cost: 3000, utility: 0.60}
      dead: {cost: 0, utility: 0}
    transitions:
      well: {well: rest, prog: p_prog, dead: 0.02}
      prog: {prog: rest, dead: 0.20}
      dead: {dead: 1}
tree:
  Treatment?:
    Chemo:
      cost: 12000
      relapses: 1
      Cured: {p: p_cure_chemo, model: survival}
      NotCured: {p: rest, model: survival, with: {p_prog: 0.25}}
    Surgery:
      cost: 30000
      Cured: {p: 0.60, model: survival}
      Relapse: {p: rest, model: survival, with: {start: prog}}
`;

test('tree normalizes: children vs payoffs vs reserved', () => {
  const m = parseModel(CHEMO);
  assert.equal(m.type, 'tree');
  const root = m.tree;
  assert.equal(root.name, 'Treatment?');
  assert.deepEqual(root.children.map(c => c.name), ['Chemo', 'Surgery']);
  const chemo = root.children[0];
  assert.deepEqual(chemo.payoffs, { cost: 12000, relapses: 1 });  // scalar non-reserved -> payoff
  assert.deepEqual(chemo.children.map(c => c.name), ['Cured', 'NotCured']);
  assert.equal(chemo.children[0].p, 'p_cure_chemo');
  assert.equal(chemo.children[0].model, 'survival');
  assert.deepEqual(chemo.children[1].with, { p_prog: 0.25 });
  assert.deepEqual(root.children[1].children[1].with, { start: 'prog' });
});

test('sub-models normalize like standalone models', () => {
  const m = parseModel(CHEMO);
  const sub = m.models.survival;
  assert.equal(sub.type, 'markov');
  assert.equal(sub.settings.cycles, 40);
  assert.deepEqual(sub.params.get('p_prog'), { value: 0.10 });
  assert.equal(sub.states.length, 3);
});

test('sub-model with top-level-only settings is rejected', () => {
  assert.throws(() => parseModel(CHEMO.replace('settings: {cycles: 40}', 'settings: {cycles: 40, wtp: 5}')), /wtp/);
});

test('serialize round-trips and never uses flow style for calls', () => {
  const m = parseModel(CHEMO);
  const text = serializeModel(m);
  assert.ok(text.includes('beta(40, 60)'));
  assert.ok(!/\{[^}]*\(/.test(text));            // no '(' inside a flow mapping
  assert.deepEqual(parseModel(text), m);
});

test('explicit children key and collision error', () => {
  const ok = parseModel(`
econeval: 1
type: tree
name: c
tree:
  Root:
    A:
      children:
        Leaf1: {p: 0.5, utility: 1}
        Leaf2: {p: rest, utility: 0}
`);
  assert.deepEqual(ok.tree.children[0].children.map(c => c.name), ['Leaf1', 'Leaf2']);
});
