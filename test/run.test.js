import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { run } from '../js/engine/run.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('markov: one result per strategy, trace included', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: strat
settings: {cycles: 3, start: well, correction: none}
params: {p_die: 0.1}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: p_die}
  dead: {dead: 1}
strategies:
  usual: {}
  better: {p_die: 0.05}
`);
  const r = run(m, {});
  close(r.strategies.usual.cost, 243.9);
  // better: survivors .95, .9025, .857375 -> sum 2.709875
  close(r.strategies.better.cost, 100 * 2.709875);
  assert.equal(r.strategies.usual.trace.length, 3);
});

test('tree with markov sub-model, with-overrides, start override', () => {
  const m = parseModel(`
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
`);
  const r = run(m, {});
  // Cure branch EV cost/entrant: 243.9 ; qaly 0.8*2.439 = 1.9512
  // Fail (p_die .5): survivors .5, .25, .125 -> 87.5 ; qaly .8*.875 = .7
  close(r.strategies.Treat.cost, 50 + 0.5 * 243.9 + 0.5 * 87.5);
  close(r.strategies.Treat.qaly, 0.5 * 1.9512 + 0.5 * 0.7);
});

test('sub-model uses top-level discount and delay', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: delay
settings:
  discount: {cost: 0.05, effect: 0}
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
    Only:
      Leaf: {p: 1, model: chronic, delay: 2}
`);
  const r = run(m, {});
  close(r.strategies.Only.cost, 100 * (0.9/1.05**3 + 0.81/1.05**4 + 0.729/1.05**5));
});

test('delay on a tree sub-model attachment throws (unsupported in v1)', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: treeDelay
models:
  acute:
    type: tree
    tree:
      R:
        Recover: {p: 0.5, utility: 1}
        Away: {p: rest, utility: 0}
tree:
  Root:
    Only:
      Leaf: {p: 1, model: acute, delay: 2}
`);
  assert.throws(() => run(m, {}), (err) => {
    assert.match(err.message, /delay on a tree sub-model attachment is not supported in v1/);
    return true;
  });
});

test('zero delay on a tree sub-model attachment is fine (delay: 0 is a no-op, not an error)', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: treeDelayZero
models:
  acute:
    type: tree
    tree:
      R:
        Recover: {p: 0.5, utility: 1}
        Away: {p: rest, utility: 0}
tree:
  Root:
    Only:
      Leaf: {p: 1, model: acute, delay: 0}
`);
  const r = run(m, {});
  close(r.strategies.Only.qaly, 0.5);
});

test('with routes through parent scope', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: scope
params: {global_p: 0.5}
models:
  s:
    type: markov
    settings: {cycles: 1, start: a, correction: none}
    params: {p: 0.1}
    states:
      a: {utility: 1}
      b: {utility: 0}
    transitions:
      a: {a: rest, b: p}
      b: {b: 1}
tree:
  Root:
    Only:
      Leaf: {p: 1, model: s, with: {p: global_p}}
`);
  const r = run(m, {});
  close(r.strategies.Only.qaly, 0.5);   // a-occupancy after 1 cycle = 1-0.5
});

test('sibling sub-model reference resolves via lexical chain (controller ruling)', () => {
  // Top-level models: defines two SIBLINGS, `chronic` (markov) and `acute` (tree, chance-rooted).
  // The main tree attaches `acute`; a terminal INSIDE acute references `chronic` by name even
  // though acute has no own `models:` block — this must fall back through the chain to the
  // top-level registry (own-first-then-ancestors, mirroring param scoping via parent envs).
  const m = parseModel(`
econeval: 1
type: tree
name: siblingScope
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
  acute:
    type: tree
    tree:
      R:
        ToChronic: {p: 0.5, model: chronic}
        Away: {p: rest, utility: 0}
tree:
  Root:
    Only:
      Leaf: {p: 1, model: acute}
`);
  const r = run(m, {});
  // chronic alone (from earlier tests): cost 243.9, qaly 0.8*2.439 = 1.9512 per entrant.
  // acute: 0.5 -> chronic, 0.5 -> a plain zero leaf.
  close(r.strategies.Only.cost, 0.5 * 243.9);
  close(r.strategies.Only.qaly, 0.5 * 1.9512);
});
