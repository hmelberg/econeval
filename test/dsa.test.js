import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { run } from '../js/engine/run.js';
import { oneWay, twoWay } from '../js/analysis/dsa.js';

const close = (a, b, tol = 1e-9, label = '') => assert.ok(Math.abs(a - b) < tol, `${label} ${a} vs ${b}`.trim());

// Task 11's `strat` markov fixture, with `low`/`high` added to p_die (per the brief) plus a
// second param (`well_cost`, narrow low/high band) with deliberately smaller NMB impact than
// p_die's sweep, to exercise the "sorted by |range| descending" requirement. Block-style YAML
// throughout (constraints.md: avoid the flow-comma trap) — the nested p_die mapping in
// particular could not be flow-style without repeating the comma-inside-braces shape.
const YAML = `
econeval: 1
type: markov
name: strat
settings: {cycles: 3, start: well, correction: none}
params:
  p_die:
    value: 0.1
    low: 0.05
    high: 0.2
  well_cost:
    value: 100
    low: 95
    high: 105
states:
  well: {cost: well_cost, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: p_die}
  dead: {dead: 1}
strategies:
  usual: {}
  better: {p_die: 0.05}
`;

const WTP = 50000;

test('oneWay: strategy-pinned param collapses to ~0 at the DSA bound it is pinned to; ordering by range; base matches independent run', () => {
  const m = parseModel(YAML);
  const rows = oneWay(m, { a: 'usual', b: 'better', wtp: WTP });

  const byParam = Object.fromEntries(rows.map((r) => [r.param, r]));

  // Interaction rule (brief): strategy overrides beat DSA overrides. `better` pins p_die: 0.05
  // unconditionally, so sweeping p_die to its `low` bound (0.05) makes `usual` (unpinned, so it
  // DOES pick up the DSA override) coincide exactly with `better` (p_die=0.05 either way,
  // well_cost=100 default for both, since this sweep only overrides p_die). Both strategies run
  // through the identical p_die=0.05/well_cost=100 computation, so lowValue should be exactly 0
  // (not just approximately) — survivors .95/.9025/.857375, sum 2.709875, for both strategies.
  close(byParam.p_die.lowValue, 0);

  // At the high bound (p_die=0.2), `usual` moves but `better` stays pinned at p_die=0.05 ->
  // the two strategies diverge, so highValue should NOT collapse to 0.
  assert.ok(Math.abs(byParam.p_die.highValue) > 1, `expected p_die.highValue to diverge from 0, got ${byParam.p_die.highValue}`);

  // well_cost is NOT pinned by either strategy's overrides, so both `usual` and `better` respond
  // to its DSA sweep — but its low/high band (95/105) is narrow, so its |range| must be smaller
  // than p_die's (0.05/0.2, plus the interaction-rule divergence above).
  const rangeOf = (r) => Math.abs(r.highValue - r.lowValue);
  assert.ok(rangeOf(byParam.p_die) > rangeOf(byParam.well_cost),
    `expected p_die's range (${rangeOf(byParam.p_die)}) > well_cost's (${rangeOf(byParam.well_cost)})`);

  // Sorted by |range| descending overall.
  assert.deepEqual(rows.map((r) => r.param), ['p_die', 'well_cost']);
  for (let i = 1; i < rows.length; i++) {
    assert.ok(rangeOf(rows[i - 1]) >= rangeOf(rows[i]), 'rows must be sorted by |range| descending');
  }

  // `low`/`high` on each row echo the param spec's bounds.
  assert.equal(byParam.p_die.low, 0.05);
  assert.equal(byParam.p_die.high, 0.2);
  assert.equal(byParam.well_cost.low, 95);
  assert.equal(byParam.well_cost.high, 105);

  // base matches an UNPERTURBED run's ΔNMB, computed independently here (not via dsa.js).
  const r0 = run(m, {});
  const nmb = (s) => WTP * r0.strategies[s].qaly - r0.strategies[s].cost;
  const expectedBase = nmb('better') - nmb('usual');
  close(byParam.p_die.base, expectedBase);
  close(byParam.well_cost.base, expectedBase);   // base is the same unperturbed run for every row
});

test('twoWay: 2x2 grid cells match direct run() calls with the same overrides; grid[i][j] = y.values[i] x x.values[j]', () => {
  const m = parseModel(YAML);
  const x = { param: 'p_die', values: [0.05, 0.2] };
  const y = { param: 'well_cost', values: [95, 105] };
  const { grid, x: xOut, y: yOut } = twoWay(m, { a: 'usual', b: 'better', wtp: WTP, x, y });

  assert.deepEqual(xOut, x);
  assert.deepEqual(yOut, y);
  assert.equal(grid.length, 2);
  assert.equal(grid[0].length, 2);

  const deltaNMB = (overrides) => {
    const r = run(m, { overrides });
    const nmb = (s) => WTP * r.strategies[s].qaly - r.strategies[s].cost;
    return nmb('better') - nmb('usual');
  };

  for (let i = 0; i < y.values.length; i++) {
    for (let j = 0; j < x.values.length; j++) {
      const expected = deltaNMB({ [x.param]: x.values[j], [y.param]: y.values[i] });
      close(grid[i][j], expected, 1e-9, `grid[${i}][${j}]`);
    }
  }
});
