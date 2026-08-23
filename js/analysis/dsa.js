/**
 * oneWay(model, {a, b, wtp}) -> [{param, low, high, lowValue, highValue, base}]
 * twoWay(model, {a, b, wtp, x:{param, values}, y:{param, values}}) -> {x, y, grid}
 *
 * Deterministic (mean-mode) sensitivity analysis over two named strategies `a` (comparator) and
 * `b` (comparison), reported as ΔNMB = NMB(b) - NMB(a), NMB = wtp*qaly - cost, per Task 11's
 * `run(model, {overrides}) -> {strategies}` contract.
 *
 * Both sweeps pass `{[param]: bound}` straight through to `run`'s `overrides` option, which
 * `makeEnv` (resolve.js) applies at the "explicit overrides" rung of its lookup chain —
 * STRATEGY overrides sit above that rung and win. So a strategy that pins a param (e.g. `better:
 * {p_die: 0.05}`) is insensitive to that param's DSA sweep: it always evaluates at its own
 * pinned value regardless of `low`/`high`. This is correct, deliberate behaviour — the tornado
 * shows the strategy comparison exactly as modelled, not as if the pin didn't exist — and is
 * exactly what the ordering/near-zero-lowValue test in test/dsa.test.js locks in.
 *
 * oneWay sweeps ONE param at a time (all others left at their model-default `value`/`dist`) —
 * the standard one-way-DSA definition. twoWay sweeps two named params jointly over their given
 * value grids (no low/high lookup — the caller supplies the axes directly).
 */
import { run } from '../engine/run.js';

function nmb(strategyResult, wtp) {
  return wtp * strategyResult.qaly - strategyResult.cost;
}

function deltaNMB(model, { a, b, wtp, overrides }) {
  const { strategies } = run(model, { overrides });
  return nmb(strategies[b], wtp) - nmb(strategies[a], wtp);
}

export function oneWay(model, { a, b, wtp }) {
  const base = deltaNMB(model, { a, b, wtp });

  const rows = [];
  for (const [param, spec] of model.params) {
    if (spec.low === undefined || spec.high === undefined) continue;
    const lowValue = deltaNMB(model, { a, b, wtp, overrides: { [param]: spec.low } });
    const highValue = deltaNMB(model, { a, b, wtp, overrides: { [param]: spec.high } });
    rows.push({ param, low: spec.low, high: spec.high, lowValue, highValue, base });
  }

  rows.sort((r1, r2) => Math.abs(r2.highValue - r2.lowValue) - Math.abs(r1.highValue - r1.lowValue));
  return rows;
}

export function twoWay(model, { a, b, wtp, x, y }) {
  // grid[i][j] = ΔNMB at y.values[i] x x.values[j] — rows indexed by y, columns by x.
  const grid = y.values.map((yv) =>
    x.values.map((xv) => deltaNMB(model, { a, b, wtp, overrides: { [x.param]: xv, [y.param]: yv } }))
  );
  return { x, y, grid };
}
