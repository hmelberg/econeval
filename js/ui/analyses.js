/**
 * js/ui/analyses.js — pure orchestration layer over the phase-1 analysis stack
 * (engine/run.js, analysis/{cea,dsa,psa,check}.js). No DOM, no Plotly — chart-spec building is
 * charts.js's job (a later task); this module only shapes engine/analysis output into the plain
 * data a UI needs, plus a couple of static availability/gating checks the UI can consult before
 * spending the cost of a full run.
 *
 * availability(model)          -> {tornado:{ok, params, reason?}, psa:{ok, reason?}, trace}
 * gate(model)                   -> {ok:true} | {ok:false, errors: findings[]}
 * runBase(model, {wtp})         -> {results, ceaRows, strategies}
 * traceFor(results, strategy)   -> {states, series, cycles}
 * runTornado(model, {a,b,wtp})  -> bars                     (= dsa.oneWay)
 * runPsa(model, {n?, seed?})    -> psaResult                (= psa.psa)
 * psaDerived(psaResult, {comparator, wtpMax=100000}) -> {plane, wtps, ceacCurves, evpi}
 */
import { run } from '../engine/run.js';
import { cea } from '../analysis/cea.js';
import { oneWay } from '../analysis/dsa.js';
import { psa, ceac, evpi, cePlane } from '../analysis/psa.js';
import { check } from '../analysis/check.js';

// Strategy names in DECLARATION order, without running the model — mirrors exactly what run()
// itself keys results by: markov iterates Object.keys(model.strategies) (run.js); a tree's
// strategies are its root node's CHILDREN, not model.strategies (tree.js — model.strategies for
// a tree model is just the implicit {base} registry and has nothing to do with the actual
// strategy names). Private helper — not part of this module's exported contract.
function strategyNamesOf(model) {
  if (model.type === 'markov') return Object.keys(model.strategies);
  if (model.type === 'tree') return (model.tree?.children ?? []).map((c) => c.name);
  return [];
}

// Top-level params only (model.params) — mirrors dsa.js's oneWay and psa.js's psa(), neither of
// which ever reaches into sub-model params: both DSA and PSA in this codebase are top-level-only.
function tornadoParamNames(model) {
  const names = [];
  for (const [name, spec] of model.params) {
    if (spec.low !== undefined && spec.high !== undefined) names.push(name);
  }
  return names;
}

function psaParamNames(model) {
  const names = [];
  for (const [name, spec] of model.params) {
    if (spec.dist !== undefined) names.push(name);
  }
  return names;
}

/**
 * availability(model) -> {tornado:{ok, params, reason?}, psa:{ok, reason?}, trace}
 *
 * tornado.ok requires >= 2 strategies AND >= 1 top-level param carrying BOTH `low` and `high`.
 * `params` always lists the qualifying param names (possibly []); `reason` is set only when not
 * ok. psa.ok requires >= 1 top-level param carrying a `dist`. `trace` is true only for markov
 * models — a tree run never carries a `trace` key (run.js's module doc).
 */
export function availability(model) {
  const strategies = strategyNamesOf(model);
  const params = tornadoParamNames(model);
  const tornadoOk = strategies.length >= 2 && params.length >= 1;
  const tornado = { ok: tornadoOk, params };
  if (!tornadoOk) {
    const reasons = [];
    if (strategies.length < 2) reasons.push('needs at least 2 strategies');
    if (params.length < 1) reasons.push("needs at least 1 param with both 'low' and 'high'");
    tornado.reason = reasons.join('; ');
  }

  const distParams = psaParamNames(model);
  const psaOk = distParams.length >= 1;
  const psaAvail = { ok: psaOk };
  if (!psaOk) psaAvail.reason = "needs at least 1 top-level param with a 'dist'";

  return { tornado, psa: psaAvail, trace: model.type === 'markov' };
}

/**
 * gate(model) -> {ok:true} | {ok:false, errors: findings[]}
 *
 * Only check() findings with level==='error' block; warnings pass, per check.js's own contract.
 */
export function gate(model) {
  const errors = check(model).filter((f) => f.level === 'error');
  return errors.length === 0 ? { ok: true } : { ok: false, errors };
}

/**
 * runBase(model, {wtp}) -> {results, ceaRows, strategies}
 *
 * `results` is run(model)'s own {strategies:{name:{cost,qaly,extras,trace?}}} shape, untouched.
 * `ceaRows` = cea() over a {name:{cost,qaly}} view reshaped from `results`, in declaration order.
 * `strategies` is that same declaration-order name list, so a caller never has to re-derive it
 * from Object.keys(results.strategies) — safe for markov (where the two happen to agree) and
 * correct for tree models too (see strategyNamesOf).
 */
export function runBase(model, { wtp } = {}) {
  const results = run(model, {});
  const strategies = strategyNamesOf(model);
  const ceaInput = {};
  for (const name of strategies) {
    const r = results.strategies[name];
    ceaInput[name] = { cost: r.cost, qaly: r.qaly };
  }
  const { rows: ceaRows } = cea(ceaInput, { wtp });
  return { results, ceaRows, strategies };
}

/**
 * traceFor(results, strategy) -> {states, series, cycles}
 *
 * `results` is a run()-shaped object (or runBase's `.results`). Pulled straight from
 * results.strategies[strategy].trace — markov only (a tree strategy carries no `trace`, so this
 * returns the empty shape {states:[], series:{}, cycles:0} for it, rather than throwing).
 * `states` is read off the first trace entry's occupancy keys, which markov.js always builds in
 * the model's own declaration order (see markov.js's collapseOccupancy /
 * Object.fromEntries(stateNamesOrig...)). series[state][i] is the occupancy share at trace entry
 * i, i.e. cycle t = i+1 — run()'s trace has no cycle-0 entry, so this series does NOT include the
 * starting distribution; a caller wanting an explicit t=0 point prepends it itself from
 * settings.start.
 */
export function traceFor(results, strategy) {
  const trace = results.strategies[strategy]?.trace ?? [];
  const states = trace.length > 0 ? Object.keys(trace[0].occupancy) : [];
  const series = {};
  for (const s of states) series[s] = trace.map((entry) => entry.occupancy[s]);
  return { states, series, cycles: trace.length };
}

/**
 * runTornado(model, {a, b, wtp}) -> bars   (= dsa.oneWay(model, {a, b, wtp}))
 */
export function runTornado(model, { a, b, wtp }) {
  return oneWay(model, { a, b, wtp });
}

/**
 * runPsa(model, {n?, seed?}) -> psaResult   (= analysis/psa.js's psa(model, {n, seed}))
 */
export function runPsa(model, opts = {}) {
  return psa(model, opts);
}

/**
 * psaDerived(psaResult, {comparator, wtpMax=100000}) -> {plane, wtps, ceacCurves, evpi}
 *
 * wtps = 26 evenly spaced points from 0 to wtpMax INCLUSIVE (25 equal steps). This function is
 * pure over an already-computed psaResult and has no access to the model that produced it, so
 * 100000 is this module's own last-resort default — the documented UI policy ("wtpMax defaults to
 * 2.5 x settings.wtp, falling back to 100000 only when settings.wtp is unset") is the CALLER's
 * job: it has the model in scope and computes that wtpMax before calling in here.
 * ceacCurves/evpi/plane all delegate to analysis/psa.js's ceac/evpi/cePlane.
 */
export function psaDerived(psaResult, { comparator, wtpMax = 100000 } = {}) {
  const wtps = Array.from({ length: 26 }, (_, i) => (wtpMax * i) / 25);
  const { curves } = ceac(psaResult, wtps);
  const evpiValues = evpi(psaResult, wtps);
  const plane = cePlane(psaResult, { comparator });
  return { plane, wtps, ceacCurves: curves, evpi: evpiValues };
}
