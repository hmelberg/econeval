import { compile } from '../core/expr.js';
import { sampleDirichlet } from '../core/dist.js';

export class MarkovError extends Error {
  constructor(message, extra = {}) {
    super(message);
    Object.assign(this, extra);
  }
}

const REST_TOL = 1e-9;

// Literal `rest` is detected on the RAW source, before compiling — `compile('rest').eval` throws
// by design (expr.js: "rest is only valid as a whole transition/branch probability"), so we must
// never hand a 'rest' source string to compile() for a probability field.
const isRestSrc = (src) => typeof src === 'string' && src.trim() === 'rest';

// Discount timing (Global Constraints, binding): accrual of cycle t (1-based) is discounted by
// (1+r)^-(delayYears + t*cycleYears) under correction:'none', and by
// (1+r)^-(delayYears + (t-0.5)*cycleYears) under 'half-cycle'/'life-table' (both = cycle-average
// basis in v1 — controller ruling amending the brief: 'life-table' is a plain alias for
// 'half-cycle', not a separate basis). One-time transition rewards ALWAYS use the end-of-cycle
// factor, regardless of the model's correction setting.
function discFactor(rate, t, cycleYears, delayYears, correction) {
  const timeOffset = correction === 'none' ? t * cycleYears : (t - 0.5) * cycleYears;
  return Math.pow(1 + rate, -(delayYears + timeOffset));
}
function endCycleFactor(rate, t, cycleYears, delayYears) {
  return Math.pow(1 + rate, -(delayYears + t * cycleYears));
}

// Compile every transition-row / state-payoff expression once, up front (not per cycle).
function compileRows(model) {
  const rows = {};
  for (const [from, row] of Object.entries(model.transitions)) {
    if (row.type === 'multinomial') {
      const names = Object.keys(row.counts);
      rows[from] = { type: 'multinomial', names, compiledCounts: names.map((n) => compile(row.counts[n])) };
      continue;
    }
    const entries = Object.entries(row.to).map(([to, e]) => {
      if (e.p === undefined)
        throw new MarkovError(`transitions.${from}.${to}: missing 'p'`, { path: `transitions.${from}.${to}` });
      const isRest = isRestSrc(e.p);
      return {
        to,
        isRest,
        p: isRest ? null : compile(e.p),
        cost: e.cost !== undefined ? compile(e.cost) : null,
        utility: e.utility !== undefined ? compile(e.utility) : null,
      };
    });
    rows[from] = { type: 'p', entries };
  }
  return rows;
}

// cost/utility are dedicated payoff fields; every other key on a state's payoff mapping is a
// tracked "extra" (per model.js's normStates comment: cost/utility/extras all live in one bucket
// on the normalized Model, this engine is what splits them apart).
function compilePayoffs(model) {
  const compiledByState = new Map();
  const extraNames = new Set();
  for (const s of model.states) {
    const compiled = {};
    for (const [k, v] of Object.entries(s.payoffs)) {
      compiled[k] = compile(v);
      if (k !== 'cost' && k !== 'utility') extraNames.add(k);
    }
    compiledByState.set(s.name, compiled);
  }
  return { compiledByState, extraNames };
}

// Multinomial rows resolve to a fixed proportion vector ONCE PER RUN (not per cycle): mean mode
// normalizes the counts; sample mode draws a single Dirichlet sample from them. Evaluated before
// the cycle loop starts (env.vars not yet seeded with t/time/age) since a multinomial row's
// counts are a per-run constant, per the brief ("one Dirichlet draw per row per run").
function resolveMultinomialProportions(rows, env) {
  const out = {};
  for (const [from, r] of Object.entries(rows)) {
    if (r.type !== 'multinomial') continue;
    const counts = r.compiledCounts.map((c) => c.eval(env));
    const props = env.mode === 'sample'
      ? sampleDirichlet(env.rand, counts)
      : counts.map((c) => c / counts.reduce((a, b) => a + b, 0));
    const obj = {};
    r.names.forEach((n, i) => { obj[n] = props[i]; });
    out[from] = obj;
  }
  return out;
}

function buildM0(stateNames, start) {
  if (!start) throw new MarkovError("settings.start is required to run a markov model", { path: 'settings.start' });
  const m = Object.fromEntries(stateNames.map((n) => [n, 0]));
  for (const [name, w] of Object.entries(start)) {
    if (!(name in m))
      throw new MarkovError(`settings.start: unknown state '${name}'`, { path: 'settings.start' });
    m[name] = w;
  }
  return m;
}

// Evaluate one cycle's transition matrix P_t: {from: {to: prob}}, filling any `rest` target per
// row. Validation (binding): the filled `rest` value must lie in [-1e-9, 1+1e-9]; a row with no
// `rest` target must itself sum to 1 (+/- 1e-9). Either failure throws, naming the cycle.
function evalRowsForCycle(rows, multinomialProportions, env, t) {
  const P = {};
  for (const [from, r] of Object.entries(rows)) {
    if (r.type === 'multinomial') { P[from] = multinomialProportions[from]; continue; }
    const row = {};
    let sumKnown = 0;
    const restTargets = [];
    for (const entry of r.entries) {
      if (entry.isRest) { restTargets.push(entry.to); continue; }
      const v = entry.p.eval(env);
      row[entry.to] = v;
      sumKnown += v;
    }
    if (restTargets.length > 1)
      throw new MarkovError(`transitions.${from}: more than one 'rest' target at cycle ${t}`, { t, path: `transitions.${from}` });
    if (restTargets.length === 1) {
      const restVal = 1 - sumKnown;
      if (restVal < -REST_TOL || restVal > 1 + REST_TOL)
        throw new MarkovError(
          `transitions.${from}: 'rest' resolves to ${restVal} at cycle ${t} (out of [0,1])`,
          { t, path: `transitions.${from}` }
        );
      row[restTargets[0]] = restVal;
    } else if (Math.abs(sumKnown - 1) > REST_TOL) {
      throw new MarkovError(
        `transitions.${from}: row sum ${sumKnown} != 1 at cycle ${t}`,
        { t, path: `transitions.${from}` }
      );
    }
    P[from] = row;
  }
  return P;
}

function advance(stateNames, m, P) {
  const mNext = Object.fromEntries(stateNames.map((n) => [n, 0]));
  for (const from of stateNames) {
    for (const [to, p] of Object.entries(P[from])) {
      mNext[to] = (mNext[to] ?? 0) + m[from] * p;
    }
  }
  return mNext;
}

/**
 * runMarkov(model, env, {discount, delayYears=0}) ->
 *   {trace:[{t, occupancy:{state:num}, cost, qaly, extras:{}}], totals:{cost, qaly, extras:{}}}
 *
 * m_0 = settings.start; for t=1..cycles: seed env.vars.{t,time,age}, evaluate P_t (filling
 * `rest`), advance m_t = m_{t-1}.P_t, accrue cost/qaly/extras on the correction's occupancy
 * basis with the correction's discount timing, plus one-time transition-reward accruals (always
 * end-of-cycle discounted) on the flow m_{t-1}[from]*P_t[from][to].
 */
export function runMarkov(model, env, { discount = {}, delayYears = 0 } = {}) {
  const { cycles, cycleYears, correction, start, age } = model.settings;
  const stateNames = model.states.map((s) => s.name);

  const rows = compileRows(model);
  for (const name of stateNames) {
    if (!rows[name])
      throw new MarkovError(`transitions.${name}: missing row for state '${name}'`, { path: `transitions.${name}` });
  }
  const { compiledByState, extraNames } = compilePayoffs(model);
  const multinomialProportions = resolveMultinomialProportions(rows, env);

  let m = buildM0(stateNames, start);

  const discCost = discount.cost ?? 0;
  const discEffect = discount.effect ?? 0;

  const trace = [];
  const totals = { cost: 0, qaly: 0, extras: Object.fromEntries([...extraNames].map((n) => [n, 0])) };

  for (let t = 1; t <= cycles; t++) {
    env.vars.t = t;
    env.vars.time = (t - 1) * cycleYears;
    if (age != null) env.vars.age = age + env.vars.time;

    const P = evalRowsForCycle(rows, multinomialProportions, env, t);
    const mNext = advance(stateNames, m, P);

    const basis = correction === 'none'
      ? mNext
      : Object.fromEntries(stateNames.map((n) => [n, (m[n] + mNext[n]) / 2]));

    const costFactor = discFactor(discCost, t, cycleYears, delayYears, correction);
    const effectFactor = discFactor(discEffect, t, cycleYears, delayYears, correction);
    const endCostFactor = endCycleFactor(discCost, t, cycleYears, delayYears);
    const endEffectFactor = endCycleFactor(discEffect, t, cycleYears, delayYears);

    // State-occupancy accruals: cost per cycle, qaly = utility * cycleYears (utility is an
    // annual weight per Global Constraints), extras accrue like cost (no cycleYears scaling).
    let costAccrual = 0, qalyAccrual = 0;
    const extrasAccrual = Object.fromEntries([...extraNames].map((n) => [n, 0]));
    for (const name of stateNames) {
      const b = basis[name];
      const cp = compiledByState.get(name);
      if (cp.cost) costAccrual += b * cp.cost.eval(env);
      if (cp.utility) qalyAccrual += b * cp.utility.eval(env) * cycleYears;
      for (const en of extraNames) {
        if (cp[en]) extrasAccrual[en] += b * cp[en].eval(env);
      }
    }

    // One-time transition-reward accruals on the flow m_{t-1}[from]*P_t[from][to]; always
    // end-of-cycle discounted, independent of the correction basis used for state occupancy.
    let rewardCost = 0, rewardQaly = 0;
    for (const from of stateNames) {
      const r = rows[from];
      if (r.type !== 'p') continue;
      for (const entry of r.entries) {
        if (!entry.cost && !entry.utility) continue;
        const flow = m[from] * P[from][entry.to];
        if (entry.cost) rewardCost += flow * entry.cost.eval(env);
        if (entry.utility) rewardQaly += flow * entry.utility.eval(env);
      }
    }

    const cost_t = costAccrual * costFactor + rewardCost * endCostFactor;
    const qaly_t = qalyAccrual * effectFactor + rewardQaly * endEffectFactor;

    // Extras: a `c_`-prefixed extra is a cost-like quantity, discounted at the cost rate on the
    // same (correction-basis) timing as cost_t; any other extra is an undiscounted count.
    const extras_t = {};
    for (const en of extraNames) {
      const factor = en.startsWith('c_') ? costFactor : 1;
      extras_t[en] = extrasAccrual[en] * factor;
    }

    trace.push({ t, occupancy: { ...mNext }, cost: cost_t, qaly: qaly_t, extras: extras_t });
    totals.cost += cost_t;
    totals.qaly += qaly_t;
    for (const en of extraNames) totals.extras[en] += extras_t[en];

    m = mNext;
  }

  return { trace, totals };
}
