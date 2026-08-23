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
// counts are a per-run constant, per the brief ("one Dirichlet draw per row per run"). `stateTimeOf`
// (Task 9, null when no state_time tunnel expansion applies) seeds vars.state_time before a
// tunnel copy's counts are evaluated — one resolution (one Dirichlet draw, if sampling) per copy
// per run, same "per run" cadence as the untunneled case, just extended per copy.
function resolveMultinomialProportions(rows, env, stateTimeOf = null) {
  const out = {};
  for (const [from, r] of Object.entries(rows)) {
    if (r.type !== 'multinomial') continue;
    if (stateTimeOf) env.vars.state_time = stateTimeOf.get(from);
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
// `stateTimeOf` (Task 9, null when no state_time tunnel expansion applies): seeds vars.state_time
// to this row's own copy index before evaluating ANY of its expressions (p, and — later in
// runMarkov's reward loop, re-set again there since that loop runs after this one completes —
// the same row's transition-reward cost/utility). `collapse` (Task 9, same nullness as
// stateTimeOf) is used ONLY to keep a thrown MarkovError's message/path referencing the
// document's own state name (plus its state_time, for context) instead of an internal tunnel-
// copy name — errors must stay clean the same way reported occupancy does.
function evalRowsForCycle(rows, multinomialProportions, env, t, stateTimeOf = null, collapse = null) {
  const P = {};
  for (const [from, r] of Object.entries(rows)) {
    if (stateTimeOf) env.vars.state_time = stateTimeOf.get(from);
    const origName = collapse ? collapse.get(from) : from;
    const label = (collapse && stateTimeOf.has(from)) ? `${origName} (state_time=${stateTimeOf.get(from)})` : origName;
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
      throw new MarkovError(`transitions.${label}: more than one 'rest' target at cycle ${t}`, { t, path: `transitions.${origName}` });
    if (restTargets.length === 1) {
      const restVal = 1 - sumKnown;
      if (restVal < -REST_TOL || restVal > 1 + REST_TOL)
        throw new MarkovError(
          `transitions.${label}: 'rest' resolves to ${restVal} at cycle ${t} (out of [0,1])`,
          { t, path: `transitions.${origName}` }
        );
      row[restTargets[0]] = restVal;
    } else if (Math.abs(sumKnown - 1) > REST_TOL) {
      throw new MarkovError(
        `transitions.${label}: row sum ${sumKnown} != 1 at cycle ${t}`,
        { t, path: `transitions.${origName}` }
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

// ---------------------------------------------------------------------------------------------
// state_time tunnel expansion (Task 9) — pre-processing step over the ALREADY-COMPILED rows/
// payoffs (compileRows/compilePayoffs run first; expressions are compiled exactly once and
// REUSED across every tunnel copy — only vars.state_time, set right before each eval, differs
// per copy). Detection (binding, per brief): a state S needs expansion if `state_time` appears in
// Expr.names on S's own payoff expressions OR on any expression of S's own transition row
// (probabilities AND transition-reward cost/utility). Expansion is purely internal: document
// keys (states/transitions in the model, and the trace/totals this engine returns) never see a
// tunnel-copy name — copy names use a Unicode Private-Use-Area separator (U+E000; deliberately
// not a NUL/control byte, which would make this source file register as binary to git and other
// text tooling), which cannot occur in a YAML mapping key produced by this project's authoring
// surface, so collision with a real state name is not a practical concern.
const TUNNEL_SEP = '';
const copyName = (s, k) => `${s}${TUNNEL_SEP}${k}`;

function detectTunnelStates(rows, compiledByState) {
  const tunnel = new Set();
  for (const [name, cp] of compiledByState.entries()) {
    for (const expr of Object.values(cp)) {
      if (expr.names.has('state_time')) { tunnel.add(name); break; }
    }
  }
  for (const [from, r] of Object.entries(rows)) {
    if (r.type === 'multinomial') {
      if (r.compiledCounts.some((c) => c.names.has('state_time'))) tunnel.add(from);
      continue;
    }
    for (const entry of r.entries) {
      if (entry.p?.names.has('state_time')) tunnel.add(from);
      if (entry.cost?.names.has('state_time')) tunnel.add(from);
      if (entry.utility?.names.has('state_time')) tunnel.add(from);
    }
  }
  return tunnel;
}

// Retarget a row's `to` fields for the expanded state space: any entry whose target is a
// tunnel-expanded state lands on that state's FIRST copy ("entry into S from elsewhere lands in
// S_1", binding — this applies uniformly to entries from non-tunnel rows AND from other tunnel
// copies' rows, and also to the model's `start` distribution via remapStart below). The row's own
// "self" entry (to === the state that owns this row, i.e. the structural "stay" edge) is instead
// pointed at `selfName` — see the Design C note on residencyAdvance() for why this is identity
// (this copy's own name), not the next copy.
function makeRetarget(from, selfName, tunnelStates) {
  return (to) => {
    if (to === from) return selfName;
    return tunnelStates.has(to) ? copyName(to, 1) : to;
  };
}

function retargetRow(r, retarget) {
  if (r.type === 'multinomial') {
    return { type: 'multinomial', names: r.names.map(retarget), compiledCounts: r.compiledCounts };
  }
  return { type: 'p', entries: r.entries.map((e) => ({ ...e, to: retarget(e.to) })) };
}

// Build the expanded {rows, compiledByState, stateNames, collapse, stateTimeOf, advanceTarget}
// structure. `collapse` maps every expanded name (tunnel copy OR untouched passthrough state)
// back to its original document state name — used to re-fold trace occupancy so reported keys
// stay exactly the document's own state names. `stateTimeOf` maps a tunnel-copy name to its own
// k (1..cycles); absent for passthrough names. `advanceTarget` is Design C's residency-advance
// map (see residencyAdvance below).
function expandStateTime(stateNamesOrig, rows, compiledByState, tunnelStates, cycles) {
  const newRows = {};
  const newCompiledByState = new Map();
  const newStateNames = [];
  const collapse = new Map();
  const stateTimeOf = new Map();
  const advanceTarget = new Map();

  for (const s of stateNamesOrig) {
    if (!tunnelStates.has(s)) {
      newStateNames.push(s);
      collapse.set(s, s);
      advanceTarget.set(s, s);
      newCompiledByState.set(s, compiledByState.get(s));
      newRows[s] = retargetRow(rows[s], makeRetarget(s, s, tunnelStates));
      continue;
    }
    for (let k = 1; k <= cycles; k++) {
      const cname = copyName(s, k);
      newStateNames.push(cname);
      collapse.set(cname, s);
      stateTimeOf.set(cname, k);
      advanceTarget.set(cname, copyName(s, Math.min(k + 1, cycles)));
      newCompiledByState.set(cname, compiledByState.get(s));
      newRows[cname] = retargetRow(rows[s], makeRetarget(s, cname, tunnelStates));
    }
  }

  return { rows: newRows, compiledByState: newCompiledByState, stateNames: newStateNames, collapse, stateTimeOf, advanceTarget };
}

// Design C (why the "stay" edge is identity, not Sk->Sk+1, inside the P-matrix): Task 8's
// `correction: 'none'` convention bills cost/qaly using `mNext` (the state occupied AFTER this
// cycle's transition, not before) — a fixed, tested invariant this task must not disturb. If the
// self-loop redirected directly to copy k+1 (the brief's most literal reading), `mNext` for a
// continuously-residing cohort would always land on copy (k+1) at cycle k, one state_time step
// AHEAD of what the cycle's payoff should read (verified by hand: it reproduces the wrong total
// for the brief's ramping-cost test, 800 instead of 600). Keeping the self-loop as an identity
// edge means `mNext`'s copy-k bucket — and thus that copy's OWN state_time=k — is exactly what
// was occupied THIS cycle, so payoffs/occupancy/rewards all read correctly with zero special-
// casing of the accrual code. The actual "Sk -> Sk+1" residency advance instead happens here, as
// a separate step between cycles: it turns this cycle's post-transition mNext into the SOURCE `m`
// the next cycle's transition-row evaluation will see (so that row correctly reads state_time =
// k+1 for the next decision) — capped at S_cycles, which maps to itself (self-loops as the final
// copy, per the brief).
function residencyAdvance(mNext, advanceTarget, stateNames) {
  const out = Object.fromEntries(stateNames.map((n) => [n, 0]));
  for (const n of stateNames) out[advanceTarget.get(n)] += mNext[n];
  return out;
}

// Fold expanded occupancy (copy names) back to the document's own state names for the reported
// trace. Folding `mNext` (this cycle's raw result, pre-residency-advance) vs. the post-advance `m`
// gives an IDENTICAL collapsed total either way — residencyAdvance only moves mass between copies
// of the SAME original state, never across different document states — so using `mNext` directly
// (already in hand at this point in the loop) is just the simpler choice, not a distinct result.
function collapseOccupancy(m, collapse, stateNamesOrig) {
  const out = Object.fromEntries(stateNamesOrig.map((n) => [n, 0]));
  for (const [name, orig] of collapse.entries()) out[orig] += m[name] ?? 0;
  return out;
}

// settings.start is entry into whatever state it names — if that state is tunnel-expanded, entry
// lands on its first copy (same "entry into S lands in S_1" rule as any other transition target).
function remapStart(start, tunnelStates) {
  if (!start) return start;
  const out = {};
  for (const [name, w] of Object.entries(start)) {
    out[tunnelStates.has(name) ? copyName(name, 1) : name] = w;
  }
  return out;
}

/**
 * runMarkov(model, env, {discount, delayYears=0}) ->
 *   {trace:[{t, occupancy:{state:num}, cost, qaly, extras:{}}], totals:{cost, qaly, extras:{}}}
 *
 * m_0 = settings.start; for t=1..cycles: seed env.vars.{t,time,age}, evaluate P_t (filling
 * `rest`), advance m_t = m_{t-1}.P_t, accrue cost/qaly/extras on the correction's occupancy
 * basis with the correction's discount timing, plus one-time transition-reward accruals (always
 * end-of-cycle discounted) on the flow m_{t-1}[from]*P_t[from][to].
 *
 * Task 9: if any state needs a state_time tunnel (detectTunnelStates), `rows`/`compiledByState`/
 * `stateNames` below are the EXPANDED (internal-only) versions — see expandStateTime/Design C.
 * When no state references state_time, `tunnelStates` is empty and every new code path below is
 * a no-op (stateTimeOf/collapse/advanceTarget all null): rows/compiledByState/stateNames are the
 * exact, unexpanded objects Task 8 built, so Task 8's tests are bit-for-bit unaffected.
 */
export function runMarkov(model, env, { discount = {}, delayYears = 0 } = {}) {
  const { cycles, cycleYears, correction, start, age } = model.settings;
  const stateNamesOrig = model.states.map((s) => s.name);

  // Guard (unconditional — not only when a tunnel is actually in use): TUNNEL_SEP is used
  // internally to build tunnel-copy names (`${origName}${TUNNEL_SEP}${k}`) and nothing stops a
  // document's own state name from containing that same character (model.js places no character
  // restriction on state names) — an unguarded collision would silently clobber `newRows`/
  // `collapse`/`compiledByState` entries during expandStateTime and produce silently wrong
  // occupancy with no error. Fail loud instead.
  for (const name of stateNamesOrig) {
    if (name.includes(TUNNEL_SEP))
      throw new MarkovError(
        `states.${name}: state names may not contain U+E000 — that character is reserved internally for state_time tunnel expansion`,
        { path: `states.${name}` }
      );
  }

  const rows0 = compileRows(model);
  for (const name of stateNamesOrig) {
    if (!rows0[name])
      throw new MarkovError(`transitions.${name}: missing row for state '${name}'`, { path: `transitions.${name}` });
  }

  // A transition target that names an undeclared state (a typo'd state name, e.g. 'daed' for
  // 'dead') would otherwise silently lose mass: advance() writes into mNext[target] for whatever
  // key the row names, but every subsequent cycle only sums over the fixed, DECLARED stateNames —
  // so a bad target's mass is credited once, then vanishes with no error at all. Fail loud instead,
  // for both p-rows (row.to targets) and multinomial rows (counts targets).
  const stateNameSet = new Set(stateNamesOrig);
  for (const [from, r] of Object.entries(rows0)) {
    const targets = r.type === 'multinomial' ? r.names : r.entries.map((e) => e.to);
    for (const target of targets) {
      if (!stateNameSet.has(target))
        throw new MarkovError(
          `transitions.${from}: transition target '${target}' is not a defined state`,
          { path: `transitions.${from}` }
        );
    }
  }
  const { compiledByState: compiledByState0, extraNames } = compilePayoffs(model);

  const tunnelStates = detectTunnelStates(rows0, compiledByState0);
  const expanded = tunnelStates.size > 0
    ? expandStateTime(stateNamesOrig, rows0, compiledByState0, tunnelStates, cycles)
    : null;

  const rows = expanded ? expanded.rows : rows0;
  const compiledByState = expanded ? expanded.compiledByState : compiledByState0;
  const stateNames = expanded ? expanded.stateNames : stateNamesOrig;
  const collapse = expanded ? expanded.collapse : null;
  const stateTimeOf = expanded ? expanded.stateTimeOf : null;
  const advanceTarget = expanded ? expanded.advanceTarget : null;

  const multinomialProportions = resolveMultinomialProportions(rows, env, stateTimeOf);

  const startForM0 = expanded ? remapStart(start, tunnelStates) : start;
  let m = buildM0(stateNames, startForM0);

  const discCost = discount.cost ?? 0;
  const discEffect = discount.effect ?? 0;

  const trace = [];
  const totals = { cost: 0, qaly: 0, extras: Object.fromEntries([...extraNames].map((n) => [n, 0])) };

  for (let t = 1; t <= cycles; t++) {
    env.vars.t = t;
    env.vars.time = (t - 1) * cycleYears;
    if (age != null) env.vars.age = age + env.vars.time;

    const P = evalRowsForCycle(rows, multinomialProportions, env, t, stateTimeOf, collapse);
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
      if (stateTimeOf) env.vars.state_time = stateTimeOf.get(name);
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
      if (stateTimeOf) env.vars.state_time = stateTimeOf.get(from);
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

    const occupancy = collapse ? collapseOccupancy(mNext, collapse, stateNamesOrig) : { ...mNext };
    trace.push({ t, occupancy, cost: cost_t, qaly: qaly_t, extras: extras_t });
    totals.cost += cost_t;
    totals.qaly += qaly_t;
    for (const en of extraNames) totals.extras[en] += extras_t[en];

    // Design C (see residencyAdvance): the NEXT cycle's transition source is not mNext verbatim
    // for a tunnel-expanded model — mass that stayed in a tunnel copy must advance to the next
    // copy (Sk -> Sk+1, capped) so the next cycle's row evaluation reads the correct state_time.
    // For an untunneled model (advanceTarget null) this is exactly Task 8's `m = mNext`.
    m = advanceTarget ? residencyAdvance(mNext, advanceTarget, stateNames) : mNext;
  }

  return { trace, totals };
}
