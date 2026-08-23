import { compile } from '../core/expr.js';

export class TreeError extends Error {
  constructor(message, extra = {}) {
    super(message);
    Object.assign(this, extra);
  }
}

const REST_TOL = 1e-9;

// Literal `rest` is detected on the RAW source, before compiling — `compile('rest').eval` throws
// by design (expr.js: "rest is only valid as a whole transition/branch probability"), so we must
// never hand a 'rest' source string to compile() for a probability field. Mirrors markov.js's
// isRestSrc exactly.
const isRestSrc = (src) => typeof src === 'string' && src.trim() === 'rest';

// Resolve one sibling group's `p` values into a parallel probs[] array (same index order as
// `node.children`). At most one 'rest' per group, filled as 1 - sum(known); either a 'rest'
// resolving outside [0,1] or (no-rest) a sibling sum outside 1±REST_TOL throws.
function resolveSiblingProbs(node, env) {
  const children = node.children;
  const probs = new Array(children.length);
  let restIndex = -1;
  let sumKnown = 0;

  children.forEach((child, i) => {
    if (child.p === undefined)
      throw new TreeError(`tree.${node.name}.${child.name}: missing 'p'`, { path: `tree.${node.name}.${child.name}` });
    if (isRestSrc(child.p)) {
      if (restIndex !== -1)
        throw new TreeError(
          `tree.${node.name}: more than one 'rest' among children of '${node.name}'`,
          { path: `tree.${node.name}` }
        );
      restIndex = i;
    } else {
      const v = compile(child.p).eval(env);
      probs[i] = v;
      sumKnown += v;
    }
  });

  if (restIndex !== -1) {
    const restVal = 1 - sumKnown;
    if (restVal < -REST_TOL || restVal > 1 + REST_TOL)
      throw new TreeError(
        `tree.${node.name}: 'rest' resolves to ${restVal} among children of '${node.name}' (out of [0,1])`,
        { path: `tree.${node.name}` }
      );
    probs[restIndex] = restVal;
  } else if (Math.abs(sumKnown - 1) > REST_TOL) {
    throw new TreeError(
      `tree.${node.name}: children p sum to ${sumKnown}, expected 1 (sum must equal 1, or replace one child's p with 'rest')`,
      { path: `tree.${node.name}` }
    );
  }

  return probs;
}

// Add one node's own payoff expressions (cost/utility/...extras) onto a running accumulator,
// returning a NEW accumulator (siblings must not share mutated state — each branch below a
// sibling group inherits the same parent accumulator and adds its own contribution independently).
function mergePayoffs(acc, payoffs, env) {
  const merged = { cost: acc.cost, utility: acc.utility, extras: { ...acc.extras } };
  for (const [k, v] of Object.entries(payoffs)) {
    const val = compile(v).eval(env);
    if (k === 'cost') merged.cost += val;
    else if (k === 'utility') merged.utility += val;
    else merged.extras[k] = (merged.extras[k] ?? 0) + val;
  }
  return merged;
}

// Recursive descent accumulating {prob, payoffs} along one path from a strategy's root down to
// each leaf. `acc` is the running accumulated payoffs for the path so far (already includes every
// ancestor's own payoffs, per node.payoffs merged as each node is visited); `totals` is the
// strategy-level running total (mutated in place — safe, since it is never shared across sibling
// recursion the way `acc` is: only one `totals` object exists per strategy).
function walk(node, prob, acc, totals, env, attach) {
  const merged = mergePayoffs(acc, node.payoffs, env);

  if (node.children.length === 0) {
    // Terminal node. A `model:` terminal delegates to a sub-model via `attach(node, env)`, which
    // returns {cost, qaly, extras} for that sub-model's own rollup (already fully resolved by the
    // caller — discounting, if any, is entirely the attach closure's concern; the tree itself
    // never discounts). `qaly` folds into this tree's `utility` accumulator (a tree's own
    // `utility` payoff IS already a QALY-like quantity — see the module doc below).
    let finalAcc = merged;
    if (node.model !== undefined) {
      if (!attach)
        throw new TreeError(
          `tree.${node.name}: terminal has 'model: ${node.model}' but no attach function was ` +
          `provided for sub-models`,
          { path: `tree.${node.name}` }
        );
      const sub = attach(node, env);
      finalAcc = { cost: merged.cost + (sub.cost ?? 0), utility: merged.utility + (sub.qaly ?? 0), extras: { ...merged.extras } };
      for (const [k, v] of Object.entries(sub.extras ?? {})) {
        finalAcc.extras[k] = (finalAcc.extras[k] ?? 0) + v;
      }
    }

    // Leaf's expected contribution = path probability × accumulated payoffs.
    totals.cost += prob * finalAcc.cost;
    totals.utility += prob * finalAcc.utility;
    for (const [k, v] of Object.entries(finalAcc.extras)) {
      totals.extras[k] = (totals.extras[k] ?? 0) + prob * v;
    }
    return;
  }

  const probs = resolveSiblingProbs(node, env);
  node.children.forEach((child, i) => walk(child, prob * probs[i], merged, totals, env, attach));
}

/**
 * runTree(model, env, {discount}, attach?) -> {strategies: {branchName: {cost, qaly, extras}}}
 *
 * Semantics (binding, per constraints.md/brief): `model.tree` is the decision (root) node; each
 * ROOT CHILD is a strategy, keyed by its own name, entered unconditionally (probability 1 — root
 * children are alternatives to compare, not a probabilistic branch, so they are NEVER subjected
 * to sibling-p validation). Below a strategy, ordinary sibling groups' `p` values weight the
 * paths (at most one 'rest' per sibling group = 1 - sum(siblings), validated in [0,1]; a group
 * with no 'rest' must itself sum to 1±1e-9, or `resolveSiblingProbs` throws mentioning 'sum').
 * Payoffs (cost, utility, and any extra scalar keys) accumulate ADDITIVELY along each path as it
 * descends; a leaf's expected contribution to its strategy's total = (that path's cumulative
 * probability) × (that path's cumulative accumulated payoffs) — see `walk`.
 *
 * Tree payoffs are INSTANTANEOUS: there is no discounting inside this engine (a tree has no time
 * axis). `discount` is accepted here only to be forwarded to sub-models — but per the module
 * contract `attach(node, env) -> {cost, qaly, extras}` takes no discount argument, so in practice
 * "pass through" means: whoever constructs the `attach` closure (Task 11's run.js) already has
 * `discount` in scope from the same top-level `run()` call and closes over it there. `runTree`
 * itself never reads `discount` — it exists in the destructured signature purely to document the
 * accepted option shape and keep call sites uniform with `runMarkov`.
 *
 * A tree's `utility` payoff is already a QALY-like quantity (per Global Constraints) — so the
 * returned totals key is `qaly` (matching runMarkov's totals shape), even though the tree's own
 * node field and YAML key are both spelled `utility`.
 *
 * A terminal node with `model:` set calls `attach(node, env)` and folds the returned
 * {cost, qaly, extras} into that leaf's accumulated payoffs before weighting by path probability.
 * If `attach` is not provided but a `model:` terminal is reached, `walk` throws (mentions 'attach'
 * and 'sub-model').
 */
export function runTree(model, env, opts = {}, attach) {
  const { discount } = opts; // accepted, forwarded only via the attach closure — see doc above.
  void discount;

  const root = model.tree;
  if (!root) throw new TreeError('tree: model has no tree', { path: 'tree' });

  const strategies = {};
  for (const strategyNode of root.children) {
    const totals = { cost: 0, utility: 0, extras: {} };
    // Root's own payoffs (if any) are part of every strategy's path — merge once before entering
    // the strategy subtree at probability 1.
    const base = mergePayoffs({ cost: 0, utility: 0, extras: {} }, root.payoffs, env);
    walk(strategyNode, 1, base, totals, env, attach);
    strategies[strategyNode.name] = { cost: totals.cost, qaly: totals.utility, extras: totals.extras };
  }

  return { strategies };
}
