import { compile } from '../core/expr.js';
import { ModelError } from '../core/model.js';
import { makeEnv } from './resolve.js';
import { runMarkov } from './markov.js';
import { runTree } from './tree.js';

const MAX_ATTACH_DEPTH = 32;

// Evaluate a tree-type sub-model as a chance SUBTREE at probability 1 (controller ruling amending
// the brief): its root's children must all carry `p` — a root whose children lack `p` looks like
// a decision (strategy-keyed) sub-model, which v1 does not support attaching, and gets a specific
// error instead of tree.js's generic "missing 'p'".
//
// Rather than re-implementing tree.js's sibling-p/'rest'/payoff-accumulation logic here (tree.js
// exports only `runTree`/`TreeError`), we reuse `runTree` itself: wrap the sub-model's real root
// as the LONE child of a synthetic decision node. `runTree` enters a decision's children
// unconditionally at probability 1 with no sibling-p validation — exactly the "probability-1
// descent" the ruling calls for — while the sub-model's real root, one level down, is walked as
// an ordinary node and so DOES get sibling-p validation on ITS OWN children, which is exactly
// "the sub-model root's children must carry p". `attach` is passed straight through, so a nested
// `model:` terminal inside the subtree recurses via the same mechanism.
function evalChanceSubtree(subModel, subEnv, attach) {
  const root = subModel.tree;
  if (root.children.length > 0 && root.children.some((c) => c.p === undefined)) {
    throw new ModelError(
      `models.${subModel.name}: tree sub-model root's children must carry 'p' (chance branches) — ` +
      `sequential decisions / decision-rooted sub-models are not supported in v1`,
      { path: `models.${subModel.name}.tree` }
    );
  }
  const wrapped = { ...subModel, tree: { name: '__root', payoffs: {}, children: [root] } };
  const { strategies } = runTree(wrapped, subEnv, {}, attach);
  return strategies[root.name];
}

// Look up node.model against a LEXICAL CHAIN of models: registries — `chain` is an array of
// `{name?, models}`-shaped registries ordered innermost-first (the current model's own `models:`
// block first, then its ancestors' up to the top-level's). Mirrors how param scoping chains via
// parent envs (resolve.js): a sub-model's own `models:` block can shadow an ancestor's, and a
// name not found locally falls back through the chain — this is what makes a SIBLING reference
// (a top-level sub-model's terminal referencing another top-level sub-model, without redeclaring
// it in its own `models:` block) resolve correctly. CONTROLLER RULING (supersedes the earlier
// per-model-local-only design): lookup is own-first-then-ancestors, not scope-local-only.
function lookupSubModel(chain, name) {
  for (const registry of chain) {
    if (registry && Object.prototype.hasOwnProperty.call(registry, name)) return registry[name];
  }
  return undefined;
}

// Build the attach(node, env) -> {cost, qaly, extras} closure for one point in the lexical chain
// (`chain[0].models` is searched first, then `chain[1].models`, etc. — see lookupSubModel).
// Recursing into a tree-type sub-model PREPENDS that sub-model onto the chain (so its own
// `models:` block can shadow an ancestor's, while ancestor names still fall through). `topDiscount`
// /`depthBox` are shared across the whole run() call: discount is always the TOP-LEVEL model's
// settings.discount (never a sub-model's — sub-models can't declare their own), and depthBox
// counts nested attach() calls so a reference cycle fails loud (structural cycle detection itself
// is check.js's job; this is just a stack-overflow guard).
function makeAttach(chain, topDiscount, depthBox) {
  return function attach(node, env) {
    if (depthBox.n > MAX_ATTACH_DEPTH)
      throw new ModelError(`tree: sub-model attachment depth exceeded ${MAX_ATTACH_DEPTH} (possible reference cycle) at 'model: ${node.model}'`, { path: `models.${node.model}` });

    const subModel = lookupSubModel(chain.map((m) => m.models), node.model);
    if (!subModel)
      throw new ModelError(`tree.${node.name}: unknown sub-model '${node.model}'`, { path: `models.${node.model}` });

    const { start, ...withParams } = node.with ?? {};
    const subEnv = makeEnv(subModel, { mode: env.mode, rand: env.rand, overrides: withParams, parent: env });
    const delayYears = compile(node.delay ?? 0).eval(env);

    depthBox.n++;
    try {
      if (subModel.type === 'markov') {
        const runModel = start === undefined
          ? subModel
          : { ...subModel, settings: { ...subModel.settings, start: { [start]: 1 } } };
        const { totals } = runMarkov(runModel, subEnv, { discount: topDiscount, delayYears });
        return { cost: totals.cost, qaly: totals.qaly, extras: totals.extras };
      }
      if (subModel.type === 'tree') {
        return evalChanceSubtree(subModel, subEnv, makeAttach([subModel, ...chain], topDiscount, depthBox));
      }
      throw new ModelError(`models.${node.model}: unsupported sub-model type '${subModel.type}'`, { path: `models.${node.model}` });
    } finally {
      depthBox.n--;
    }
  };
}

/**
 * run(model, {mode='mean', rand?, overrides?}) -> {strategies: {name: {cost, qaly, extras, trace?}}}
 *
 * markov: one makeEnv + runMarkov per strategy name in model.strategies, discounted with the
 * model's own settings.discount; per-strategy result includes `trace`.
 * tree: a single makeEnv, then runTree with an attach() closure that resolves `model:` terminals
 * into sub-models (markov sub-model -> runMarkov; tree sub-model -> a chance-subtree EV via
 * evalChanceSubtree, per controller ruling) — always discounted at the TOP-LEVEL model's
 * settings.discount, never a sub-model's own (sub-models can't declare one). No `trace` key.
 */
export function run(model, opts = {}) {
  const { mode = 'mean', rand, overrides } = opts;

  if (model.type === 'markov') {
    const strategies = {};
    for (const name of Object.keys(model.strategies)) {
      const env = makeEnv(model, { strategy: name, mode, rand, overrides });
      const { trace, totals } = runMarkov(model, env, { discount: model.settings.discount });
      strategies[name] = { cost: totals.cost, qaly: totals.qaly, extras: totals.extras, trace };
    }
    return { strategies };
  }

  if (model.type === 'tree') {
    const env = makeEnv(model, { mode, rand, overrides });
    const attach = makeAttach([model], model.settings.discount, { n: 0 });
    const { strategies } = runTree(model, env, { discount: model.settings.discount }, attach);
    return { strategies };
  }

  throw new ModelError(`run: unknown model type '${model.type}'`, { path: 'type' });
}
