import { compile, ExprError } from '../core/expr.js';

// Module-level compile cache: expression sources (param values, dist strings, override
// strings) repeat heavily across cycles/strategies/PSA draws — compile once, reuse the Expr.
const compileCache = new Map();
function getCompiled(src) {
  let c = compileCache.get(src);
  if (c === undefined) {
    c = compile(src);
    compileCache.set(src, c);
  }
  return c;
}

// Param spec -> value, given the already-current env (params may reference other params,
// resolved recursively through env.get -> DAG). Mean mode: numeric/string `value` wins over
// `dist`; dist-only -> analytic mean. Sample mode: `dist` (whether or not `value` is also
// present) draws via env.rand; value-only still resolves the value normally.
function resolveParamSpec(name, spec, env) {
  const hasValue = spec.value !== undefined;
  const hasDist = spec.dist !== undefined;
  if (env.mode === 'sample' && hasDist) return getCompiled(spec.dist).eval(env);
  if (hasValue) return getCompiled(spec.value).eval(env);
  if (hasDist) return getCompiled(spec.dist).eval(env);
  throw new ExprError(`param '${name}' has neither a value nor a dist to resolve`);
}

export function makeEnv(model, opts = {}) {
  const { strategy, mode = 'mean', rand, overrides = {}, parent } = opts;
  let strategyOverrides = {};
  if (strategy !== undefined) {
    const s = model.strategies?.[strategy];
    if (!s) throw new ExprError(`unknown strategy '${strategy}'`);
    strategyOverrides = s.overrides ?? {};
  }

  const memo = new Map();
  const inProgress = new Set();      // cycle detection, in resolution order

  const env = {
    vars: {},                                        // engines set t/time/age/state_time here
    tables: model.tables,
    cycleYears: model.settings?.cycleYears,
    rand,
    mode,
    get(name) {
      if (Object.prototype.hasOwnProperty.call(env.vars, name)) return env.vars[name];
      if (memo.has(name)) return memo.get(name);
      if (inProgress.has(name))
        throw new ExprError(`cycle in param resolution: ${[...inProgress, name].join(' -> ')}`);

      let resolve;
      if (Object.prototype.hasOwnProperty.call(strategyOverrides, name)) {
        const src = strategyOverrides[name];
        resolve = () => getCompiled(src).eval(env);
      } else if (Object.prototype.hasOwnProperty.call(overrides, name)) {
        const src = overrides[name];
        resolve = () => getCompiled(src).eval(env);
      } else if (model.params?.has(name)) {
        const spec = model.params.get(name);
        resolve = () => resolveParamSpec(name, spec, env);
      }

      if (resolve) {
        inProgress.add(name);
        try {
          const value = resolve();
          memo.set(name, value);
          return value;
        } finally {
          inProgress.delete(name);
        }
      }

      if (parent) return parent.get(name);
      throw new ExprError(`unknown name: ${name}`);
    },
  };

  return env;
}
