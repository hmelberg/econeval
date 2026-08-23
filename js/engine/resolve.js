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

// Param spec -> {value, isDist}. `isDist` marks the dist path (mean() or a sample() draw) —
// that path always memoizes unconditionally regardless of taint (see makeEnv below); only the
// `value`-expression path is subject to taint-based memoization.
// Mean mode: numeric/string `value` wins over `dist`; dist-only -> analytic mean. Sample mode:
// `dist` (whether or not `value` is also present) draws via env.rand; value-only still resolves
// the value normally.
function resolveParamSpec(name, spec, env) {
  const hasValue = spec.value !== undefined;
  const hasDist = spec.dist !== undefined;
  if (env.mode === 'sample' && hasDist) return { value: getCompiled(spec.dist).eval(env), isDist: true };
  if (hasValue) return { value: getCompiled(spec.value).eval(env), isDist: false };
  if (hasDist) return { value: getCompiled(spec.dist).eval(env), isDist: true };
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

  // Taint-aware memoization: a value-expression that (transitively) reads a cycle-mutated var
  // (t/time/age/state_time, or any other name the engine ever sets on env.vars) must NOT be
  // memoized, or cycles 2..N would silently keep reusing the cycle-1 value after the engine
  // mutates env.vars. `taintStack` holds one frame per resolution currently in progress on THIS
  // env; markTaint() flags the innermost (currently resolving) frame. A var access always
  // taints; so does depending on a name whose own resolution turned out tainted (propagated via
  // markTaint() right after that dependency's frame is popped) — this is how taint travels
  // through a DAG (`a: 't + 1'`, `b: 'a * 10'` -> b is tainted too). Dist-path resolutions
  // (mean() or a sample() draw) are exempt — they always memoize, preserving "one draw per env".
  const taintStack = [];
  function markTaint() {
    if (taintStack.length > 0) taintStack[taintStack.length - 1].tainted = true;
  }

  const env = {
    vars: {},                                        // engines set t/time/age/state_time here
    tables: model.tables,
    cycleYears: model.settings?.cycleYears,
    rand,
    mode,
    get(name) {
      if (Object.prototype.hasOwnProperty.call(env.vars, name)) {
        markTaint();                                  // reading a cycle-mutated var taints the caller
        return env.vars[name];
      }
      if (memo.has(name)) return memo.get(name);       // memoized values are, by construction, safe
      if (inProgress.has(name))
        throw new ExprError(`cycle in param resolution: ${[...inProgress, name].join(' -> ')}`);

      let resolve;
      if (Object.prototype.hasOwnProperty.call(strategyOverrides, name)) {
        const src = strategyOverrides[name];
        resolve = () => ({ value: getCompiled(src).eval(env), isDist: false });
      } else if (Object.prototype.hasOwnProperty.call(overrides, name)) {
        const src = overrides[name];
        resolve = () => ({ value: getCompiled(src).eval(env), isDist: false });
      } else if (model.params?.has(name)) {
        const spec = model.params.get(name);
        resolve = () => resolveParamSpec(name, spec, env);
      }

      if (resolve) {
        inProgress.add(name);
        const frame = { tainted: false };
        taintStack.push(frame);
        let result;
        try {
          result = resolve();
        } finally {
          taintStack.pop();
          inProgress.delete(name);
        }
        if (result.isDist || !frame.tainted) {
          memo.set(name, result.value);
        } else {
          markTaint();                                // this name is itself unstable -> propagate
        }
        return result.value;
      }

      if (parent) {
        markTaint();                                  // parent-scoped value: unknown stability, be safe
        return parent.get(name);
      }
      throw new ExprError(`unknown name: ${name}`);
    },
  };

  return env;
}
