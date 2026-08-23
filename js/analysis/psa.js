/**
 * psa(model, {n?, seed?}) -> {strategies, samples:[{cost:{s}, qaly:{s}}]}
 * ceac(psaResult, wtps)   -> {wtps, curves:{strategy: number[]}}
 * evpi(psaResult, wtps)   -> number[]
 * cePlane(psaResult, {comparator}) -> {strategy: [{dcost, dqaly}]}
 *
 * Probabilistic sensitivity analysis. ONE `rand = rng(seed)` stream drives the whole run.
 * Per iteration:
 *   1. Build `overrides` by sampling every dist-bearing GLOBAL param (model.params entries with
 *      a `dist:`) exactly once, in declaration order:
 *        - params named in settings.psa.correlations are drawn together as a Gaussian copula:
 *          independent standard normals z0 (one normalQuantile(rand()) draw per listed param,
 *          in declaration order) -> correlated z = L·z0 (L = Cholesky factor of the correlation
 *          matrix built over the listed params; unlisted pairs among them are independent,
 *          diagonal 1) -> u = Phi(z) via cdf(normal(0,1), z) -> value = quantile(paramDist, u).
 *        - every other dist-bearing param is drawn independently via quantile(paramDist,
 *          rand()) — quantile-based (not dist.js's `sample`) so a draw only ever depends on one
 *          uniform from the shared stream, keeping reproducibility independent of any sampler's
 *          internals.
 *   2. `run(model, {mode:'sample', rand, overrides})`. Numeric overrides sit ABOVE model params
 *      in resolve.js's lookup chain (so a pre-drawn global never gets re-sampled from its own
 *      `dist:` inside run()) but BELOW strategy overrides (a strategy that pins a param is
 *      insensitive to its PSA draw, by the same rule DSA relies on) — a strategy-override
 *      *expression* containing a dist call still samples per iteration from the same shared
 *      `rand`. For markov models run() iterates strategies in `Object.keys(model.strategies)`
 *      order (stable/declared order), so that per-strategy sampling stays reproducible too.
 *
 * A param's `dist:` field is a single distribution call string (e.g. "beta(10, 90)") — the same
 * shape expr.js's evalCall recognizes via DIST_NAMES. expr.js only exposes eval-to-a-number, not
 * the raw {name, args}, so it's parsed independently here (arguments run through expr.js's own
 * `compile`/`eval` in a plain mean-mode env, so an argument may itself reference other params).
 */
import { rng, quantile, cdf, normalQuantile, DIST_NAMES } from '../core/dist.js';
import { compile } from '../core/expr.js';
import { makeEnv } from '../engine/resolve.js';
import { run } from '../engine/run.js';

// User-model-facing PSA errors (a malformed settings.psa.correlations block) carry a `path`,
// mirroring ModelError/MarkovError — see the two throw sites below. Errors below this point that
// stay plain `Error` (e.g. inside parseDistCall) are internal/assertion-style: they only fire on
// a dist spec shape that expr.js's own compile/eval already accepts, so they aren't a distinct
// user-facing validation surface.
export class PsaError extends Error {
  constructor(message, extra = {}) {
    super(message);
    Object.assign(this, extra);
  }
}

function splitTopLevelArgs(s) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '(') depth++;
    else if (c === ')') depth--;
    else if (c === ',' && depth === 0) { parts.push(s.slice(start, i)); start = i + 1; }
  }
  const last = s.slice(start).trim();
  if (last !== '') parts.push(last);
  return parts.map((p) => p.trim());
}

function parseDistCall(src, env) {
  const m = /^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$/s.exec(src);
  if (!m) throw new Error(`psa: cannot parse dist spec '${src}' as a distribution call`);
  const [, name, argsSrc] = m;
  if (!DIST_NAMES.has(name)) throw new Error(`psa: unknown distribution '${name}' in dist spec '${src}'`);
  const args = splitTopLevelArgs(argsSrc).map((a) => compile(a).eval(env));
  return { name, args };
}

// R = correlation matrix over `names` (declaration order): diagonal 1, entries from
// `correlations` (both directions), everything else 0 (independent, per the brief).
function buildCorrMatrix(names, correlations) {
  const idx = new Map(names.map((n, i) => [n, i]));
  const k = names.length;
  const R = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)));
  for (const { a, b, r } of correlations) {
    const i = idx.get(a), j = idx.get(b);
    R[i][j] = r;
    R[j][i] = r;
  }
  return R;
}

// Cholesky decomposition R = L·Lᵀ (R symmetric). Throws (PsaError, user-facing: a malformed
// settings.psa.correlations block) if R is not positive definite (a non-positive pivot on the
// diagonal) — e.g. inconsistent correlations among the listed params.
function cholesky(R) {
  const k = R.length;
  const L = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = R[i][j];
      for (let m = 0; m < j; m++) sum -= L[i][m] * L[j][m];
      if (i === j) {
        if (sum <= 0)
          throw new PsaError(
            `settings.psa.correlations: correlation matrix is not positive definite (row ${i}) — ` +
            `check for inconsistent correlation coefficients among the listed params`,
            { path: 'settings.psa.correlations' }
          );
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  return L;
}

function matVec(L, x) {
  return L.map((row) => row.reduce((s, v, m) => s + v * x[m], 0));
}

export function psa(model, opts = {}) {
  const psaSettings = model.settings.psa ?? {};
  const n = opts.n ?? psaSettings.n;
  const seed = opts.seed ?? psaSettings.seed;
  const correlations = psaSettings.correlations ?? [];
  const rand = rng(seed);

  // Dist-bearing global params, in declaration order — the only params PSA touches.
  const distParamNames = [...model.params.keys()].filter((name) => model.params.get(name).dist !== undefined);
  const distNameSet = new Set(distParamNames);

  const corrNameSet = new Set();
  for (const { a, b } of correlations) { corrNameSet.add(a); corrNameSet.add(b); }
  for (const name of corrNameSet) {
    if (!distNameSet.has(name))
      throw new PsaError(
        `settings.psa.correlations references '${name}', which has no 'dist:' in params`,
        { path: 'settings.psa.correlations' }
      );
  }
  // Same declaration order as distParamNames, so the copula's index order is deterministic.
  const corrNames = distParamNames.filter((name) => corrNameSet.has(name));
  const L = corrNames.length > 0 ? cholesky(buildCorrMatrix(corrNames, correlations)) : [];

  // Each dist call's numeric {name, args} is fixed across iterations (no cycle-mutated vars are
  // visible here) — resolve once, up front, via a plain mean-mode env.
  const meanEnv = makeEnv(model, { mode: 'mean' });
  const distOf = new Map(distParamNames.map((name) => [name, parseDistCall(model.params.get(name).dist, meanEnv)]));

  const strategies = Object.keys(model.strategies);
  const samples = [];

  for (let iter = 0; iter < n; iter++) {
    const overrides = {};

    for (const name of distParamNames) {
      if (Object.prototype.hasOwnProperty.call(overrides, name)) continue; // already drawn as part of a group

      if (corrNameSet.has(name)) {
        const z0 = corrNames.map(() => normalQuantile(rand()));
        const z = matVec(L, z0);
        corrNames.forEach((cn, i) => {
          const u = cdf({ name: 'normal', args: [0, 1] }, z[i]);
          overrides[cn] = quantile(distOf.get(cn), u);
        });
      } else {
        overrides[name] = quantile(distOf.get(name), rand());
      }
    }

    const { strategies: results } = run(model, { mode: 'sample', rand, overrides });
    const cost = {}, qaly = {};
    for (const s of strategies) { cost[s] = results[s].cost; qaly[s] = results[s].qaly; }
    samples.push({ cost, qaly });
  }

  return { strategies, samples };
}

function nmbOf(sample, strategy, wtp) { return wtp * sample.qaly[strategy] - sample.cost[strategy]; }

export function ceac(psaResult, wtps) {
  const { strategies, samples } = psaResult;
  const curves = Object.fromEntries(strategies.map((s) => [s, new Array(wtps.length).fill(0)]));

  wtps.forEach((wtp, w) => {
    for (const sample of samples) {
      const nmbs = strategies.map((s) => nmbOf(sample, s, wtp));
      const maxNmb = Math.max(...nmbs);
      const winners = strategies.filter((s, i) => nmbs[i] === maxNmb);
      const share = 1 / winners.length;
      for (const s of winners) curves[s][w] += share;
    }
    for (const s of strategies) curves[s][w] /= samples.length;
  });

  return { wtps, curves };
}

export function evpi(psaResult, wtps) {
  const { strategies, samples } = psaResult;
  const n = samples.length;

  return wtps.map((wtp) => {
    let sumMax = 0;
    const sumByStrategy = Object.fromEntries(strategies.map((s) => [s, 0]));
    for (const sample of samples) {
      let maxNmb = -Infinity;
      for (const s of strategies) {
        const v = nmbOf(sample, s, wtp);
        sumByStrategy[s] += v;
        if (v > maxNmb) maxNmb = v;
      }
      sumMax += maxNmb;
    }
    const meanMax = sumMax / n;
    const maxMean = Math.max(...strategies.map((s) => sumByStrategy[s] / n));
    return meanMax - maxMean;
  });
}

export function cePlane(psaResult, { comparator }) {
  const { strategies, samples } = psaResult;
  const out = {};
  for (const s of strategies) {
    if (s === comparator) continue;
    out[s] = samples.map((sample) => ({
      dcost: sample.cost[s] - sample.cost[comparator],
      dqaly: sample.qaly[s] - sample.qaly[comparator],
    }));
  }
  return out;
}
