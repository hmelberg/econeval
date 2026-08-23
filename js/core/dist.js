function sfc32(a, b, c, d) {
  return function () {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = (a + b | 0) + d | 0;
    d = d + 1 | 0;
    a = b ^ b >>> 9;
    b = c + (c << 3) | 0;
    c = (c << 21 | c >>> 11) + t | 0;
    return (t >>> 0) / 4294967296;
  };
}

export function rng(seed) {
  const r = sfc32(0x9e3779b9, 0x243f6a88, 0xb7e15162, seed >>> 0);
  for (let i = 0; i < 12; i++) r();   // decorrelate nearby seeds
  return r;
}

function sampleNormal01(rand) {           // Marsaglia polar
  let u, v, s;
  do { u = 2*rand() - 1; v = 2*rand() - 1; s = u*u + v*v; } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

function sampleGammaShape(rand, k) {      // Marsaglia–Tsang; scale 1
  if (k < 1) return sampleGammaShape(rand, k + 1) * Math.pow(rand(), 1 / k);
  const d = k - 1/3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = sampleNormal01(rand); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = rand();
    if (u < 1 - 0.0331 * x**4) return d * v;
    if (Math.log(u) < 0.5 * x*x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// gamma(mean, sd) -> shape/scale
function gammaShapeScale(m, sd) { return { shape: (m/sd)**2, scale: sd*sd/m }; }

const DISTS = {
  beta: {
    arity: 2,
    mean: (a, b) => a / (a + b),
    sample: (rand, a, b) => {
      const x = sampleGammaShape(rand, a), y = sampleGammaShape(rand, b);
      return x / (x + y);
    },
  },
  gamma: {
    arity: 2,
    mean: (m) => m,
    sample: (rand, m, sd) => { const {shape, scale} = gammaShapeScale(m, sd); return sampleGammaShape(rand, shape) * scale; },
  },
  normal: {
    arity: 2,
    mean: (m) => m,
    sample: (rand, m, sd) => m + sd * sampleNormal01(rand),
  },
  lognormal: {
    arity: 2,
    mean: (mu, sig) => Math.exp(mu + sig*sig/2),
    sample: (rand, mu, sig) => Math.exp(mu + sig * sampleNormal01(rand)),
  },
  uniform: {
    arity: 2,
    mean: (lo, hi) => (lo + hi) / 2,
    sample: (rand, lo, hi) => lo + (hi - lo) * rand(),
  },
  triangular: {
    arity: 3,
    mean: (lo, mode, hi) => (lo + mode + hi) / 3,
    sample: (rand, lo, mode, hi) => {
      const u = rand(), f = (mode - lo) / (hi - lo);
      return u < f ? lo + Math.sqrt(u * (hi - lo) * (mode - lo))
                   : hi - Math.sqrt((1 - u) * (hi - lo) * (hi - mode));
    },
  },
};

export const DIST_NAMES = new Set(Object.keys(DISTS));

function get(d) {
  const spec = DISTS[d.name];
  if (!spec) throw new Error(`unknown distribution: ${d.name}`);
  if (d.args.length !== spec.arity) throw new Error(`${d.name} takes ${spec.arity} arguments, got ${d.args.length}`);
  return spec;
}

export function mean(d) { return get(d).mean(...d.args); }
export function sample(d, rand) { return get(d).sample(rand, ...d.args); }

export function sampleDirichlet(rand, counts) {
  const g = counts.map(c => sampleGammaShape(rand, c));
  const s = g.reduce((a, b) => a + b, 0);
  return g.map(x => x / s);
}
