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

// ---- special functions (Numerical Recipes-style) ----
function gammln(x) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += cof[j] / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function gammPLower(a, x) {                 // regularized P(a,x)
  if (x <= 0) return 0;
  if (x < a + 1) {                          // series
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 500; n++) {
      ap++; del *= x / ap; sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - gammln(a));
  }
  // continued fraction for Q(a,x)
  const FPMIN = 1e-300;
  let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
  for (let i = 1; i <= 500; i++) {
    const an = -i * (i - a);
    b += 2; d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - gammln(a)) * h;
}

function betacf(a, b, x) {
  const FPMIN = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d; let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}

function betaInc(a, b, x) {                 // regularized I_x(a,b)
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(gammln(a + b) - gammln(a) - gammln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a
                                   : 1 - bt * betacf(b, a, 1 - x) / b;
}

function normCdf(z) {
  return z >= 0 ? 0.5 + 0.5 * gammPLower(0.5, z * z / 2)
                : 0.5 - 0.5 * gammPLower(0.5, z * z / 2);
}

export function normalQuantile(p) {         // Acklam's algorithm
  if (p <= 0 || p >= 1) throw new Error(`normalQuantile: p out of (0,1): ${p}`);
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, x;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  } else if (p <= phigh) {
    q = p - 0.5; const r = q * q;
    x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
        (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
         ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  }
  // one Halley refinement using normCdf
  const e = normCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

const CDFS = {
  beta: (x, a, b) => betaInc(a, b, x),
  gamma: (x, m, sd) => { const { shape, scale } = gammaShapeScale(m, sd); return gammPLower(shape, x / scale); },
  normal: (x, m, sd) => normCdf((x - m) / sd),
  lognormal: (x, mu, sig) => x <= 0 ? 0 : normCdf((Math.log(x) - mu) / sig),
  uniform: (x, lo, hi) => Math.min(1, Math.max(0, (x - lo) / (hi - lo))),
  triangular: (x, lo, mode, hi) => {
    if (x <= lo) return 0;
    if (x >= hi) return 1;
    return x <= mode ? (x - lo) ** 2 / ((hi - lo) * (mode - lo))
                     : 1 - (hi - x) ** 2 / ((hi - lo) * (hi - mode));
  },
};

export function cdf(d, x) {
  const f = CDFS[d.name];
  if (!f) throw new Error(`unknown distribution: ${d.name}`);
  return f(x, ...d.args);
}

// support bounds for quantile bisection
function bounds(d) {
  const [p1, p2, p3] = d.args;
  switch (d.name) {
    case 'beta': return [0, 1];
    case 'gamma': return [0, p1 + 40 * p2];
    case 'normal': return [p1 - 40 * p2, p1 + 40 * p2];
    case 'lognormal': return [0, Math.exp(p1 + 40 * p2)];
    case 'uniform': return [p1, p2];
    case 'triangular': return [p1, p3];
    default: throw new Error(`unknown distribution: ${d.name}`);
  }
}

export function quantile(d, p) {
  if (d.name === 'normal') return d.args[0] + d.args[1] * normalQuantile(p);
  if (d.name === 'lognormal') return Math.exp(d.args[0] + d.args[1] * normalQuantile(p));
  let [lo, hi] = bounds(d);
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(d, mid) < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
