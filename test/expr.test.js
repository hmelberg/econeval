import test from 'node:test';
import assert from 'node:assert/strict';
import { compile, ExprError } from '../js/core/expr.js';
import { rng } from '../js/core/dist.js';

const env = (vars = {}, extra = {}) => ({
  get: (n) => { if (n in vars) return vars[n]; throw new ExprError(`unknown name: ${n}`, 0); },
  mode: 'mean', cycleYears: 1, ...extra,
});

test('arithmetic and precedence', () => {
  assert.equal(compile('2 + 3 * 4').eval(env()), 14);
  assert.equal(compile('(1 + 2) * 3').eval(env()), 9);
  assert.equal(compile('2 ^ 3 ^ 2').eval(env()), 512);      // right-assoc
  assert.equal(compile('-2 ^ 2').eval(env()), -4);
  assert.equal(compile(7).eval(env()), 7);                   // number passthrough
});

test('comparisons, if, min/max', () => {
  assert.equal(compile('1 < 2').eval(env()), 1);
  assert.equal(compile('if(t >= 3, 1, 0)').eval(env({ t: 3 })), 1);
  assert.equal(compile('min(3, 1, 2)').eval(env()), 1);
  assert.equal(compile('max(3, 1, 2)').eval(env()), 3);
});

test('names resolve via env.get; names set is collected', () => {
  const e = compile('p_AB * rr + 1');
  assert.deepEqual([...e.names].sort(), ['p_AB', 'rr']);
  assert.equal(e.eval(env({ p_AB: 0.2, rr: 0.5 })), 1.1);
  assert.throws(() => compile('nope').eval(env()), /unknown name: nope/);
});

test('rate/prob conversions use cycleYears', () => {
  const e1 = compile('rate_to_prob(0.2)');
  assert.ok(Math.abs(e1.eval(env({}, { cycleYears: 1 })) - (1 - Math.exp(-0.2))) < 1e-12);
  assert.ok(Math.abs(e1.eval(env({}, { cycleYears: 1/12 })) - (1 - Math.exp(-0.2/12))) < 1e-12);
  const p = 0.3;
  const rt = compile('prob_to_rate(0.3)').eval(env({}, { cycleYears: 1 }));
  assert.ok(Math.abs(rt - (-Math.log(1 - p))) < 1e-12);
  // probability observed over 5 years, converted to a 1-year cycle
  const rs = compile('rescale_prob(0.3, 5)').eval(env({}, { cycleYears: 1 }));
  assert.ok(Math.abs(rs - (1 - Math.pow(1 - 0.3, 1/5))) < 1e-12);
});

test('lookup interpolates and clamps', () => {
  const tables = { mort: { age: [60, 61, 62], rate: [0.1, 0.2, 0.3] } };
  const ev = (src, vars) => compile(src).eval(env(vars, { tables }));
  assert.equal(ev('lookup(mort, 61)', {}), 0.2);
  assert.ok(Math.abs(ev('lookup(mort, 60.5)', {}) - 0.15) < 1e-12);
  assert.equal(ev('lookup(mort, 59)', {}), 0.1);              // clamped
  assert.equal(ev('lookup(mort, 99)', {}), 0.3);              // clamped
  assert.equal(ev('lookup(mort, 61, rate)', {}), 0.2);        // explicit column
});

test('distribution calls: mean in mean mode, sampled in sample mode', () => {
  const e = compile('beta(2, 8)');
  assert.ok(Math.abs(e.eval(env()) - 0.2) < 1e-12);
  const r1 = e.eval(env({}, { mode: 'sample', rand: rng(5) }));
  const r2 = e.eval(env({}, { mode: 'sample', rand: rng(5) }));
  assert.equal(r1, r2);                                       // seeded => reproducible
  assert.notEqual(r1, 0.2);
});

test('errors carry position; rest rejected here', () => {
  try { compile('2 + * 3'); assert.fail('should throw'); }
  catch (err) { assert.ok(err instanceof ExprError); assert.equal(typeof err.pos, 'number'); }
  assert.throws(() => compile('rest').eval(env()), /rest/);
  assert.throws(() => compile('foo(1)').eval(env()), /foo/);
});
