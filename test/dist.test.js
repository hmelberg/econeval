import test from 'node:test';
import assert from 'node:assert/strict';
import { rng, mean, sample, sampleDirichlet, DIST_NAMES } from '../js/core/dist.js';

test('rng is deterministic per seed and in [0,1)', () => {
  const a = rng(42), b = rng(42), c = rng(43);
  const seqA = [a(), a(), a()], seqB = [b(), b(), b()];
  assert.deepEqual(seqA, seqB);
  assert.notDeepEqual(seqA, [c(), c(), c()]);
  for (const x of seqA) assert.ok(x >= 0 && x < 1);
});

test('means are analytic', () => {
  assert.ok(Math.abs(mean({name:'beta', args:[202, 798]}) - 0.202) < 1e-12);
  assert.equal(mean({name:'gamma', args:[2756, 50]}), 2756);
  assert.equal(mean({name:'normal', args:[5, 2]}), 5);
  assert.ok(Math.abs(mean({name:'lognormal', args:[-0.675, 0.173]}) - Math.exp(-0.675 + 0.173**2/2)) < 1e-12);
  assert.equal(mean({name:'uniform', args:[2, 4]}), 3);
  assert.equal(mean({name:'triangular', args:[0, 1, 5]}), 2);
});

test('sample means converge to analytic means', () => {
  const N = 20000;
  for (const d of [
    {name:'beta', args:[2, 8]}, {name:'gamma', args:[100, 30]},
    {name:'normal', args:[5, 2]}, {name:'lognormal', args:[0, 0.5]},
    {name:'uniform', args:[2, 4]}, {name:'triangular', args:[0, 1, 5]},
  ]) {
    const rand = rng(7);
    let s = 0;
    for (let i = 0; i < N; i++) s += sample(d, rand);
    const m = mean(d);
    const tol = Math.max(0.02 * Math.abs(m), 0.02);
    assert.ok(Math.abs(s/N - m) < tol, `${d.name}: ${s/N} vs ${m}`);
  }
});

test('beta samples stay in (0,1); gamma positive', () => {
  const rand = rng(1);
  for (let i = 0; i < 1000; i++) {
    const x = sample({name:'beta', args:[0.5, 0.5]}, rand);
    assert.ok(x > 0 && x < 1);
    assert.ok(sample({name:'gamma', args:[10, 20]}, rand) > 0);
  }
});

test('dirichlet: normalized, deterministic, mean ~ counts/total', () => {
  const rand = rng(9);
  const one = sampleDirichlet(rand, [721, 202, 67, 10]);
  assert.ok(Math.abs(one.reduce((a,b)=>a+b, 0) - 1) < 1e-12);
  const acc = [0,0,0,0];
  const rand2 = rng(11);
  for (let i = 0; i < 5000; i++) sampleDirichlet(rand2, [721, 202, 67, 10]).forEach((v,j)=>acc[j]+=v);
  assert.ok(Math.abs(acc[0]/5000 - 0.721) < 0.01);
});

test('unknown distribution throws', () => {
  assert.throws(() => mean({name:'weibull', args:[1,1]}), /weibull/);
  assert.ok(DIST_NAMES.has('beta') && !DIST_NAMES.has('weibull'));
});
