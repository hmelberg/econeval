import test from 'node:test';
import assert from 'node:assert/strict';
import { cdf, quantile, normalQuantile } from '../js/core/dist.js';

test('normalQuantile matches known values', () => {
  assert.ok(Math.abs(normalQuantile(0.975) - 1.959963985) < 1e-6);
  assert.ok(Math.abs(normalQuantile(0.5)) < 1e-9);
  assert.ok(Math.abs(normalQuantile(0.025) + 1.959963985) < 1e-6);
});

test('cdf known points', () => {
  assert.ok(Math.abs(cdf({name:'uniform', args:[0, 2]}, 0.5) - 0.25) < 1e-12);
  assert.ok(Math.abs(cdf({name:'normal', args:[0, 1]}, 1.959963985) - 0.975) < 1e-7);
  assert.ok(Math.abs(cdf({name:'beta', args:[1, 1]}, 0.3) - 0.3) < 1e-9);   // beta(1,1)=uniform
  // gamma(mean,sd) with mean=sd is exponential(1/mean): cdf(x) = 1-exp(-x/mean)
  assert.ok(Math.abs(cdf({name:'gamma', args:[2, 2]}, 3) - (1 - Math.exp(-1.5))) < 1e-8);
});

test('quantile inverts cdf for every distribution', () => {
  const dists = [
    {name:'beta', args:[202, 798]}, {name:'beta', args:[0.5, 0.5]},
    {name:'gamma', args:[2756, 500]}, {name:'normal', args:[5, 2]},
    {name:'lognormal', args:[-0.675, 0.173]}, {name:'uniform', args:[2, 4]},
    {name:'triangular', args:[0, 1, 5]},
  ];
  for (const d of dists)
    for (const p of [0.01, 0.1, 0.5, 0.9, 0.99])
      assert.ok(Math.abs(cdf(d, quantile(d, p)) - p) < 1e-8, `${d.name} p=${p}`);
});
