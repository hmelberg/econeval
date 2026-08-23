import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv } from '../js/engine/resolve.js';
import { parseModel } from '../js/core/model.js';
import { compile } from '../js/core/expr.js';
import { rng } from '../js/core/dist.js';

const M = parseModel(`
econeval: 1
type: markov
name: r
settings: {cycles: 1}
params:
  a: 2
  b: a * 3
  c:
    value: 10
    dist: beta(2, 8)
  d:
    dist: beta(2, 8)
states: {s: {utility: 1}}
transitions: {s: {s: 1}}
strategies:
  base: {}
  alt:
    a: 5
`);

test('params resolve through the DAG', () => {
  const env = makeEnv(M, {});
  assert.equal(env.get('a'), 2);
  assert.equal(env.get('b'), 6);
  assert.equal(compile('b + 1').eval(env), 7);
});

test('mean mode: value wins over dist; dist-only uses mean', () => {
  const env = makeEnv(M, {});
  assert.equal(env.get('c'), 10);
  assert.ok(Math.abs(env.get('d') - 0.2) < 1e-12);
});

test('sample mode: dist params sample, reproducibly', () => {
  const e1 = makeEnv(M, { mode: 'sample', rand: rng(3) });
  const e2 = makeEnv(M, { mode: 'sample', rand: rng(3) });
  const v1 = e1.get('c');
  assert.equal(v1, e2.get('c'));
  assert.notEqual(v1, 10);
  const memo = e1.get('c');
  assert.equal(memo, v1);                      // memoized: one draw per env
});

test('strategy overrides apply, and beat explicit overrides', () => {
  const env = makeEnv(M, { strategy: 'alt' });
  assert.equal(env.get('a'), 5);
  assert.equal(env.get('b'), 15);
  // strategy pins win: PSA/DSA sweeping 'a' must not unpin a strategy that fixes it
  const env2 = makeEnv(M, { strategy: 'alt', overrides: { a: 7 } });
  assert.equal(env2.get('b'), 15);
  // without a strategy pin, explicit overrides apply
  const env3 = makeEnv(M, { overrides: { a: 7 } });
  assert.equal(env3.get('b'), 21);
});

test('parent scoping (sub-model chain)', () => {
  const parent = makeEnv(M, {});
  const child = makeEnv(M, { overrides: { b: 'a + 100' }, parent });
  assert.equal(child.get('b'), 102);
  assert.equal(child.get('a'), 2);             // falls through to own params first
});

test('vars (t, state_time...) take precedence', () => {
  const env = makeEnv(M, {});
  env.vars.t = 4;
  assert.equal(compile('t * a').eval(env), 8);
});

test('cycle detection names the chain', () => {
  const bad = parseModel(`
econeval: 1
type: markov
name: cyc
settings: {cycles: 1}
params: {x: y + 1, y: x + 1}
states: {s: {utility: 1}}
transitions: {s: {s: 1}}
`);
  assert.throws(() => makeEnv(bad, {}).get('x'), /x.*y.*x|cycle/);
});

// --- Regression: memoized params referencing time vars must not go stale across cycles ---
// (a param whose value-expression touches t/time/age/state_time — directly or transitively
// through another param — must re-evaluate on every get(), tracking env.vars mutation by the
// engine; only dist DRAWS still memoize unconditionally, per PSA "one draw per env" semantics)

const TM = parseModel(`
econeval: 1
type: markov
name: taint
settings: {cycles: 1}
params:
  p: t * 2
  a: t + 1
  b: a * 10
states: {s: {utility: 1}}
transitions: {s: {s: 1}}
`);

test('taint: a param referencing t re-evaluates, not stale after t changes', () => {
  const env = makeEnv(TM, {});
  env.vars.t = 1;
  assert.equal(env.get('p'), 2);
  env.vars.t = 3;
  assert.equal(env.get('p'), 6);               // must NOT reuse the cycle-1 memoized value
});

test('taint: propagates transitively through param references', () => {
  const env = makeEnv(TM, {});
  env.vars.t = 1;
  assert.equal(env.get('b'), 20);              // a = t+1 = 2, b = a*10 = 20
  env.vars.t = 2;
  assert.equal(env.get('b'), 30);              // a = t+1 = 3, b = a*10 = 30 (b re-evaluated too)
});

test('taint: dist draws still memoize unconditionally (one draw per env, unchanged)', () => {
  const e1 = makeEnv(M, { mode: 'sample', rand: rng(3) });
  const e2 = makeEnv(M, { mode: 'sample', rand: rng(3) });
  const v1 = e1.get('c');
  assert.equal(v1, e2.get('c'));
  assert.notEqual(v1, 10);
  const memo = e1.get('c');
  assert.equal(memo, v1);                      // memoized: one draw per env
});
