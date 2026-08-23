import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { makeEnv } from '../js/engine/resolve.js';
import { runMarkov } from '../js/engine/markov.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('age-dependent mortality via lookup', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: age
settings: {cycles: 3, start: alive, correction: none, age: 60}
tables:
  mort:
    age: [60, 61, 62]
    rate: [0.1, 0.2, 0.3]
states:
  alive: {utility: 1}
  dead: {utility: 0}
transitions:
  alive:
    alive: rest
    dead: lookup(mort, age)
  dead: {dead: 1}
`);
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  // t=1 age 60 -> p .1 ; t=2 age 61 -> .2 ; t=3 age 62 -> .3
  close(r.trace[0].occupancy.alive, 0.9);
  close(r.trace[1].occupancy.alive, 0.72);
  close(r.trace[2].occupancy.alive, 0.504);
});

test('state_time tunnels: forced exit after 3 cycles in state', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: st
settings: {cycles: 4, start: sick, correction: none}
states:
  sick: {utility: 0.5}
  dead: {utility: 0}
transitions:
  sick:
    sick: rest
    dead: if(state_time >= 3, 1, 0)
  dead: {dead: 1}
`);
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(r.trace[0].occupancy.sick, 1);    // st=1: stay
  close(r.trace[1].occupancy.sick, 1);    // st=2: stay
  close(r.trace[2].occupancy.dead, 1);    // st=3: die
  close(r.trace[3].occupancy.dead, 1);
});

test('state_time in payoffs (ramping cost)', () => {
  const m = parseModel(`
econeval: 1
type: markov
name: stp
settings: {cycles: 3, start: sick, correction: none}
states:
  sick: {cost: 100 * state_time, utility: 0}
  dead: {utility: 0}
transitions:
  sick: {sick: 1}
  dead: {dead: 1}
`);
  const r = runMarkov(m, makeEnv(m, {}), { discount: { cost: 0, effect: 0 } });
  close(r.totals.cost, 100 * (1 + 2 + 3));
});
