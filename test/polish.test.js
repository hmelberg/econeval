import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel, formatCycle } from '../js/core/model.js';

// Task 7, item 1: cycle-unit round-trip formatting. Unit table (shared with parseCycle):
// year=1, month=1/12, week=7/365.25, day=1/365.25 — tolerance 1e-9 on the fraction match.

test('formatCycle: exact-unit fractions format as the friendliest matching unit', () => {
  assert.equal(formatCycle(1 / 12), '1 month');
  assert.equal(formatCycle(6 / 12), '6 months');
  assert.equal(formatCycle(1), '1 year');
  assert.equal(formatCycle(2), '2 years');
  assert.equal(formatCycle(7 / 365.25), '1 week');
  assert.equal(formatCycle(2 * (7 / 365.25)), '2 weeks');
  assert.equal(formatCycle(1 / 365.25), '1 day');
  assert.equal(formatCycle(3 * (1 / 365.25)), '3 days');
});

test('formatCycle: a non-matching fraction stays decimal years, always singular', () => {
  assert.equal(formatCycle(0.3), '0.3 year');
});

// --- settings.cycle: parse -> serialize round-trips through the SAME formatter ---

const withCycle = (cycleLine) => `
econeval: 1
type: markov
name: cyc
settings: {cycles: 3, cycle: ${cycleLine}}
states: {a: {utility: 1}}
transitions: {a: {a: 1}}
`;

for (const [input, expected] of [
  ['1 month', '1 month'],
  ['6 months', '6 months'],
  ['2 years', '2 years'],
  ['1 week', '1 week'],
]) {
  test(`settings.cycle round-trip: '${input}' serializes back containing 'cycle: ${expected}'`, () => {
    const m = parseModel(withCycle(input));
    const text = serializeModel(m);
    assert.ok(text.includes(`cycle: ${expected}`), text);
    assert.deepEqual(parseModel(text), m); // parse must accept exactly what serialize emits
  });
}

test('settings.cycle round-trip: an odd fraction (0.3 year) stays decimal on serialize', () => {
  const m = parseModel(withCycle('0.3 year'));
  const text = serializeModel(m);
  assert.ok(text.includes('cycle: 0.3 year'), text);
  assert.deepEqual(parseModel(text), m);
});

// --- tree node delay (stored in years, same parseCycle pathway) uses the same formatter ---

test('tree node delay round-trips through the same cycle formatter as settings.cycle', () => {
  const m = parseModel(`
econeval: 1
type: tree
name: d
tree:
  Root:
    A: {p: 1, delay: 1 month, utility: 5}
    B: {p: rest, delay: 2 weeks, utility: 0}
`);
  assert.ok(Math.abs(m.tree.children[0].delay - 1 / 12) < 1e-9);
  const text = serializeModel(m);
  assert.ok(text.includes('delay: 1 month'), text);
  assert.ok(text.includes('delay: 2 weeks'), text);
  assert.deepEqual(parseModel(text), m);
});
