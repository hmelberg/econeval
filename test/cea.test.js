import test from 'node:test';
import assert from 'node:assert/strict';
import { cea } from '../js/analysis/cea.js';

const R = {
  A: { cost: 0, qaly: 0 },
  B: { cost: 1000, qaly: 0.5 },
  D: { cost: 1500, qaly: 0.4 },   // dominated by B
  E: { cost: 1800, qaly: 0.55 },  // extended dominated (ICER 16000 then C at 4000)
  C: { cost: 2000, qaly: 0.6 },
};

test('dominance, extended dominance, ICERs', () => {
  const { rows } = cea(R, { wtp: 20000 });
  const by = Object.fromEntries(rows.map(r => [r.strategy, r]));
  assert.equal(by.D.status, 'dominated');
  assert.equal(by.E.status, 'extended');
  assert.equal(by.A.icer, null);
  assert.ok(Math.abs(by.B.icer - 2000) < 1e-9);     // 1000/0.5
  assert.ok(Math.abs(by.C.icer - 10000) < 1e-9);    // (2000-1000)/(0.6-0.5)
  assert.ok(Math.abs(by.B.nmb - (20000*0.5 - 1000)) < 1e-9);
  assert.deepEqual(rows.map(r => r.strategy), ['A', 'B', 'D', 'E', 'C']);  // cost order
});

test('no wtp -> nmb omitted', () => {
  const { rows } = cea({ A: { cost: 0, qaly: 0 }, B: { cost: 10, qaly: 1 } }, {});
  assert.equal(rows[1].nmb, undefined);
});

test('exact-duplicate strategies: icer is null, never NaN', () => {
  const { rows } = cea({
    A: { cost: 0, qaly: 0 },
    B1: { cost: 1000, qaly: 0.5 },
    B2: { cost: 1000, qaly: 0.5 },
    C: { cost: 2000, qaly: 0.8 },
  }, {});
  const by = Object.fromEntries(rows.map(r => [r.strategy, r]));

  for (const r of rows) {
    if (r.icer !== null) assert.equal(Number.isNaN(r.icer), false, `${r.strategy}.icer is NaN`);
  }
  assert.equal(by.B2.icer, null);
  assert.notEqual(by.B1.status, 'dominated');
  assert.notEqual(by.B2.status, 'dominated');
  assert.equal(typeof by.C.icer, 'number');
  assert.equal(Number.isFinite(by.C.icer), true);
});
