import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseModel } from '../js/core/model.js';
import { run } from '../js/engine/run.js';
import { check } from '../js/analysis/check.js';
import { cea } from '../js/analysis/cea.js';

const load = (f) => parseModel(readFileSync(new URL(`../examples/${f}`, import.meta.url), 'utf8'));

test('examples pass validation with zero errors', () => {
  for (const f of ['hiv.yaml', 'surgery.yaml'])
    assert.deepEqual(check(load(f)).filter(x => x.level === 'error'), []);
});

test('hiv: engine matches an independent dense reference implementation', () => {
  const m = load('hiv.yaml');
  const r = run(m, {});
  // reference: hand-built numeric matrices, no engine code shared
  const ref = (rr, cDrug) => {
    const P = [
      [.721, .202 * rr, .067 * rr, .010 * rr],
      [0, 0, .407 * rr, .012 * rr],
      [0, 0, 0, .250 * rr],
      [0, 0, 0, 1],
    ];
    P[0][0] = 1 - (P[0][1] + P[0][2] + P[0][3]);
    P[1][1] = 1 - (P[1][2] + P[1][3]);
    P[2][2] = 1 - P[2][3];
    const cost = [2756 + cDrug, 3052 + cDrug, 9007 + cDrug, 0];
    let mvec = [1, 0, 0, 0], totC = 0, totQ = 0;
    for (let t = 1; t <= 20; t++) {
      const next = [0, 0, 0, 0];
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) next[j] += mvec[i] * P[i][j];
      mvec = next;
      const dfC = Math.pow(1.06, -t);
      for (let s = 0; s < 4; s++) { totC += mvec[s] * cost[s] * dfC; totQ += mvec[s] * (s < 3 ? 1 : 0); }
    }
    return { cost: totC, qaly: totQ };
  };
  const mono = ref(1, 2278), combo = ref(0.509, 2278 + 2086);
  assert.ok(Math.abs(r.strategies.mono.cost - mono.cost) < 1e-6, `${r.strategies.mono.cost} vs ${mono.cost}`);
  assert.ok(Math.abs(r.strategies.mono.qaly - mono.qaly) < 1e-9);
  assert.ok(Math.abs(r.strategies.combo.cost - combo.cost) < 1e-6);
  assert.ok(Math.abs(r.strategies.combo.qaly - combo.qaly) < 1e-9);
  // sanity band, not a literature claim: ICER per life-year in single-digit thousands (GBP, 1997 costs)
  const { rows } = cea({ mono: r.strategies.mono, combo: r.strategies.combo }, {});
  const icer = rows.find(x => x.strategy === 'combo').icer;
  assert.ok(icer > 3000 && icer < 15000, `ICER ${icer} outside sanity band`);
});

test('surgery tree: exact hand-computed results', () => {
  const r = run(load('surgery.yaml'), {});
  assert.ok(Math.abs(r.strategies.Surgery.cost - 5200) < 1e-9);
  assert.ok(Math.abs(r.strategies.Surgery.qaly - 0.895) < 1e-9);
  assert.ok(Math.abs(r.strategies.Medication.cost - 800) < 1e-9);
  const { rows } = cea({ Surgery: r.strategies.Surgery, Medication: r.strategies.Medication }, {});
  assert.ok(Math.abs(rows.find(x => x.strategy === 'Surgery').icer - 4400 / 0.155) < 1e-6);
});

test('no Math.random anywhere under js/ (excluding js/vendor — third-party code we don\'t author)', async () => {
  const { readdirSync, readFileSync: read } = await import('node:fs');
  const walk = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap(e =>
    e.isDirectory() ? walk(`${dir}/${e.name}`) : [`${dir}/${e.name}`]);
  for (const f of walk(new URL('../js', import.meta.url).pathname)) {
    if (f.includes('/js/vendor/')) continue; // vendored libs (js-yaml, plotly) aren't ours to constrain
    assert.ok(!read(f, 'utf8').includes('Math.random'), `${f} uses Math.random`);
  }
});
