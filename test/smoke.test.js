import test from 'node:test';
import assert from 'node:assert/strict';
import { load } from 'js-yaml';

test('yaml round-trips and keeps expressions as strings', () => {
  const doc = load('p: p_AB * rr\nn: 3');
  assert.equal(doc.p, 'p_AB * rr');
  assert.equal(doc.n, 3);
});
