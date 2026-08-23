import test from 'node:test';
import assert from 'node:assert/strict';

test('vendored js-yaml is importable and matches the npm package API', async () => {
  const vendored = await import('../js/vendor/js-yaml.mjs');
  assert.equal(typeof vendored.load, 'function');
  assert.equal(typeof vendored.dump, 'function');
  const npm = await import('js-yaml');
  assert.deepEqual(vendored.load('a: 1'), npm.load('a: 1'));
});
