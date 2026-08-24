import test from 'node:test';
import assert from 'node:assert/strict';
import {
  NODE_R, GRID, hitShapeFor, isInside, pickNode, snapToGrid, fitBox,
} from '../js/ui/canvas/geometry.js';

test('hitShapeFor maps every node kind to its real outline', () => {
  assert.deepEqual(hitShapeFor('state'), { shape: 'circle', r: NODE_R });
  assert.deepEqual(hitShapeFor('chance'), { shape: 'circle', r: NODE_R });
  assert.deepEqual(hitShapeFor('root'), { shape: 'rect', w: 52, h: 52 });
  assert.deepEqual(hitShapeFor('submodel'), { shape: 'stadium', w: 120, h: 32 });
  assert.deepEqual(hitShapeFor('terminal'), { shape: 'rect', w: 16, h: 30 });
});

test('hitShapeFor throws on unrecognized node kind', () => {
  assert.throws(() => hitShapeFor('unknown'), /unrecognized node kind/);
  assert.throws(() => hitShapeFor('circle'), /unrecognized node kind/);
});

test('isInside throws on a non-[x,y]-array point instead of silently NaN-ing every distance', () => {
  // Regression (task-4 review): connectDrop once passed pickNode/isInside an {x, y} object (from
  // clientToUser) instead of a [x, y] array. point[0]/point[1] on that object are undefined, every
  // distance becomes NaN, every `NaN <= ...` is false, and the hit test always misses with no error
  // anywhere -- markov Connect-drag silently could never create a transition. "Errors surfaced,
  // never swallowed" means this shape mismatch must throw, not return a wrong answer.
  const c = hitShapeFor('state');
  assert.throws(() => isInside({ x: 1, y: 2 }, [0, 0], c), /must be a \[x, y\] array/);
  assert.throws(() => isInside([1], [0, 0], c), /must be a \[x, y\] array/);
});

test('circle hit: inside, on the rim, inside the slack, outside', () => {
  const c = hitShapeFor('state');   // r 26, slack 6
  assert.equal(isInside([100, 100], [100, 100], c), true);
  assert.equal(isInside([126, 100], [100, 100], c), true);   // exactly on the rim
  assert.equal(isInside([131, 100], [100, 100], c), true);   // within slack
  assert.equal(isInside([133, 100], [100, 100], c), false);  // past r + slack
});

test('rect hit respects width and height separately', () => {
  const r = hitShapeFor('terminal');  // 16 x 30, slack 6 -> +-14 x, +-21 y
  assert.equal(isInside([100, 100], [100, 100], r), true);
  assert.equal(isInside([113, 100], [100, 100], r), true);
  assert.equal(isInside([116, 100], [100, 100], r), false);
  assert.equal(isInside([100, 120], [100, 100], r), true);
  assert.equal(isInside([100, 123], [100, 100], r), false);
});

test('stadium hit rounds its end caps instead of squaring them', () => {
  const s = hitShapeFor('submodel');  // 120 x 32; caps are circles of r 16 at x = +-44
  assert.equal(isInside([100, 100], [100, 100], s, 0), true);
  assert.equal(isInside([158, 100], [100, 100], s, 0), true);   // 2px inside the cap's far edge (160)
  assert.equal(isInside([161, 100], [100, 100], s, 0), false);
  // The corner of the bounding box is OUTSIDE the pill, which a rect test would wrongly accept.
  assert.equal(isInside([159, 115], [100, 100], s, 0), false);
});

test('pickNode returns the topmost (last) match when shapes overlap', () => {
  const index = [
    { key: 'under', xy: [100, 100], hit: hitShapeFor('state') },
    { key: 'over', xy: [110, 100], hit: hitShapeFor('state') },
  ];
  assert.equal(pickNode([105, 100], index).key, 'over');
  assert.equal(pickNode([70, 100], index).key, 'under');
  assert.equal(pickNode([400, 400], index), null);
});

test('pickNode is forgiving by HIT_SLACK by default', () => {
  // Probe BETWEEN the bare radius and radius + slack: a default of 0 would miss this, a default of
  // HIT_SLACK finds it. Probing inside the bare radius cannot tell the two defaults apart.
  const index = [{ key: 'only', xy: [100, 100], hit: hitShapeFor('state') }];
  assert.equal(pickNode([129, 100], index).key, 'only');   // 29 > r 26, <= r + slack 32
  assert.equal(pickNode([129, 100], index, 0), null);      // explicit 0 refuses it
});

test('snapToGrid rounds to the nearest 12px multiple', () => {
  assert.equal(GRID, 12);
  assert.deepEqual(snapToGrid([0, 0]), [0, 0]);
  assert.deepEqual(snapToGrid([17, 5]), [12, 0]);
  assert.deepEqual(snapToGrid([18, 19]), [24, 24]);
  // -5 / 12 rounds to -0; assert/strict tells -0 and 0 apart, so this pins the normalization.
  assert.deepEqual(snapToGrid([-17, -5]), [-12, 0]);
  assert.ok(Object.is(snapToGrid([-17, -5])[1], 0), 'negative zero must be normalized to +0');
});

test('fitBox wraps every position with padding', () => {
  const box = fitBox([[100, 100], [300, 200]], 50);
  assert.deepEqual(box, { x: 50, y: 50, w: 300, h: 200 });
});

test('fitBox falls back to the base viewBox when there is nothing to fit', () => {
  assert.deepEqual(fitBox([], 60), { x: 0, y: 0, w: 900, h: 640 });
});

test('fitBox never produces a zero-sized box for a single node', () => {
  const box = fitBox([[100, 100]], 60);
  assert.equal(box.w, 120);
  assert.equal(box.h, 120);
});
