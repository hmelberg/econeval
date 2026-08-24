import test from 'node:test';
import assert from 'node:assert/strict';
import { clampMenuPosition } from '../js/ui/canvas/menu-position.js';

// Fixed viewport for every case below: 1000 x 800, margin 8 -> right/bottom overflow thresholds
// (maxX/maxY) at 992/792.
const VW = 1000;
const VH = 800;
const M = 8;

test('no flip: menu fits below-right of the click, placed at the click point unchanged', () => {
  // width 200, height 150: clientX 100 + width 300 <= maxX 992; clientY 100 + height 250 <= maxY 792.
  assert.deepEqual(clampMenuPosition(100, 100, 200, 150, VW, VH, M), { left: 100, top: 100 });
});

test('flips left when the un-flipped placement would overflow the right edge', () => {
  // clientX 950 + width 200 = 1150 > maxX 992 -> flip: left = 950 - 200 = 750. Y untouched.
  assert.deepEqual(clampMenuPosition(950, 100, 200, 150, VW, VH, M), { left: 750, top: 100 });
});

test('flips up when the un-flipped placement would overflow the bottom edge', () => {
  // clientY 750 + height 150 = 900 > maxY 792 -> flip: top = 750 - 150 = 600. X untouched.
  assert.deepEqual(clampMenuPosition(100, 750, 200, 150, VW, VH, M), { left: 100, top: 600 });
});

test('flips both axes at once when the click is near the bottom-right corner', () => {
  assert.deepEqual(clampMenuPosition(950, 750, 200, 150, VW, VH, M), { left: 750, top: 600 });
});

test('clamps to the margin when a flip would push the menu off the left edge', () => {
  // width 900 is wider than clientX 100 is far from the left edge: overflow triggers a flip
  // (100 + 900 = 1000 > maxX 992), but the flipped x (100 - 900 = -800) is negative, so it must
  // clamp to the margin rather than escape off-screen to the left.
  assert.deepEqual(clampMenuPosition(100, 100, 900, 150, VW, VH, M), { left: M, top: 100 });
});

test('clamps to the margin when a flip would push the menu off the top edge', () => {
  // height 700 similarly overflows the bottom (100 + 700 = 800 > maxY 792) and the flipped y
  // (100 - 700 = -600) is negative, so it clamps to the margin instead.
  assert.deepEqual(clampMenuPosition(100, 100, 200, 700, VW, VH, M), { left: 100, top: M });
});

test('a menu larger than the viewport itself always clamps to the top-left margin', () => {
  // width 1200 > innerWidth 1000, height 1000 > innerHeight 800: even clicking near the origin
  // (10, 10) triggers a flip on both axes, and both flipped coordinates go deeply negative, so the
  // only sane placement left is pinned to the margin on both axes.
  assert.deepEqual(clampMenuPosition(10, 10, 1200, 1000, VW, VH, M), { left: M, top: M });
});
