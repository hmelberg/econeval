// Tests for the pure half of js/ui/panels.js — the layout state machine and its localStorage
// serialization. The DOM-wiring half (initPanels) is exercised by manual verification only, per
// constraints.md ("Pure logic ... lives in DOM-free modules with tests; DOM modules ... hold no
// business logic" — nothing under test here touches the DOM).

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clampPane, nextLayoutState, defaultLayoutState, serializeLayout, parseLayout,
  saveLayout, loadLayout, PANE_BOUNDS, LAYOUT_KEY,
} from '../js/ui/panels.js';

// A Map-backed storage shim, mirroring the pattern files.js's tests use for injectable storage —
// no real localStorage touched.
function fakeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    _map: m,
  };
}

// --- clampPane ---

test('clampPane: value inside bounds passes through unchanged', () => {
  assert.equal(clampPane(300, { min: 200, max: 560 }), 300);
});

test('clampPane: clamps below min', () => {
  assert.equal(clampPane(0, { min: 200, max: 560 }), 200);
});

test('clampPane: clamps above max', () => {
  assert.equal(clampPane(9999, { min: 200, max: 560 }), 560);
});

// --- drag clamps at both bounds (required) ---

test('drag: yaml pane clamps at the lower bound (200)', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag', pane: 'yaml', dx: -10000 });
  assert.equal(next.yaml.w, PANE_BOUNDS.yaml.min);
});

test('drag: yaml pane clamps at the upper bound (560)', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag', pane: 'yaml', dx: 10000 });
  assert.equal(next.yaml.w, PANE_BOUNDS.yaml.max);
});

test('drag: inspector pane clamps at the lower bound (240) — dragging the right splitter right shrinks it', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag', pane: 'insp', dx: 10000 });
  assert.equal(next.insp.w, PANE_BOUNDS.insp.min);
});

test('drag: inspector pane clamps at the upper bound (480) — dragging the right splitter left grows it', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag', pane: 'insp', dx: -10000 });
  assert.equal(next.insp.w, PANE_BOUNDS.insp.max);
});

test('drag: a small in-bounds delta is not clamped, and does not mutate the input state', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag', pane: 'yaml', dx: 20 });
  assert.equal(next.yaml.w, 320);
  assert.equal(s.yaml.w, 300); // original untouched
});

// --- toggle-yaml reopens at last width (required) ---

test('toggle-yaml: opens the (default-closed) pane at its default width (300)', () => {
  const s = defaultLayoutState();
  assert.equal(s.yaml.open, false);
  const next = nextLayoutState(s, { type: 'toggle-yaml' });
  assert.equal(next.yaml.open, true);
  assert.equal(next.yaml.w, 300);
});

test('toggle-yaml: closing then reopening restores the last dragged width, not the default', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-yaml' }); // open at 300
  s = nextLayoutState(s, { type: 'drag', pane: 'yaml', dx: 100 }); // -> 400
  assert.equal(s.yaml.w, 400);
  s = nextLayoutState(s, { type: 'toggle-yaml' }); // close
  assert.equal(s.yaml.open, false);
  assert.equal(s.yaml.w, 400); // width preserved while closed
  s = nextLayoutState(s, { type: 'toggle-yaml' }); // reopen
  assert.equal(s.yaml.open, true);
  assert.equal(s.yaml.w, 400); // reopened at last width, not reset to the 300 default
});

test('toggle-yaml: closing clears a minimized flag and un-maximizes yaml (no orphaned collapsed state)', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-yaml' }); // open
  s = nextLayoutState(s, { type: 'minimize', pane: 'yaml' });
  assert.equal(s.yaml.min, true);
  s = nextLayoutState(s, { type: 'toggle-yaml' }); // close
  assert.equal(s.yaml.open, false);
  assert.equal(s.yaml.min, false);
});

// --- maximize/restore round-trip (required) ---

test('maximize/restore: maximize sets state.maximized, restore clears it back to the original state', () => {
  const s = defaultLayoutState();
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'canvas' });
  assert.equal(maxed.maximized, 'canvas');
  const restored = nextLayoutState(maxed, { type: 'restore' });
  assert.deepEqual(restored, s);
});

test('maximize: maximizing the yaml pane also opens it and clears its minimized flag', () => {
  const s = defaultLayoutState(); // yaml closed
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'yaml' });
  assert.equal(maxed.maximized, 'yaml');
  assert.equal(maxed.yaml.open, true);
  assert.equal(maxed.yaml.min, false);
});

test('maximize when already maximized -> restore (required contract line, verbatim)', () => {
  const s = defaultLayoutState();
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'inspector' });
  assert.equal(maxed.maximized, 'inspector');
  const toggledOff = nextLayoutState(maxed, { type: 'maximize', pane: 'inspector' });
  assert.equal(toggledOff.maximized, null);
});

test('maximize: maximizing a different pane while one is already maximized simply switches', () => {
  const s = defaultLayoutState();
  const yamlMax = nextLayoutState(s, { type: 'maximize', pane: 'yaml' });
  const canvasMax = nextLayoutState(yamlMax, { type: 'maximize', pane: 'canvas' });
  assert.equal(canvasMax.maximized, 'canvas');
});

// --- minimize on a maximized pane restores first (required) ---

test('minimize a maximized pane: restores first, then minimizes', () => {
  const s = defaultLayoutState();
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'yaml' });
  assert.equal(maxed.maximized, 'yaml');
  const next = nextLayoutState(maxed, { type: 'minimize', pane: 'yaml' });
  assert.equal(next.maximized, null); // restored first
  assert.equal(next.yaml.min, true); // then minimized
});

test('minimize: toggles back off (un-minimizes) on a second call — the strip-click restore path', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'minimize', pane: 'inspector' });
  assert.equal(s.insp.min, true);
  s = nextLayoutState(s, { type: 'minimize', pane: 'inspector' });
  assert.equal(s.insp.min, false);
});

test('minimize: canvas cannot minimize — a minimize action targeting it is a no-op', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'minimize', pane: 'canvas' });
  assert.deepEqual(next, s);
});

// --- misc state-machine robustness ---

test('nextLayoutState never mutates the state object it was given', () => {
  const s = defaultLayoutState();
  const snapshot = JSON.parse(JSON.stringify(s));
  nextLayoutState(s, { type: 'drag', pane: 'yaml', dx: 50 });
  nextLayoutState(s, { type: 'maximize', pane: 'canvas' });
  nextLayoutState(s, { type: 'minimize', pane: 'inspector' });
  assert.deepEqual(s, snapshot);
});

test('nextLayoutState throws on an unrecognized action type (fail fast on a wiring bug)', () => {
  const s = defaultLayoutState();
  assert.throws(() => nextLayoutState(s, { type: 'nope' }), /unknown action type/);
});

test('defaultLayoutState returns a fresh object each call (no shared references)', () => {
  const a = defaultLayoutState();
  const b = defaultLayoutState();
  a.yaml.w = 999;
  assert.equal(b.yaml.w, 300);
});

// --- serialize/parse round-trip (required) ---

test('serialize/parse round-trip: a well-formed state survives unchanged', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-yaml' });
  s = nextLayoutState(s, { type: 'drag', pane: 'yaml', dx: 45 });
  s = nextLayoutState(s, { type: 'minimize', pane: 'inspector' });
  s = { ...s, outline: { collapsed: ['group:parameters'], filter: 'cost' } };
  const json = serializeLayout(s);
  assert.equal(typeof json, 'string');
  const parsed = parseLayout(json);
  assert.deepEqual(parsed, s);
});

test('parseLayout("garbage") -> null (required)', () => {
  assert.equal(parseLayout('garbage'), null);
});

test('parseLayout: non-object JSON values (null, array, number) are also treated as unparseable', () => {
  assert.equal(parseLayout('null'), null);
  assert.equal(parseLayout('[1,2,3]'), null);
  assert.equal(parseLayout('42'), null);
});

test('parseLayout: a valid-but-empty object recovers to full defaults, field by field', () => {
  assert.deepEqual(parseLayout('{}'), defaultLayoutState());
});

test('parseLayout: a partial blob keeps its given fields and fills only what is missing', () => {
  const partial = JSON.stringify({ yaml: { w: 350, open: true, min: false } });
  const parsed = parseLayout(partial);
  assert.equal(parsed.yaml.w, 350);
  assert.equal(parsed.yaml.open, true);
  assert.equal(parsed.insp.w, 300); // default, since insp was absent
  assert.deepEqual(parsed.outline, { collapsed: [], filter: '' }); // default, since outline was absent
});

test('parseLayout: an out-of-range stored width is clamped back into bounds on load', () => {
  const stale = JSON.stringify({ yaml: { w: 9999, open: true, min: false }, insp: { w: 1, min: false } });
  const parsed = parseLayout(stale);
  assert.equal(parsed.yaml.w, PANE_BOUNDS.yaml.max);
  assert.equal(parsed.insp.w, PANE_BOUNDS.insp.min);
});

test('parseLayout: an invalid maximized value falls back to null', () => {
  const bad = JSON.stringify({ maximized: 'nonsense' });
  assert.equal(parseLayout(bad).maximized, null);
});

// --- saveLayout/loadLayout over injected storage ---

test('saveLayout writes under the documented localStorage key', () => {
  const storage = fakeStorage();
  saveLayout(defaultLayoutState(), storage);
  assert.ok(storage._map.has(LAYOUT_KEY));
  assert.equal(storage._map.get(LAYOUT_KEY), serializeLayout(defaultLayoutState()));
});

test('loadLayout with nothing stored yet returns defaults', () => {
  const storage = fakeStorage();
  assert.deepEqual(loadLayout(storage), defaultLayoutState());
});

test('saveLayout/loadLayout round-trip through injected storage', () => {
  const storage = fakeStorage();
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'maximize', pane: 'canvas' });
  s = { ...s, outline: { collapsed: ['group:settings'], filter: '' } };
  saveLayout(s, storage);
  assert.deepEqual(loadLayout(storage), s);
});

test('loadLayout falls back to defaults when the stored blob is corrupt JSON', () => {
  const storage = fakeStorage();
  storage.setItem(LAYOUT_KEY, 'not json{{{');
  assert.deepEqual(loadLayout(storage), defaultLayoutState());
});

test('saveLayout never throws even if the storage backend does', () => {
  const angryStorage = { getItem: () => null, setItem: () => { throw new Error('quota'); } };
  assert.doesNotThrow(() => saveLayout(defaultLayoutState(), angryStorage));
});

test('loadLayout never throws even if the storage backend does', () => {
  const angryStorage = { getItem: () => { throw new Error('blocked'); } };
  assert.doesNotThrow(() => loadLayout(angryStorage));
  assert.deepEqual(loadLayout(angryStorage), defaultLayoutState());
});

// --- results drawer (phase-3 task 1) ---

test('defaultLayoutState: results starts closed, h 300, not minimized', () => {
  const s = defaultLayoutState();
  assert.deepEqual(s.results, { h: 300, open: false, min: false });
});

// --- drag-results clamps at both bounds (required) ---

test('drag-results: clamps at the lower bound (160)', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag-results', dy: 10000 });
  assert.equal(next.results.h, PANE_BOUNDS.results.min);
});

test('drag-results: clamps at the upper bound (600)', () => {
  const s = defaultLayoutState();
  const next = nextLayoutState(s, { type: 'drag-results', dy: -10000 });
  assert.equal(next.results.h, PANE_BOUNDS.results.max);
});

test('drag-results: a small in-bounds delta is not clamped, and does not mutate the input state', () => {
  const s = defaultLayoutState();
  // dy negative (drag up) grows the drawer, same sign convention as the right splitter's dx.
  const next = nextLayoutState(s, { type: 'drag-results', dy: -20 });
  assert.equal(next.results.h, 320);
  assert.equal(s.results.h, 300); // original untouched
});

// --- toggle-results reopens at last h (required) ---

test('toggle-results: opens the (default-closed) drawer at its default height (300)', () => {
  const s = defaultLayoutState();
  assert.equal(s.results.open, false);
  const next = nextLayoutState(s, { type: 'toggle-results' });
  assert.equal(next.results.open, true);
  assert.equal(next.results.h, 300);
});

test('toggle-results: closing then reopening restores the last dragged height, not the default', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-results' }); // open at 300
  s = nextLayoutState(s, { type: 'drag-results', dy: -100 }); // -> 400
  assert.equal(s.results.h, 400);
  s = nextLayoutState(s, { type: 'toggle-results' }); // close
  assert.equal(s.results.open, false);
  assert.equal(s.results.h, 400); // height preserved while closed
  s = nextLayoutState(s, { type: 'toggle-results' }); // reopen
  assert.equal(s.results.open, true);
  assert.equal(s.results.h, 400); // reopened at last height, not reset to the 300 default
});

test('toggle-results: closing clears a minimized flag and un-maximizes results (no orphaned collapsed state)', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-results' }); // open
  s = nextLayoutState(s, { type: 'minimize', pane: 'results' });
  assert.equal(s.results.min, true);
  s = nextLayoutState(s, { type: 'toggle-results' }); // close
  assert.equal(s.results.open, false);
  assert.equal(s.results.min, false);
});

// --- maximize 'results' / restore (required) ---

test('maximize/restore: maximize(results) sets state.maximized, restore clears it back to the original state', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-results' }); // open first, so restore lands back here exactly
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'results' });
  assert.equal(maxed.maximized, 'results');
  const restored = nextLayoutState(maxed, { type: 'restore' });
  assert.deepEqual(restored, s);
});

test('maximize: maximizing the results drawer also opens it and clears its minimized flag', () => {
  const s = defaultLayoutState(); // results closed
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'results' });
  assert.equal(maxed.maximized, 'results');
  assert.equal(maxed.results.open, true);
  assert.equal(maxed.results.min, false);
});

test('maximize(results) when already maximized -> restore (required contract line, verbatim)', () => {
  const s = defaultLayoutState();
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'results' });
  assert.equal(maxed.maximized, 'results');
  const toggledOff = nextLayoutState(maxed, { type: 'maximize', pane: 'results' });
  assert.equal(toggledOff.maximized, null);
});

// --- minimize results on maximized restores first (required) ---

test('minimize a maximized results drawer: restores first, then minimizes', () => {
  const s = defaultLayoutState();
  const maxed = nextLayoutState(s, { type: 'maximize', pane: 'results' });
  assert.equal(maxed.maximized, 'results');
  const next = nextLayoutState(maxed, { type: 'minimize', pane: 'results' });
  assert.equal(next.maximized, null); // restored first
  assert.equal(next.results.min, true); // then minimized
});

test('minimize(results): toggles back off (un-minimizes) on a second call — the strip-click restore path', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'minimize', pane: 'results' });
  assert.equal(s.results.min, true);
  s = nextLayoutState(s, { type: 'minimize', pane: 'results' });
  assert.equal(s.results.min, false);
});

// --- serialize/parse round-trip includes results (required) ---

test('serialize/parse round-trip: results survives unchanged alongside yaml/insp/outline', () => {
  let s = defaultLayoutState();
  s = nextLayoutState(s, { type: 'toggle-results' });
  s = nextLayoutState(s, { type: 'drag-results', dy: -50 });
  s = { ...s, outline: { collapsed: ['group:parameters'], filter: 'x' } };
  const json = serializeLayout(s);
  const parsed = parseLayout(json);
  assert.deepEqual(parsed, s);
  assert.deepEqual(parsed.results, { h: 350, open: true, min: false });
});

test('parseLayout: an out-of-range stored results height is clamped back into bounds on load', () => {
  const stale = JSON.stringify({ results: { h: 1, open: true, min: false } });
  const parsed = parseLayout(stale);
  assert.equal(parsed.results.h, PANE_BOUNDS.results.min);
});

test('parseLayout: an invalid maximized value of "results" is accepted (a valid pane id)', () => {
  const blob = JSON.stringify({ maximized: 'results' });
  assert.equal(parseLayout(blob).maximized, 'results');
});

// --- parseLayout backward-compat: phase-2 blob (no results field) parses to defaults, not null ---

test('parseLayout: a phase-2 blob without a results field parses results to defaults (backward compat)', () => {
  const phase2Blob = JSON.stringify({
    yaml: { w: 320, open: true, min: false },
    insp: { w: 280, min: false },
    maximized: null,
    tab: 'parameters', // a pre-Task-10 field, no longer part of the schema — simply ignored
  });
  const parsed = parseLayout(phase2Blob);
  assert.notEqual(parsed, null);
  assert.deepEqual(parsed.results, defaultLayoutState().results);
  // fields that WERE present in the phase-2 blob still round-trip correctly
  assert.equal(parsed.yaml.w, 320);
  assert.equal(parsed.yaml.open, true);
  assert.equal(parsed.insp.w, 280);
  // the dead `tab` field is dropped, not carried through; outline recovers to its own defaults
  assert.deepEqual(parsed.outline, defaultLayoutState().outline);
});

test('loadLayout: a phase-2 blob (no results key) in storage loads with results defaulted', () => {
  const storage = fakeStorage();
  const phase2Blob = JSON.stringify({
    yaml: { w: 300, open: false, min: false },
    insp: { w: 300, min: false },
    maximized: null,
    tab: 'selection', // a pre-Task-10 field, no longer part of the schema — simply ignored
  });
  storage.setItem(LAYOUT_KEY, phase2Blob);
  const loaded = loadLayout(storage);
  assert.deepEqual(loaded.results, defaultLayoutState().results);
});
