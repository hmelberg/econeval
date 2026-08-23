// Tests for js/ui/files.js — the model registry (versioned local storage) and autosave. Pure
// over an injected storage object, per constraints.md ("pure logic ... lives in DOM-free modules
// with tests") — no real localStorage touched; a Map-backed shim stands in for it, same pattern
// test/panels.test.js already uses for its own injectable storage.

import test from 'node:test';
import assert from 'node:assert/strict';
import { createRegistry } from '../js/ui/files.js';

function fakeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    _map: m,
  };
}

// A deterministic, strictly-increasing clock for tests that care about ordering/eviction, so
// nothing depends on real wall-clock resolution or Date.now() ties within a fast test run.
function fakeClock(start = 0) {
  let t = start;
  return () => t++;
}

// --- list() on an empty / fresh store ---

test('list() on a fresh store is an empty array', () => {
  const reg = createRegistry(fakeStorage());
  assert.deepEqual(reg.list(), []);
});

// --- saveVersion: new entry (null id) ---

test('saveVersion(null, ...) creates a new entry and returns a string id', () => {
  const reg = createRegistry(fakeStorage());
  const id = reg.saveVersion(null, 'HIV model', 'econeval: 1\ntype: markov\n');
  assert.equal(typeof id, 'string');
  assert.ok(id.length > 0);

  const list = reg.list();
  assert.equal(list.length, 1);
  assert.equal(list[0].id, id);
  assert.equal(list[0].name, 'HIV model');
  assert.equal(list[0].versionCount, 1);
});

test('two saveVersion(null, ...) calls produce two distinct entries with distinct ids', () => {
  const reg = createRegistry(fakeStorage());
  const id1 = reg.saveVersion(null, 'Model one', 'text1');
  const id2 = reg.saveVersion(null, 'Model two', 'text2');
  assert.notEqual(id1, id2);
  assert.equal(reg.list().length, 2);
});

// --- saveVersion: existing id prepends a version, updates name ---

test('saveVersion(id, ...) on an existing entry prepends a new version (versionCount grows)', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  const id = reg.saveVersion(null, 'Model', 'v0');
  reg.saveVersion(id, 'Model', 'v1');
  reg.saveVersion(id, 'Model', 'v2');

  const list = reg.list();
  assert.equal(list.length, 1);
  assert.equal(list[0].versionCount, 3);
  // default load() = latest = most recently saved
  assert.equal(reg.load(id).text, 'v2');
});

test('saveVersion(id, name, ...) updates the entry display name on every call', () => {
  const reg = createRegistry(fakeStorage());
  const id = reg.saveVersion(null, 'Original name', 'v0');
  reg.saveVersion(id, 'Renamed', 'v1');
  assert.equal(reg.list()[0].name, 'Renamed');
  assert.equal(reg.load(id).name, 'Renamed');
});

test('saveVersion accepts an optional label and stores it on the version', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  const id = reg.saveVersion(null, 'Model', 'v0', 'first cut');
  // load() doesn't surface label per the brief's {text, name} shape, but saving with a label
  // must not throw and must not corrupt the text/name round-trip.
  assert.equal(reg.load(id).text, 'v0');
  assert.equal(reg.load(id).name, 'Model');
});

// --- version cap: 20 per model, oldest evicted, prepend order (newest first) ---

test('saveVersion caps at 20 versions per model, evicting the oldest, prepending the newest', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  let id = null;
  const tsAtFirstSave = 0; // fakeClock starts at 0; first saveVersion call consumes ts=0
  for (let i = 0; i < 21; i += 1) {
    id = reg.saveVersion(id, 'Model', `v${i}`);
  }
  const list = reg.list();
  assert.equal(list[0].versionCount, 20);
  // Newest save (v20) is the default load
  assert.equal(reg.load(id).text, 'v20');
  // The very first version (ts=0, text 'v0') was evicted by the cap
  assert.throws(() => reg.load(id, tsAtFirstSave), /no such version/);
});

// --- load: default latest vs. explicit versionTs ---

test('load(id) with no versionTs returns the latest (most recently saved) version', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  const id = reg.saveVersion(null, 'Model', 'v0');
  reg.saveVersion(id, 'Model', 'v1');
  assert.equal(reg.load(id).text, 'v1');
});

test('load(id, versionTs) returns the specific version matching that ts', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  const id = reg.saveVersion(null, 'Model', 'v0'); // ts=0
  reg.saveVersion(id, 'Model', 'v1'); // ts=1
  reg.saveVersion(id, 'Model', 'v2'); // ts=2
  assert.equal(reg.load(id, 0).text, 'v0');
  assert.equal(reg.load(id, 1).text, 'v1');
});

test('load() on an unknown id throws', () => {
  const reg = createRegistry(fakeStorage());
  assert.throws(() => reg.load('nope'), /no such model/);
});

test('load(id, versionTs) with an unmatched ts throws', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  const id = reg.saveVersion(null, 'Model', 'v0');
  assert.throws(() => reg.load(id, 999999), /no such version/);
});

// --- listVersions: not in the brief's literal method list, but required to build the "Open"
// dialog's version picker (list registry + versions + delete) — reg.list() alone only exposes a
// version COUNT, not each version's own ts/label, and reg.load(id, ts) needs a ts to be handed to
// it from somewhere. Small addition on top of the contract, exercised the same as everything else.

test('listVersions(id) returns each version (newest first) with ts + label, no text payload', () => {
  const reg = createRegistry(fakeStorage(), { now: fakeClock() });
  const id = reg.saveVersion(null, 'Model', 'v0', 'first');
  reg.saveVersion(id, 'Model', 'v1'); // no label
  const versions = reg.listVersions(id);
  assert.deepEqual(versions, [
    { ts: 1, label: null },
    { ts: 0, label: 'first' },
  ]);
});

test('listVersions() on an unknown id returns an empty array, not a throw', () => {
  const reg = createRegistry(fakeStorage());
  assert.deepEqual(reg.listVersions('nope'), []);
});

// --- remove ---

test('remove(id) deletes the entry; a later list()/load() reflects that', () => {
  const reg = createRegistry(fakeStorage());
  const id = reg.saveVersion(null, 'Model', 'v0');
  reg.remove(id);
  assert.deepEqual(reg.list(), []);
  assert.throws(() => reg.load(id), /no such model/);
});

test('remove() on an unknown id is a safe no-op (does not throw)', () => {
  const reg = createRegistry(fakeStorage());
  assert.doesNotThrow(() => reg.remove('never-existed'));
});

test('remove(id) only affects that entry, leaving other models intact', () => {
  const reg = createRegistry(fakeStorage());
  const id1 = reg.saveVersion(null, 'Model one', 'v0');
  const id2 = reg.saveVersion(null, 'Model two', 'v0');
  reg.remove(id1);
  const list = reg.list();
  assert.equal(list.length, 1);
  assert.equal(list[0].id, id2);
});

// --- autosave / readAutosave ---

test('readAutosave() with nothing saved yet returns null', () => {
  const reg = createRegistry(fakeStorage());
  assert.equal(reg.readAutosave(), null);
});

test('autosave(text) / readAutosave() round-trips the exact text', () => {
  const reg = createRegistry(fakeStorage());
  reg.autosave('econeval: 1\ntype: markov\n');
  assert.equal(reg.readAutosave(), 'econeval: 1\ntype: markov\n');
});

test('autosave overwrites the previous autosave (only the latest is kept)', () => {
  const reg = createRegistry(fakeStorage());
  reg.autosave('first');
  reg.autosave('second');
  assert.equal(reg.readAutosave(), 'second');
});

test('autosave/readAutosave use a separate storage key from the model registry', () => {
  const storage = fakeStorage();
  const reg = createRegistry(storage);
  reg.saveVersion(null, 'Model', 'v0');
  reg.autosave('draft text');
  assert.equal(reg.readAutosave(), 'draft text');
  assert.equal(reg.list().length, 1); // the registry entry above is untouched
  assert.ok(storage._map.has('econeval.autosave.v1'));
  assert.ok(storage._map.has('econeval.models.v1'));
});

// --- corrupt JSON in storage: empty registry, no crash ---

test('corrupt JSON under the registry key -> list() returns [] without throwing', () => {
  const storage = fakeStorage();
  storage.setItem('econeval.models.v1', '{not valid json');
  const reg = createRegistry(storage);
  assert.doesNotThrow(() => reg.list());
  assert.deepEqual(reg.list(), []);
});

test('corrupt JSON under the registry key -> saveVersion still works afterward (self-heals)', () => {
  const storage = fakeStorage();
  storage.setItem('econeval.models.v1', 'totally not json {{{');
  const reg = createRegistry(storage);
  const id = reg.saveVersion(null, 'Model', 'v0');
  assert.equal(reg.list().length, 1);
  assert.equal(reg.load(id).text, 'v0');
});

test('a well-formed JSON value that is not a plausible registry object (array) -> empty registry', () => {
  const storage = fakeStorage();
  storage.setItem('econeval.models.v1', '[1,2,3]');
  const reg = createRegistry(storage);
  assert.deepEqual(reg.list(), []);
});

test('corrupt JSON under the registry key does not affect autosave (separate key)', () => {
  const storage = fakeStorage();
  storage.setItem('econeval.models.v1', '{{{broken');
  const reg = createRegistry(storage);
  reg.autosave('draft');
  assert.equal(reg.readAutosave(), 'draft');
});

// --- Final-review fix: saveVersion propagates a storage.setItem failure (e.g. quota exceeded)
// rather than swallowing it. Unlike autosave() (deliberately best-effort — see its own comment
// above), saveVersion is the user's explicit, durable "Save" action; app.js's caller wraps this in
// its own try/catch to alert() the user and skip markSaved(), so the failure must actually reach
// the caller instead of looking like a silent success. writeIndex() has no try/catch of its own —
// this test is really pinning down that "don't add one", not exercising new files.js code.

test('saveVersion propagates a storage.setItem failure instead of swallowing it', () => {
  const storage = fakeStorage();
  storage.setItem = () => { throw new Error('QuotaExceededError'); };
  const reg = createRegistry(storage);
  assert.throws(() => reg.saveVersion(null, 'Model', 'v0'), /QuotaExceededError/);
});
