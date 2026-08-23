import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel, ModelError } from '../js/core/model.js';
import * as ops from '../js/ui/ops.js';
import { createStore } from '../js/ui/store.js';

const GOOD = `
econeval: 1
type: markov
name: m
settings: {cycles: 3, start: well}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`;

const BAD = `
econeval: 1
type: markov
name: m
states: [this is not a mapping
`;

test('initial parse: get() reflects the parsed model, no parseError, not dirty', () => {
  const store = createStore(GOOD);
  const s = store.get();
  assert.equal(s.text, GOOD);
  assert.equal(s.parseError, null);
  assert.ok(s.model);
  assert.equal(s.model.name, 'm');
  assert.deepEqual(s.selection, { kind: null, id: null });
  assert.equal(s.dirty, false);
  assert.equal(s.canUndo, false);
  assert.equal(s.canRedo, false);
});

test('setText with good text swaps the model and marks dirty', () => {
  const store = createStore(GOOD);
  const next = GOOD.replace('name: m', 'name: renamed');
  store.setText(next);
  const s = store.get();
  assert.equal(s.text, next);
  assert.equal(s.parseError, null);
  assert.equal(s.model.name, 'renamed');
  assert.equal(s.dirty, true);
});

test('setText with bad text keeps the last-good model and sets parseError with a line', () => {
  const store = createStore(GOOD);
  store.setText(BAD);
  const s = store.get();
  assert.equal(s.text, BAD);
  assert.ok(s.parseError instanceof ModelError);
  assert.equal(typeof s.parseError.line, 'number');
  assert.equal(s.model.name, 'm'); // last GOOD parse kept
});

test('applyOp serializes the new model to text and marks dirty', () => {
  const store = createStore(GOOD);
  store.applyOp((m) => ops.addState(m), { label: 'Add state' });
  const s = store.get();
  assert.ok(s.model.states.some((st) => st.name === 'state1'));
  assert.equal(s.text, serializeModel(s.model));
  assert.equal(s.dirty, true);
  assert.equal(s.parseError, null);
});

test('applyOp with a throwing fn leaves the store untouched and rethrows', () => {
  const store = createStore(GOOD);
  const before = store.get();
  assert.throws(() => store.applyOp((m) => ops.deleteState(m, 'nope')), /not found/);
  assert.deepEqual(store.get(), before);
});

test('undo restores prior text and model; redo re-applies the change', () => {
  const store = createStore(GOOD);
  const initialText = store.get().text;
  store.applyOp((m) => ops.addState(m));
  const afterAddText = store.get().text;
  assert.equal(store.get().canUndo, true);

  store.undo();
  const afterUndo = store.get();
  assert.equal(afterUndo.text, initialText);
  assert.equal(afterUndo.model.states.length, 2);
  assert.equal(afterUndo.canRedo, true);

  store.redo();
  const afterRedo = store.get();
  assert.equal(afterRedo.text, afterAddText);
  assert.ok(afterRedo.model.states.some((s) => s.name === 'state1'));
});

test('redo is cleared once a new change is made after undo', () => {
  const store = createStore(GOOD);
  store.applyOp((m) => ops.addState(m));
  store.undo();
  assert.equal(store.get().canRedo, true);

  store.applyOp((m) => ops.addState(m, 'other'));
  assert.equal(store.get().canRedo, false);
  // redo is a no-op now (nothing to redo)
  const before = store.get();
  store.redo();
  assert.deepEqual(store.get(), before);
});

test('undo history caps at 100 entries: the 101st push drops the oldest', () => {
  const store = createStore(GOOD);
  // Op #1 gives a distinguishable model (adds 'state1'); undo-100-times from 101 ops should
  // land on the result of op #1, not the original initial text (which was dropped).
  for (let i = 1; i <= 101; i += 1) {
    store.applyOp((m) => ops.addState(m, `s${i}`));
  }
  for (let i = 0; i < 100; i += 1) store.undo();
  const s = store.get();
  assert.equal(s.canUndo, false); // history exhausted (oldest snapshot was dropped)
  // Landed on the state right after op #1: states well, dead, s1 only.
  assert.deepEqual(s.model.states.map((st) => st.name).sort(), ['dead', 's1', 'well']);
});

test('subscribe fires once per change and not after unsubscribe', () => {
  const store = createStore(GOOD);
  let calls = 0;
  const unsubscribe = store.subscribe(() => { calls += 1; });

  store.applyOp((m) => ops.addState(m));
  assert.equal(calls, 1);

  store.setText(GOOD.replace('name: m', 'name: m2'));
  assert.equal(calls, 2);

  store.undo();
  assert.equal(calls, 3);

  unsubscribe();
  store.redo();
  assert.equal(calls, 3); // no further calls after unsubscribe
});

test('subscribe listener receives no payload (pull state via get())', () => {
  const store = createStore(GOOD);
  let argCount = -1;
  store.subscribe((...args) => { argCount = args.length; });
  store.applyOp((m) => ops.addState(m));
  assert.equal(argCount, 0);
});

test('selection is cleared when the selected state is deleted via applyOp(ops.deleteState)', () => {
  const store = createStore(GOOD);
  store.select({ kind: 'state', id: 'dead' });
  assert.deepEqual(store.get().selection, { kind: 'state', id: 'dead' });

  store.applyOp((m) => ops.deleteState(m, 'dead'));
  assert.deepEqual(store.get().selection, { kind: null, id: null });
});

test('selection is preserved across a change when the referent still exists', () => {
  const store = createStore(GOOD);
  store.select({ kind: 'state', id: 'well' });
  store.applyOp((m) => ops.addState(m)); // unrelated change
  assert.deepEqual(store.get().selection, { kind: 'state', id: 'well' });
});

test('markSaved clears dirty', () => {
  const store = createStore(GOOD);
  store.applyOp((m) => ops.addState(m));
  assert.equal(store.get().dirty, true);
  store.markSaved();
  assert.equal(store.get().dirty, false);
});
