import test from 'node:test';
import assert from 'node:assert/strict';
import { scopedStore, scopedStoreFor } from '../js/ui/scoped-store.js';

// Minimal store-shaped fake: enough surface for the wrapper's contract.
function fakeStore(model) {
  const state = { model, selection: { kind: null, id: null } };
  const calls = [];
  return {
    calls,
    get: () => state,
    applyOp(fn) { state.model = fn(state.model); calls.push('applyOp'); },
    select(sel) { state.selection = sel; calls.push('select'); },
    undo() { calls.push('undo'); },
    redo() { calls.push('redo'); },
    markSaved() { calls.push('markSaved'); },
    subscribe() { calls.push('subscribe'); },
  };
}

const doc = () => ({ name: 'top', models: { inner: { name: 'inner', models: { deep: { name: 'deep' } } } } });

test('get() returns the named sub-model as .model', () => {
  const s = scopedStore(fakeStore(doc()), 'inner');
  assert.equal(s.get().model.name, 'inner');
});

test('get() returns null when the sub-model is absent', () => {
  const s = scopedStore(fakeStore({ name: 'top' }), 'missing');
  assert.equal(s.get().model, null);
});

test('applyOp splices the edited sub-model back into a fresh top-level model', () => {
  const base = fakeStore(doc());
  const s = scopedStore(base, 'inner');
  s.applyOp((m) => ({ ...m, name: 'edited' }));
  assert.equal(base.get().model.models.inner.name, 'edited');
  assert.equal(base.get().model.name, 'top');           // outer untouched
});

test('applyOp throws when the sub-model is gone', () => {
  const s = scopedStore(fakeStore({ name: 'top' }), 'missing');
  assert.throws(() => s.applyOp((m) => m), /missing/);
});

test('select prepends this wrapper name onto modelPath', () => {
  const base = fakeStore(doc());
  scopedStore(base, 'inner').select({ kind: 'state', id: 'Well' });
  assert.deepEqual(base.get().selection.modelPath, ['inner']);
});

test('chained wrappers reach nested models and stamp both names in order', () => {
  const base = fakeStore(doc());
  const s = scopedStoreFor(base, ['inner', 'deep']);
  assert.equal(s.get().model.name, 'deep');
  s.select({ kind: 'state', id: 'X' });
  assert.deepEqual(base.get().selection.modelPath, ['inner', 'deep']);
});

test('scopedStoreFor with an empty path returns the base store itself', () => {
  const base = fakeStore(doc());
  assert.equal(scopedStoreFor(base, []), base);
});

test('undo/redo/markSaved/subscribe pass through', () => {
  const base = fakeStore(doc());
  const s = scopedStore(base, 'inner');
  s.undo(); s.redo(); s.markSaved(); s.subscribe(() => {});
  assert.deepEqual(base.calls, ['undo', 'redo', 'markSaved', 'subscribe']);
});
