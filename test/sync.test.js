// Tests for js/ui/sync.js — the DOM-free debounce/coalesce layer between the textarea and the
// store. Uses a real createStore (already covered by store.test.js) so these tests exercise real
// setText()/applyOp() semantics rather than a hand-rolled mock, plus a fake timer harness that
// captures the callback passed to the injected setTimer and fires it manually — no real waiting.

import test from 'node:test';
import assert from 'node:assert/strict';
import * as ops from '../js/ui/ops.js';
import { createStore } from '../js/ui/store.js';
import { createSync } from '../js/ui/sync.js';

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

// A fake timer pair: setTimer records {id, cb} and returns the id; clearTimer removes it by id
// (mirroring real clearTimeout — a cleared timer never fires again, even if something still holds
// a reference to it); fireAll() invokes every still-pending callback, in the order scheduled, and
// removes them (one-shot, like real timers).
function createFakeTimers() {
  let nextId = 0;
  const pending = new Map(); // id -> cb
  return {
    setTimer(cb) {
      const id = ++nextId;
      pending.set(id, cb);
      return id;
    },
    clearTimer(id) {
      pending.delete(id);
    },
    fireAll() {
      const callbacks = [...pending.values()];
      pending.clear();
      for (const cb of callbacks) cb();
    },
    pendingCount() {
      return pending.size;
    },
  };
}

test('onUserInput debounces store.setText: no change before the timer fires, commits after', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { debounceMs: 400, setTimer: timers.setTimer, clearTimer: timers.clearTimer });

  const edited = GOOD.replace('name: m', 'name: edited');
  sync.onUserInput(edited);
  assert.equal(store.get().text, GOOD, 'no store change before the debounce fires');
  assert.equal(timers.pendingCount(), 1);

  timers.fireAll();
  assert.equal(store.get().text, edited);
  assert.equal(store.get().model.name, 'edited');
});

test('rapid onUserInput calls coalesce into a single setText call carrying the latest text', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { setTimer: timers.setTimer, clearTimer: timers.clearTimer });

  let setTextCalls = 0;
  const realSetText = store.setText.bind(store);
  store.setText = (t) => { setTextCalls += 1; realSetText(t); };

  sync.onUserInput(GOOD.replace('name: m', 'name: a'));
  sync.onUserInput(GOOD.replace('name: m', 'name: ab'));
  sync.onUserInput(GOOD.replace('name: m', 'name: abc'));
  assert.equal(timers.pendingCount(), 1, 'earlier timers were cancelled, only the latest remains scheduled');

  timers.fireAll();
  assert.equal(setTextCalls, 1);
  assert.equal(store.get().model.name, 'abc');
});

test('flush commits pending input immediately and cancels the timer', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { setTimer: timers.setTimer, clearTimer: timers.clearTimer });

  const edited = GOOD.replace('name: m', 'name: edited');
  sync.onUserInput(edited);
  assert.equal(timers.pendingCount(), 1);

  sync.flush();
  assert.equal(store.get().text, edited);
  assert.equal(timers.pendingCount(), 0, 'the debounce timer was cancelled by flush');

  // A stray late fire (simulating a timer that already fired concurrently with flush in a real
  // runtime) must not double-commit or throw.
  assert.doesNotThrow(() => timers.fireAll());
});

test('flush() with nothing pending is a safe no-op', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { setTimer: timers.setTimer, clearTimer: timers.clearTimer });
  const before = store.get();

  assert.doesNotThrow(() => sync.flush());
  assert.deepEqual(store.get(), before);
});

test('a store change from applyOp while NOT typing updates textForView (model wins)', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { setTimer: timers.setTimer, clearTimer: timers.clearTimer });

  assert.deepEqual(sync.textForView(), { text: GOOD, dirtyFromModel: false });

  store.applyOp((m) => ops.addState(m));
  const view = sync.textForView();
  assert.equal(view.text, store.get().text);
  assert.equal(view.dirtyFromModel, true, 'a model-originated change must be flagged so the view overwrites the textarea');

  // Once the view has "seen" the update, it is no longer flagged dirty on a repeat read.
  const view2 = sync.textForView();
  assert.equal(view2.dirtyFromModel, false);
});

test('a store change arriving while a debounce is pending does not clobber the pending user text', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { setTimer: timers.setTimer, clearTimer: timers.clearTimer });

  const typed = GOOD.replace('name: m', 'name: typed');
  sync.onUserInput(typed);

  // A model-originated change arrives mid-debounce (e.g. a canvas gesture applied concurrently).
  store.applyOp((m) => ops.addState(m));

  const view = sync.textForView();
  assert.equal(view.text, typed, "the user's in-flight text must win, never be clobbered");
  assert.equal(view.dirtyFromModel, false);

  // The debounce fires: the user's edit is what gets committed, resolving on top of the
  // concurrent model change (the applyOp's state1 addition is superseded).
  timers.fireAll();
  assert.equal(store.get().text, typed);
  assert.equal(store.get().model.name, 'typed');
  assert.equal(store.get().model.states.some((s) => s.name === 'state1'), false);
});

test('dispose cancels any pending timer and stops further commits', () => {
  const store = createStore(GOOD);
  const timers = createFakeTimers();
  const sync = createSync(store, { setTimer: timers.setTimer, clearTimer: timers.clearTimer });

  const typed = GOOD.replace('name: m', 'name: typed');
  sync.onUserInput(typed);
  assert.equal(timers.pendingCount(), 1);

  sync.dispose();
  assert.equal(timers.pendingCount(), 0);

  assert.doesNotThrow(() => timers.fireAll());
  assert.equal(store.get().text, GOOD, 'the pending edit was discarded, not committed, by dispose');
});

test('createSync works with real timers and default injected functions when none are passed', async () => {
  const store = createStore(GOOD);
  const sync = createSync(store, { debounceMs: 5 });
  const edited = GOOD.replace('name: m', 'name: edited');

  sync.onUserInput(edited);
  assert.equal(store.get().text, GOOD);

  await new Promise((resolve) => setTimeout(resolve, 25));
  assert.equal(store.get().text, edited);

  sync.dispose();
});
