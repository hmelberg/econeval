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

// --- Review fixes: critical + important issues found in the task-5 review. ---

test('a bad setText clears redoStack too, so a later redo/undo cannot crash on corrupted history', () => {
  // Reproduces the reported crash sequence exactly: applyOp -> undo -> setText(bad) -> redo ->
  // undo. Before the round-1 fix, setText's failure branch left a stale redoStack entry in place;
  // redo() would then push the BAD text onto undoStack, and the final undo() would try to
  // re-parse it and throw an uncaught ModelError. (Without the intervening redo() call this
  // sequence never reproduces the crash at all — undo() on an empty stack is always a safe no-op
  // — so the redo() call is included here even though it isn't spelled out step-by-step in the
  // ticket.)
  //
  // NOTE (round 2): with the parseError-guard added to undo()/redo() (see the "round-2 review
  // fix" tests below), this redo() call no longer merely no-ops on an empty redoStack — since
  // parseError is set at that point, it takes the guard branch and actively reverts the bad
  // buffer to the last-good model's own text. That's the correct, improved behavior (redo()
  // symmetric with undo() — see the controller's ruling), so this test's final assertions were
  // updated accordingly: the point of THIS test is still "the whole sequence never throws", not
  // "the bad text survives untouched".
  const store = createStore(GOOD);
  store.applyOp((m) => ops.addState(m));
  store.undo(); // back to GOOD; redoStack now holds the post-addState text
  assert.equal(store.get().canRedo, true);

  store.setText('not: valid: yaml: [');
  assert.equal(store.get().canRedo, false); // the direct regression check: ANY setText clears redo

  assert.doesNotThrow(() => store.redo()); // parseError is set -> reverts the bad buffer; no throw
  let s = store.get();
  assert.equal(s.parseError, null);
  assert.equal(s.text, serializeModel(s.model));

  assert.doesNotThrow(() => store.undo()); // no-op: nothing left on undoStack — this used to throw
  s = store.get();
  // The last-good MODEL is unaffected by the whole redo()/undo() detour: still the pre-bad-edit
  // model (no 'state1' — that was undone before the bad edit was ever typed).
  assert.equal(s.model.states.some((st) => st.name === 'state1'), false);
});

test('a throwing subscriber does not suppress other subscribers or break the mutation', () => {
  const store = createStore(GOOD);
  const originalConsoleError = console.error;
  const loggedErrors = [];
  console.error = (...args) => { loggedErrors.push(args); };
  try {
    let secondCalled = false;
    store.subscribe(() => { throw new Error('boom'); });
    store.subscribe(() => { secondCalled = true; });

    assert.doesNotThrow(() => store.applyOp((m) => ops.addState(m)));
    assert.equal(secondCalled, true);
    assert.equal(loggedErrors.length, 1);
    assert.equal(loggedErrors[0][0], 'store listener failed');
  } finally {
    console.error = originalConsoleError;
  }
});

test('applyOp commits the model reparsed from the serialized text, catching op/serializer drift', () => {
  // A hand-crafted "op" that produces a structurally-valid Model whose serialized text is
  // rejected on reparse: a payoff value with unbalanced parentheses passes straight through
  // serializeModel (it does no expression/paren validation) but trips normStates's
  // unbalanced-parens guard when the store reparses the text it just produced.
  const store = createStore(GOOD);
  const before = store.get();
  const brokenFn = (m) => {
    const m2 = structuredClone(m);
    m2.states.find((s) => s.name === 'well').payoffs.cost = 'lookup(x, y';
    return m2;
  };
  assert.throws(() => store.applyOp(brokenFn), /unbalanced parentheses/);
  assert.deepEqual(store.get(), before); // untouched, same as a throwing fn
});

test('applyOp: the committed model equals parseModel(serializeModel(fn(model))) for a normal op', () => {
  const store = createStore(GOOD);
  store.applyOp((m) => ops.addState(m));
  const s = store.get();
  assert.deepEqual(s.model, parseModel(serializeModel(s.model)));
});

// --- Round-2 review fix: the symmetric crash on the undo() side. ---
//
// Round 1 fixed setText(bad) failing to clear redoStack. But undo() itself had the same class of
// bug: applyOp -> setText(bad) -> undo() -> redo() still threw, because undo()'s
// `redoStack.push(text)` ran unconditionally and captured the BAD buffer, which redo() then
// reparsed unguarded. The controller's ruling: the real invariant is "the stacks only ever hold
// good text" — enforced by guarding at the top of BOTH undo() and redo(): while parseError is
// set, revert the buffer to the current (last-good) model's own text and return, touching neither
// stack. This also gives sensible UX: the first undo() backs out invalid typing; the next one
// pops the real snapshot.

test('undo() when parseError is set reverts the bad buffer instead of consuming a real snapshot', () => {
  const store = createStore(GOOD);
  const initialText = store.get().text;

  assert.doesNotThrow(() => store.applyOp((m) => ops.addState(m)));
  assert.ok(store.get().model.states.some((s) => s.name === 'state1'));

  assert.doesNotThrow(() => store.setText(BAD));
  assert.ok(store.get().parseError);

  // First undo(): backs out the bad typing only — reverts the buffer to the last-good model's
  // own text, without touching either stack.
  assert.doesNotThrow(() => store.undo());
  let s = store.get();
  assert.equal(s.text, serializeModel(s.model));
  assert.ok(s.model.states.some((st) => st.name === 'state1')); // still the post-addState model
  assert.equal(s.parseError, null);
  assert.equal(s.canUndo, true); // the real snapshot (initialText) is still there, untouched
  assert.equal(s.canRedo, false); // nothing was pushed to redo by this buffer-revert

  // Second undo(): now pops the real snapshot, landing back on the initial text.
  assert.doesNotThrow(() => store.undo());
  s = store.get();
  assert.equal(s.text, initialText);
  assert.equal(s.model.states.some((st) => st.name === 'state1'), false);

  // redo(): replays the addState change.
  assert.doesNotThrow(() => store.redo());
  s = store.get();
  assert.ok(s.model.states.some((st) => st.name === 'state1'));
  assert.equal(s.parseError, null);
});

// --- Task 10 controller ruling: scoped selection (selection.modelPath). ---
//
// A canvas gesture made while the user has drilled into a sub-model (via scopedStore) needs its
// selection to survive edits made elsewhere in the document, and to be invalidated when the
// sub-model it points into stops existing — not when some unrelated top-level state of the same
// name does/doesn't exist. isSelectionValid now resolves the SCOPED model first (by walking
// model.models along selection.modelPath) and validates kind/id against THAT, not the top-level
// model. Default modelPath (omitted, i.e. undefined) must keep resolving to the top-level model,
// so every pre-existing test above (none of which ever set modelPath) is unaffected.

const NESTED_TEXT = `
econeval: 1
type: markov
name: top
settings: {cycles: 3, start: a}
states:
  a: {cost: 0, utility: 1}
  b: {cost: 0, utility: 1}
transitions:
  a: {a: rest, b: 0.1}
  b: {b: 1}
models:
  chronic:
    type: markov
    settings: {cycles: 3, start: well}
    states:
      well: {cost: 0, utility: 1}
      sick: {cost: 0, utility: 1}
    transitions:
      well: {well: rest, sick: 0.1}
      sick: {sick: 1}
`;

test('a scoped selection survives a top-level applyOp that does not touch its sub-model', () => {
  const store = createStore(NESTED_TEXT);
  store.select({ kind: 'state', id: 'well', modelPath: ['chronic'] });
  assert.deepEqual(store.get().selection, { kind: 'state', id: 'well', modelPath: ['chronic'] });

  // Top-level op: adds a state to 'top', never touches models.chronic at all. Under the OLD
  // (unscoped) isSelectionValid, this would check the TOP-LEVEL model's states for 'well' (not
  // found there) and wrongly clear the selection.
  store.applyOp((m) => ops.addState(m));

  assert.deepEqual(store.get().selection, { kind: 'state', id: 'well', modelPath: ['chronic'] });
});

test('a scoped selection is cleared when the sub-model it points into is itself deleted from models', () => {
  const store = createStore(NESTED_TEXT);
  store.select({ kind: 'state', id: 'well', modelPath: ['chronic'] });
  assert.deepEqual(store.get().selection, { kind: 'state', id: 'well', modelPath: ['chronic'] });

  // No ops.js op deletes a whole sub-model; a raw applyOp fn (same pattern as the earlier
  // "commits the reparsed model" test's brokenFn) does it directly. Deleting the WHOLE 'chronic'
  // key round-trips cleanly (model.js only emits `models:` when it's non-empty; normModels treats
  // a missing `models:` as {}), so this is a legitimate, reparse-clean op.
  store.applyOp((m) => {
    const m2 = structuredClone(m);
    delete m2.models.chronic;
    return m2;
  });

  assert.deepEqual(store.get().selection, { kind: null, id: null });
});

test('setText(bad) then redo() never throws and never leaks bad text onto a stack, with or without a prior redo entry', () => {
  // Case A: nothing to redo at all (redoStack already empty when the bad edit happens).
  const storeA = createStore(GOOD);
  storeA.setText(BAD);
  assert.equal(storeA.get().canRedo, false);
  assert.doesNotThrow(() => storeA.redo());
  const a = storeA.get();
  assert.equal(a.parseError, null); // buffer reverted, not left dangling on a bad parse
  assert.equal(a.canRedo, false);
  assert.equal(a.canUndo, false);

  // Case B: a real redo entry exists (from undoing a good change) at the moment of the bad edit.
  const storeB = createStore(GOOD);
  storeB.applyOp((m) => ops.addState(m));
  storeB.undo();
  assert.equal(storeB.get().canRedo, true); // the good entry is there, before the bad edit

  storeB.setText(BAD);
  assert.equal(storeB.get().canRedo, false); // cleared by setText itself (round-1 fix)

  assert.doesNotThrow(() => storeB.redo()); // parseError-guard branch: reverts buffer, no stack touched
  const b = storeB.get();
  assert.equal(b.parseError, null);
  assert.equal(b.canRedo, false);
  assert.equal(b.canUndo, false); // the good entry was never resurrected onto undoStack either
  assert.equal(b.model.states.some((st) => st.name === 'state1'), false); // still pre-addState model

  // Stacks are provably clean afterward: a further undo() is a safe no-op, not a crash.
  assert.doesNotThrow(() => storeB.undo());
});
