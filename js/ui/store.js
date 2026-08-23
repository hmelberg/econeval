// DOM-free document store: canonical text + last-good parsed model + undo/redo + selection.
// Every model mutation the app makes flows through here (setText from the YAML pane, applyOp
// from canvas/inspector gestures) so the two panes and undo/redo stay in sync off one source of
// truth. See constraints.md: "Every model mutation flows through store.apply(...)".

import { parseModel, serializeModel } from '../core/model.js';
import { nodeAt } from './ops.js';

const UNDO_CAP = 100;

// Task 10 (scoped-selection fix, controller ruling): selection carries an optional modelPath
// (array of sub-model names, chained via model.models) — [] / undefined means the top-level
// model itself, matching every pre-Task-10 selection (none of which ever set modelPath). Walks
// model.models along modelPath to find the model the selection's kind/id actually refers to; a
// missing sub-model anywhere along the way makes the selection invalid, same as a missing
// state/node/param does today.
function resolveScopedModel(model, modelPath) {
  let m = model;
  for (const name of modelPath ?? []) {
    if (!m || !m.models || !(name in m.models)) return undefined;
    m = m.models[name];
  }
  return m;
}

function isSelectionValid(model, selection) {
  if (!selection || selection.kind == null) return true;
  const scoped = resolveScopedModel(model, selection.modelPath);
  if (!scoped) return false;
  const { kind, id } = selection;
  if (kind === 'state') {
    return scoped.type === 'markov' && scoped.states.some((s) => s.name === id);
  }
  if (kind === 'edge') {
    const row = scoped.transitions[id.from];
    if (!row) return false;
    return row.type === 'multinomial' ? id.to in row.counts : id.to in row.to;
  }
  if (kind === 'node') {
    if (scoped.type !== 'tree') return false;
    try { nodeAt(scoped, id); return true; } catch { return false; }
  }
  if (kind === 'param') {
    return scoped.params.has(id);
  }
  return true;
}

export function createStore(initialText) {
  let text = initialText;
  let model = null;
  let parseError = null;
  try {
    model = parseModel(initialText);
  } catch (e) {
    parseError = e;
  }

  let selection = { kind: null, id: null };
  let dirty = false;
  const undoStack = []; // snapshots of text BEFORE a change; oldest at index 0
  let redoStack = [];
  const listeners = new Set();

  function notify() {
    // Isolate subscribers from each other: a throwing listener (a bug in one panel's wiring)
    // must not suppress the remaining listeners, nor propagate out of a mutator that has already
    // committed successfully. "Errors surfaced, never swallowed" (constraints.md) is about
    // user/model errors — not about letting a broken subscriber crash sibling panes.
    for (const listener of listeners) {
      try {
        listener();
      } catch (err) {
        console.error('store listener failed', err);
      }
    }
  }

  function reconcileSelection(newModel) {
    if (!isSelectionValid(newModel, selection)) selection = { kind: null, id: null };
  }

  // Commits a new good (text, model) pair as one undo-able change: snapshots the current text,
  // caps the undo stack, clears redo, reconciles selection, marks dirty, notifies.
  function commit(newText, newModel) {
    undoStack.push(text);
    if (undoStack.length > UNDO_CAP) undoStack.shift();
    redoStack = [];
    text = newText;
    model = newModel;
    parseError = null;
    dirty = true;
    reconcileSelection(model);
    notify();
  }

  return {
    get() {
      return {
        text,
        model,
        parseError,
        selection,
        dirty,
        canUndo: undoStack.length > 0,
        canRedo: redoStack.length > 0,
      };
    },

    setText(newText) {
      let newModel;
      try {
        newModel = parseModel(newText);
      } catch (e) {
        text = newText;
        parseError = e;
        dirty = true;
        // A bad edit invalidates redo too — not just "no new undo snapshot". If it didn't, a
        // stale redoStack entry (from before this edit) could later be pushed onto undoStack by
        // redo(), and an undo() after that would try to re-parse an unvalidated snapshot and
        // throw uncaught. Any setText call, good or bad, is "a new change" for redo purposes.
        redoStack = [];
        notify();
        return;
      }
      commit(newText, newModel);
    },

    applyOp(fn, _opts) {
      const newModel = fn(model); // throws propagate untouched — nothing below has run yet
      const newText = serializeModel(newModel);
      // Enforce the snapshots-good invariant at the source: commit the model reparsed FROM the
      // serialized text, not fn's raw return. This guarantees every undo/redo snapshot really
      // does round-trip (catches op/serializer drift here, not as a later undo() surprise), and
      // matches the throwing-fn contract — if the reparse throws, nothing has been committed yet,
      // so the store is left untouched and the error propagates to the caller.
      const reparsed = parseModel(newText);
      commit(newText, reparsed);
    },

    select(sel) {
      selection = sel;
      notify();
    },

    undo() {
      // Invariant: undoStack/redoStack only ever hold GOOD text (each entry must re-parse
      // cleanly). The live `text` buffer can currently be bad (parseError set) without either
      // stack knowing about it — e.g. after setText(bad). In that case undo/redo must never push
      // that unvalidated buffer onto a stack; instead, "undo" the bad typing itself: snap the
      // buffer back to the current (last-good) model's own text, touching neither stack. The next
      // undo() call then proceeds as normal, popping the real snapshot.
      if (parseError) {
        text = serializeModel(model);
        parseError = null;
        dirty = true;
        notify();
        return;
      }
      if (undoStack.length === 0) return;
      const prevText = undoStack.pop();
      redoStack.push(text);
      text = prevText;
      model = parseModel(prevText); // always good — only good states are ever snapshotted
      parseError = null;
      dirty = true;
      reconcileSelection(model);
      notify();
    },

    redo() {
      // Same invariant/guard as undo() above — see its comment.
      if (parseError) {
        text = serializeModel(model);
        parseError = null;
        dirty = true;
        notify();
        return;
      }
      if (redoStack.length === 0) return;
      const nextText = redoStack.pop();
      undoStack.push(text);
      if (undoStack.length > UNDO_CAP) undoStack.shift();
      text = nextText;
      model = parseModel(nextText);
      parseError = null;
      dirty = true;
      reconcileSelection(model);
      notify();
    },

    markSaved() {
      dirty = false;
      notify();
    },

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
