// DOM-free debounce/coalesce adapter between the textarea and the store. The app wires a
// textarea's 'input' event to onUserInput(text), blur/Run/Save to flush(), and reads
// textForView() (e.g. after every store.subscribe() notification) to decide what the textarea
// should show. See constraints.md: "Every model mutation flows through store.apply(...)" — this
// module never mutates the model itself, it only decides *when* to call store.setText().
//
// Two origins produce a store text change: this module's own debounced/flushed commit, and
// everything else (store.applyOp from canvas/inspector gestures, or another caller of setText
// entirely). textForView() must tell them apart: model-originated changes always replace the
// view; the module's own commits must not be reported back as "the model changed under you" the
// moment after it happens. Origins are distinguished by remembering the text this module itself
// last pushed into the store (lastSyncedText) and comparing it to the store's current text.

export function createSync(store, {
  debounceMs = 400,
  now = Date.now, // accepted for interface parity / future use; current debounce logic only
                   // needs relative scheduling via setTimer, not wall-clock time.
  setTimer = setTimeout,
  clearTimer = clearTimeout,
} = {}) {
  let pendingText = null;   // non-null while a debounce is pending: the latest text from onUserInput
  let timerHandle = null;   // handle returned by setTimer, or null when no debounce is scheduled
  let lastSyncedText = store.get().text; // text this module believes the view currently shows

  function commitPending() {
    const text = pendingText;
    pendingText = null;
    if (timerHandle !== null) {
      clearTimer(timerHandle);
      timerHandle = null;
    }
    // Set lastSyncedText to the text we're about to commit BEFORE calling store.setText, not
    // after. store.setText() calls the store's subscribers synchronously (notify() runs inside
    // it, before the call returns), so a subscriber that reads textForView() from inside its own
    // listener would otherwise see the store's new text compared against the STALE
    // lastSyncedText (still the pre-commit value) and misreport this as a model-originated
    // change — a false dirtyFromModel: true for our own commit, which in the real textarea
    // binding means an unwanted .value reassignment (cursor jump) on every debounce fire.
    // store.setText(text), on both the good- and bad-parse paths, always sets store.get().text to
    // exactly `text`, so recording it here first is equivalent to reading it back after, but
    // without the synchronous-notification race.
    lastSyncedText = text;
    store.setText(text);
  }

  return {
    onUserInput(text) {
      pendingText = text;
      if (timerHandle !== null) clearTimer(timerHandle);
      timerHandle = setTimer(() => {
        timerHandle = null;
        commitPending();
      }, debounceMs);
    },

    flush() {
      if (pendingText === null) return; // nothing pending: a safe no-op
      commitPending();
    },

    textForView() {
      // Mid-typing: the pending text is always what the view shows, regardless of what the store
      // reports underneath it — never clobbered until the debounce (or a flush) resolves it.
      if (pendingText !== null) {
        return { text: pendingText, dirtyFromModel: false };
      }
      const text = store.get().text;
      const dirtyFromModel = text !== lastSyncedText;
      lastSyncedText = text; // the view is presumed to adopt this text now that it's been read
      return { text, dirtyFromModel };
    },

    dispose() {
      if (timerHandle !== null) {
        clearTimer(timerHandle);
        timerHandle = null;
      }
      pendingText = null; // any in-flight edit is discarded, not committed
    },
  };
}
