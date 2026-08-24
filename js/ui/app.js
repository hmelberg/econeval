// App wiring: boots the store/sync/panels/canvas/inspector, binds the YAML textarea, and drives
// the topbar (New/Open/Save version/Examples/Import/Export), keyboard shortcuts, autosave, and
// the unsaved-changes guard. This is the one module allowed to be "just wiring" — every piece of
// real logic it touches (parsing, ops, layout, registry) lives in the DOM-free modules it imports.
//
// #btn-yaml is deliberately NOT wired here — js/ui/panels.js's initPanels() already owns that
// button's click handler (toggle-yaml is a layout action, kept in one place). This module only
// calls initPanels() once at boot, same as every other createXxx(...) constructor below.

import { createStore } from './store.js';
import { createSync } from './sync.js';
import { initPanels } from './panels.js';
import { createCanvas } from './canvas/index.js';
import { createInspector } from './inspector.js';
import { createResults } from './results.js';
import { layoutFor } from './layouts.js';
import { createRegistry } from './files.js';

// ================================================================================================
// ---------- Blank templates (New menu; also the boot fallback when there's no autosave) ----------
// ================================================================================================
// Both are valid per parseModel()/check() as written — no placeholder gaps for the user to trip
// over. The markov template follows the same "two states + start + one rest row" shape as the
// phase-1/2 test fixtures' own minimal GOOD model (test/store.test.js, test/sync.test.js): one
// row uses 'rest' (state1), the other is a plain absorbing row (state2) — nothing here needs a
// param or a strategy block to parse cleanly.

const BLANK_MARKOV = `econeval: 1
type: markov
name: New model

settings:
  cycles: 10
  start: state1

states:
  state1: {cost: 0, utility: 1}
  state2: {cost: 0, utility: 0}

transitions:
  state1: {state1: rest, state2: 0.1}
  state2: {state2: 1}
`;

// A single root node with no children is a fully valid tree (normTree only requires exactly one
// root key; the body may be an empty mapping) — the user builds branches via canvas gestures from
// here (Add tool), per the e2e checklist's "New -> Tree -> build 2 branches via gestures".
const BLANK_TREE = `econeval: 1
type: tree
name: New decision tree

tree:
  Decision?: {}
`;

// ================================================================================================
// ---------- Boot ----------
// ================================================================================================

const reg = createRegistry(window.localStorage);
const initialText = reg.readAutosave() ?? BLANK_MARKOV;

const store = createStore(initialText);
store.markSaved(); // boot state is a freshly loaded document, not an unsaved edit

const sync = createSync(store);

// The registry id backing the CURRENTLY open document, or null when the document has never been
// saved as a version yet (a fresh boot, New, Examples, or Import) or was loaded from a source
// other than the registry. Save version writes through this id (or mints one, on first save).
let currentModelId = null;

// ================================================================================================
// ---------- DOM refs ----------
// ================================================================================================

const nameEl = document.getElementById('model-name');
const typeBadgeEl = document.getElementById('type-badge');
const yamlText = document.getElementById('yaml-text');
const yamlErrorEl = document.getElementById('yaml-error');
const dlgNew = document.getElementById('dlg-new');
const dlgOpen = document.getElementById('dlg-open');
const dlgExamples = document.getElementById('dlg-examples');
const fileImport = document.getElementById('file-import');

// ================================================================================================
// ---------- Small shared helpers ----------
// ================================================================================================

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return true;
  return !!el.isContentEditable;
}

function confirmDiscardIfDirty() {
  if (!store.get().dirty) return true;
  return window.confirm('Discard unsaved changes?');
}

function formatParseError(err) {
  if (!err) return '';
  let text = err.line != null ? `Line ${err.line}: ${err.message}` : err.message;
  if (err.hint) text += ` — ${err.hint}`;
  return text;
}

function formatTs(ts) {
  return new Date(ts).toLocaleString();
}

function slugify(name) {
  const s = (name ?? '').toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return s || 'model';
}

// Loads a brand-new document text into the store as a clean (non-dirty) state: cancels any
// pending debounced YAML edit first (an in-flight keystroke from the PREVIOUS document must never
// land on the new one a moment later), then commits and immediately marks it saved — New/Open/
// Examples/Import all represent "a document just got loaded", not "the user just edited it".
function loadDocument(text) {
  sync.dispose();
  store.setText(text);
  store.resetHistory(); // a freshly loaded document starts with clean undo/redo, never the previous document's stacks
  store.markSaved();
}

// A tiny "name" op, inline — mirrors js/ui/inspector.js's own setNameOp exactly (name lives
// top-level on the model, not under settings, so it isn't a setSetting keyPath). Kept as a
// separate local copy rather than exported/shared from inspector.js, per the brief's "keep
// consistent" instruction, since inspector.js's copy is intentionally private to that module.
function setNameOp(model, newName) {
  const trimmed = (newName ?? '').trim();
  if (!trimmed) throw new Error('name: must not be empty');
  const m = structuredClone(model);
  m.name = trimmed;
  return m;
}

// ================================================================================================
// ---------- Topbar: model name (editable), type badge, dirty dot ----------
// ================================================================================================

// index.html already declares contenteditable="plaintext-only" (item 2: plain-text-only paste,
// role=textbox, aria-label) — this JS assignment only needs to exist for browsers where the
// attribute failed to parse; setting it to the SAME value here (never 'true') keeps a pasted
// rich-text name from smuggling markup into the model name field.
nameEl.contentEditable = 'plaintext-only';

nameEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    nameEl.blur();
  } else if (e.key === 'Escape') {
    e.preventDefault();
    nameEl.textContent = store.get().model?.name ?? '';
    nameEl.blur();
  }
});

nameEl.addEventListener('blur', () => {
  const current = store.get().model?.name ?? '';
  const next = nameEl.textContent.trim();
  if (!next || next === current) {
    nameEl.textContent = current; // revert a blank/no-op edit rather than committing it
    return;
  }
  sync.flush();
  try {
    store.applyOp((m) => setNameOp(m, next));
  } catch (err) {
    window.alert(err.message);
    nameEl.textContent = current;
  }
});

function renderTopbar() {
  const { model, dirty } = store.get();
  if (document.activeElement !== nameEl) {
    nameEl.textContent = model?.name ?? '';
  }
  typeBadgeEl.textContent = model ? model.type.toUpperCase() : '';
  nameEl.classList.toggle('dirty', dirty);
}

// ================================================================================================
// ---------- YAML pane binding ----------
// ================================================================================================

yamlText.addEventListener('input', () => sync.onUserInput(yamlText.value));
yamlText.addEventListener('blur', () => sync.flush());

function renderYamlPane() {
  const { text, dirtyFromModel } = sync.textForView();
  // dirtyFromModel alone isn't a sufficient trigger for two edge cases sync.js has no way to see:
  // (1) the very first render — sync captures lastSyncedText = store.get().text at ITS OWN
  //     construction, so the first read here always trivially reports dirtyFromModel: false (the
  //     native <textarea> starts genuinely empty, and nothing else would ever paint it);
  //     (2) a discarded pending edit that happens to coincide with whatever sync.js believes is
  //     already synced (loadDocument() -> sync.dispose() drops an uncommitted edit, but if the
  //     freshly loaded text happens to equal lastSyncedText, textForView() reports
  //     dirtyFromModel: false even though the textarea's DOM value still shows the discarded
  //     typing). Comparing against the DOM value directly closes both gaps; it's a safe no-op
  //     during normal typing, since yamlText.value always already equals `text` in that case
  //     (text IS pendingText, which onUserInput captured FROM yamlText.value).
  if (dirtyFromModel || yamlText.value !== text) yamlText.value = text;

  const { parseError } = store.get();
  if (parseError) {
    yamlErrorEl.hidden = false;
    yamlErrorEl.textContent = formatParseError(parseError);
  } else {
    yamlErrorEl.hidden = true;
    yamlErrorEl.textContent = '';
  }
}

// ================================================================================================
// ---------- Panels, canvas, inspector ----------
// ================================================================================================

const panels = initPanels();

const canvas = createCanvas(document.getElementById('canvas'), store, {
  layoutFor,
  flush: sync.flush,
});

const inspector = createInspector(document.getElementById('inspector-body'), document.getElementById('inspector-tabs'), store, {
  flush: sync.flush,
});

// selectOnCanvas: the Validation tab's click-through target (js/ui/results.js). Selects the entity
// (store.select — the same call every canvas gesture/inspector picker already makes), then
// switches the inspector to its Selection tab via the EXACT function its own tab button's click
// handler calls (inspector.setActiveTab — saveLayout's read-merge-write persist + a render(), both
// already implemented there; nothing duplicated here).
const results = createResults(document.getElementById('pane-results'), store, {
  flush: sync.flush,
  plotly: window.Plotly,
  selectOnCanvas: (sel) => {
    // Item 4 (final-review, ruling): open the sub-model's scope on canvas FIRST — canvas.openScope
    // sets currentModelPath and re-renders, so the halo-matching check inside canvas.js's own
    // render sees the right scope by the time store.select's notification arrives just below.
    canvas.openScope(sel.modelPath ?? []);
    store.select(sel);
    inspector.setActiveTab('selection');
  },
});

// ================================================================================================
// ---------- Run (results drawer) ----------
// ================================================================================================
// #btn-run opens the results drawer (via panels' own toggle-results action, only if it's currently
// closed — both the gate-error and success paths in results.js's runBase() end with the drawer
// open, so it's simplest to just always ensure it's open here rather than have results.js reach
// back into panels, which its module contract (paneEl, store, {flush, plotly, selectOnCanvas}) has
// no room for). The actual run is deferred one tick (setTimeout 0) so the disabled/aria-busy state
// set just below actually PAINTS before the sync run computation blocks the main thread — without
// this, a fast run (typical case — no PSA involved) would set and clear "busy" within the same
// frame, never visibly rendering it.

const btnRun = document.getElementById('btn-run');

function runModel() {
  if (!panels.getState().results.open) panels.dispatch({ type: 'toggle-results' });
  // Item 2 (final-review): a maximized OTHER pane, or a minimized results drawer, would otherwise
  // let the run compute invisibly — the drawer opens above per the toggle-results call just above,
  // but a maximized yaml/canvas/inspector pane still covers it, and a minimized drawer still shows
  // only its 28px strip. Un-maximize whatever isn't already 'results' (maximizing results itself is
  // exactly the visible state we want, so leave that alone), then un-minimize the drawer if needed
  // — 'minimize' toggles, so dispatching it again is the reducer's own un-minimize path.
  const layoutState = panels.getState();
  if (layoutState.maximized && layoutState.maximized !== 'results') {
    panels.dispatch({ type: 'restore' });
  }
  if (panels.getState().results.min) {
    panels.dispatch({ type: 'minimize', pane: 'results' });
  }
  btnRun.disabled = true;
  btnRun.setAttribute('aria-busy', 'true');
  setTimeout(() => {
    try {
      results.runBase();
    } finally {
      btnRun.disabled = false;
      btnRun.removeAttribute('aria-busy');
    }
  }, 0);
}

btnRun.addEventListener('click', runModel);

// ================================================================================================
// ---------- Store subscription: drives every render + the debounced autosave ----------
// ================================================================================================

let autosaveTimer = null;

function scheduleAutosave() {
  if (autosaveTimer !== null) clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => {
    autosaveTimer = null;
    reg.autosave(store.get().text);
  }, 1000);
}

store.subscribe(() => {
  renderYamlPane();
  renderTopbar();
  scheduleAutosave();
});

renderYamlPane(); // see renderYamlPane's own comment for why this first call is safe/correct too
renderTopbar();

// ================================================================================================
// ---------- Dialog plumbing (New / Open / Examples) ----------
// ================================================================================================
// All three dialogs are built fresh into their <dialog> element on open (their content is entirely
// data-driven — the registry list, the examples list — so there's nothing useful to keep static in
// index.html beyond the empty <dialog> shells already there, matching #dlg-open's own pre-existing
// pattern). Clicking the backdrop closes the dialog (e.target === the dialog element itself is only
// true for a backdrop click, never a click on its content); Escape already closes a <dialog> shown
// via showModal() natively, no extra wiring needed.

function wireBackdropClose(dlg) {
  dlg.addEventListener('click', (e) => {
    if (e.target === dlg) dlg.close();
  });
}
wireBackdropClose(dlgNew);
wireBackdropClose(dlgOpen);
wireBackdropClose(dlgExamples);

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// ---------- New ----------

function renderNewDialog() {
  dlgNew.replaceChildren(
    h('h2', { class: 'dlg-title' }, 'New model'),
    h('div', { class: 'dlg-body' },
      h('button', {
        type: 'button', class: 'dlg-template-btn', 'data-template': 'markov',
        onclick: () => chooseTemplate(BLANK_MARKOV),
      }, h('strong', {}, 'Markov model'), h('span', {}, 'States and cyclical transitions.')),
      h('button', {
        type: 'button', class: 'dlg-template-btn', 'data-template': 'tree',
        onclick: () => chooseTemplate(BLANK_TREE),
      }, h('strong', {}, 'Decision tree'), h('span', {}, 'A root decision with branching outcomes.')),
    ),
    h('div', { class: 'dlg-actions' },
      h('button', { type: 'button', onclick: () => dlgNew.close() }, 'Cancel'),
    ),
  );
}

function chooseTemplate(text) {
  if (!confirmDiscardIfDirty()) return;
  dlgNew.close();
  currentModelId = null;
  loadDocument(text);
}

document.getElementById('btn-new').addEventListener('click', () => {
  renderNewDialog();
  dlgNew.showModal();
});

// ---------- Open ----------

function renderOpenDialog() {
  const items = reg.list();
  const body = items.length === 0
    ? h('p', { class: 'insp-empty' }, 'No saved models yet.')
    : h('ul', { class: 'dlg-list' }, ...items.map(renderOpenItem));

  dlgOpen.replaceChildren(
    h('h2', { class: 'dlg-title' }, 'Open'),
    h('div', { class: 'dlg-body' }, body),
    h('div', { class: 'dlg-actions' },
      h('button', { type: 'button', onclick: () => dlgOpen.close() }, 'Close'),
    ),
  );
}

function renderOpenItem(item) {
  const versions = reg.listVersions(item.id);
  const select = h('select', { class: 'insp-font-data', 'aria-label': `Version of ${item.name}` },
    ...versions.map((v) => h('option', { value: String(v.ts) },
      `${formatTs(v.ts)}${v.label ? ' — ' + v.label : ''}`)),
  );

  const loadOne = () => {
    if (!confirmDiscardIfDirty()) return;
    const ts = Number(select.value);
    const { text } = reg.load(item.id, ts);
    dlgOpen.close();
    currentModelId = item.id;
    loadDocument(text);
  };

  const deleteOne = () => {
    if (!window.confirm(`Delete "${item.name}" and all its versions? This cannot be undone.`)) return;
    reg.remove(item.id);
    if (currentModelId === item.id) currentModelId = null;
    renderOpenDialog();
  };

  return h('li', { class: 'dlg-list-item' },
    h('span', { class: 'dlg-list-name' }, item.name),
    h('span', { class: 'dlg-list-meta' },
      `${item.versionCount} version${item.versionCount === 1 ? '' : 's'} · updated ${formatTs(item.updated)}`),
    select,
    h('button', { type: 'button', class: 'dlg-load-btn', onclick: loadOne }, 'Load'),
    h('button', {
      type: 'button', class: 'insp-remove-btn dlg-del-btn', 'aria-label': `Delete ${item.name}`,
      title: 'Delete', onclick: deleteOne,
    }, '✕'),
  );
}

document.getElementById('btn-open').addEventListener('click', () => {
  renderOpenDialog();
  dlgOpen.showModal();
});

// ---------- Save version ----------

function saveVersion() {
  sync.flush();
  const { text, model, parseError } = store.get();
  if (parseError) {
    window.alert('Fix the YAML error before saving a version.');
    return;
  }
  const raw = window.prompt('Label for this version (optional):', '');
  if (raw === null) return; // cancelled
  const label = raw.trim() || undefined;
  // Review fix (Important): reg.saveVersion writes through localStorage.setItem, which can throw
  // (quota exceeded, private-browsing storage blocked, ...). Previously that exception propagated
  // uncaught out of this click/keydown handler — silently failing (nothing visible changes) while
  // still leaving the document marked dirty, which is at least honest, but gives the user no idea
  // the save never happened. Surface it, and do NOT markSaved (the document truly wasn't saved).
  try {
    currentModelId = reg.saveVersion(currentModelId, model.name, text, label);
  } catch (err) {
    window.alert('Could not save: ' + err.message);
    return;
  }
  store.markSaved();
}

document.getElementById('btn-save').addEventListener('click', saveVersion);

// ---------- Examples ----------

function renderExamplesBody(content) {
  dlgExamples.replaceChildren(
    h('h2', { class: 'dlg-title' }, 'Examples'),
    h('div', { class: 'dlg-body' }, content),
    h('div', { class: 'dlg-actions' },
      h('button', { type: 'button', onclick: () => dlgExamples.close() }, 'Cancel'),
    ),
  );
}

async function loadExample(item) {
  if (!confirmDiscardIfDirty()) return;
  try {
    const res = await fetch(`examples/${item.file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    dlgExamples.close();
    currentModelId = null;
    loadDocument(text);
  } catch (err) {
    window.alert(`Could not load example "${item.name}": ${err.message}`);
  }
}

async function openExamplesDialog() {
  renderExamplesBody(h('p', { class: 'insp-empty' }, 'Loading…'));
  dlgExamples.showModal();
  try {
    const res = await fetch('examples/index.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const items = await res.json();
    renderExamplesBody(h('ul', { class: 'dlg-list' },
      ...items.map((item) => h('li', {},
        h('button', { type: 'button', class: 'dlg-example-btn', onclick: () => loadExample(item) },
          item.name),
      )),
    ));
  } catch (err) {
    renderExamplesBody(h('p', { class: 'insp-empty' }, `Could not load the examples list: ${err.message}`));
  }
}

document.getElementById('btn-examples').addEventListener('click', openExamplesDialog);

// ---------- Import ----------

document.getElementById('btn-import').addEventListener('click', () => {
  fileImport.value = ''; // allow re-importing the same filename twice in a row
  fileImport.click();
});

fileImport.addEventListener('change', async () => {
  const file = fileImport.files[0];
  if (!file) return;
  if (!confirmDiscardIfDirty()) {
    fileImport.value = '';
    return;
  }
  const text = await file.text();
  currentModelId = null;
  // Parse errors in an imported file are not special-cased: loadDocument -> store.setText(text)
  // surfaces them in the YAML pane's own error strip exactly like a bad hand-edit would, per the
  // brief ("parse errors surface in the YAML pane's strip") — never a separate silent failure.
  loadDocument(text);
});

// ---------- Export ----------

document.getElementById('btn-export').addEventListener('click', () => {
  sync.flush();
  const { text, model } = store.get();
  const filename = `${slugify(model?.name)}.yaml`;
  const blob = new Blob([text], { type: 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// ================================================================================================
// ---------- Keyboard shortcuts ----------
// ================================================================================================
// Undo/redo only when focus is NOT in a text field (brief, verbatim) — a native textarea/input has
// its own undo stack that must keep working normally; the app's document-level undo only engages
// when focus is elsewhere (canvas, topbar, etc.). Ctrl/Cmd+S has no such carve-out: it must always
// override the browser's native "Save page" behavior, everywhere, including while typing. Item 5
// (final-review ruling): Ctrl/Cmd+Enter is the same — runModel() -> results.runBase() calls
// flush() first (via runNow), so a pending debounced YAML edit is always synced before the run
// reads the model; there's no staleness hazard the typing-target guard was protecting against, and
// a user mid-edit in the YAML pane wanting to run immediately is exactly the common case. The Z/Y
// undo/redo guard below is intentionally left untouched.

window.addEventListener('keydown', (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (!mod) return;
  const key = e.key.toLowerCase();

  if (key === 's') {
    e.preventDefault();
    saveVersion();
    return;
  }

  if (key === 'enter') {
    e.preventDefault();
    runModel();
    return;
  }

  if (key !== 'z' && key !== 'y') return;
  if (isTypingTarget(document.activeElement)) return;

  e.preventDefault();
  sync.flush();
  if (key === 'y' || (key === 'z' && e.shiftKey)) store.redo();
  else store.undo();
});

// ================================================================================================
// ---------- Unsaved-changes guard ----------
// ================================================================================================

window.addEventListener('beforeunload', (e) => {
  if (!store.get().dirty) return;
  e.preventDefault();
  e.returnValue = '';
});

// Exposed only for the manual/e2e verification pass (never imported by any module) — lets a
// browser console poke at live state without re-deriving it from the DOM.
window.__econeval = { store, sync, canvas, inspector, reg, panels, results };
