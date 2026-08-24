// Panel manager: splitters, maximize, minimize, layout persistence.
//
// Pure state machine (DOM-free, exported + covered by test/panels.test.js):
//   clampPane, nextLayoutState, defaultLayoutState, serializeLayout, parseLayout, saveLayout,
//   loadLayout, PANE_BOUNDS, LAYOUT_KEY.
// Thin DOM wiring (below the banner further down; untested by node:test — exercised by manual
// verification, see the task-8 and task-1 (phase-3) reports): initPanels(root) wires
// #split-l/#split-r/#split-b drag + keyboard, the yaml/inspector/results panel-head
// maximize/minimize buttons, the canvas toolbar's maximize button, and #btn-yaml, applying every
// resulting state to CSS vars + data attributes and persisting it on every change.
//
// State shape (the brief's contract, verbatim; `results` added phase-3 task 1):
//   {yaml: {w, open, min}, insp: {w, min}, results: {h, open, min},
//    maximized: null|'yaml'|'canvas'|'inspector'|'results'}
// nextLayoutState always spreads the incoming state as its base and only overwrites the field(s)
// an action actually changes, so an extra field a caller has tacked on survives every dispatch
// untouched. The DOM layer below relies on this: it carries a fourth field, `outline`, through the
// same object so the outline sidebar's collapsed-group set and last filter string (Task 10)
// persist in the same localStorage blob (see the "Persistence" comment on initPanels for how the
// two writers stay out of each other's way). `outline` replaces a phase-3 `tab` field (the old
// three-tab inspector's active tab) that Task 10's single-outline redesign made meaningless.
//
// Actions: {type:'drag', pane:'yaml'|'insp', dx}, {type:'toggle-yaml'},
// {type:'drag-results', dy}, {type:'toggle-results'}, {type:'maximize', pane},
// {type:'restore'}, {type:'minimize', pane}. `maximize`/`minimize`'s `pane` uses the DOM-facing
// vocabulary ('yaml'|'canvas'|'inspector'|'results', matching #pane-yaml/#pane-canvas/
// #pane-inspector/#pane-results) — different from `drag`'s pane vocabulary ('yaml'|'insp',
// matching the state's own geometry keys) because only two of those panes have a WIDTH to drag
// via the shared 'drag' action; results has a HEIGHT instead, so it gets its own 'drag-results'
// action (dy, not dx) rather than overloading 'drag' with a third pane id. 'results' is the one
// pane id that's spelled the same in both vocabularies — there's only one results concept, so no
// separate abbreviation was needed the way yaml/insp or yaml/canvas/inspector differ.

export const PANE_BOUNDS = {
  yaml: { min: 200, max: 560 },
  insp: { min: 240, max: 480 },
  results: { min: 160, max: 600 },
};

export const LAYOUT_KEY = 'econeval.layout.v1';

// ============================================================================================
// ---------- Pure: geometry ----------
// ============================================================================================

export function clampPane(px, { min, max }) {
  return Math.min(max, Math.max(min, px));
}

export function defaultLayoutState() {
  // A fresh object every call — callers (including nextLayoutState's spread-based updates) must
  // never share this by reference, or a mutation on one "default" would leak into every other use
  // of it.
  return {
    yaml: { w: 300, open: false, min: false },
    insp: { w: 300, min: false },
    results: { h: 300, open: false, min: false },
    maximized: null,
    outline: { collapsed: [], filter: '' },
  };
}

// Maps the DOM-facing pane id used by maximize/minimize ('yaml'|'canvas'|'inspector'|'results')
// to the state's own geometry key ('yaml'|'insp'|'results'). 'canvas' has no geometry entry — it
// isn't resizable, it just fills whatever space the other two leave — so it maps to null.
function geometryKey(pane) {
  if (pane === 'yaml') return 'yaml';
  if (pane === 'inspector') return 'insp';
  if (pane === 'results') return 'results';
  return null;
}

export function nextLayoutState(state, action) {
  switch (action.type) {
    case 'drag': {
      const { pane, dx } = action;
      if (pane === 'yaml') {
        const w = clampPane(state.yaml.w + dx, PANE_BOUNDS.yaml);
        return { ...state, yaml: { ...state.yaml, w } };
      }
      if (pane === 'insp') {
        // The right splitter sits between canvas and inspector: dragging it right (dx > 0) grows
        // canvas and shrinks the inspector, so dx is subtracted rather than added here.
        const w = clampPane(state.insp.w - dx, PANE_BOUNDS.insp);
        return { ...state, insp: { ...state.insp, w } };
      }
      return state;
    }

    case 'toggle-yaml': {
      const open = !state.yaml.open;
      // Closing the pane makes "it's minimized" or "it's the maximized pane" meaningless — clear
      // both so a later reopen never has to reconcile a stale collapsed/maximized flag. Width is
      // deliberately left untouched either way: that's what makes reopening land "at last width".
      const yaml = { ...state.yaml, open, min: open ? state.yaml.min : false };
      const maximized = (!open && state.maximized === 'yaml') ? null : state.maximized;
      return { ...state, yaml, maximized };
    }

    case 'drag-results': {
      // The results drawer sits at the bottom, above which #split-b is the boundary: dragging it
      // DOWN (dy > 0, clientY grows downward) moves that boundary toward the drawer's own floor,
      // shrinking it, so dy is subtracted — same "opposite of its own edge" convention as the
      // right splitter's dx above.
      const { dy } = action;
      const h = clampPane(state.results.h - dy, PANE_BOUNDS.results);
      return { ...state, results: { ...state.results, h } };
    }

    case 'toggle-results': {
      const open = !state.results.open;
      // Mirrors toggle-yaml exactly: closing clears minimized/maximized-self; height (h) is left
      // untouched either way, so reopening lands at the last dragged height, not the 300 default.
      const results = { ...state.results, open, min: open ? state.results.min : false };
      const maximized = (!open && state.maximized === 'results') ? null : state.maximized;
      return { ...state, results, maximized };
    }

    case 'maximize': {
      const { pane } = action;
      if (state.maximized === pane) return { ...state, maximized: null }; // toggle -> restore
      const key = geometryKey(pane);
      const patch = { maximized: pane };
      // A maximized pane can't simultaneously read as minimized or closed.
      if (key === 'yaml') patch.yaml = { ...state.yaml, open: true, min: false };
      else if (key === 'insp') patch.insp = { ...state.insp, min: false };
      else if (key === 'results') patch.results = { ...state.results, open: true, min: false };
      return { ...state, ...patch };
    }

    case 'restore':
      return { ...state, maximized: null };

    case 'minimize': {
      const { pane } = action;
      const key = geometryKey(pane);
      if (!key) return state; // canvas cannot minimize; the DOM layer never offers the control
      // "minimize a maximized pane -> restore first" (brief, verbatim). Minimizing an
      // already-minimized pane un-minimizes it — this doubles as the "click the strip to restore"
      // mechanism: the DOM layer just dispatches the same minimize action again.
      const restored = state.maximized === pane ? { ...state, maximized: null } : state;
      return { ...restored, [key]: { ...restored[key], min: !restored[key].min } };
    }

    default:
      throw new Error(`nextLayoutState: unknown action type "${action.type}"`);
  }
}

// ============================================================================================
// ---------- Pure: persistence ----------
// ============================================================================================

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function sanitizePane(value, fallback, bounds, hasOpen) {
  const src = (value && typeof value === 'object') ? value : {};
  const w = clampPane(isFiniteNumber(src.w) ? src.w : fallback.w, bounds);
  const min = typeof src.min === 'boolean' ? src.min : fallback.min;
  return hasOpen
    ? { w, open: typeof src.open === 'boolean' ? src.open : fallback.open, min }
    : { w, min };
}

// Same shape as sanitizePane above but for the results drawer's own field name ('h', not 'w') —
// kept as a separate small function rather than generalizing sanitizePane's hard-coded 'w', to
// leave the existing yaml/insp path (and its tests) untouched.
function sanitizeResults(value, fallback, bounds) {
  const src = (value && typeof value === 'object') ? value : {};
  const h = clampPane(isFiniteNumber(src.h) ? src.h : fallback.h, bounds);
  const min = typeof src.min === 'boolean' ? src.min : fallback.min;
  const open = typeof src.open === 'boolean' ? src.open : fallback.open;
  return { h, open, min };
}

// The outline sidebar's own small bit of persisted state (Task 10): which group ids are
// collapsed, and the last filter string typed. `collapsed` recovers to `fallback.collapsed`
// wholesale unless it's an array (and is then filtered down to just its string entries — a stray
// non-string entry from a hand-edited or future-schema blob is dropped rather than rejecting the
// whole field); `filter` recovers field-by-field like every other leaf here.
function sanitizeOutline(value, fallback) {
  const src = (value && typeof value === 'object') ? value : {};
  const collapsed = Array.isArray(src.collapsed)
    ? src.collapsed.filter((x) => typeof x === 'string')
    : fallback.collapsed;
  const filter = typeof src.filter === 'string' ? src.filter : fallback.filter;
  return { collapsed, filter };
}

const MAXIMIZED_VALUES = new Set([null, 'yaml', 'canvas', 'inspector', 'results']);

// Normalizes anything object-shaped into the canonical 5-field layout blob, filling each missing
// or invalid field from defaultLayoutState() independently. This is what lets a corrupt-but-
// parseable stored blob (an old schema, a hand-edited value, a future field this version doesn't
// know about yet) recover field-by-field instead of losing the whole saved layout to one bad key.
// This is also the backward-compat path for a phase-2 blob that predates `results` entirely: raw
// simply has no `results` key, sanitizeResults(undefined, ...) falls through to `fallback` (=
// defaultLayoutState().results) for every one of its fields, and the blob parses normally instead
// of being rejected — never null just because an older schema is missing the newer field.
function sanitizeLayout(raw) {
  const d = defaultLayoutState();
  return {
    yaml: sanitizePane(raw.yaml, d.yaml, PANE_BOUNDS.yaml, true),
    insp: sanitizePane(raw.insp, d.insp, PANE_BOUNDS.insp, false),
    results: sanitizeResults(raw.results, d.results, PANE_BOUNDS.results),
    maximized: MAXIMIZED_VALUES.has(raw.maximized) ? raw.maximized : null,
    outline: sanitizeOutline(raw.outline, d.outline),
  };
}

export function serializeLayout(state) {
  return JSON.stringify(sanitizeLayout(state && typeof state === 'object' ? state : {}));
}

// bad JSON -> null (never throws); a well-formed value that isn't a plausible layout object
// (null, an array, a primitive) is treated the same way — both are the caller's cue to fall back
// to defaultLayoutState().
export function parseLayout(str) {
  let raw;
  try {
    raw = JSON.parse(str);
  } catch {
    return null;
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  return sanitizeLayout(raw);
}

function safeStorage(storage) {
  if (storage) return storage;
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null; // storage access itself can throw (e.g. blocked in some private-browsing modes)
  }
}

export function saveLayout(state, storage) {
  const store = safeStorage(storage);
  if (!store) return;
  try {
    store.setItem(LAYOUT_KEY, serializeLayout(state));
  } catch {
    // Best-effort persistence only — a full/blocked storage quota isn't a user-facing model error;
    // constraints.md's "errors surfaced, never swallowed" is about YAML/expression/check errors.
  }
}

export function loadLayout(storage) {
  const store = safeStorage(storage);
  if (!store) return defaultLayoutState();
  let raw;
  try {
    raw = store.getItem(LAYOUT_KEY);
  } catch {
    return defaultLayoutState();
  }
  if (raw == null) return defaultLayoutState();
  return parseLayout(raw) ?? defaultLayoutState();
}

// ============================================================================================
// ---------- DOM wiring (thin — no business logic; everything above owns that) ----------
// ============================================================================================
//
// initPanels(root = document, {storage} = {}) -> {dispatch, getState}
//
// Boots from loadLayout(storage), applies it to the DOM, wires every gesture below, and
// re-applies + persists on every dispatch. Also wires #btn-yaml (dispatches 'toggle-yaml') even
// though that button physically lives in the topbar, not #workspace: it's a layout action through
// and through, so it's kept here rather than split across this file and app.js (Task 12 should
// NOT attach its own click handler to #btn-yaml — it only needs to call initPanels() once at
// boot, same as it calls the other createXxx(...) constructors).
//
// Persistence: this module and inspector.js both write the SAME localStorage blob (this module
// owns yaml/insp/maximized; inspector.js owns `outline`). Every save here re-reads the current
// `outline` from storage immediately before writing, so an outline change (a filter keystroke, a
// group collapsed/expanded) made elsewhere is never clobbered by a later resize/maximize/minimize
// here. inspector.js mirrors this: load the full blob, overwrite only `outline`, save the full
// blob back — so it never clobbers panel geometry either. Both sides doing read-merge-write per
// discrete user gesture is safe because these are synchronous localStorage calls on a single UI
// thread; there is no window for the two writers to race.

function toggleAttr(el, name, on) {
  if (on) el.setAttribute(name, '');
  else el.removeAttribute(name);
}

// hidden removes the element from layout AND (per the HTML spec) from the tab order on its own;
// tabIndex is set explicitly too, belt-and-suspenders, so "not focusable" doesn't rely solely on
// that implicit behavior. This is the fix for the carried Task-2 finding: #split-l must not remain
// a visible 4px track / keyboard tab-stop once #pane-yaml is closed, and the same rule is applied
// generally (minimized or maximized-elsewhere also removes a splitter from the tab order).
function setSplitterVisible(el, visible) {
  el.hidden = !visible;
  el.tabIndex = visible ? 0 : -1;
}

function effectiveWidth(pane) {
  return pane.min ? 28 : pane.w;
}

function effectiveHeight(pane) {
  return pane.min ? 28 : pane.h;
}

// The minimized strip has no button in it (the panel-ctl buttons are CSS-hidden — no room), so it
// needs its own keyboard entry point: made a focusable, labeled, Enter/Space-operable "button"
// only while minimized. Not focusable at all otherwise, so it never becomes a stray tab-stop.
function setStripFocusable(headEl, on, label) {
  if (on) {
    headEl.tabIndex = 0;
    headEl.setAttribute('role', 'button');
    headEl.setAttribute('aria-label', label);
  } else {
    headEl.removeAttribute('tabindex');
    headEl.removeAttribute('role');
    headEl.removeAttribute('aria-label');
  }
}

function makeCtlButton(glyph, label) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = glyph;
  btn.setAttribute('aria-label', label);
  btn.title = label;
  return btn;
}

function wireSplitter(el, pane, dispatch) {
  let dragging = false;
  let lastX = 0;

  function move(e) {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    lastX = e.clientX;
    if (dx !== 0) dispatch({ type: 'drag', pane, dx });
  }
  function up(e) {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('dragging');
    document.body.style.removeProperty('user-select');
    try { el.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  }

  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return; // primary button/touch only
    dragging = true;
    lastX = e.clientX;
    el.setPointerCapture(e.pointerId);
    el.classList.add('dragging');
    document.body.style.userSelect = 'none'; // avoid text-selection glitches while dragging
  });
  el.addEventListener('pointermove', move);
  el.addEventListener('pointerup', up);
  el.addEventListener('pointercancel', up);

  // ArrowLeft/ArrowRight nudge +-16px using the same sign convention as physically dragging this
  // splitter that far: for the right splitter, ArrowRight *shrinks* the inspector, same as
  // dragging it right does, because both are "move this splitter right".
  el.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); dispatch({ type: 'drag', pane, dx: -16 }); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); dispatch({ type: 'drag', pane, dx: 16 }); }
  });
}

// Mirrors wireSplitter above but for #split-b: it drags vertically (against clientY) and
// dispatches the dedicated 'drag-results' action instead of the generic 'drag' — the results row
// is sized by height (h), not width, so it doesn't fit drag's {pane, dx} vocabulary. Kept as its
// own function (rather than generalizing wireSplitter with an axis flag) to leave the working,
// already-covered-by-manual-verification split-l/split-r wiring untouched.
function wireResultsSplitter(el, dispatch) {
  let dragging = false;
  let lastY = 0;

  function move(e) {
    if (!dragging) return;
    const dy = e.clientY - lastY;
    lastY = e.clientY;
    if (dy !== 0) dispatch({ type: 'drag-results', dy });
  }
  function up(e) {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('dragging');
    document.body.style.removeProperty('user-select');
    try { el.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  }

  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return; // primary button/touch only
    dragging = true;
    lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
    el.classList.add('dragging');
    document.body.style.userSelect = 'none'; // avoid text-selection glitches while dragging
  });
  el.addEventListener('pointermove', move);
  el.addEventListener('pointerup', up);
  el.addEventListener('pointercancel', up);

  // ArrowUp grows the drawer (same direction as physically dragging this splitter up); ArrowDown
  // shrinks it — matching drag-results' h = h - dy convention the same way wireSplitter's
  // ArrowLeft/ArrowRight mirrors its own splitter's dx convention above.
  el.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') { e.preventDefault(); dispatch({ type: 'drag-results', dy: -16 }); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); dispatch({ type: 'drag-results', dy: 16 }); }
  });
}

function wireStripRestore(paneEl, headEl, isMinimized, restore) {
  // Ignore clicks that originated on a button (the panel-ctl maximize/minimize buttons live
  // inside headEl, which is inside paneEl, so their clicks bubble up here too). Without this
  // guard, clicking "Minimize" self-undoes in the very same click: the button's own listener
  // sets min=true synchronously, then this bubbled handler sees isMinimized()===true and calls
  // restore() before the event finishes propagating. The buttons are CSS-hidden once actually
  // minimized (no room in the 28px strip), so this only ever excludes the moment of that click.
  paneEl.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    if (isMinimized()) restore();
  });
  headEl.addEventListener('keydown', (e) => {
    if (!isMinimized()) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); restore(); }
  });
}

function applyLayout(state, els) {
  const {
    workspace, paneYaml, splitL, splitR, paneInsp, btnYaml, yamlHead, inspHead,
    yamlMaxBtn, yamlMinBtn, inspMaxBtn, inspMinBtn, canvasMaxBtn,
    paneResults, splitB, resultsHead, resMaxBtn, resMinBtn,
  } = els;

  paneYaml.hidden = !state.yaml.open;
  workspace.style.setProperty('--w-yaml', state.yaml.open ? `${effectiveWidth(state.yaml)}px` : '0px');
  workspace.style.setProperty('--w-insp', `${effectiveWidth(state.insp)}px`);

  paneResults.hidden = !state.results.open;
  workspace.style.setProperty('--h-results', state.results.open ? `${effectiveHeight(state.results)}px` : '0px');

  toggleAttr(paneYaml, 'data-min', state.yaml.open && state.yaml.min);
  toggleAttr(paneInsp, 'data-min', state.insp.min);
  toggleAttr(paneResults, 'data-min', state.results.open && state.results.min);

  if (state.maximized) workspace.setAttribute('data-max', state.maximized);
  else workspace.removeAttribute('data-max');

  const splitLVisible = state.yaml.open && !state.yaml.min && !state.maximized;
  const splitRVisible = !state.insp.min && !state.maximized;
  // #split-b: hidden whenever the drawer is closed or minimized, or whenever ANY pane is
  // maximized (mirrors splitL/splitR — a maximized pane hides every other pane and splitter,
  // including its own, since there's nothing left to drag against).
  const splitBVisible = state.results.open && !state.results.min && !state.maximized;
  setSplitterVisible(splitL, splitLVisible);
  setSplitterVisible(splitR, splitRVisible);
  setSplitterVisible(splitB, splitBVisible);

  setStripFocusable(yamlHead, state.yaml.open && state.yaml.min, 'Restore YAML panel');
  setStripFocusable(inspHead, state.insp.min, 'Restore inspector panel');
  setStripFocusable(resultsHead, state.results.open && state.results.min, 'Restore results panel');

  if (btnYaml) btnYaml.setAttribute('aria-pressed', String(state.yaml.open));

  yamlMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'yaml'));
  yamlMaxBtn.title = state.maximized === 'yaml' ? 'Restore panel' : 'Maximize panel';
  yamlMinBtn.setAttribute('aria-pressed', String(state.yaml.min));

  inspMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'inspector'));
  inspMaxBtn.title = state.maximized === 'inspector' ? 'Restore panel' : 'Maximize panel';
  inspMinBtn.setAttribute('aria-pressed', String(state.insp.min));

  canvasMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'canvas'));
  canvasMaxBtn.title = state.maximized === 'canvas' ? 'Restore panel' : 'Maximize panel';

  resMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'results'));
  resMaxBtn.title = state.maximized === 'results' ? 'Restore panel' : 'Maximize panel';
  resMinBtn.setAttribute('aria-pressed', String(state.results.min));
}

export function initPanels(root = document, { storage } = {}) {
  const workspace = root.getElementById('workspace');
  const paneYaml = root.getElementById('pane-yaml');
  const splitL = root.getElementById('split-l');
  const canvasToolbar = root.getElementById('canvas-toolbar');
  const splitR = root.getElementById('split-r');
  const paneInsp = root.getElementById('pane-inspector');
  const inspectorTabs = root.getElementById('inspector-tabs'); // this IS inspector's .panel-head
  const btnYaml = root.getElementById('btn-yaml');
  const paneResults = root.getElementById('pane-results');
  const splitB = root.getElementById('split-b');

  const yamlHead = paneYaml.querySelector('.panel-head');
  const yamlCtl = yamlHead.querySelector('.panel-ctl');
  const yamlMaxBtn = makeCtlButton('⤢', 'Maximize panel'); // ⤢
  const yamlMinBtn = makeCtlButton('—', 'Minimize panel'); // —
  yamlCtl.append(yamlMaxBtn, yamlMinBtn);

  // Inspector's panel-head (#inspector-tabs) carries its own static "Model" text (index.html) but
  // no pre-existing .panel-ctl span (unlike yaml's) — appended here as a sibling, same shape as
  // yaml's. inspector.js never touches #inspector-tabs at all (Task 10 replaced the old three-tab
  // strip with the single outline; there is nothing left to render into the head), so there is no
  // "coexist with a later render()" concern here any more — this span is simply the only content
  // this element's JS-driven side ever gets.
  const inspHead = inspectorTabs;
  const inspCtl = document.createElement('span');
  inspCtl.className = 'panel-ctl';
  const inspMaxBtn = makeCtlButton('⤢', 'Maximize panel');
  const inspMinBtn = makeCtlButton('—', 'Minimize panel');
  inspCtl.append(inspMaxBtn, inspMinBtn);
  inspectorTabs.appendChild(inspCtl);

  // Canvas has no .panel-head at all — its maximize button (no minimize; canvas can't minimize)
  // goes straight into the toolbar per the brief.
  const canvasMaxBtn = makeCtlButton('⤢', 'Maximize panel');
  canvasToolbar.appendChild(canvasMaxBtn);

  // Results drawer's panel-head has the same static shape as yaml's (literal "Results" text plus
  // a pre-existing .panel-ctl span in index.html) — mirror yaml's wiring exactly.
  const resultsHead = paneResults.querySelector('.panel-head');
  const resultsCtl = resultsHead.querySelector('.panel-ctl');
  const resMaxBtn = makeCtlButton('⤢', 'Maximize panel');
  const resMinBtn = makeCtlButton('—', 'Minimize panel');
  resultsCtl.append(resMaxBtn, resMinBtn);

  let state = loadLayout(storage);

  function persist() {
    const latest = loadLayout(storage); // don't clobber `outline` written elsewhere since our last save
    saveLayout({ ...state, outline: latest.outline }, storage);
  }

  function render() {
    applyLayout(state, {
      workspace, paneYaml, splitL, splitR, paneInsp, btnYaml, yamlHead, inspHead,
      yamlMaxBtn, yamlMinBtn, inspMaxBtn, inspMinBtn, canvasMaxBtn,
      paneResults, splitB, resultsHead, resMaxBtn, resMinBtn,
    });
  }

  function dispatch(action) {
    state = nextLayoutState(state, action);
    render();
    persist();
  }

  wireSplitter(splitL, 'yaml', dispatch);
  wireSplitter(splitR, 'insp', dispatch);
  wireResultsSplitter(splitB, dispatch);

  yamlMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'yaml' }));
  yamlMinBtn.addEventListener('click', () => dispatch({ type: 'minimize', pane: 'yaml' }));
  inspMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'inspector' }));
  inspMinBtn.addEventListener('click', () => dispatch({ type: 'minimize', pane: 'inspector' }));
  canvasMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'canvas' }));
  resMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'results' }));
  resMinBtn.addEventListener('click', () => dispatch({ type: 'minimize', pane: 'results' }));

  wireStripRestore(paneYaml, yamlHead, () => state.yaml.min, () => dispatch({ type: 'minimize', pane: 'yaml' }));
  wireStripRestore(paneInsp, inspHead, () => state.insp.min, () => dispatch({ type: 'minimize', pane: 'inspector' }));
  wireStripRestore(paneResults, resultsHead, () => state.results.min, () => dispatch({ type: 'minimize', pane: 'results' }));

  if (btnYaml) btnYaml.addEventListener('click', () => dispatch({ type: 'toggle-yaml' }));

  // A generic "step back" affordance: Escape restores whichever pane is maximized, regardless of
  // which panel-head's button triggered it. Harmless alongside Task 10's canvas-tool Escape
  // handling (both may fire on the same keypress; neither depends on the other).
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.maximized) dispatch({ type: 'restore' });
  });

  render();
  persist(); // normalize + write back a canonical blob immediately, even if nothing changed yet

  return {
    dispatch,
    getState: () => state,
  };
}
