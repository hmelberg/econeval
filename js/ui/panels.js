// Panel manager: splitters, maximize, minimize, layout persistence.
//
// Pure state machine (DOM-free, exported + covered by test/panels.test.js):
//   clampPane, nextLayoutState, defaultLayoutState, serializeLayout, parseLayout, saveLayout,
//   loadLayout, PANE_BOUNDS, LAYOUT_KEY.
// Thin DOM wiring (below the banner further down; untested by node:test — exercised by manual
// verification, see the task-8 report): initPanels(root) wires #split-l/#split-r drag + keyboard,
// the yaml/inspector panel-head maximize/minimize buttons, the canvas toolbar's maximize button,
// and #btn-yaml, applying every resulting state to CSS vars + data attributes and persisting it on
// every change.
//
// State shape (the brief's contract, verbatim):
//   {yaml: {w, open, min}, insp: {w, min}, maximized: null|'yaml'|'canvas'|'inspector'}
// nextLayoutState always spreads the incoming state as its base and only overwrites the field(s)
// an action actually changes, so an extra field a caller has tacked on survives every dispatch
// untouched. The DOM layer below relies on this: it carries a fourth field, `tab`, through the
// same object so Task 11's active-inspector-tab selection persists in the same localStorage blob
// (see the "Persistence" comment on initPanels for how the two writers stay out of each other's
// way).
//
// Actions: {type:'drag', pane:'yaml'|'insp', dx}, {type:'toggle-yaml'}, {type:'maximize', pane},
// {type:'restore'}, {type:'minimize', pane}. `maximize`/`minimize`'s `pane` uses the DOM-facing
// vocabulary ('yaml'|'canvas'|'inspector', matching #pane-yaml/#pane-canvas/#pane-inspector) —
// different from `drag`'s pane vocabulary ('yaml'|'insp', matching the state's own geometry keys)
// because only two of the three panes have a width to drag, but all three can be maximized.

export const PANE_BOUNDS = {
  yaml: { min: 200, max: 560 },
  insp: { min: 240, max: 480 },
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
    maximized: null,
    tab: 'selection',
  };
}

// Maps the DOM-facing pane id used by maximize/minimize ('yaml'|'canvas'|'inspector') to the
// state's own geometry key ('yaml'|'insp'). 'canvas' has no geometry entry — it isn't resizable,
// it just fills whatever space the other two leave — so it maps to null.
function geometryKey(pane) {
  if (pane === 'yaml') return 'yaml';
  if (pane === 'inspector') return 'insp';
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

    case 'maximize': {
      const { pane } = action;
      if (state.maximized === pane) return { ...state, maximized: null }; // toggle -> restore
      const key = geometryKey(pane);
      const patch = { maximized: pane };
      // A maximized pane can't simultaneously read as minimized or closed.
      if (key === 'yaml') patch.yaml = { ...state.yaml, open: true, min: false };
      else if (key === 'insp') patch.insp = { ...state.insp, min: false };
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

const MAXIMIZED_VALUES = new Set([null, 'yaml', 'canvas', 'inspector']);

// Normalizes anything object-shaped into the canonical 4-field layout blob, filling each missing
// or invalid field from defaultLayoutState() independently. This is what lets a corrupt-but-
// parseable stored blob (an old schema, a hand-edited value, a future field this version doesn't
// know about yet) recover field-by-field instead of losing the whole saved layout to one bad key.
function sanitizeLayout(raw) {
  const d = defaultLayoutState();
  return {
    yaml: sanitizePane(raw.yaml, d.yaml, PANE_BOUNDS.yaml, true),
    insp: sanitizePane(raw.insp, d.insp, PANE_BOUNDS.insp, false),
    maximized: MAXIMIZED_VALUES.has(raw.maximized) ? raw.maximized : null,
    tab: typeof raw.tab === 'string' ? raw.tab : d.tab,
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
// Persistence: this module and Task 11's inspector.js both write the SAME localStorage blob (this
// module owns yaml/insp/maximized; inspector.js owns `tab`). Every save here re-reads the current
// `tab` from storage immediately before writing, so a tab change made elsewhere is never clobbered
// by a later resize/maximize/minimize here. inspector.js should mirror this: load the full blob,
// overwrite only `tab`, save the full blob back — so it never clobbers panel geometry either. Both
// sides doing read-merge-write per discrete user gesture is safe because these are synchronous
// localStorage calls on a single UI thread; there is no window for the two writers to race.

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

function wireStripRestore(paneEl, headEl, isMinimized, restore) {
  paneEl.addEventListener('click', () => { if (isMinimized()) restore(); });
  headEl.addEventListener('keydown', (e) => {
    if (!isMinimized()) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); restore(); }
  });
}

function applyLayout(state, els) {
  const {
    workspace, paneYaml, splitL, splitR, paneInsp, btnYaml, yamlHead, inspHead,
    yamlMaxBtn, yamlMinBtn, inspMaxBtn, inspMinBtn, canvasMaxBtn,
  } = els;

  paneYaml.hidden = !state.yaml.open;
  workspace.style.setProperty('--w-yaml', state.yaml.open ? `${effectiveWidth(state.yaml)}px` : '0px');
  workspace.style.setProperty('--w-insp', `${effectiveWidth(state.insp)}px`);

  toggleAttr(paneYaml, 'data-min', state.yaml.open && state.yaml.min);
  toggleAttr(paneInsp, 'data-min', state.insp.min);

  if (state.maximized) workspace.setAttribute('data-max', state.maximized);
  else workspace.removeAttribute('data-max');

  const splitLVisible = state.yaml.open && !state.yaml.min && !state.maximized;
  const splitRVisible = !state.insp.min && !state.maximized;
  setSplitterVisible(splitL, splitLVisible);
  setSplitterVisible(splitR, splitRVisible);

  setStripFocusable(yamlHead, state.yaml.open && state.yaml.min, 'Restore YAML panel');
  setStripFocusable(inspHead, state.insp.min, 'Restore inspector panel');

  if (btnYaml) btnYaml.setAttribute('aria-pressed', String(state.yaml.open));

  yamlMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'yaml'));
  yamlMaxBtn.title = state.maximized === 'yaml' ? 'Restore panel' : 'Maximize panel';
  yamlMinBtn.setAttribute('aria-pressed', String(state.yaml.min));

  inspMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'inspector'));
  inspMaxBtn.title = state.maximized === 'inspector' ? 'Restore panel' : 'Maximize panel';
  inspMinBtn.setAttribute('aria-pressed', String(state.insp.min));

  canvasMaxBtn.setAttribute('aria-pressed', String(state.maximized === 'canvas'));
  canvasMaxBtn.title = state.maximized === 'canvas' ? 'Restore panel' : 'Maximize panel';
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

  const yamlHead = paneYaml.querySelector('.panel-head');
  const yamlCtl = yamlHead.querySelector('.panel-ctl');
  const yamlMaxBtn = makeCtlButton('⤢', 'Maximize panel'); // ⤢
  const yamlMinBtn = makeCtlButton('—', 'Minimize panel'); // —
  yamlCtl.append(yamlMaxBtn, yamlMinBtn);

  // Inspector's panel-head (#inspector-tabs) has no pre-existing .panel-ctl in index.html (unlike
  // yaml's) — Task 11 renders the tab buttons into #inspector-tabs itself. Mirroring yaml's shape
  // by appending a sibling .panel-ctl span here (rather than mixing these buttons in with the tab
  // buttons) is the most predictable thing for that later render() to coexist with: it should
  // append/update tab buttons without clobbering this span's innerHTML wholesale.
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

  let state = loadLayout(storage);

  function persist() {
    const latest = loadLayout(storage); // don't clobber a `tab` written elsewhere since our last save
    saveLayout({ ...state, tab: latest.tab }, storage);
  }

  function render() {
    applyLayout(state, {
      workspace, paneYaml, splitL, splitR, paneInsp, btnYaml, yamlHead, inspHead,
      yamlMaxBtn, yamlMinBtn, inspMaxBtn, inspMinBtn, canvasMaxBtn,
    });
  }

  function dispatch(action) {
    state = nextLayoutState(state, action);
    render();
    persist();
  }

  wireSplitter(splitL, 'yaml', dispatch);
  wireSplitter(splitR, 'insp', dispatch);

  yamlMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'yaml' }));
  yamlMinBtn.addEventListener('click', () => dispatch({ type: 'minimize', pane: 'yaml' }));
  inspMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'inspector' }));
  inspMinBtn.addEventListener('click', () => dispatch({ type: 'minimize', pane: 'inspector' }));
  canvasMaxBtn.addEventListener('click', () => dispatch({ type: 'maximize', pane: 'canvas' }));

  wireStripRestore(paneYaml, yamlHead, () => state.yaml.min, () => dispatch({ type: 'minimize', pane: 'yaml' }));
  wireStripRestore(paneInsp, inspHead, () => state.insp.min, () => dispatch({ type: 'minimize', pane: 'inspector' }));

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
