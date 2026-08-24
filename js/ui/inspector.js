// Inspector: a single filterable outline listing every state/edge/tree-node, parameter, sub-model,
// and setting — replacing the earlier three-tab (Selection/Parameters/Settings) design (Task 10).
// Exactly one row's fields are shown at a time (the selected one, indented beneath its row) except
// the terminal "Settings" group, whose fields are its own group's "content" and show/hide with its
// own collapse toggle instead (see the "Expansion" section below).
//
// createInspector(rootEl, headEl, store, {flush, openScope}) -> {render(), revealSelection()}
//   rootEl:     #inspector-body — the filter bar + row list go here, rebuilt on every structural
//               render().
//   headEl:     #inspector-tabs, the pane's .panel-head. panels.js already writes its own
//               .panel-ctl (maximize/minimize) span into it and gives it the static text "Model"
//               (index.html) — this module never touches headEl at all; it's accepted only to
//               keep the constructor's shape stable for callers (app.js) and any future use.
//   flush:      optional () => void, called before EVERY op commit (a canvas/inspector edit must
//               never race a pending debounced YAML-pane edit).
//   openScope:  optional (modelPath) => void, provided by app.js as canvas.openScope. A row click
//               drills the canvas into the row's sub-model scope before selecting — inspector.js
//               holds no canvas reference (importing one would rebuild exactly the backwards peer
//               dependency scoped-store.js exists to remove), so app.js injects this instead.
//               Defaults to a no-op (safe for tests / a caller that doesn't need it).
//
// Row source: js/ui/outline/build.js's buildOutline(model) — pure, DOM-free, fully tested. This
// module is only the view on top of it: buildOutline is always called on the TOP-LEVEL model
// (modelPath: [] for every row), matching every existing outline-build.test.js fixture. A row's
// own `sel.modelPath` is therefore always [] today; scopedStoreFor(store, row.sel.modelPath) below
// is written generally (not hardcoded to `store`) so a future deeper-nesting outline would not
// need this module's edit logic to change, but it is a no-op in v1.
//
// Selection: STRUCTURE rows (state/edge/node) are canvas entities — a click calls
// openScope(row.sel.modelPath) then store.select(row.sel), the same ordering results.js's
// Validation click-through already depends on (so the canvas halo's sameModelPath check sees the
// right scope before the selection notification arrives). PARAM rows carry sel: null in
// buildOutline (params are not canvas entities), but this module still drives their expansion
// through the very same store.select mechanism: a click synthesizes
// {kind:'param', id: name, modelPath: []} — store.js's isSelectionValid already resolves that kind
// (`scoped.params.has(id)`), so reusing it here (rather than a second, parallel "local selection"
// concept) gets a deleted-parameter's field editor collapsed for free, on the same rule as a
// deleted state. SUBMODEL rows have no fields at all; a click only drills the canvas in
// (openScope([row.label])) — the outline itself never recurses into a sub-model's own content.
// GROUP header rows (including "Settings") only ever toggle collapse; they never carry a
// selection.
//
// Expansion: `rowForSelection(allRows, store.get().selection)` (js/ui/outline/build.js) names the
// one row whose `.otl-fields` shows the entity's editor. The "Settings" group is the one
// exception — build.js gives it exactly one row (there is no per-setting row to select), so its
// fields render whenever the group itself is not collapsed, exactly mirroring how any other
// group's collapse toggle shows/hides its children — Settings simply has no child ROWS, only
// child FIELDS. This can coexist with a real selected-entity's own fields being shown elsewhere in
// the same outline (a Structure row expanded AND Settings expanded is normal, not a conflict) —
// "exactly one row expanded" governs the SELECTED-ENTITY editor only, not group content.
//
// Scope: STRUCTURE rows edit through scopedStoreFor(store, row.sel.modelPath); PARAMETERS and
// SETTINGS always edit the top-level store (row.sel.modelPath is [] there always, by definition —
// build.js never gives params/settings a non-empty modelPath).
//
// Render discipline: a full render() rebuilds the row list, which would steal focus/cursor out
// from under a field the user is actively editing. shouldSkipRender() skips the rebuild whenever
// focus is on a real input inside rootEl, REGARDLESS of what triggered the store change — our own
// field commit, a canvas gesture, a YAML-pane edit, undo/redo — with the single exception that the
// entity the focused field belongs to (`renderedSelection`, captured at the top of the last
// render()) is no longer resolvable against the fresh model (deleted out from under the user, not
// merely deselected), in which case the skip is lifted and render() proceeds immediately.
// `committingSelf` (true only for the synchronous span of a field commit) always forces the skip
// on its own — most usefully for an in-place rename, which briefly makes the OLD selection
// unresolvable the instant the commit lands (store.js nulls out an invalid selection as part of
// the very same commit); the field keeps focus through that instant and a blur-scheduled render()
// reconciles a tick later.
//
// The filter input and the "Only findings" toggle are LOCAL UI state, not store state — typing in
// the filter box never touches the store, so it never goes through shouldSkipRender at all; it
// calls render() directly on every keystroke. Because render() rebuilds rootEl wholesale
// (including the filter bar itself), render() explicitly captures the filter input's focus +
// cursor position + rootEl.scrollTop before replaceChildren() and restores all three after —
// otherwise every filter keystroke would visibly steal its own input's focus and jump the scroll
// position. The collapsed-group Set and filter string are also the two pieces of outline state
// persisted through panels.js's saveLayout (read-merge-write, same pattern panels.js's own
// persist() uses) — replacing the old three-tab design's now-dead `tab` field. `onlyFindings`
// itself is NOT persisted (a session-only view preference).
//
// Findings (Task 11): check(model) is re-run on a 300ms debounce after every store notification
// (scheduleFindingsCheck, mirroring results.js's own scheduleValidationBadge) into `latestFindings`,
// then mapped onto the outline via js/ui/outline/build.js's attachFindings(allRows, latestFindings)
// — never reimplemented here. The result drives three things, all patched onto the EXISTING DOM by
// paintFindings() (see its own doc, just above its definition) and NEVER via render(): a colored dot
// on the row that owns a finding (`byRow`), a rolled-up errors/warnings badge on group headers
// (`counts`), and a "Model findings" list pinned at the bottom of the outline for `residual`
// findings that match no row at all — nothing check() reports is ever silently dropped. A finding
// whose path also matches a field currently rendered in the expanded row's `.otl-fields` block shows
// a THIRD time, inline beneath that field, via the pre-existing fieldSlots/splitFindings mechanism —
// unrelated to attachFindings, and unchanged by this task. The "Only findings" toggle (wired here)
// composes with the text filter in a fixed order: filterRows(rows, query) first, then narrow to
// rows with a non-zero attachFindings count (own finding, or an ancestor of one) — see render()'s
// own comment at the composition site for why `counts` must be computed against the FULL row set,
// not the already-filtered one.

import { compile, ExprError } from '../core/expr.js';
import { formatCycle } from '../core/model.js';
import { check } from '../analysis/check.js';
import { scopedStoreFor } from './scoped-store.js';
import { loadLayout, saveLayout } from './panels.js';
import * as ops from './ops.js';
import {
  buildOutline, filterRows, scopePrefix, nodePathToCheckPath, rowForSelection, collapseFilter,
  ancestorIds, addAfterIndex, attachFindings,
} from './outline/build.js';

// ================================================================================================
// ---------- Pure helpers (exported + tested in test/inspector-match.test.js) ----------
// ================================================================================================

export function countByLevel(findings) {
  let errors = 0;
  let warnings = 0;
  for (const f of findings) {
    if (f.level === 'error') errors += 1;
    else if (f.level === 'warning') warnings += 1;
  }
  return { errors, warnings };
}

// splitFindings: findings whose `path` exactly matches a path in `renderedPaths` (a Set of check-
// path strings currently backing a rendered field) go into `inline` (keyed by path, preserving
// order, multiple findings per path stack); everything else goes into `rest`, in original order.
// Consumed by paintFindings() below (Task 11) for the inline-under-a-rendered-field mechanism —
// note this is a SEPARATE, field-scoped concern from attachFindings' row-scoped `byRow`/`residual`
// (js/ui/outline/build.js): a finding can be both the dot on a row AND, when that row happens to be
// the one currently expanded, an inline message under one of its fields at the same time.
export function splitFindings(findings, renderedPaths) {
  const inline = new Map();
  const rest = [];
  for (const f of findings) {
    if (renderedPaths.has(f.path)) {
      if (!inline.has(f.path)) inline.set(f.path, []);
      inline.get(f.path).push(f);
    } else {
      rest.push(f);
    }
  }
  return { inline, rest };
}

// ================================================================================================
// ---------- DOM micro-helper ----------
// ================================================================================================

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

// A tiny "name" op, inline per the controller ruling (name lives top-level on the model, not
// under settings — not a setSetting keyPath).
function setNameOp(model, newName) {
  const trimmed = (newName ?? '').trim();
  if (!trimmed) throw new Error('name: must not be empty');
  const m = structuredClone(model);
  m.name = trimmed;
  return m;
}

const PARAM_FIELDS = [
  ['value', 'Value'],
  ['low', 'Low'],
  ['high', 'High'],
  ['dist', 'Dist'],
  ['source', 'Source'],
];

// ================================================================================================
// ---------- createInspector ----------
// ================================================================================================

export function createInspector(rootEl, headEl, store, { flush = () => {}, openScope = () => {} } = {}) {
  let committingSelf = false;
  let fieldSlots = new Map(); // check-path -> error <div> element, from the last structural render (Task 11 will consume this; unused for now)
  let rowElements = new Map(); // row id -> its <button class="otl-row"> element, from the last render — used by revealSelection's scrollIntoView
  let selectedFieldsEl = null; // the SELECTED ENTITY's own .otl-fields container from the last render (never Settings' — see buildRowFields/render()) — used by shouldSkipRender to scope its force-reconcile exception to what's actually focused, not just "some selection somewhere went stale"
  let pendingFocusEl = null; // a just-added blank kv-row's key input (or a newly added param's Name input), focused at the end of render()
  let pendingPayoffRef = { count: 0 };
  let pendingWithRef = { count: 0 };
  let pendingParamFocusName = null; // set right before store.select()-ing a just-added param, consumed by appendParamFields on the render() that call triggers
  let lastSelKey = null;
  let renderedSelection = { kind: null, id: null }; // the selection the CURRENTLY DISPLAYED fields were built from, set at the top of every render()

  const initialOutline = loadLayout().outline ?? { collapsed: [], filter: '' };
  let collapsedGroups = new Set(initialOutline.collapsed ?? []);
  let filterQuery = initialOutline.filter ?? '';
  let onlyFindings = false; // wired below (buildFilterBar's click handler + render()'s onlyFindings composition). Not persisted — a session-only view preference, unlike the filter text.

  // ---------- findings state (Task 11) ----------
  //
  // latestFindings: the last check(model) result, refreshed by the 300ms-debounced
  // scheduleFindingsCheck() below on every store notification — never computed synchronously on
  // every keystroke (check() walks the whole model; debouncing coalesces a burst of rapid store
  // notifications, e.g. every keystroke of a debounced YAML-pane edit flushing, into one run).
  // Mirrors results.js's own validationFindings/scheduleValidationBadge exactly (same interval,
  // same "self-contained, not shared" module boundary), except this copy also drives the outline's
  // dots/counts/residual list, not just a tab badge.
  let latestFindings = [];
  let findingsTimer = null;
  // allRows: the FULL, unfiltered row list from the last render() (buildOutline(topModel) with no
  // filterRows/onlyFindings/collapseFilter applied) — paintFindings() below recomputes
  // attachFindings(allRows, latestFindings) against THIS cached list rather than calling
  // buildOutline() fresh, so it always stays paired with rowElements/fieldSlots (also captured at
  // the same last render()) even when a render() itself was skipped (shouldSkipRender, mid-typing)
  // while the debounced check() still ran against the live (possibly newer) model. Using a fresh
  // buildOutline() here instead could hand back row ids that don't match what's actually in the DOM
  // right now (e.g. a state renamed elsewhere while render() was skipped), silently failing to find
  // the row to patch — this way patching only ever targets rows genuinely present in rowElements.
  let allRows = [];
  let modelFindingsListEl = null; // the residual "Model findings" <ul>, from the last render()
  let modelFindingsWrapEl = null; // its wrapping <div> (hidden when residual is empty)

  // ---------- op commit plumbing ----------

  // Re-checks whether `sel` still resolves against the CURRENT model — the same rule store.js's
  // private isSelectionValid applies (duplicated here in miniature since store.js doesn't export
  // it; it's small, pure, and read-only). Used by the render-skip decision below: a selection with
  // kind:null is trivially "resolvable" (nothing to orphan).
  function selectionResolvable(sel) {
    if (!sel || sel.kind == null) return true;
    const scopeModel = scopedStoreFor(store, sel.modelPath ?? []).get().model;
    if (!scopeModel) return false;
    if (sel.kind === 'state') return scopeModel.type === 'markov' && scopeModel.states.some((s) => s.name === sel.id);
    if (sel.kind === 'edge') {
      const row = scopeModel.transitions[sel.id.from];
      if (!row) return false;
      return row.type === 'multinomial' ? sel.id.to in row.counts : sel.id.to in row.to;
    }
    if (sel.kind === 'node') {
      if (scopeModel.type !== 'tree') return false;
      try {
        ops.nodeAt(scopeModel, sel.id);
        return true;
      } catch {
        return false;
      }
    }
    if (sel.kind === 'param') return scopeModel.params.has(sel.id);
    return true;
  }

  function isTypingTarget(el) {
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
  }

  // Render-skip decision (controller ruling, generalized beyond "own commit only"): skip the full
  // structural rebuild whenever the user is mid-interaction with a field inside rootEl, REGARDLESS
  // of what triggered this store notification (our own field commit, a canvas gesture, a YAML-pane
  // edit, undo/redo, ...) — a canvas-origin selection change must not be able to yank a field out
  // from under someone who's mid-typing elsewhere. Two cases still force an immediate render even
  // while typing:
  //   - Not our own commit AND focus sits inside the SELECTED ENTITY'S OWN `.otl-fields` block
  //     (`selectedFieldsEl`, captured at the end of the last render()) AND that entity
  //     (`renderedSelection`) is no longer resolvable — it was deleted out from under the user
  //     (Delete tool, undo, a YAML edit...), so there is nothing sensible left to keep showing;
  //     reconcile immediately instead of leaving an orphaned field on screen. Review fix
  //     (Important, Task 10 review): unlike the old three-tab inspector, a selected entity's
  //     fields and the always-visible Settings fields can be on screen at once — checking only
  //     "is renderedSelection resolvable" (with no check on what's actually focused) meant typing
  //     in an unrelated Settings field, while some OTHER selected entity got deleted elsewhere,
  //     would force-reconcile and yank focus out of the Settings field for no reason connected to
  //     it. Scoping to `selectedFieldsEl.contains(active)` fixes that: only the field editor that
  //     could actually be showing something orphaned gets force-reconciled. Reselecting a
  //     DIFFERENT-but-still-existing entity (e.g. clicking another canvas node) does NOT trigger
  //     this either way — the field being edited is untouched, just no longer the active
  //     selection, so it's left alone until its own blur reconciles it.
  //   - Focus isn't a typing target inside rootEl at all — nothing to protect.
  function shouldSkipRender() {
    const active = document.activeElement;
    if (!isTypingTarget(active) || !rootEl.contains(active)) return false;
    if (committingSelf) return true;
    if (!selectedFieldsEl || !selectedFieldsEl.contains(active)) return true;
    return selectionResolvable(renderedSelection);
  }

  // Buttons (add/remove/delete row): no focus to protect, always let the store's own subscribe
  // handler do a full, immediate re-render.
  function commitOp(targetStore, fn) {
    try {
      flush();
      targetStore.applyOp(fn);
      return null;
    } catch (e) {
      return e;
    }
  }

  // Text/select field commits (change/Enter): marks the synchronous commit window so the
  // subscribe handler can skip stealing focus if the user is still inside this panel.
  function commitFieldOp(targetStore, fn) {
    committingSelf = true;
    try {
      flush();
      targetStore.applyOp(fn);
      return null;
    } catch (e) {
      return e;
    } finally {
      committingSelf = false;
    }
  }

  function registerField(path, errEl) {
    if (!path) return;
    fieldSlots.set(path, errEl);
  }

  // ---------- field wiring ----------

  function wireExprInput(inputEl, errEl) {
    inputEl.addEventListener('input', () => {
      const v = inputEl.value;
      if (v.trim() === '') {
        inputEl.classList.remove('insp-field-invalid');
        errEl.hidden = true;
        errEl.textContent = '';
        return;
      }
      try {
        compile(v);
        inputEl.classList.remove('insp-field-invalid');
        errEl.hidden = true;
        errEl.textContent = '';
      } catch (e) {
        if (e instanceof ExprError) {
          inputEl.classList.add('insp-field-invalid');
          errEl.hidden = false;
          errEl.textContent = e.message;
        }
      }
    });
  }

  function wireCommit(inputEl, commitFn) {
    inputEl.addEventListener('change', commitFn);
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        commitFn();
      }
    });
    inputEl.addEventListener('blur', () => {
      setTimeout(() => {
        // Review fix (Critical, carried from the pre-outline inspector): if focus landed on
        // ANOTHER typing target inside rootEl (the user tabbed/clicked from this field straight
        // into the next one), a full render() here would rebuild every field element out from
        // under it a tick later, destroying its focus mid-entry. Reuses shouldSkipRender()'s own
        // typing-target arm so the two stay in lockstep. Blurring OUT of the panel entirely
        // (canvas, body, a button, ...) still renders, so the panel reconciles promptly.
        const active = document.activeElement;
        if (isTypingTarget(active) && rootEl.contains(active)) return;
        render();
      }, 0);
    });
  }

  function makeCommitter(inputEl, errEl, targetStore, buildOp) {
    return () => {
      const raw = inputEl.value;
      const err = commitFieldOp(targetStore, (m) => buildOp(m, raw));
      if (err) {
        errEl.hidden = false;
        errEl.textContent = err.message;
        inputEl.classList.add('insp-field-invalid');
      } else {
        errEl.hidden = true;
        errEl.textContent = '';
      }
    };
  }

  function fieldRow(label, inputEl) {
    const row = h('div', { class: 'insp-row' });
    const lab = h('label', { class: 'insp-row-label' }, label);
    lab.appendChild(inputEl);
    row.appendChild(lab);
    const err = h('div', { class: 'insp-field-error', hidden: '' });
    row.appendChild(err);
    return { row, err };
  }

  function appendNameField(container, targetStore, currentValue, buildOp, label = 'Name') {
    const input = h('input', { type: 'text', value: currentValue, 'aria-label': label });
    const { row, err } = fieldRow(label, input);
    wireCommit(input, makeCommitter(input, err, targetStore, buildOp));
    container.appendChild(row);
  }

  // ---------- key/value editors (payoffs, tree `with` overrides) ----------

  function kvRow({ targetStore, key, value, isNew, pathFor, setOp, isExpr, pendingRef }) {
    const row = h('div', { class: 'insp-kv-row' });
    const keyInput = h('input', {
      type: 'text', class: 'insp-kv-key', value: key,
      placeholder: isNew ? 'name' : undefined, 'aria-label': 'field name',
    });
    const valInput = h('input', {
      type: 'text', class: isExpr ? 'insp-kv-val insp-font-data' : 'insp-kv-val',
      value: value === undefined ? '' : String(value), 'aria-label': 'field value',
    });
    const err = h('div', { class: 'insp-field-error', hidden: '' });
    const rm = h('button', {
      type: 'button', class: 'insp-remove-btn', title: 'Remove', 'aria-label': `Remove ${key || 'field'}`,
    }, '−');

    function commitKey() {
      const newKey = keyInput.value.trim();
      if (!newKey) return;
      if (isNew) {
        const v = valInput.value.trim() === '' ? 0 : valInput.value;
        const e = commitFieldOp(targetStore, (m) => setOp(m, newKey, v));
        if (e) {
          err.hidden = false;
          err.textContent = e.message;
          keyInput.classList.add('insp-field-invalid');
        } else if (pendingRef) {
          pendingRef.count = Math.max(0, pendingRef.count - 1);
        }
        return;
      }
      if (newKey === key) return;
      const e = commitFieldOp(targetStore, (m) => setOp(setOp(m, key, null), newKey, valInput.value));
      if (e) {
        err.hidden = false;
        err.textContent = e.message;
        keyInput.classList.add('insp-field-invalid');
      }
    }

    function commitVal() {
      const k = isNew ? keyInput.value.trim() : key;
      if (!k) return;
      const e = commitFieldOp(targetStore, (m) => setOp(m, k, valInput.value));
      if (e) {
        err.hidden = false;
        err.textContent = e.message;
        valInput.classList.add('insp-field-invalid');
      } else {
        err.hidden = true;
        err.textContent = '';
        if (isNew && pendingRef) pendingRef.count = Math.max(0, pendingRef.count - 1);
      }
    }

    if (isExpr) wireExprInput(valInput, err);
    wireCommit(keyInput, commitKey);
    wireCommit(valInput, commitVal);

    rm.addEventListener('click', () => {
      if (!key) {
        if (pendingRef) pendingRef.count = Math.max(0, pendingRef.count - 1);
        render();
        return;
      }
      commitOp(targetStore, (m) => setOp(m, key, null));
    });

    if (!isNew) registerField(pathFor(key), err);

    row.append(keyInput, valInput, err, rm);
    return row;
  }

  function keyValueEditor({ container, targetStore, obj, pathFor, setOp, addLabel, pendingRef, isExpr, subhead }) {
    const wrap = h('div', { class: 'insp-kv' });
    if (subhead) wrap.appendChild(h('div', { class: 'insp-subhead' }, subhead));
    const list = h('div', { class: 'insp-kv-list' });
    for (const key of Object.keys(obj)) {
      list.appendChild(kvRow({ targetStore, key, value: obj[key], isNew: false, pathFor, setOp, isExpr }));
    }
    for (let i = 0; i < pendingRef.count; i += 1) {
      const row = kvRow({ targetStore, key: '', value: '', isNew: true, pathFor, setOp, isExpr, pendingRef });
      list.appendChild(row);
      pendingFocusEl = row.querySelector('.insp-kv-key');
    }
    wrap.appendChild(list);
    const addBtn = h('button', { type: 'button', class: 'insp-add-row' }, addLabel);
    addBtn.addEventListener('click', () => {
      pendingRef.count += 1;
      render();
    });
    wrap.appendChild(addBtn);
    container.appendChild(wrap);
  }

  // ---------- entity fields: state / edge / node ----------

  function renderStateFields(container, targetStore, model, name, prefix) {
    const st = model.states.find((s) => s.name === name);
    if (!st) {
      container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
      return;
    }
    appendNameField(container, targetStore, name, (m, v) => ops.renameState(m, name, v));

    // Row-level findings (E_NO_ROW / E_ROWSUM / E_TWO_RESTS) for THIS state's transitions row.
    const rowErr = h('div', { class: 'insp-field-error', hidden: '' });
    container.appendChild(rowErr);
    registerField(`${prefix}transitions.${name}`, rowErr);

    keyValueEditor({
      container, targetStore, obj: st.payoffs,
      pathFor: (k) => `${prefix}states.${name}.${k}`,
      setOp: (m, k, v) => ops.setStatePayoff(m, name, k, v),
      addLabel: '+ add payoff', pendingRef: pendingPayoffRef, isExpr: true, subhead: 'Payoffs',
    });
  }

  function renderEdgeFields(container, targetStore, model, id, prefix) {
    const row = model.transitions[id.from];
    const entry = row && row.type !== 'multinomial' ? row.to[id.to] : undefined;
    if (!entry) {
      container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
      return;
    }
    container.appendChild(h('div', { class: 'insp-subhead' }, `Transition: ${id.from} → ${id.to}`));

    {
      const input = h('input', { type: 'text', class: 'insp-font-data', value: entry.p === undefined ? '' : String(entry.p) });
      const { row: r, err } = fieldRow('p', input);
      wireExprInput(input, err);
      wireCommit(input, makeCommitter(input, err, targetStore, (m, v) => ops.setTransitionAttr(m, id.from, id.to, 'p', v)));
      registerField(`${prefix}transitions.${id.from}.${id.to}`, err);
      container.appendChild(r);
    }

    for (const key of ['cost', 'utility']) {
      const has = entry[key] !== undefined;
      const input = h('input', { type: 'text', class: 'insp-font-data', value: has ? String(entry[key]) : '' });
      const { row: r, err } = fieldRow(key === 'cost' ? 'Cost' : 'Utility', input);
      wireExprInput(input, err);
      wireCommit(input, makeCommitter(input, err, targetStore, (m, v) => ops.setTransitionAttr(m, id.from, id.to, key, v)));
      registerField(`${prefix}transitions.${id.from}.${id.to}.${key}`, err);
      if (has) {
        const rm = h('button', { type: 'button', class: 'insp-remove-btn', title: 'Remove', 'aria-label': `Remove ${key}` }, '−');
        rm.addEventListener('click', () => commitOp(targetStore, (m) => ops.setTransitionAttr(m, id.from, id.to, key, null)));
        r.querySelector('label').appendChild(rm);
      }
      container.appendChild(r);
    }
  }

  function renderNodeFields(container, targetStore, model, path, prefix) {
    let node;
    try {
      node = ops.nodeAt(model, path);
    } catch {
      container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
      return;
    }
    const checkPath = `${prefix}${nodePathToCheckPath(path)}`;
    const isRootChild = path.length === 2;

    appendNameField(container, targetStore, node.name, (m, v) => ops.renameNode(m, path, v));

    // Node-level findings (E_TREE_PSUM/E_DECISION_BELOW_ROOT for this node's OWN children, this
    // node's own p-value eval errors, E_SUBMODEL_MISSING/E_SUBMODEL_CYCLE for its `model:`
    // attachment) all share this node's bare check-path in check.js.
    const nodeErr = h('div', { class: 'insp-field-error', hidden: '' });
    container.appendChild(nodeErr);
    registerField(checkPath, nodeErr);

    keyValueEditor({
      container, targetStore, obj: node.payoffs,
      pathFor: (k) => `${checkPath}.${k}`,
      setOp: (m, k, v) => ops.setNodePayoff(m, path, k, v),
      addLabel: '+ add payoff', pendingRef: pendingPayoffRef, isExpr: true, subhead: 'Payoffs',
    });

    if (!isRootChild) {
      const input = h('input', { type: 'text', class: 'insp-font-data', value: node.p === undefined ? '' : String(node.p) });
      const { row, err } = fieldRow('p', input);
      wireExprInput(input, err);
      wireCommit(input, makeCommitter(input, err, targetStore, (m, v) => ops.setNodeAttr(m, path, 'p', v)));
      container.appendChild(row);
    } else {
      container.appendChild(h('p', { class: 'insp-hint' }, 'Strategies are entered unconditionally (no p).'));
    }

    {
      const input = h('input', { type: 'text', value: node.delay === undefined ? '' : String(node.delay) });
      const { row, err } = fieldRow('Delay (e.g. "1 year", "6 months")', input);
      wireCommit(input, makeCommitter(input, err, targetStore, (m, v) => (
        v.trim() === '' ? ops.setNodeAttr(m, path, 'delay', null) : ops.setNodeAttr(m, path, 'delay', v)
      )));
      container.appendChild(row);
    }

    {
      const input = h('input', { type: 'text', value: node.model ?? '' });
      const { row, err } = fieldRow('Sub-model', input);
      wireCommit(input, makeCommitter(input, err, targetStore, (m, v) => (
        v.trim() === '' ? ops.setNodeAttr(m, path, 'model', null) : ops.setNodeAttr(m, path, 'model', v)
      )));
      container.appendChild(row);
    }

    keyValueEditor({
      container, targetStore, obj: node.with ?? {},
      pathFor: (k) => `${checkPath}.with.${k}`,
      setOp: (m, k, v) => ops.setWith(m, path, k, v),
      addLabel: '+ add with', pendingRef: pendingWithRef, isExpr: true, subhead: 'With (sub-model overrides)',
    });
  }

  // ---------- entity fields: parameter (rewritten — was a <tr> of 7 <td>s, now vertical fieldRows) ----------

  function appendParamFields(container, name, spec, targetStore) {
    appendNameField(container, targetStore, name, (m, v) => ops.renameParam(m, name, v));
    if (name === pendingParamFocusName) {
      pendingParamFocusName = null;
      const nameInput = container.querySelector('input[aria-label="Name"]');
      if (nameInput) pendingFocusEl = nameInput;
    }

    for (const [field, label] of PARAM_FIELDS) {
      const isExpr = field !== 'source';
      const val = spec[field];
      const input = h('input', {
        type: 'text', class: isExpr ? 'insp-font-data' : '',
        value: val === undefined ? '' : String(val), 'aria-label': `${label} for ${name}`,
      });
      const { row, err } = fieldRow(label, input);
      if (isExpr) wireExprInput(input, err);
      wireCommit(input, makeCommitter(input, err, targetStore, (m, raw) => {
        const v = raw.trim() === '' ? null : raw;
        return ops.setParam(m, name, field, v);
      }));
      // check.js only ever emits param findings at .value/.dist (js/analysis/check.js's tryEval
      // calls) — never at .low/.high, so registering those two as inline finding slots would be
      // dead: they could never receive a match from splitFindings.
      if (field === 'value' || field === 'dist') registerField(`params.${name}.${field}`, err);
      container.appendChild(row);
    }

    const delBtn = h('button', { type: 'button', class: 'otl-delete-btn' }, `Delete parameter "${name}"`);
    delBtn.addEventListener('click', () => commitOp(targetStore, (m) => ops.deleteParam(m, name)));
    container.appendChild(delBtn);
  }

  function buildAddParamButton() {
    const btn = h('button', { type: 'button', class: 'insp-add-row otl-add-param' }, '+ Add parameter');
    btn.addEventListener('click', () => {
      const before = new Set(store.get().model.params.keys());
      const e = commitOp(store, (m) => ops.addParam(m));
      if (!e) {
        const after = store.get().model.params;
        const added = [...after.keys()].find((k) => !before.has(k));
        if (added) {
          pendingParamFocusName = added;
          // Selecting the new param makes its row "the expanded one" so its fields actually
          // render — this call re-enters render() synchronously (store.select -> notify ->
          // onStoreChange -> render(), not skipped: no typing target has focus mid-button-click),
          // which is what lets appendParamFields see pendingParamFocusName and stash the Name
          // input into pendingFocusEl before THIS render() call returns.
          store.select({ kind: 'param', id: added, modelPath: [] });
        }
      }
    });
    return btn;
  }

  // ---------- settings fields (always top-level; there is no per-setting row, only the one
  // "Settings" group row — see the module doc's "Expansion" section) ----------

  function appendNumberField(container, label, currentValue, keyPath, { nullable = false } = {}) {
    const input = h('input', {
      type: 'text', class: 'insp-font-data',
      value: currentValue === null || currentValue === undefined ? '' : String(currentValue),
      'aria-label': label,
    });
    const { row, err } = fieldRow(label, input);
    wireCommit(input, makeCommitter(input, err, store, (m, raw) => {
      const t = raw.trim();
      if (t === '') {
        if (!nullable) throw new Error(`${label}: must not be empty`);
        return ops.setSetting(m, keyPath, null);
      }
      const n = Number(t);
      if (!Number.isFinite(n)) throw new Error(`${label}: must be a number (got '${raw}')`);
      return ops.setSetting(m, keyPath, n);
    }));
    registerField(`settings.${keyPath}`, err);
    container.appendChild(row);
  }

  function appendTextSettingField(container, label, currentValue, keyPath) {
    const input = h('input', { type: 'text', value: currentValue ?? '', 'aria-label': label });
    const { row, err } = fieldRow(label, input);
    wireCommit(input, makeCommitter(input, err, store, (m, raw) => ops.setSetting(m, keyPath, raw)));
    registerField(`settings.${keyPath}`, err);
    container.appendChild(row);
  }

  function renderSettingsFieldsInto(container, topModel) {
    appendNameField(container, store, topModel.name, (m, v) => setNameOp(m, v));

    container.appendChild(h('div', { class: 'insp-row' },
      h('span', { class: 'insp-row-label' }, 'Type'),
      h('span', { class: 'insp-type-badge' }, topModel.type.toUpperCase()),
    ));

    appendNumberField(container, 'Cycles', topModel.settings.cycles, 'cycles', { nullable: true });
    appendTextSettingField(
      container, 'Cycle length (e.g. "1 year", "6 months")',
      formatCycle(topModel.settings.cycleYears),
      'cycle',
    );
    appendNumberField(container, 'Discount: cost', topModel.settings.discount.cost, 'discount.cost');
    appendNumberField(container, 'Discount: effect', topModel.settings.discount.effect, 'discount.effect');

    {
      const select = h('select', { 'aria-label': 'Correction' });
      for (const opt of ['half-cycle', 'life-table', 'none']) {
        select.appendChild(h('option', { value: opt, selected: topModel.settings.correction === opt ? '' : undefined }, opt));
      }
      const { row, err } = fieldRow('Correction', select);
      select.addEventListener('change', makeCommitter(select, err, store, (m, v) => ops.setSetting(m, 'correction', v)));
      container.appendChild(row);
    }

    appendNumberField(container, 'Willingness to pay (wtp)', topModel.settings.wtp, 'wtp', { nullable: true });
    appendNumberField(container, 'Age', topModel.settings.age, 'age', { nullable: true });
    appendNumberField(container, 'PSA: n', topModel.settings.psa.n, 'psa.n');
    appendNumberField(container, 'PSA: seed', topModel.settings.psa.seed, 'psa.seed');

    if (topModel.type === 'markov') {
      const startKeys = Object.keys(topModel.settings.start ?? {});
      const currentStart = startKeys.length === 1 ? startKeys[0] : '';
      const select = h('select', { 'aria-label': 'Start state' });
      select.appendChild(h('option', { value: '', selected: currentStart === '' ? '' : undefined }, '(none)'));
      for (const s of topModel.states) {
        select.appendChild(h('option', { value: s.name, selected: s.name === currentStart ? '' : undefined }, s.name));
      }
      const { row, err } = fieldRow('Start state', select);
      if (startKeys.length > 1) {
        row.appendChild(h('div', { class: 'insp-hint' }, 'Multiple start states are set — edit the distribution via the YAML pane.'));
      }
      select.addEventListener('change', () => {
        const v = select.value;
        const e = commitFieldOp(store, (m) => ops.setSetting(m, 'start', v === '' ? null : { [v]: 1 }));
        if (e) {
          err.hidden = false;
          err.textContent = e.message;
        } else {
          err.hidden = true;
          err.textContent = '';
        }
      });
      registerField('settings.start', err);
      container.appendChild(row);
    }
  }

  // ---------- row fields dispatcher ----------

  function buildRowFields(container, row, topModel) {
    if (row.kind === 'state' || row.kind === 'edge' || row.kind === 'node') {
      const modelPath = row.sel.modelPath ?? [];
      const targetStore = scopedStoreFor(store, modelPath);
      const scopeModel = targetStore.get().model;
      const prefix = scopePrefix(modelPath);
      if (!scopeModel) {
        container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
        return;
      }
      if (row.kind === 'state') renderStateFields(container, targetStore, scopeModel, row.sel.id, prefix);
      else if (row.kind === 'edge') renderEdgeFields(container, targetStore, scopeModel, row.sel.id, prefix);
      else renderNodeFields(container, targetStore, scopeModel, row.sel.id, prefix);
    } else if (row.kind === 'param') {
      const spec = topModel.params.get(row.label);
      if (!spec) return; // deleted out from under this render pass shouldn't happen — buildOutline just ran against this same model
      appendParamFields(container, row.label, spec, store);
    } else if (row.id === 'group:settings') {
      renderSettingsFieldsInto(container, topModel);
    }
  }

  // ---------- findings: row dots, group-header counts, inline field messages, residual list ----------
  //
  // paintFindings() is a DOM PATCH ONLY — it never calls render(), never rebuilds a row, never
  // touches the visible row SET. It only flips hidden/class/textContent/title on elements that
  // already exist (rowElements, fieldSlots, modelFindingsListEl), all captured at the last
  // render(). This is the one rule the whole task turns on: a debounced check() firing mid-keystroke
  // must never steal focus, and calling render() from here would do exactly that by a different
  // door (see the module doc's render-discipline section). Called from two places: at the end of
  // render() itself (so a freshly built row list shows correct dots/counts immediately, not blank
  // for up to 300ms), and from scheduleFindingsCheck()'s timeout (so a debounced model change gets
  // reflected even when shouldSkipRender() skipped the structural render that would otherwise show
  // it).
  //
  // Dot vs. count badge: EVERY row (group or not) can carry `.otl-dot` OR `.otl-count` — outlineRow()
  // below decides which element a given row gets, never both, so there's exactly one thing to patch
  // per row. Non-group rows get `.otl-dot`: attachFindings' `byRow` map assigns each finding to the
  // single most-specific row that owns it (js/ui/outline/build.js's longest-checkPath-match rule),
  // so a dot means "this exact row has an own finding" — colored --danger if any of them is an
  // error, --warn otherwise, `title` the joined messages (verbatim per the brief). GROUP headers
  // (including `group:settings`, whose one row is also where settings-scoped findings land — there
  // is no per-setting row to carry a dot instead) get `.otl-count` in place of a dot: attachFindings'
  // `counts` map already sums a row's OWN findings together with every descendant's (see its own
  // doc in build.js), so for a leaf group like Settings the badge alone already reflects its direct
  // findings — no separate dot needed or shown for any group row. `counts` also has non-zero entries
  // for ordinary structural rows with children (a state's edges, a tree node's descendants) — those
  // are deliberately never shown as a badge (the brief's own wording: "counts... on group headers"),
  // only used for the onlyFindings composition below.
  function paintFindings() {
    const { byRow, counts, residual } = attachFindings(allRows, latestFindings);

    for (const [id, el] of rowElements) {
      const dot = el.querySelector('.otl-dot');
      if (dot) {
        const findings = byRow.get(id);
        if (findings && findings.length) {
          const anyError = findings.some((f) => f.level === 'error');
          dot.hidden = false;
          dot.classList.toggle('otl-dot-error', anyError);
          dot.classList.toggle('otl-dot-warn', !anyError);
          dot.title = findings.map((f) => f.message).join(' · ');
        } else {
          dot.hidden = true;
          dot.classList.remove('otl-dot-error', 'otl-dot-warn');
          dot.removeAttribute('title');
        }
        continue;
      }
      const countEl = el.querySelector('.otl-count');
      if (!countEl) continue; // defensive; every row has exactly one of the two, per outlineRow()
      const c = counts.get(id);
      countEl.replaceChildren();
      if (c) {
        countEl.hidden = false;
        if (c.errors > 0) countEl.appendChild(h('span', { class: 'insp-badge insp-badge-error' }, `${c.errors} error${c.errors === 1 ? '' : 's'}`));
        if (c.warnings > 0) countEl.appendChild(h('span', { class: 'insp-badge insp-badge-warn' }, `${c.warnings} warning${c.warnings === 1 ? '' : 's'}`));
      } else {
        countEl.hidden = true;
      }
    }

    // Inline field messages — predates the outline (see splitFindings' own doc above): a finding
    // whose path exactly matches a check-path currently backing a RENDERED field (fieldSlots, only
    // populated for the one expanded row's fields, or Settings') shows directly beneath that field,
    // in addition to (never instead of) the dot on its row.
    const renderedPaths = new Set(fieldSlots.keys());
    const { inline } = splitFindings(latestFindings, renderedPaths);
    for (const [path, errEl] of fieldSlots) {
      const msgs = inline.get(path);
      if (msgs && msgs.length) {
        errEl.hidden = false;
        errEl.textContent = msgs.map((m) => m.message).join(' · ');
        const anyError = msgs.some((m) => m.level === 'error');
        errEl.classList.toggle('insp-lvl-error', anyError);
        errEl.classList.toggle('insp-lvl-warn', !anyError);
      } else {
        errEl.hidden = true;
        errEl.textContent = '';
        errEl.classList.remove('insp-lvl-error', 'insp-lvl-warn');
      }
    }

    // Residual — findings matching NO row at all (attachFindings' own contract: never swallowed).
    // Pinned at the bottom of the outline, independent of collapse/filter/onlyFindings state — a
    // finding with nowhere else to go must stay visible regardless of what the row list is doing.
    if (modelFindingsListEl) {
      modelFindingsListEl.replaceChildren();
      for (const f of residual) {
        modelFindingsListEl.appendChild(h('li', { class: f.level === 'error' ? 'insp-lvl-error' : 'insp-lvl-warn' },
          h('code', {}, f.path || '(model)'), ` — ${f.message}`));
      }
      if (modelFindingsWrapEl) modelFindingsWrapEl.hidden = residual.length === 0;
    }
  }

  // Debounced 300ms, mirroring results.js's own scheduleValidationBadge() exactly (same interval,
  // same "coalesce rapid store notifications" motivation) — but a SEPARATE timer/state, per that
  // module's own "results.js stays self-contained" ruling (and symmetrically here: this module never
  // reads results.js's validationFindings either).
  function scheduleFindingsCheck() {
    if (findingsTimer) clearTimeout(findingsTimer);
    findingsTimer = setTimeout(() => {
      findingsTimer = null;
      const m = store.get().model;
      latestFindings = m ? check(m) : [];
      paintFindings();
    }, 300);
  }

  // ---------- outline chrome: filter bar, group collapse, row list ----------

  function persistOutlineState() {
    const blob = loadLayout();
    saveLayout({ ...blob, outline: { collapsed: [...collapsedGroups], filter: filterQuery } });
  }

  function toggleCollapsed(id) {
    if (collapsedGroups.has(id)) collapsedGroups.delete(id);
    else collapsedGroups.add(id);
    persistOutlineState();
    render();
  }

  function buildFilterBar() {
    const bar = h('div', { class: 'otl-filterbar' });
    const filterInput = h('input', {
      type: 'search', id: 'otl-filter', class: 'otl-filter-input',
      placeholder: 'Filter…', 'aria-label': 'Filter outline', value: filterQuery,
    });
    filterInput.addEventListener('input', () => {
      filterQuery = filterInput.value;
      persistOutlineState();
      render(); // local UI state, not a store change -- bypasses shouldSkipRender entirely, on purpose
    });
    const onlyBtn = h('button', {
      type: 'button', class: 'otl-only-findings', 'aria-pressed': String(onlyFindings),
      title: 'Show only rows with findings',
    }, 'Only findings');
    onlyBtn.addEventListener('click', () => {
      onlyFindings = !onlyFindings;
      render();
    });
    bar.append(filterInput, onlyBtn);
    return bar;
  }

  // One row. Indent comes from `depth` via a custom property, so nesting needs no wrapper elements
  // and the whole list stays a flat sequence of siblings — which is what keeps scroll restoration
  // and findings patching simple. Every row gets exactly ONE of `.otl-dot` (a single finding-colored
  // dot, for any NON-group row with its own findings) or `.otl-count` (a rolled-up errors/warnings
  // badge, EVERY group row, including `group:settings` — see paintFindings' own doc for why a leaf
  // group's badge alone already covers its direct findings, no dot needed) — never both, so
  // paintFindings() has exactly one thing to patch per row. Both start hidden/empty; paintFindings()
  // (called at the end of every render() and again 300ms after every store change) fills them in
  // place, never rebuilding a row.
  function outlineRow(row, { expanded, hasChildren, collapsed }) {
    const btn = h('button', {
      type: 'button', class: 'otl-row', id: `otl-${row.id}`,
      'data-kind': row.kind, style: `--depth:${row.depth}`,
      'aria-expanded': hasChildren ? String(!collapsed) : undefined,
      'aria-current': expanded ? 'true' : undefined,
    });
    btn.append(
      h('span', { class: 'otl-twisty' }, hasChildren ? (collapsed ? '▸' : '▾') : ''),
      h('span', { class: 'otl-label' }, row.label),
      h('span', { class: 'otl-detail' }, row.detail),
      row.kind === 'group' ? h('span', { class: 'otl-count', hidden: '' }) : h('span', { class: 'otl-dot', hidden: '' }),
    );
    btn.addEventListener('click', () => {
      if (row.kind === 'group') { toggleCollapsed(row.id); return; }
      if (row.kind === 'submodel') { openScope([row.label]); return; }
      if (row.sel) { openScope(row.sel.modelPath ?? []); store.select(row.sel); return; }
      if (row.kind === 'param') store.select({ kind: 'param', id: row.label, modelPath: [] });
    });
    return btn;
  }

  // ---------- top-level render ----------

  function render() {
    // Captured BEFORE replaceChildren() destroys whatever currently has focus — restored after,
    // below. See the module doc's render-discipline section for why this exists.
    const scrollTop = rootEl.scrollTop;
    const active = document.activeElement;
    const preserveFilterFocus = !!(active && active.id === 'otl-filter' && rootEl.contains(active));
    const filterSelStart = preserveFilterFocus ? active.selectionStart : null;
    const filterSelEnd = preserveFilterFocus ? active.selectionEnd : null;

    const state = store.get();
    renderedSelection = state.selection ?? { kind: null, id: null };

    const selKey = JSON.stringify(state.selection ?? null);
    if (selKey !== lastSelKey) {
      pendingPayoffRef = { count: 0 };
      pendingWithRef = { count: 0 };
      lastSelKey = selKey;
    }

    fieldSlots = new Map();
    rowElements = new Map();
    selectedFieldsEl = null;
    modelFindingsListEl = null;
    modelFindingsWrapEl = null;
    allRows = [];
    rootEl.replaceChildren();

    const topModel = state.model;
    if (!topModel) {
      rootEl.appendChild(h('p', { class: 'insp-empty' }, 'No model loaded.'));
      return;
    }

    rootEl.appendChild(buildFilterBar());
    const list = h('div', { class: 'otl-list' });
    rootEl.appendChild(list);

    allRows = buildOutline(topModel);
    const filtered = filterRows(allRows, filterQuery);
    // onlyFindings composition (per the controller's "things the brief cannot know" note): applied
    // AFTER filterRows, never before — a row survives onlyFindings when IT ITSELF has a finding OR
    // it's an ancestor of one. attachFindings' `counts` (js/ui/outline/build.js) already rolls every
    // descendant's findings up into each ancestor (group headers AND ordinary structural parents
    // alike — a state with an erroring edge gets a non-zero count too, even though only group
    // headers ever SHOW a badge for it, see outlineRow's doc), so "has a non-zero count" is exactly
    // "has a finding or is an ancestor of one" — no second tree walk needed. `counts` is computed
    // against `allRows` (the FULL, unfiltered set), not `filtered` — attachFindings needs every row
    // present to resolve a finding to its true longest-checkPath match and to roll counts up
    // correctly; computing it against an already-text-filtered subset could misattribute a finding
    // to the wrong (shorter-checkPath) surviving ancestor, or roll a count into a group header that
    // the true owning row (filtered out by the text query) was actually responsible for.
    const { counts } = attachFindings(allRows, latestFindings);
    const findingsFiltered = onlyFindings ? filtered.filter((r) => counts.has(r.id)) : filtered;
    const visible = collapseFilter(findingsFiltered, collapsedGroups);
    const expandedRow = rowForSelection(allRows, state.selection);
    const expandedId = expandedRow ? expandedRow.id : null;

    const addParamAfterIndex = addAfterIndex(visible, 'group:parameters', collapsedGroups);

    visible.forEach((row, i) => {
      // Review fix (Important, Task 10 review): only GROUP headers are individually collapsible
      // (toggleCollapsed is only ever called for row.kind === 'group', below) — a markov state
      // with edges, or a non-leaf tree node, has children in the `parentId` sense but clicking it
      // always selects, never collapses. Gating hasChildren to group rows keeps the twisty/
      // aria-expanded truthful: no permanent, non-functional "expanded" affordance on the
      // majority of structural rows.
      const hasChildren = row.kind === 'group' && allRows.some((r) => r.parentId === row.id);
      const collapsed = collapsedGroups.has(row.id);
      const expanded = row.kind !== 'group' && row.id === expandedId;
      const btn = outlineRow(row, { expanded, hasChildren, collapsed });
      list.appendChild(btn);
      rowElements.set(row.id, btn);

      const showFields = row.id === 'group:settings' ? !collapsed : row.id === expandedId;
      if (showFields) {
        const fieldsEl = h('div', { class: 'otl-fields', style: `--depth:${row.depth}` });
        buildRowFields(fieldsEl, row, topModel);
        list.appendChild(fieldsEl);
        // Settings' own fields block is never "the selected entity" (rowForSelection can never
        // return the settings row — see build.js) — only mark it here for the actual selected row,
        // so shouldSkipRender's force-reconcile exception stays scoped to it (Finding 3).
        if (row.id === expandedId) selectedFieldsEl = fieldsEl;
      }

      if (i === addParamAfterIndex) list.appendChild(buildAddParamButton());
    });

    // "Model findings" — pinned at the bottom of the outline (below the row list, always present
    // in the DOM so paintFindings() can patch it without a rebuild; hidden via `hidden` whenever
    // there's nothing residual to show). Independent of the filter/onlyFindings/collapse state
    // above: a residual finding has no row to attach to at all, so nothing narrows it out of view.
    const findingsWrap = h('div', { class: 'insp-model-findings', hidden: '' });
    findingsWrap.appendChild(h('div', { class: 'insp-model-findings-head' }, 'Model findings'));
    const findingsList = h('ul', { class: 'insp-model-findings-list' });
    findingsWrap.appendChild(findingsList);
    rootEl.appendChild(findingsWrap);
    modelFindingsListEl = findingsList;
    modelFindingsWrapEl = findingsWrap;

    rootEl.scrollTop = scrollTop;
    if (preserveFilterFocus) {
      const el = rootEl.querySelector('#otl-filter');
      if (el) {
        el.focus();
        if (filterSelStart != null) el.setSelectionRange(filterSelStart, filterSelEnd);
      }
    }

    // Paints dots/counts/inline messages/residual list onto the row list just built above — a DOM
    // patch only (see paintFindings' own doc), so it's safe to call unconditionally at the end of
    // every structural render, keeping findings visible immediately rather than blank for up to
    // 300ms until the next debounced check() happens to fire.
    paintFindings();

    if (pendingFocusEl) {
      const toFocus = pendingFocusEl;
      pendingFocusEl = null;
      toFocus.focus();
    }
  }

  // Expands the selected row (uncollapsing any collapsed ancestor group above it) and scrolls it
  // into view. Exposed for results.js's Validation-tab click-through (via app.js's
  // selectOnCanvas): canvas.openScope(...) -> store.select(sel) already happened by the time this
  // is called, so the row list already reflects the new selection (unless shouldSkipRender
  // happened to skip that notification) — this call forces a render() unconditionally to make
  // sure, then scrolls.
  function revealSelection() {
    const state = store.get();
    if (!state.model) return;
    const allRows = buildOutline(state.model);
    const row = rowForSelection(allRows, state.selection);
    if (!row) return;
    const toUncollapse = ancestorIds(allRows, row).filter((id) => collapsedGroups.has(id));
    for (const id of toUncollapse) collapsedGroups.delete(id);
    if (toUncollapse.length > 0) persistOutlineState();
    render();
    const el = rowElements.get(row.id);
    if (el) el.scrollIntoView({ block: 'nearest' });
  }

  function onStoreChange() {
    // Always reschedule the debounced findings check, REGARDLESS of shouldSkipRender's decision —
    // the model may have changed even when the structural render itself is skipped (mid-typing
    // elsewhere in this panel), and paintFindings() patches dots/counts/messages in place without
    // touching row structure, so it's always safe to let it fire independently of render().
    scheduleFindingsCheck();
    if (shouldSkipRender()) return;
    render();
  }
  store.subscribe(onStoreChange);

  // Initial findings computation is synchronous (mirrors results.js's own boot-time check() call)
  // — the very first render must show correct dots/counts/residual immediately, not a blank state
  // for the first 300ms until the debounced path first fires.
  const initModel = store.get().model;
  latestFindings = initModel ? check(initModel) : [];

  render();

  return { render, revealSelection };
}
