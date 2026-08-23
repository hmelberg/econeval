// Inspector: Selection / Parameters / Settings tabs.
//
// createInspector(rootEl, tabsEl, store, {flush}) -> {render()}
//   rootEl:  #inspector-body — the tab CONTENT goes here (rebuilt on every structural render()).
//   tabsEl:  #inspector-tabs — panels.js's own .panel-ctl (maximize/minimize) span already lives
//            here (see js/ui/panels.js's initPanels comment); this module appends its own tab
//            strip as a SIBLING wrapper (never touches/clears panels.js's span), inserted as the
//            first child so the tab buttons sit left of the panel-ctl icons regardless of which
//            module wires up first.
//   flush:   optional () => void, called before EVERY op commit (Task 12 passes sync.flush — a
//            canvas/inspector edit must never race a pending debounced YAML-pane edit).
//
// Scope (controller ruling): the Selection tab operates on the model named by
// store.get().selection.modelPath, built via the exported scopedStore chain from canvas.js
// (empty/absent modelPath = top-level). Parameters and Settings ALWAYS operate top-level in v1;
// when a sub-model selection is active they show a muted hint pointing at the YAML pane instead.
//
// Findings: check(model) (always against the TOP-LEVEL model — findings are whole-document) runs
// debounced 300ms after every store change. The tab strip carries an error/warning-count badge
// updated by a lightweight DOM patch (never a structural rebuild, so it can never steal focus);
// a finding whose `path` matches one of the fields the active tab currently renders is shown
// inline under that field, everything else is listed under "Model findings" at the tab's bottom.
//
// Render discipline: a full render() rebuilds every field element, which would steal focus/cursor
// out from under a field the user is actively editing. The store.subscribe handler
// (shouldSkipRender) skips the full rebuild whenever focus is on a real input/select inside
// rootEl, REGARDLESS of what triggered the store change — our own field commit, a canvas gesture
// (e.g. clicking a different node with the Select tool while mid-typing elsewhere), a YAML-pane
// edit, undo/redo. The one exception: on the Selection tab, if the entity the focused field
// belongs to (captured as `renderedSelection` at the top of the last render()) is no longer
// resolvable against the fresh model — it was deleted out from under the user, not just
// deselected — the skip is lifted and render() proceeds immediately, since there's nothing
// sensible left to keep showing. `committingSelf` (true only for the synchronous span of a field
// commit, never a button click) always forces the skip on its own — most usefully for an in-place
// rename, which can otherwise make the OLD selection.id briefly unresolvable the instant the
// commit lands. Every field also schedules its own reconciling render() on blur (deferred one
// tick, after the browser has settled the blur/focus transition) so nothing goes stale for long.

import { compile, ExprError } from '../core/expr.js';
import { check } from '../analysis/check.js';
import { scopedStore } from './canvas.js';
import { loadLayout, saveLayout } from './panels.js';
import * as ops from './ops.js';

// ================================================================================================
// ---------- Pure helpers (exported + tested in test/inspector-match.test.js) ----------
// ================================================================================================

// scopePrefix(['a','b']) -> 'models.a.models.b.' — matches checkModelContent's own recursive
// pathPrefix building (js/analysis/check.js) exactly, since selection.modelPath segments are
// `models:` registry keys chained the same way (canvas.js's currentModelPath/scopedStore).
export function scopePrefix(modelPath) {
  return (modelPath ?? []).map((name) => `models.${name}.`).join('');
}

// check.js's tree-content paths start at 'tree' and OMIT the root node's own name (walkTreeNode is
// first called with path=`${pathPrefix}tree` for the root itself, then `${path}.${child.name}` for
// each descendant) — different from ops.nodeAt's path convention (root name included at path[0])
// and from the layout-key convention (root name included, '/'-joined). This is the one place those
// three conventions actually diverge; converts an ops-style node path to check.js's convention.
export function nodePathToCheckPath(path) {
  const rest = (path ?? []).slice(1);
  return rest.length ? `tree.${rest.join('.')}` : 'tree';
}

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

function cssEscape(s) {
  if (typeof CSS !== 'undefined' && CSS.escape) return CSS.escape(s);
  return String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => `\\${c}`);
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

const TABS = [
  ['selection', 'Selection'],
  ['parameters', 'Parameters'],
  ['settings', 'Settings'],
];

// ================================================================================================
// ---------- createInspector ----------
// ================================================================================================

export function createInspector(rootEl, tabsEl, store, { flush = () => {} } = {}) {
  let activeTab = loadLayout().tab;
  let committingSelf = false;
  let latestFindings = [];
  let findingsTimer = null;
  let fieldSlots = new Map(); // check-path -> error <div> element, from the last structural render
  let modelFindingsListEl = null;
  let modelFindingsWrapEl = null;
  let badgeEl = null;
  let pendingFocusEl = null; // a just-added blank kv-row's key input, focused at the end of render()
  let pendingPayoffRef = { count: 0 };
  let pendingWithRef = { count: 0 };
  let lastSelKey = null;
  let renderedSelection = { kind: null, id: null }; // the selection the CURRENTLY DISPLAYED Selection-tab fields were built from, set at the top of every render()

  // ---------- op commit plumbing ----------

  function scopedTarget(modelPath) {
    return (modelPath ?? []).reduce((s, name) => scopedStore(s, name), store);
  }

  // Re-checks whether `sel` still resolves against the CURRENT model — the same rule store.js's
  // private isSelectionValid applies (duplicated here in miniature since store.js doesn't export
  // it; it's small, pure, and read-only). Used by the render-skip decision below: a selection with
  // kind:null is trivially "resolvable" (nothing to orphan).
  function selectionResolvable(sel) {
    if (!sel || sel.kind == null) return true;
    const scopeModel = scopedTarget(sel.modelPath).get().model;
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
  //   - Not our own commit AND on the Selection tab AND the entity the focused field belongs to
  //     (`renderedSelection`, captured at the top of the last render()) is no longer resolvable —
  //     it was deleted out from under the user (Delete tool, undo, a YAML edit...), so there is
  //     nothing sensible left to keep showing; reconcile immediately instead of leaving an orphaned
  //     field on screen. Reselecting a DIFFERENT-but-still-existing entity (e.g. clicking another
  //     canvas node) does NOT trigger this — the field being edited is untouched, just no longer
  //     the active selection, so it's left alone until its own blur reconciles it.
  //   - Focus isn't a typing target inside rootEl at all — nothing to protect.
  // Parameters/Settings fields are never selection-entity-bound (ruling 1: always top-level), so
  // for those two tabs a non-own-commit change is always skipped while typing — there is no
  // "field may no longer exist" risk to check there.
  function shouldSkipRender() {
    const active = document.activeElement;
    if (!isTypingTarget(active) || !rootEl.contains(active)) return false;
    if (committingSelf) return true;
    if (activeTab !== 'selection') return true;
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
        // Review fix (Critical): if focus landed on ANOTHER typing target inside rootEl (the user
        // tabbed/clicked from this field straight into the next one), a full render() here would
        // rebuild every field element out from under it a tick later, destroying its focus (falls
        // to body) mid-entry. Reuses shouldSkipRender()'s own typing-target arm so the two stay in
        // lockstep. Blurring OUT of the panel entirely (canvas, body, a button, ...) still renders,
        // so the panel reconciles promptly as before.
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

  // ---------- Selection tab: state / edge / node ----------

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

  // Review fix (Important): the canvas is the only pre-existing way to create a selection — a
  // keyboard-only or pointer-free user had no way to reach the Selection tab's fields at all.
  // Small, deliberately simple addition: in the empty state, a labeled <select> listing every
  // markov state / tree node (path, root inclusive — same entities a canvas click can select) of
  // the CURRENT TOP-LEVEL model; choosing one calls store.select with modelPath: [] (this picker
  // only ever addresses the top-level model, matching the "current top-level model" brief — a
  // sub-model selection is still made by drilling in via canvas, same as today).
  //
  // Walks a tree collecting every node's path (root inclusive, in document order) — the exact
  // path convention ops.nodeAt/canvas.js's own nodeIndex use (array of names from the root).
  function collectTreePaths(node, path, out) {
    out.push(path);
    for (const child of node.children) collectTreePaths(child, [...path, child.name], out);
  }

  function renderSelectionPicker(container, topModel) {
    const options = []; // { label, sel } in <select> order
    if (topModel.type === 'markov') {
      for (const s of topModel.states) {
        options.push({ label: s.name, sel: { kind: 'state', id: s.name, modelPath: [] } });
      }
    } else if (topModel.type === 'tree' && topModel.tree) {
      const paths = [];
      collectTreePaths(topModel.tree, [topModel.tree.name], paths);
      for (const path of paths) {
        options.push({ label: path.join(' → '), sel: { kind: 'node', id: path, modelPath: [] } });
      }
    }
    if (options.length === 0) return; // nothing to offer (e.g. a markov model with zero states)

    const select = h('select', { 'aria-label': 'Select on canvas, or choose here' },
      h('option', { value: '' }, '(choose)'),
      ...options.map((o, i) => h('option', { value: String(i) }, o.label)),
    );
    select.addEventListener('change', () => {
      if (select.value === '') return;
      store.select(options[Number(select.value)].sel);
      // Structural change (empty state -> a whole different set of fields), not a value tweak on
      // an existing field — the native <select> already shows the new value on its own, but
      // nothing else re-renders on its behalf. onStoreChange's own shouldSkipRender() would in
      // fact SKIP this notification (a <select> is a typing target, and the OLD (empty) selection
      // is trivially "resolvable"), so render() is called directly here rather than relying on it.
      render();
    });
    const { row } = fieldRow('Select on canvas, or choose here:', select);
    container.appendChild(row);
  }

  function appendScopeHint(container, modelPath) {
    if (!modelPath || modelPath.length === 0) return;
    const label = modelPath.join(' → ');
    container.appendChild(h('p', { class: 'insp-hint' },
      `The current selection is inside sub-model "${label}". Parameters and Settings here are ` +
      'always top-level — edit sub-model parameters/settings via the YAML pane.'));
  }

  function renderSelectionTab(container, state, topModel) {
    const selection = state.selection ?? { kind: null, id: null };
    if (!selection.kind) {
      container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
      renderSelectionPicker(container, topModel);
      renderModelFindingsSection(container);
      return;
    }
    const modelPath = selection.modelPath ?? [];
    const targetStore = scopedTarget(modelPath);
    const scopeModel = targetStore.get().model;
    const prefix = scopePrefix(modelPath);

    if (!scopeModel) {
      container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
      renderModelFindingsSection(container);
      return;
    }

    if (selection.kind === 'state') renderStateFields(container, targetStore, scopeModel, selection.id, prefix);
    else if (selection.kind === 'edge') renderEdgeFields(container, targetStore, scopeModel, selection.id, prefix);
    else if (selection.kind === 'node') renderNodeFields(container, targetStore, scopeModel, selection.id, prefix);
    else if (selection.kind === 'param') {
      container.appendChild(h('p', { class: 'insp-empty' }, 'This is a parameter — edit it on the Parameters tab.'));
    } else {
      container.appendChild(h('p', { class: 'insp-empty' }, 'Select a state, branch, or transition on the canvas.'));
    }

    renderModelFindingsSection(container);
  }

  // ---------- Parameters tab ----------

  function paramRow(name, spec) {
    const tr = h('tr', { 'data-param': name });

    const nameInput = h('input', { type: 'text', class: 'insp-name-input', value: name, 'aria-label': 'Parameter name' });
    const nameErr = h('div', { class: 'insp-field-error', hidden: '' });
    wireCommit(nameInput, () => {
      const v = nameInput.value.trim();
      if (!v || v === name) return;
      const e = commitFieldOp(store, (m) => ops.renameParam(m, name, v));
      if (e) {
        nameErr.hidden = false;
        nameErr.textContent = e.message;
        nameInput.classList.add('insp-field-invalid');
      }
    });
    const nameTd = h('td', {}, nameInput, nameErr);

    function cell(field, isExpr) {
      const val = spec[field];
      const input = h('input', {
        type: 'text', class: isExpr ? 'insp-font-data' : '',
        value: val === undefined ? '' : String(val), 'aria-label': `${field} for ${name}`,
      });
      const err = h('div', { class: 'insp-field-error', hidden: '' });
      if (isExpr) wireExprInput(input, err);
      wireCommit(input, makeCommitter(input, err, store, (m, raw) => {
        const v = raw.trim() === '' ? null : raw;
        return ops.setParam(m, name, field, v);
      }));
      if (isExpr) registerField(`params.${name}.${field}`, err);
      return h('td', {}, input, err);
    }

    const delBtn = h('button', {
      type: 'button', class: 'insp-remove-btn', title: 'Delete parameter', 'aria-label': `Delete parameter ${name}`,
    }, '×');
    delBtn.addEventListener('click', () => commitOp(store, (m) => ops.deleteParam(m, name)));

    tr.append(nameTd, cell('value', true), cell('low', true), cell('high', true), cell('dist', true), cell('source', false), h('td', { class: 'insp-col-del' }, delBtn));
    return tr;
  }

  function renderParametersTab(container, state, topModel) {
    appendScopeHint(container, state.selection?.modelPath);

    const table = h('table', { class: 'insp-table' });
    table.appendChild(h('thead', {}, h('tr', {},
      h('th', {}, 'Name'), h('th', {}, 'Value'), h('th', {}, 'Low'), h('th', {}, 'High'),
      h('th', {}, 'Dist'), h('th', {}, 'Source'), h('th', { class: 'insp-col-del' }, ''),
    )));
    const tbody = h('tbody');
    for (const [name, spec] of topModel.params) tbody.appendChild(paramRow(name, spec));
    table.appendChild(tbody);
    container.appendChild(table);

    const addBtn = h('button', { type: 'button', class: 'insp-add-row' }, '+ add parameter');
    addBtn.addEventListener('click', () => {
      const before = new Set(topModel.params.keys());
      const e = commitOp(store, (m) => ops.addParam(m));
      if (!e) {
        const after = store.get().model.params;
        const added = [...after.keys()].find((k) => !before.has(k));
        if (added) {
          const input = rootEl.querySelector(`[data-param="${cssEscape(added)}"] .insp-name-input`);
          if (input) input.focus();
        }
      }
    });
    container.appendChild(addBtn);

    renderModelFindingsSection(container);
  }

  // ---------- Settings tab ----------

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

  function renderSettingsTab(container, state, topModel) {
    appendScopeHint(container, state.selection?.modelPath);

    appendNameField(container, store, topModel.name, (m, v) => setNameOp(m, v));

    container.appendChild(h('div', { class: 'insp-row' },
      h('span', { class: 'insp-row-label' }, 'Type'),
      h('span', { class: 'insp-type-badge' }, topModel.type.toUpperCase()),
    ));

    appendNumberField(container, 'Cycles', topModel.settings.cycles, 'cycles', { nullable: true });
    appendTextSettingField(
      container, 'Cycle length (e.g. "1 year", "6 months")',
      topModel.settings.cycleYears === 1 ? '1 year' : `${topModel.settings.cycleYears} year`,
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

    renderModelFindingsSection(container);
  }

  // ---------- findings: badge + inline + "Model findings" ----------

  function renderModelFindingsSection(container) {
    const wrap = h('div', { class: 'insp-model-findings' });
    wrap.appendChild(h('div', { class: 'insp-model-findings-head' }, 'Model findings'));
    const list = h('ul', { class: 'insp-model-findings-list' });
    wrap.appendChild(list);
    container.appendChild(wrap);
    modelFindingsListEl = list;
    modelFindingsWrapEl = wrap;
  }

  function updateBadge() {
    if (!badgeEl) return;
    badgeEl.replaceChildren();
    const { errors, warnings } = countByLevel(latestFindings);
    if (errors > 0) {
      badgeEl.appendChild(h('span', { class: 'insp-badge insp-badge-error' }, `${errors} error${errors === 1 ? '' : 's'}`));
    }
    if (warnings > 0) {
      badgeEl.appendChild(h('span', { class: 'insp-badge insp-badge-warn' }, `${warnings} warning${warnings === 1 ? '' : 's'}`));
    }
  }

  function applyFindingsToDom() {
    updateBadge();
    const renderedPaths = new Set(fieldSlots.keys());
    const { inline, rest } = splitFindings(latestFindings, renderedPaths);
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
    if (modelFindingsListEl) {
      modelFindingsListEl.replaceChildren();
      for (const f of rest) {
        modelFindingsListEl.appendChild(h('li', { class: f.level === 'error' ? 'insp-lvl-error' : 'insp-lvl-warn' },
          h('code', {}, f.path || '(model)'), ` — ${f.message}`));
      }
      if (modelFindingsWrapEl) modelFindingsWrapEl.hidden = rest.length === 0;
    }
  }

  function scheduleFindingsCheck() {
    if (findingsTimer) clearTimeout(findingsTimer);
    findingsTimer = setTimeout(() => {
      findingsTimer = null;
      const m = store.get().model;
      latestFindings = m ? check(m) : [];
      applyFindingsToDom();
    }, 300);
  }

  // ---------- tab strip ----------

  function setActiveTab(tab) {
    if (tab === activeTab) return;
    activeTab = tab;
    const blob = loadLayout();
    saveLayout({ ...blob, tab });
    render();
  }

  function renderTabStrip() {
    let mine = tabsEl.querySelector('#insp-tabs-own');
    if (!mine) {
      mine = h('div', { id: 'insp-tabs-own', class: 'insp-tabstrip', role: 'tablist', 'aria-label': 'Inspector tabs' });
      tabsEl.insertBefore(mine, tabsEl.firstChild);
    }
    mine.replaceChildren();
    for (const [key, label] of TABS) {
      const btn = h('button', {
        type: 'button', class: 'insp-tab', role: 'tab', id: `insp-tab-${key}`,
        'aria-selected': String(activeTab === key),
      }, label);
      btn.addEventListener('click', () => setActiveTab(key));
      mine.appendChild(btn);
    }
    badgeEl = h('span', { class: 'insp-findings-badge' });
    mine.appendChild(badgeEl);
    updateBadge();
  }

  // ---------- top-level render ----------

  function render() {
    const state = store.get();
    renderedSelection = state.selection ?? { kind: null, id: null };

    const selKey = JSON.stringify(state.selection ?? null);
    if (selKey !== lastSelKey) {
      pendingPayoffRef = { count: 0 };
      pendingWithRef = { count: 0 };
      lastSelKey = selKey;
    }

    fieldSlots = new Map();
    renderTabStrip();

    rootEl.replaceChildren();
    const topModel = state.model;
    if (!topModel) {
      rootEl.appendChild(h('p', { class: 'insp-empty' }, 'No model loaded.'));
    } else if (activeTab === 'selection') {
      renderSelectionTab(rootEl, state, topModel);
    } else if (activeTab === 'parameters') {
      renderParametersTab(rootEl, state, topModel);
    } else {
      renderSettingsTab(rootEl, state, topModel);
    }

    applyFindingsToDom();

    if (pendingFocusEl) {
      const toFocus = pendingFocusEl;
      pendingFocusEl = null;
      toFocus.focus();
    }
  }

  function onStoreChange() {
    scheduleFindingsCheck();
    if (shouldSkipRender()) return;
    render();
  }
  store.subscribe(onStoreChange);

  const initModel = store.get().model;
  latestFindings = initModel ? check(initModel) : [];
  render();

  return { render };
}
