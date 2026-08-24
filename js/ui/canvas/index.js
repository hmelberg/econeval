// SVG canvas renderer for econeval models.
//
// Four pieces now, per constraints.md ("DOM modules ... hold no business logic") and the design
// spec's §5 module split:
//   1. Pure geometry (canvas/geometry.js, covered by test/canvas-geometry.test.js): geometry
//      helpers, shape descriptors, hit-testing, snap-to-grid, fit-box. None touch DOM.
//   2. DOM building (canvas/render.js): buildSvg(svgEl, model, {positions, selection, handlers})
//      constructs the <g> tree of nodes/edges and wires their pointerdown listeners to call back
//      into the `handlers` this module passes in — it never touches the store or applies an op.
//   3. The gesture state machine (canvas/gestures.js, Task 5): pan / node move / node connect /
//      click-select / double-click-create, all resolved from ONE pointer gesture captured on
//      svgEl, decided at drop time rather than by a globally-selected tool. Owns the Space latch.
// This file (createCanvas(svgEl, store, {layoutFor, flush})) is the fourth piece: store/selection
// resolution, scope/breadcrumb, the view (pan/zoom state), inline rename, keyboard shortcuts
// (Escape/Delete/Backspace) and the toast strip — wiring gestures.js's callbacks to the store and
// passing its `handlers` straight into buildSvg. Reads store.get().model (or, once a sub-model has
// been entered, model.models[name] via the same currentModelPath chase), positions nodes via the
// injected layoutFor, and re-renders in full on every store.subscribe notification (no vDOM — fine
// at this scale, per the brief).
//
// Scope: Task 9 delivered render + click-to-select + double-click-to-enter-a-sub-model + pan/zoom.
// Task 10 added editing via a 4-tool toolbar (Select/Add/Connect/Delete) and Task 4 split the file
// in two (this module + render.js), replacing the old scalar `hitR` with `hit` shape descriptors
// from geometry.hitShapeFor. Task 5 deleted the toolbar entirely (modes were friction — see the
// design spec's "What and why") and moved the gesture logic into canvas/gestures.js.

import {
  renameState, deleteState, deleteTransition, renameNode, deleteNode, setLayout, clearLayout,
} from '../ops.js';

import { BASE_W, BASE_H, GRID, fitBox } from './geometry.js';
import { el, buildSvg } from './render.js';
import { createGestures } from './gestures.js';

import { scopedStoreFor } from '../scoped-store.js';

const ZOOM_MIN = 0.5;
const ZOOM_MAX = 2.5;

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

// createCanvas(svgEl, store, opts) options contract (Task 10 addition, binding — Task 12 consumes
// this literally):
//   layoutFor: (model) -> {key: [x,y]}   — required, unchanged from Task 9.
//   flush: () => void                    — optional, defaults to a no-op. Controller ruling:
//     every gesture that ends up calling store.applyOp (directly, or via the scoped store) must
//     call flush() FIRST — Task 12 passes sync.flush so a pending debounced YAML edit is
//     committed into the store before a canvas gesture reads/mutates the model, so the gesture
//     never operates on a stale model and never gets silently overwritten by a debounce that
//     fires moments later. Called at the top of every node/edge/background pointerdown handler
//     (the start of a pointer gesture) and at the top of the Delete/Backspace key handler and the
//     rename-commit handler (the two op-producing gestures that don't begin with a pointerdown).
export function createCanvas(svgEl, store, { layoutFor, flush = () => {} }) {
  const breadcrumbEl = typeof document !== 'undefined' ? document.getElementById('breadcrumb') : null;
  const toastEl = typeof document !== 'undefined' ? document.getElementById('canvas-toast') : null;
  const toolbarEl = typeof document !== 'undefined' ? document.getElementById('canvas-toolbar') : null;

  const currentModelPath = []; // names into .models, chained; [] = the top-level model itself

  const view = { x: 0, y: 0, w: BASE_W, h: BASE_H };
  let zoom = 1;
  let nodeIndex = [];       // rebuilt every render (buildSvg's return value): [{kind, key|path, xy,
                             // hit, el, ...}] — used for hit-testing (drop-target resolution, via
                             // gestures.js's getNodeIndex callback) and for re-resolving a node's
                             // CURRENT element/position at gesture-start (after flush() may have
                             // re-rendered) rather than trusting a stale closure reference.
  let activeRename = null;  // { fo, input, target, currentName } while an inline rename is open
  let toastTimer = null;

  function applyViewBox() {
    svgEl.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);
  }
  applyViewBox();

  // -------- model/store resolution for the currently-entered scope --------

  function resolveActiveModel() {
    let m = store.get().model;
    for (const name of currentModelPath) {
      if (!m || !m.models || !(name in m.models)) return null;
      m = m.models[name];
    }
    return m;
  }

  function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
    return true;
  }

  // selection.modelPath is stamped by scopedStore.select (js/ui/scoped-store.js) with the exact
  // chain of sub-model names the selection was made through; [] / undefined means the top-level
  // model. A selection only renders a halo when it matches the scope CURRENTLY on screen —
  // otherwise an unrelated sub-model's selection would incorrectly highlight something in the
  // wrong view (or nothing at all, if the id happens not to resolve here).
  function sameModelPath(a, b) {
    return arraysEqual(a ?? [], b ?? []);
  }

  // Item 4 (final-review, Validation click-through): jumps the canvas straight to the scope a
  // finding's modelPath names — the same currentModelPath + render() the breadcrumb pop-to and the
  // double-click sub-model drill-in (enterSubModel, below, called from gestures.js) already use,
  // just settable to an arbitrary depth in one call instead of one level at a time. app.js's
  // selectOnCanvas calls this BEFORE store.select(sel), so the halo-matching sameModelPath check
  // above sees the right scope by the time the resulting store notification re-renders.
  function openScope(modelPath) {
    const path = modelPath ?? [];
    if (sameModelPath(currentModelPath, path)) return;
    currentModelPath.length = 0;
    currentModelPath.push(...path);
    render();
  }

  // -------- toast strip (transient op-error messages; "errors surfaced, never swallowed") --------

  function showToast(message) {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.hidden = false;
    if (toastTimer !== null) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, 3500);
  }

  function runOp(activeStore, fn) {
    try {
      activeStore.applyOp(fn);
    } catch (err) {
      showToast(err && err.message ? err.message : String(err));
    }
  }

  // -------- breadcrumb --------

  function renderBreadcrumb() {
    if (!breadcrumbEl) return;
    breadcrumbEl.replaceChildren();
    if (currentModelPath.length === 0) {
      breadcrumbEl.hidden = true;
      return;
    }
    breadcrumbEl.hidden = false;
    const segs = ['main', ...currentModelPath];
    segs.forEach((label, i) => {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'sep';
        sep.textContent = '/';
        breadcrumbEl.appendChild(sep);
      }
      const seg = document.createElement('span');
      seg.className = 'crumb';
      seg.textContent = label;
      seg.tabIndex = 0;
      seg.setAttribute('role', 'button');
      const popTo = () => {
        if (currentModelPath.length === i) return; // already at this depth
        currentModelPath.length = i;
        render();
      };
      seg.addEventListener('click', popTo);
      seg.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          popTo();
        }
      });
      breadcrumbEl.appendChild(seg);
    });
  }

  // -------- escape: cancel rename, ELSE cancel gesture, ELSE clear selection (design spec's
  // gesture table: "cancel gesture / rename, else deselect" — priority, not all three at once.
  // Cancelling a rename on a node, or a gesture dragging one, must not also silently drop the
  // selection of that same node as a side effect.) --------

  function escapeAll() {
    if (activeRename) { cancelRename(); return; }
    if (gestures.cancelGesture()) return;
    selectTarget(null);
  }

  // -------- keyboard shortcuts (global; ignored while typing in an input/textarea/foreignObject
  // input, incl. the rename box itself — that box handles its own Enter/Escape locally) --------

  function isTypingTarget(t) {
    if (!t) return false;
    const tag = t.tagName ? String(t.tagName).toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
    return !!t.isContentEditable;
  }

  function deleteSelection() {
    const { selection } = store.get();
    if (!selection || selection.kind == null) return;
    if (!sameModelPath(selection.modelPath, currentModelPath)) return; // not visible in this scope
    flush();
    const activeStore = scopedStoreFor(store, currentModelPath);
    if (selection.kind === 'edge') {
      runOp(activeStore, (m) => deleteTransition(m, selection.id.from, selection.id.to));
    } else if (selection.kind === 'state') {
      runOp(activeStore, (m) => deleteState(m, selection.id));
    } else if (selection.kind === 'node') {
      runOp(activeStore, (m) => deleteNode(m, selection.id));
    }
  }

  // Resolves the CURRENT selection to its nodeIndex entry (the shape startRename/setLayout need:
  // {kind, key|path, xy, ...}) — an edge selection has no such entry (nodeIndex only carries
  // 'state'/'node' entries; a tree "edge" IS its child node's selection), so this returns null for
  // one, which both callers below treat as a no-op rather than an error.
  function findSelectedEntry(selection) {
    if (selection.kind === 'state') return nodeIndex.find((n) => n.kind === 'state' && n.key === selection.id);
    if (selection.kind === 'node') return nodeIndex.find((n) => n.kind === 'node' && arraysEqual(n.path, selection.id));
    return null;
  }

  // Enter: open the inline rename for the selected node. No-op for an edge selection (no node to
  // rename) or a selection not visible in the scope currently on screen.
  function renameSelection() {
    const { selection } = store.get();
    if (!selection || selection.kind == null) return;
    if (!sameModelPath(selection.modelPath, currentModelPath)) return;
    const entry = findSelectedEntry(selection);
    if (!entry) return;
    startRename(entry);
  }

  // Arrow keys: nudge the selected node by one GRID step (four with Shift) as a single setLayout
  // op — one keypress, one undo entry, the deliberate granularity for a deliberate key press. No-op
  // for an edge selection (no position) or a selection not visible in this scope.
  const NUDGE_DELTA = {
    ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0],
  };

  function nudgeSelection(key, big) {
    const { selection } = store.get();
    if (!selection || selection.kind == null) return;
    if (!sameModelPath(selection.modelPath, currentModelPath)) return;
    flush();
    const entry = findSelectedEntry(selection);
    if (!entry) return; // edge selection (no position), or vanished under the flush() above
    const step = (big ? GRID * 4 : GRID);
    const [ux, uy] = NUDGE_DELTA[key];
    const at = [entry.xy[0] + ux * step, entry.xy[1] + uy * step];
    const layoutKey = entry.kind === 'state' ? entry.key : entry.path.join('/');
    const activeStore = scopedStoreFor(store, currentModelPath);
    runOp(activeStore, (m) => setLayout(m, layoutKey, at));
  }

  if (typeof document !== 'undefined') {
    document.addEventListener('keydown', (e) => {
      // Review fix (Important): a modal <dialog> (New/Open/Examples) owns keyboard input while
      // open — without this guard, Escape's preventDefault() stopped the dialog's own native
      // close, and Delete/Backspace could delete the canvas's current selection from underneath a
      // dialog the user is filling in. Early-return before any of this handler's own key handling.
      if (document.querySelector('dialog[open]')) return;
      if (isTypingTarget(e.target)) return;
      if (e.key === 'Escape') { e.preventDefault(); escapeAll(); return; }
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteSelection(); return; }
      // Task 5 removed the V/A/C/D tool-switch arm here — the toolbar it drove no longer exists.

      // Task 6: view shortcuts, gated on meta/ctrl so they never collide with a plain keypress.
      // Every OTHER meta/ctrl chord (notably ⌘Z/⌘Y/⌘S/⌘Enter) is left completely untouched — no
      // preventDefault, no return-with-side-effect — so app.js's own window-level keydown handler
      // (registered separately; this one never calls stopPropagation) still sees it.
      if (e.metaKey || e.ctrlKey) {
        if (e.key === '0') { e.preventDefault(); fitToView(); }
        else if (e.key === '=' || e.key === '+') { e.preventDefault(); zoomAtCentre(1.1); }
        else if (e.key === '-') { e.preventDefault(); zoomAtCentre(1 / 1.1); }
        return; // whether matched above or not: a meta/ctrl chord is never also an editing key below
      }

      // Task 7: keyboard editing of the current selection — Enter renames, arrows nudge by one
      // grid step (four with Shift). Both apply only when the selection is visible in the scope
      // currently on screen (sameModelPath), the same rule deleteSelection above already uses.
      if (e.key === 'Enter') { e.preventDefault(); renameSelection(); return; }
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        nudgeSelection(e.key, e.shiftKey);
      }
    });
  }

  // -------- inline rename (SVG foreignObject over the node) --------

  function cancelRename() {
    // Review fix (Critical): null activeRename BEFORE removing the foreignObject, mirroring
    // commitRename's existing order. Removing a FOCUSED element fires 'blur' on it synchronously
    // (real browser behavior) — and the rename input's blur handler does
    // `if (activeRename) commitRename();`. With the old remove-then-null order, that handler ran
    // while activeRename was still set, re-entering commitRename() and silently COMMITTING the
    // edited text instead of cancelling it — on Escape, and on any external re-render while a
    // rename was open (render()'s own cancelRename() cleanup call runs this same path).
    const rec = activeRename;
    if (!rec) return;
    activeRename = null;
    try { rec.fo.remove(); } catch { /* already detached */ }
  }

  function commitRename() {
    const rec = activeRename;
    if (!rec) return;
    activeRename = null;
    try { rec.fo.remove(); } catch { /* already detached */ }
    const newName = rec.input.value;
    if (newName === rec.currentName) return; // no-op: nothing changed
    flush();
    const activeStore = scopedStoreFor(store, currentModelPath);
    if (rec.target.kind === 'state') runOp(activeStore, (m) => renameState(m, rec.target.key, newName));
    else runOp(activeStore, (m) => renameNode(m, rec.target.path, newName));
  }

  function startRename(target) {
    cancelRename();
    const currentName = target.kind === 'state' ? target.key : target.node.name;
    const [cx, cy] = target.xy;
    const width = 120;
    const height = 24;
    const fo = el('foreignObject', {
      x: cx - width / 2, y: cy - height / 2, width, height, class: 'rename-fo',
    });
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'rename-input';
    input.value = currentName;
    fo.appendChild(input);
    svgEl.appendChild(fo);
    activeRename = { fo, input, target, currentName };
    if (typeof input.focus === 'function') input.focus();
    if (typeof input.select === 'function') input.select();

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); commitRename(); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); cancelRename(); }
    });
    input.addEventListener('blur', () => { if (activeRename) commitRename(); });
  }

  // -------- gesture callbacks: selection, sub-model entry, scoped selection lookup, pan --------
  // These are the pieces of gesture behaviour that genuinely belong to index.js (they touch the
  // store, currentModelPath or the view) and are handed to gestures.js as callbacks — see
  // createGestures's call below. Node/edge pointerdown routing, double-click detection, the drag
  // preview and the Space latch all live in canvas/gestures.js now (Task 5).

  function selectTarget(target) {
    const activeStore = scopedStoreFor(store, currentModelPath);
    if (!target) { activeStore.select({ kind: null, id: null }); return; }
    if (target.kind === 'state') { activeStore.select({ kind: 'state', id: target.key }); return; }
    if (target.kind === 'node') { activeStore.select({ kind: 'node', id: target.path }); return; }
    if (target.variant === 'markov') {
      activeStore.select({ kind: 'edge', id: { from: target.from, to: target.to } });
      return;
    }
    activeStore.select({ kind: 'node', id: target.path }); // a tree "edge" IS its child node
  }

  function enterSubModel(target) {
    currentModelPath.push(target.node.model);
    render();
  }

  // The selection filtered to the scope the canvas is CURRENTLY showing (controller ruling — a
  // selection made inside a different sub-model, or at the top level while drilled into one, does
  // not count here): used both by render()'s halo and by gestures.js's background-double-click
  // create (a tree needs a selected node in THIS scope to parent onto).
  function scopedSelection() {
    const raw = store.get().selection;
    return sameModelPath(raw?.modelPath, currentModelPath) ? raw : null;
  }

  function panBy(dxPx, dyPx) {
    const rect = svgEl.getBoundingClientRect();
    const scaleX = rect.width ? view.w / rect.width : 1;
    const scaleY = rect.height ? view.h / rect.height : 1;
    view.x -= dxPx * scaleX;
    view.y -= dyPx * scaleY;
    applyViewBox();
  }

  // -------- top-level render (delegates DOM construction to render.js's buildSvg) --------

  function render() {
    const top = store.get().model;
    if (!top) {
      cancelRename();
      svgEl.replaceChildren();
      nodeIndex = [];
      renderBreadcrumb();
      return;
    }
    let model = resolveActiveModel();
    if (!model) {
      currentModelPath.length = 0; // the path no longer resolves (e.g. edited away) — bail to main
      model = top;
    }
    renderBreadcrumb();
    cancelRename(); // buildSvg rebuilds the whole svg subtree; tear down any open rename first —
                     // DOM cleanup that belongs here, not in render.js's pure DOM-construction.
    const positions = layoutFor(model);
    nodeIndex = buildSvg(svgEl, model, {
      positions,
      selection: scopedSelection(),
      handlers: gestures.handlers,
    });
  }

  // -------- pan/zoom to cursor, fit to view, tidy layout (Task 6) --------

  function clientToUser(clientX, clientY) {
    if (typeof svgEl.createSVGPoint !== 'function' || typeof svgEl.getScreenCTM !== 'function') {
      return { x: clientX, y: clientY };
    }
    const pt = svgEl.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svgEl.getScreenCTM();
    if (!ctm) return { x: clientX, y: clientY };
    const p = pt.matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  }

  // zoomAt: the shared zoom-toward-a-client-point primitive — pinch (ctrl/meta+wheel) and the
  // toolbar/keyboard shortcuts (which zoom toward the canvas's own centre, via zoomAtCentre below)
  // both route through this one function so their math can never drift apart.
  function zoomAt(clientX, clientY, factor) {
    const newZoom = clamp(zoom * factor, ZOOM_MIN, ZOOM_MAX);
    if (newZoom === zoom) return;
    const { x: ux, y: uy } = clientToUser(clientX, clientY);
    // Scale the CURRENT view.w/view.h, not BASE_W/BASE_H — deliberate departure from the
    // pre-Task-6 formula (which rebuilt w/h from scratch as BASE_W/newZoom, BASE_H/newZoom every
    // time). That was fine while zoomAt was the only thing that ever touched w/h, so view.w:view.h
    // never left the BASE_W:BASE_H ratio. fitToView (below) breaks that invariant on purpose — it
    // fits the viewBox tightly to the content's own bounding box, whatever its aspect ratio. Scaling
    // from BASE_W/BASE_H after a Fit snapped the aspect ratio back toward BASE_W:BASE_H on the very
    // next zoom (wheel or button), a visible jump discovered in browser verification. Scaling the
    // CURRENT w/h instead preserves whatever aspect ratio is on screen — Fit's tight fit included —
    // while `zoom` itself is still tracked the same way, so ZOOM_MIN/ZOOM_MAX stays meaningful.
    const scale = zoom / newZoom;
    const newW = view.w * scale;
    const newH = view.h * scale;
    view.x = ux - (ux - view.x) * scale;
    view.y = uy - (uy - view.y) * scale;
    view.w = newW;
    view.h = newH;
    zoom = newZoom;
    applyViewBox();
  }

  // The toolbar zoom buttons and ⌘+/⌘0/⌘- have no cursor position to zoom toward the way
  // wheel/pinch does — zoom toward the canvas's own on-screen centre instead.
  function zoomAtCentre(factor) {
    const rect = svgEl.getBoundingClientRect();
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, factor);
  }

  // Fit to view (⌘0 / the toolbar button): reframe the viewBox around every node's CURRENT
  // position in the scope on screen (fitBox's own 60px pad; an empty model falls back to the base
  // viewBox) and reset zoom to match, so the wheel/pinch zoom clamp (ZOOM_MIN/ZOOM_MAX, expressed
  // relative to BASE_W) stays meaningful immediately afterwards.
  function fitToView() {
    const box = fitBox(Object.values(layoutFor(resolveActiveModel() ?? {})), 60);
    view.x = box.x; view.y = box.y; view.w = box.w; view.h = box.h;
    zoom = BASE_W / box.w;
    applyViewBox();
  }

  // Tidy layout (the toolbar button): drop every explicit position in the current scope's
  // model.layout, handing positioning back to layouts.js's auto-layout. flush() first (op-producing
  // gesture) and one runOp = one undo entry, same rule as every other op-producing action here.
  function tidyLayout() {
    flush();
    const activeStore = scopedStoreFor(store, currentModelPath);
    runOp(activeStore, (m) => clearLayout(m));
  }

  function makeToolbarButton(id, glyph, label) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.id = id;
    btn.textContent = glyph;
    btn.setAttribute('aria-label', label);
    btn.title = label;
    return btn;
  }

  // The four view-control buttons, appended as siblings into #canvas-toolbar. panels.js's
  // initPanels() runs before createCanvas() (see js/ui/app.js) and has already written its own
  // maximize button into this same node — append only, never clear/replace it.
  if (toolbarEl) {
    const zoomOutBtn = makeToolbarButton('view-zoom-out', '−', 'Zoom out');
    const zoomInBtn = makeToolbarButton('view-zoom-in', '+', 'Zoom in');
    const fitBtn = makeToolbarButton('view-fit', '⤢', 'Fit to view');
    const tidyBtn = makeToolbarButton('view-tidy', '⌗', 'Tidy layout');
    zoomOutBtn.addEventListener('click', () => zoomAtCentre(1 / 1.1));
    zoomInBtn.addEventListener('click', () => zoomAtCentre(1.1));
    fitBtn.addEventListener('click', fitToView);
    tidyBtn.addEventListener('click', tidyLayout);
    toolbarEl.append(zoomOutBtn, zoomInBtn, fitBtn, tidyBtn);
  }

  // The gesture state machine itself (canvas/gestures.js, Task 5): owns pointerdown/move/up/cancel
  // on svgEl and the Space latch; everything it needs back from here (store access, selection,
  // scope entry, rename, panning) arrives as the callbacks below. `gestures.handlers` is what
  // render() passes into buildSvg, above, so a node/edge's own pointerdown (wired by render.js,
  // which calls e.stopPropagation()) still reaches this machinery.
  const gestures = createGestures(svgEl, {
    getNodeIndex: () => nodeIndex,
    getModel: resolveActiveModel,
    getActiveStore: () => scopedStoreFor(store, currentModelPath),
    clientToUser,
    flush,
    runOp,
    render,
    showToast,
    panBy,
    startRename,
    enterSubModel,
    selectTarget,
    getScopedSelection: scopedSelection,
  });

  // Plain wheel / two-finger trackpad scroll pans; ctrl/meta+wheel zooms to the cursor (a
  // trackpad pinch arrives as ctrl+wheel — the platform convention this flips wheel to match).
  svgEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.1 : 1 / 1.1);
      return;
    }
    const rect = svgEl.getBoundingClientRect();
    view.x += e.deltaX * (rect.width ? view.w / rect.width : 1);
    view.y += e.deltaY * (rect.height ? view.h / rect.height : 1);
    applyViewBox();
  }, { passive: false });

  store.subscribe(render);
  render();

  return {
    render,
    currentModelPath,
    openScope,
  };
}
