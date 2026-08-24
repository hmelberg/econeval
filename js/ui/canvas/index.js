// SVG canvas renderer for econeval models.
//
// Two halves, per constraints.md ("DOM modules ... hold no business logic"):
//   1. Pure geometry (canvas/geometry.js, covered by test/canvas-geometry.test.js): geometry
//      helpers, shape descriptors, hit-testing, snap-to-grid, fit-box. None touch DOM.
//   2. DOM building (canvas/render.js): buildSvg(svgEl, model, {positions, selection, handlers})
//      constructs the <g> tree of nodes/edges and wires their pointerdown listeners to call back
//      into the `handlers` this module passes in — it never touches the store or applies an op.
// This file (createCanvas(svgEl, store, {layoutFor, flush})) is the third piece (Task 4 split it
// out of the old js/ui/canvas.js, which combined all three): store/selection resolution, the
// pointer-gesture state machine (pan / node drag-to-move / connect drag), the toolbar, inline
// rename, keyboard shortcuts and the toast strip. Reads store.get().model (or, once a sub-model
// has been entered, model.models[name] via the same currentModelPath chase), positions nodes via
// the injected layoutFor, and re-renders in full on every store.subscribe notification (no vDOM —
// fine at this scale, per the brief).
//
// Scope: Task 9 delivered render + click-to-select + double-click-to-enter-a-sub-model + pan/zoom.
// Task 10 added the editing gestures: a 4-tool toolbar (Select/Add/Connect/Delete, appended into
// #canvas-toolbar — panels.js's maximize button already lives there, never replaced), node
// drag-to-move (live preview + one setLayout op on release), inline foreignObject rename, Add/
// Connect/Delete tool click & drag gestures, a transient toast strip for op errors, and the
// store's optional {flush} callback (Task 12 passes sync.flush — see the controller ruling below).
// Task 4 split the file in two (this module + render.js) and replaced the old scalar `hitR` with
// `hit` shape descriptors from geometry.hitShapeFor, so every node/edge gets a real hit area
// instead of a uniform circle (or, for edges, only the painted 1.5px stroke) — setTool/the toolbar
// itself are unchanged here and are deleted in Task 5.

import {
  addState, renameState, deleteState, addTransition, deleteTransition, setLayout,
  addChild, renameNode, deleteNode,
} from '../ops.js';

import { BASE_W, BASE_H, edgePath, pickNode } from './geometry.js';
import { el, buildSvg, treeTrimRadius } from './render.js';

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
  const toolbarEl = typeof document !== 'undefined' ? document.getElementById('canvas-toolbar') : null;
  const toastEl = typeof document !== 'undefined' ? document.getElementById('canvas-toast') : null;

  let tool = 'select';
  const currentModelPath = []; // names into .models, chained; [] = the top-level model itself

  const view = { x: 0, y: 0, w: BASE_W, h: BASE_H };
  let zoom = 1;
  let gesture = null;       // the in-flight pointer gesture (pan / node move / connect drag), or null
  let nodeIndex = [];       // rebuilt every render (buildSvg's return value): [{kind, key|path, xy,
                             // hit, el, ...}] — used for hit-testing (Connect-tool drop target) and
                             // for re-resolving a node's CURRENT element/position at gesture-start
                             // (after flush() may have re-rendered) rather than trusting a stale
                             // closure reference.
  let lastDown = null;      // { kind, key|path, time } — for hand-rolled double-click detection
                             // (self-timed rather than relying on native dblclick synthesis, which
                             // interacts unpredictably with pointer capture; see task-10 report)
  let activeRename = null;  // { fo, input, target, currentName } while an inline rename is open
  let toastTimer = null;

  function applyViewBox() {
    svgEl.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);
  }
  applyViewBox();
  svgEl.setAttribute('data-tool', tool);

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
  // double-click sub-model drill-in (handleNodeDoubleClick, below) already use, just settable to an
  // arbitrary depth in one call instead of one level at a time. app.js's selectOnCanvas calls this
  // BEFORE store.select(sel), so the halo-matching sameModelPath check above sees the right scope
  // by the time the resulting store notification re-renders.
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

  // -------- toolbar (Select/Add/Connect/Delete — appended into #canvas-toolbar; panels.js's own
  // maximize button already lives there and is never touched/replaced) --------

  const TOOL_DEFS = [
    { name: 'select', glyph: '↖', label: 'Select', key: 'V' },   // north-west arrow (pointer)
    { name: 'add', glyph: '+', label: 'Add', key: 'A' },
    { name: 'connect', glyph: '→', label: 'Connect', key: 'C' }, // rightwards arrow
    { name: 'delete', glyph: '✕', label: 'Delete', key: 'D' },   // multiplication x
  ];
  const toolButtons = {};
  if (toolbarEl) {
    for (const def of TOOL_DEFS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.id = `tool-${def.name}`;
      btn.textContent = def.glyph;
      btn.setAttribute('aria-label', def.label);
      btn.setAttribute('aria-pressed', String(def.name === tool));
      btn.title = `${def.label} (${def.key})`;
      btn.addEventListener('click', () => setTool(def.name));
      toolbarEl.appendChild(btn);
      toolButtons[def.name] = btn;
    }
  }

  function updateToolButtons() {
    for (const name of Object.keys(toolButtons)) {
      toolButtons[name].setAttribute('aria-pressed', String(tool === name));
    }
  }

  function cancelGesture() {
    if (!gesture) return;
    if (gesture.ghostEl) { try { gesture.ghostEl.remove(); } catch { /* already detached */ } }
    try { svgEl.releasePointerCapture(gesture.pointerId); } catch { /* already released */ }
    gesture = null;
    render(); // discard any live-preview transform by re-rendering from the model's real layout
  }

  function setTool(t) {
    if (tool === t) return;
    cancelGesture();
    cancelRename();
    tool = t;
    svgEl.setAttribute('data-tool', tool);
    updateToolButtons();
  }

  function escapeAll() {
    cancelRename();
    cancelGesture();
    if (tool !== 'select') {
      tool = 'select';
      svgEl.setAttribute('data-tool', tool);
      updateToolButtons();
    }
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
      if (e.metaKey || e.ctrlKey || e.altKey) return; // don't hijack browser/OS chords
      const k = e.key.toLowerCase();
      if (k === 'v') { e.preventDefault(); setTool('select'); }
      else if (k === 'a') { e.preventDefault(); setTool('add'); }
      else if (k === 'c') { e.preventDefault(); setTool('connect'); }
      else if (k === 'd') { e.preventDefault(); setTool('delete'); }
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

  // -------- node/edge identity / double-click detection / handlers passed into buildSvg --------

  const DOUBLE_CLICK_MS = 400;

  function sameNodeId(a, b) {
    if (a.kind !== b.kind) return false;
    return a.kind === 'state' ? a.key === b.key : arraysEqual(a.path, b.path);
  }

  function handleNodeDoubleClick(target) {
    if (target.kind === 'node' && target.treeKind === 'submodel') {
      currentModelPath.push(target.node.model);
      render();
      return;
    }
    if (tool !== 'select') return;
    startRename(target);
  }

  function handleNodePointerDown(e, id) {
    flush(); // controller ruling: sync any pending debounced YAML edit before this gesture starts
    const fresh = nodeIndex.find((n) => sameNodeId(n, id));
    if (!fresh) return; // vanished under us (e.g. the flush() above triggered an edit that removed it)
    const now = Date.now();
    const isDbl = !!(lastDown && sameNodeId(lastDown, id) && (now - lastDown.time) < DOUBLE_CLICK_MS);
    lastDown = isDbl ? null : { ...id, time: now };
    if (isDbl) { handleNodeDoubleClick(fresh); return; }
    startGesture(e, { domKind: 'node', ...fresh });
  }

  function handleEdgePointerDown(e, target) {
    flush(); // controller ruling: sync any pending debounced YAML edit before this gesture starts
    startGesture(e, { domKind: 'edge', ...target });
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
    // A selection only renders a halo when its modelPath matches the scope on screen right now
    // (controller ruling — see sameModelPath above): a selection made inside a different
    // sub-model, or at the top level while we're drilled into one, must not paint here.
    const rawSelection = store.get().selection;
    const selection = sameModelPath(rawSelection?.modelPath, currentModelPath) ? rawSelection : null;
    nodeIndex = buildSvg(svgEl, model, {
      positions,
      selection,
      handlers: { onNodePointerDown: handleNodePointerDown, onEdgePointerDown: handleEdgePointerDown },
    });
  }

  // -------- pan / node-move / connect-drag (one unified pointer gesture) + wheel zoom to cursor --------
  //
  // Every pointer gesture (background pan, node drag-to-move, Connect-tool drag) is captured on
  // svgEl itself (never on the individual node/edge, even though it's a node's own pointerdown
  // that STARTS a node-move/connect gesture — see startGesture) so that pointermove/pointerup
  // keep firing reliably even if the pointer leaves the node's small hit area mid-drag. Because
  // capture is always on svgEl, e.target on pointerup is svgEl regardless of what's visually
  // under the cursor — so the Connect tool's drop target is found via GEOMETRY hit-testing
  // (geometry.pickNode) against nodeIndex, never via e.target. This also sidesteps every ambiguity
  // around native click/dblclick synthesis interacting with pointer capture: nothing in this file
  // relies on a native 'click' or 'dblclick' event anywhere.

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

  function startGesture(e, target) {
    gesture = {
      target,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startViewX: view.x,
      startViewY: view.y,
      moved: false,
      pointerId: e.pointerId,
    };
    if (tool === 'connect' && target.domKind === 'node') {
      gesture.ghostEl = el('path', { class: 'ghost-edge', d: edgePath(target.xy, target.xy, 0) });
      svgEl.appendChild(gesture.ghostEl);
    }
    try { svgEl.setPointerCapture(e.pointerId); } catch { /* not supported in this environment */ }
  }

  function moveNodeEnd(activeStore, target, cur) {
    const key = target.kind === 'state' ? target.key : target.path.join('/'); // layout-key rule
    const xy = [Math.round(cur.x), Math.round(cur.y)];
    runOp(activeStore, (m) => setLayout(m, key, xy));
  }

  function addOnBackground(activeStore, cur) {
    const model = activeStore.get().model;
    if (!model || model.type !== 'markov') return; // tree: empty-space click is a no-op (brief)
    runOp(activeStore, (m) => {
      const m1 = addState(m);
      const added = m1.states[m1.states.length - 1]; // addState always pushes the new state last
      return setLayout(m1, added.name, [Math.round(cur.x), Math.round(cur.y)]);
    });
  }

  function addOnNode(activeStore, target) {
    const model = activeStore.get().model;
    if (!model || model.type !== 'tree' || target.kind !== 'node') return; // markov node click: no-op
    runOp(activeStore, (m) => addChild(m, target.path));
  }

  function connectDrop(activeStore, fromTarget, cur) {
    const model = activeStore.get().model;
    if (!model) return;
    if (model.type === 'markov') {
      const target = pickNode([cur.x, cur.y], nodeIndex); // pickNode indexes point[0]/point[1] — an
                                                            // {x,y} object here silently NaNs every
                                                            // distance and always returns null
      if (!target) return; // dropped on empty space: silently cancel (no gesture-level self-loop
                            // ban — A->A IS allowed; ops.addTransition is what actually validates it)
      runOp(activeStore, (m) => addTransition(m, fromTarget.key, target.key));
    } else if (model.type === 'tree') {
      const target = pickNode([cur.x, cur.y], nodeIndex);
      if (target) { showToast('trees are trees'); return; } // dropped on an existing node: invalid
      runOp(activeStore, (m) => addChild(m, fromTarget.path));
    }
  }

  function endGesture(e) {
    if (!gesture) return;
    const g = gesture;
    gesture = null;
    try { svgEl.releasePointerCapture(g.pointerId); } catch { /* already released */ }
    if (g.ghostEl) { try { g.ghostEl.remove(); } catch { /* already detached */ } }

    const cur = clientToUser(e.clientX, e.clientY);
    const activeStore = scopedStoreFor(store, currentModelPath);

    if (g.target.domKind === 'background') {
      if (tool === 'select' && !g.moved) activeStore.select({ kind: null, id: null });
      else if (tool === 'add' && !g.moved) addOnBackground(activeStore, cur);
      return;
    }

    if (g.target.domKind === 'edge') {
      if (g.moved) return; // edges aren't draggable; ignore anything but a plain click
      if (tool === 'select') {
        if (g.target.variant === 'markov') {
          activeStore.select({ kind: 'edge', id: { from: g.target.from, to: g.target.to } });
        } else {
          activeStore.select({ kind: 'node', id: g.target.path }); // a tree "edge" IS its child node
        }
      } else if (tool === 'delete') {
        if (g.target.variant === 'markov') runOp(activeStore, (m) => deleteTransition(m, g.target.from, g.target.to));
        else runOp(activeStore, (m) => deleteNode(m, g.target.path));
      }
      return;
    }

    if (g.target.domKind === 'node') {
      if (tool === 'select') {
        if (g.moved) moveNodeEnd(activeStore, g.target, cur);
        else {
          const sel = g.target.kind === 'state'
            ? { kind: 'state', id: g.target.key }
            : { kind: 'node', id: g.target.path };
          activeStore.select(sel);
        }
      } else if (tool === 'add' && !g.moved) {
        addOnNode(activeStore, g.target);
      } else if (tool === 'connect' && g.moved) {
        connectDrop(activeStore, g.target, cur);
      } else if (tool === 'delete' && !g.moved) {
        if (g.target.kind === 'state') runOp(activeStore, (m) => deleteState(m, g.target.key));
        else runOp(activeStore, (m) => deleteNode(m, g.target.path));
      }
    }
  }

  svgEl.addEventListener('pointerdown', (e) => {
    if (e.target !== svgEl) return; // only the bare background — nodes/edges stopPropagation()
    flush();
    startGesture(e, { domKind: 'background' });
  });

  svgEl.addEventListener('pointermove', (e) => {
    if (!gesture) return;
    const dxPx = e.clientX - gesture.startClientX;
    const dyPx = e.clientY - gesture.startClientY;
    if (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3) gesture.moved = true;

    if (gesture.target.domKind === 'background' && tool === 'select') {
      const rect = svgEl.getBoundingClientRect();
      const scaleX = rect.width ? view.w / rect.width : 1;
      const scaleY = rect.height ? view.h / rect.height : 1;
      view.x = gesture.startViewX - dxPx * scaleX;
      view.y = gesture.startViewY - dyPx * scaleY;
      applyViewBox();
      return;
    }

    if (gesture.target.domKind === 'node' && tool === 'select' && gesture.moved) {
      const cur = clientToUser(e.clientX, e.clientY);
      gesture.target.el.setAttribute('transform', `translate(${cur.x},${cur.y})`); // live preview only
      return;
    }

    if (gesture.target.domKind === 'node' && tool === 'connect' && gesture.ghostEl) {
      const cur = clientToUser(e.clientX, e.clientY);
      // Same per-shape trim used to lay an edge's endpoint on a node's outline (render.js) — the
      // ghost line is a preview of exactly that edge, so it reuses treeTrimRadius rather than the
      // (unrelated) hit-testing shape on gesture.target.hit. treeTrimRadius(undefined) falls
      // through to NODE_R, which is also the right trim for a markov state (treeKind is undefined
      // there) — see render.js's own comment on that function.
      gesture.ghostEl.setAttribute('d', edgePath(gesture.target.xy, [cur.x, cur.y], treeTrimRadius(gesture.target.treeKind)));
    }
  });

  svgEl.addEventListener('pointerup', endGesture);
  svgEl.addEventListener('pointercancel', () => {
    // Same cleanup as cancelGesture(): a cancelled gesture never reaches endGesture, so a
    // node-move's live-preview transform (set directly via setAttribute during pointermove, with
    // no setLayout op ever committed) would otherwise stay stuck on screen out of sync with the
    // model. render() discards it by rebuilding from the model's real (unchanged) layout.
    if (gesture?.ghostEl) { try { gesture.ghostEl.remove(); } catch { /* ignore */ } }
    gesture = null;
    render();
  });

  svgEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    const { x: ux, y: uy } = clientToUser(e.clientX, e.clientY);
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const newZoom = clamp(zoom * factor, ZOOM_MIN, ZOOM_MAX);
    if (newZoom === zoom) return;
    const newW = BASE_W / newZoom;
    const newH = BASE_H / newZoom;
    view.x = ux - (ux - view.x) * (newW / view.w);
    view.y = uy - (uy - view.y) * (newH / view.h);
    view.w = newW;
    view.h = newH;
    zoom = newZoom;
    applyViewBox();
  }, { passive: false });

  store.subscribe(render);
  render();

  return {
    render,
    setTool,
    currentModelPath,
    openScope,
  };
}
