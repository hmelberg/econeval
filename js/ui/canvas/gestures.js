// The modeless gesture state machine — Task 5. Replaces the old 4-tool toolbar entirely: ONE
// pointer gesture, captured on svgEl, decides its own meaning at drop time (pan / node move / node
// connect / click-select / double-click-create), per docs/superpowers/specs/
// 2026-08-24-econeval-editor-design.md §1. index.js still owns the store, the view (pan/zoom
// state), scope (currentModelPath) and rename; everything this module needs from index.js arrives
// as a callback (see createGestures's options below) — this module never reads the store or the
// DOM directly beyond svgEl (plus the Space latch's own document/window listeners, called out
// explicitly below).
//
// createGestures(svgEl, {
//   getNodeIndex, getModel, getActiveStore, clientToUser, flush, runOp,
//   render, showToast, panBy, startRename, enterSubModel, selectTarget, getScopedSelection,
//   openContextMenu,
// }) -> { handlers, cancelGesture, destroy }
//   handlers: {onNodePointerDown(e, id), onEdgePointerDown(e, target)} — the exact shape
//   render.js's buildSvg expects. index.js passes this object straight through as buildSvg's
//   `handlers` on every render, so a node/edge's OWN pointerdown listener (attached by render.js,
//   which calls e.stopPropagation()) still routes into this module's gesture machinery.
//   cancelGesture(): aborts any in-flight gesture without committing anything — index.js's Escape
//   handler calls this. destroy(): removes every listener this module registered.
//   `openContextMenu` has no implementation until Task 8 — defaulted to a no-op here so this
//   task's app is complete on its own; nothing calls it yet (Task 8 also wires the `contextmenu`
//   listener, so the whole menu lands in one reviewable diff).
//
// The real node NEVER moves during a drag (index.js's header + the design spec's §1.1): a
// translucent clone (`.drag-ghost`) follows the cursor instead, so switching between the "move"
// and "connect" readings is a crossfade between two overlays, never the real node snapping back.

import { pickNode, isInside, snapToGrid, edgePath } from './geometry.js';
import { el, treeTrimRadius } from './render.js';
import { addState, addTransition, addChild, moveNode, setLayout, nodeAt } from '../ops.js';

const DOUBLE_CLICK_MS = 400;

// Small, deliberately duplicated from index.js: the Space latch is one of the few things this
// module touches on `document` directly (see the header comment above and the brief's explicit
// "owns... the Space keydown/keyup latch"), so it needs its own copy of this guard rather than a
// callback thread just for this.
function isTypingTarget(t) {
  if (!t) return false;
  const tag = t.tagName ? String(t.tagName).toLowerCase() : '';
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  return !!t.isContentEditable;
}

function sameNodeId(a, b) {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'state') return a.key === b.key;
  return arraysEqualPath(a.path, b.path);
}

function arraysEqualPath(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

export function createGestures(svgEl, opts) {
  const {
    getNodeIndex, getModel, getActiveStore, clientToUser, flush, runOp,
    render, showToast, panBy, startRename, enterSubModel, selectTarget, getScopedSelection,
    // Accepted (defaulted to a no-op) for the options contract's sake, per the brief — Task 8
    // wires both the `contextmenu` listener and this handler; nothing calls it in this task.
    openContextMenu = () => {},
  } = opts;

  let gesture = null;    // the in-flight pointer gesture (background/edge/node), or null
  let lastDown = null;   // {kind, key|path, time} (node) — hand-rolled double-click detection,
                          // checked/set at pointerDOWN (see onNodePointerDown); background
                          // double-clicks are tracked separately (lastBackgroundClick, below)
  let lastBackgroundClick = null; // {time} — background's own double-click timer (checked/set at
                                   // POINTERUP, unlike a node's, which is checked/set at pointerdown
                                   // — see the brief: a plain background click must deselect
                                   // immediately, so there is nothing to defer to pointerdown time)
  let spaceHeld = false;

  // Every point downstream needs BOTH forms at once: geometry.js's pickNode/isInside/snapToGrid
  // index a point as a [x, y] array (clientToUser returns an {x, y} object — see index.js's header
  // for the NaN trap that bit once already), while this module's own drop-decision arithmetic
  // (grabDX/DY offsets, the metaKey-round branches) reads cur.x/cur.y. Building one hybrid value —
  // an array that also carries its own .x/.y properties — satisfies both call shapes from a single
  // conversion per event, so nothing downstream has to convert again.
  function toPoint(e) {
    const { x, y } = clientToUser(e.clientX, e.clientY);
    const p = [x, y];
    p.x = x;
    p.y = y;
    return p;
  }

  function isNodeTarget(target) {
    return !!target && !('variant' in target);
  }

  function capture(e) {
    try { svgEl.setPointerCapture(e.pointerId); } catch { /* not supported in this environment */ }
  }

  function release(g) {
    try { svgEl.releasePointerCapture(g.pointerId); } catch { /* already released */ }
  }

  function makeGesture(e, target, extra = {}) {
    return {
      target,
      startClientX: e.clientX,
      startClientY: e.clientY,
      moved: false,
      leftSource: false,
      pointerId: e.pointerId,
      ghostNodeEl: null,
      ghostEdgeEl: null,
      grabDX: 0,
      grabDY: 0,
      ringEl: null,          // the node currently wearing .drop-ring, if any
      lastClientX: e.clientX, // background pan only: last pointermove position (panBy is incremental)
      lastClientY: e.clientY,
      ...extra,
    };
  }

  // -------- ghost overlays (node gestures only) --------

  function clearRing(g) {
    if (g.ringEl) {
      g.ringEl.classList.remove('drop-ring');
      g.ringEl = null;
    }
  }

  function setRing(g, target) {
    const el2 = target ? target.el : null;
    if (g.ringEl === el2) return;
    clearRing(g);
    if (el2) {
      el2.classList.add('drop-ring');
      g.ringEl = el2;
    }
  }

  function removeGhosts(g) {
    if (g.ghostNodeEl) { try { g.ghostNodeEl.remove(); } catch { /* already detached */ } }
    if (g.ghostEdgeEl) { try { g.ghostEdgeEl.remove(); } catch { /* already detached */ } }
    clearRing(g);
  }

  // -------- node/edge/background pointerdown entry points --------
  // onNodePointerDown/onEdgePointerDown are exposed (via `handlers`, below) to render.js's buildSvg
  // exactly as index.js's own handlers used to be; the background pointerdown listener is wired
  // directly onto svgEl further down.

  function startNodeGesture(e, entry) {
    const cur = toPoint(e);
    const ghostNodeEl = entry.el.cloneNode(true);
    ghostNodeEl.classList.add('drag-ghost');
    ghostNodeEl.style.display = 'none';
    svgEl.appendChild(ghostNodeEl);
    const ghostEdgeEl = el('path', { class: 'ghost-edge', d: edgePath(entry.xy, entry.xy, 0) });
    ghostEdgeEl.style.display = 'none';
    svgEl.appendChild(ghostEdgeEl);
    gesture = makeGesture(e, entry, {
      ghostNodeEl,
      ghostEdgeEl,
      grabDX: entry.xy[0] - cur.x,
      grabDY: entry.xy[1] - cur.y,
    });
    capture(e);
  }

  function onNodePointerDown(e, id) {
    flush(); // controller ruling: sync any pending debounced YAML edit before this gesture starts
    const fresh = getNodeIndex().find((n) => sameNodeId(n, id));
    if (!fresh) return; // vanished under us (e.g. the flush() above triggered an edit that removed it)
    const now = Date.now();
    const isDbl = !!(lastDown && sameNodeId(lastDown, id) && (now - lastDown.time) < DOUBLE_CLICK_MS);
    lastDown = isDbl ? null : { ...id, time: now };
    if (isDbl) {
      if (fresh.kind === 'node' && fresh.treeKind === 'submodel') { enterSubModel(fresh); return; }
      startRename(fresh);
      return;
    }
    startNodeGesture(e, fresh);
  }

  function onEdgePointerDown(e, target) {
    flush(); // controller ruling: sync any pending debounced YAML edit before this gesture starts
    gesture = makeGesture(e, target);
    capture(e);
  }

  svgEl.addEventListener('pointerdown', onBackgroundPointerDown);
  function onBackgroundPointerDown(e) {
    if (e.target !== svgEl) return; // nodes/edges stopPropagation() their own pointerdown
    flush(); // controller ruling: sync any pending debounced YAML edit before this gesture starts
    gesture = makeGesture(e, null);
    capture(e);
  }

  // -------- pointermove: pan (background) / move-vs-connect preview (node) --------

  svgEl.addEventListener('pointermove', onPointerMove);
  function onPointerMove(e) {
    if (!gesture) return;
    const g = gesture;
    const dxPx = e.clientX - g.startClientX;
    const dyPx = e.clientY - g.startClientY;
    if (!g.moved && (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3)) g.moved = true;

    if (!g.target) { // background: pan, incrementally, by the delta since the last move event
      panBy(e.clientX - g.lastClientX, e.clientY - g.lastClientY);
      g.lastClientX = e.clientX;
      g.lastClientY = e.clientY;
      return;
    }

    if (!isNodeTarget(g.target)) return; // edge: not draggable — nothing to preview

    const cur = toPoint(e);
    const source = g.target;
    if (!g.leftSource && !isInside(cur, source.xy, source.hit)) g.leftSource = true;

    const forceArrow = spaceHeld;
    const forceMove = e.altKey;
    const over = pickNode(cur, getNodeIndex());
    const target = over && (over !== source || g.leftSource) ? over : null;
    const connecting = !forceMove && (forceArrow || target !== null);

    if (connecting) {
      g.ghostNodeEl.style.display = 'none';
      g.ghostEdgeEl.style.display = '';
      // Same per-shape trim used to lay an edge's endpoint on a node's outline (render.js) —
      // treeTrimRadius(undefined) falls through to NODE_R, which is also the right trim for a
      // markov state (treeKind is undefined there; see render.js's own comment on that function).
      g.ghostEdgeEl.setAttribute('d', edgePath(source.xy, cur, treeTrimRadius(source.treeKind)));
      setRing(g, target);
    } else {
      g.ghostEdgeEl.style.display = 'none';
      setRing(g, null);
      const xy = [cur.x + g.grabDX, cur.y + g.grabDY];
      const at = e.metaKey ? [Math.round(xy[0]), Math.round(xy[1])] : snapToGrid(xy);
      g.ghostNodeEl.setAttribute('transform', `translate(${at[0]},${at[1]})`);
      g.ghostNodeEl.style.display = '';
    }
  }

  // -------- pointerup: commit --------

  function endNodeGesture(g, cur, e) {
    const model = getModel();
    const source = g.target;
    const over = pickNode(cur, getNodeIndex());
    const target = over && (over !== source || g.leftSource) ? over : null;
    const connecting = !e.altKey && (spaceHeld || target !== null);
    const store = getActiveStore();

    if (!g.moved) { selectTarget(source); return; }          // a plain click still just selects

    if (!connecting) {
      const key = source.kind === 'state' ? source.key : source.path.join('/');
      const xy = [cur.x + g.grabDX, cur.y + g.grabDY];
      const at = e.metaKey ? [Math.round(xy[0]), Math.round(xy[1])] : snapToGrid(xy);
      runOp(store, (m) => setLayout(m, key, at));
      return;
    }

    if (model.type === 'markov') {
      if (target) {
        runOp(store, (m) => addTransition(m, source.key, target.key));
      } else {
        // One op: add the state, place it where the pointer was released, and connect to it. Read
        // the invented name back off the model addState returns — it always pushes the new state last.
        const at = e.metaKey ? [Math.round(cur.x), Math.round(cur.y)] : snapToGrid([cur.x, cur.y]);
        runOp(store, (m) => {
          const m1 = addState(m);
          const added = m1.states[m1.states.length - 1].name;
          return addTransition(setLayout(m1, added, at), source.key, added);
        });
      }
      return;
    }

    // tree
    if (target) {
      runOp(store, (m) => moveNode(m, source.path, target.path));
    } else {
      const at = e.metaKey ? [Math.round(cur.x), Math.round(cur.y)] : snapToGrid([cur.x, cur.y]);
      runOp(store, (m) => {
        const m1 = addChild(m, source.path);
        const parent = nodeAt(m1, source.path);
        const added = parent.children[parent.children.length - 1].name;
        return setLayout(m1, [...source.path, added].join('/'), at);
      });
    }
  }

  // createAt(cur, priorSelection): the background double-click. Same shape as endNodeGesture's
  // create branches — markov is one runOp doing addState + setLayout; tree is addChild(selection)
  // + setLayout, or a toast when there is no usable selection to parent onto. `priorSelection` is
  // passed in rather than read fresh here — see the comment on lastBackgroundClick below for why.
  function createAt(cur, priorSelection) {
    const model = getModel();
    if (!model) return;
    const store = getActiveStore();

    if (model.type === 'markov') {
      const at = snapToGrid([cur.x, cur.y]);
      runOp(store, (m) => {
        const m1 = addState(m);
        const added = m1.states[m1.states.length - 1].name;
        return setLayout(m1, added, at);
      });
      return;
    }

    const sel = priorSelection;
    if (!sel || sel.kind !== 'node') {
      showToast('Select a parent node first.');
      return;
    }
    const at = snapToGrid([cur.x, cur.y]);
    runOp(store, (m) => {
      const m1 = addChild(m, sel.id);
      const parent = nodeAt(m1, sel.id);
      const added = parent.children[parent.children.length - 1].name;
      return setLayout(m1, [...sel.id, added].join('/'), at);
    });
  }

  svgEl.addEventListener('pointerup', onPointerUp);
  function onPointerUp(e) {
    if (!gesture) return;
    const g = gesture;
    gesture = null;
    release(g);
    removeGhosts(g);

    if (!g.target) { // background
      if (!g.moved) {
        const now = Date.now();
        const isDbl = !!(lastBackgroundClick && (now - lastBackgroundClick.time) < DOUBLE_CLICK_MS);
        if (isDbl) {
          // Use the selection as it stood BEFORE the first click of this pair deselected it, not
          // whatever is selected now (nothing — the first click's own pointerup, below, already
          // cleared it). Without this, "select a node, then double-click empty to add its child"
          // — the gesture table's own normal case — would always toast "Select a parent node
          // first.": the first click of every double-click deselects on its own pointerup, so by
          // the time the second click confirms the double-click, the selection is already gone.
          createAt(toPoint(e), lastBackgroundClick.priorSelection);
          lastBackgroundClick = null;
          return;
        }
        lastBackgroundClick = { time: now, priorSelection: getScopedSelection() };
        selectTarget(null); // a plain click deselects
      }
      return;
    }

    if (!isNodeTarget(g.target)) { // edge — not draggable; a plain click selects it
      if (!g.moved) selectTarget(g.target);
      return;
    }

    endNodeGesture(g, toPoint(e), e);
  }

  svgEl.addEventListener('pointercancel', onPointerCancel);
  function onPointerCancel() {
    if (gesture) removeGhosts(gesture);
    gesture = null;
    spaceHeld = false; // a lost keyup must not leave the latch stuck on for the next gesture
    render(); // discard any live-preview state by re-rendering from the model's real (unchanged) layout
  }

  // -------- Space latch: forces the connect reading for the whole gesture --------
  // Guarded by the same isTypingTarget/dialog[open] rules index.js's own keydown handler uses, and
  // claimed only while a node gesture is actually in flight — a bare Space press elsewhere (no
  // gesture, or a background/edge gesture) is left untouched, so the page still scrolls normally.

  function isNodeGestureActive() {
    return !!gesture && isNodeTarget(gesture.target);
  }

  function onKeyDown(e) {
    if (e.key !== ' ' && e.code !== 'Space') return;
    if (document.querySelector('dialog[open]')) return;
    if (isTypingTarget(e.target)) return;
    if (!isNodeGestureActive()) return;
    e.preventDefault();
    spaceHeld = true;
  }

  function onKeyUp(e) {
    if (e.key !== ' ' && e.code !== 'Space') return;
    spaceHeld = false;
  }

  function onBlur() {
    spaceHeld = false; // a lost keyup (e.g. Cmd-Tab away while holding Space) must not stick
  }

  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);

  return {
    handlers: { onNodePointerDown, onEdgePointerDown },

    // Aborts any in-flight gesture without committing anything — index.js's Escape handler calls
    // this (escapeAll: "cancelling the rename and the gesture and now also clears the selection").
    cancelGesture() {
      if (!gesture) return;
      const g = gesture;
      gesture = null;
      release(g);
      removeGhosts(g);
      render();
    },

    destroy() {
      svgEl.removeEventListener('pointerdown', onBackgroundPointerDown);
      svgEl.removeEventListener('pointermove', onPointerMove);
      svgEl.removeEventListener('pointerup', onPointerUp);
      svgEl.removeEventListener('pointercancel', onPointerCancel);
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    },
  };
}
