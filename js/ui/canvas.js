// SVG canvas renderer for econeval models.
//
// Two halves, per constraints.md ("DOM modules ... hold no business logic"):
//   1. Pure geometry (exported + covered by test/canvas-model.test.js): NODE_R, edgePath,
//      selfLoopPath, edgeLabelPos, scopedStore. None of these touch the DOM — safe to import
//      from node:test.
//   2. createCanvas(svgEl, store, {layoutFor}) — the DOM-heavy renderer. Reads store.get().model
//      (or, once a sub-model has been entered, model.models[name] via the same currentModelPath
//      chase), positions nodes via the injected layoutFor, and re-renders in full on every
//      store.subscribe notification (no vDOM — fine at this scale, per the brief).
//
// Scope (binding, task-9 brief): render + click-to-select + double-click-to-enter-a-sub-model +
// pan/zoom only. No editing gestures (drag-to-move, add/connect/delete tools) — those land in
// Task 10, which extends this same file and reuses everything below.

import { nodeAt } from './ops.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------------------------------------
// Node vocabulary sizes (design tokens doc: "markov state = circle r 26; tree decision root =
// square; chance = circle; terminal = short vertical end-bar; sub-model attachment = stadium
// double-stroke"). Exported where a consumer (Task 10's hit-testing/Add tool) plausibly needs the
// same radius used here.
// ---------------------------------------------------------------------------------------------

export const NODE_R = 26;          // markov state circle, and tree chance-node circle
const ROOT_HALF = 26;              // tree decision root: square, side = ROOT_HALF*2 (52) — same
                                    // footprint as the circle so root/chance read as one family
const TERMINAL_HALF = 15;          // tree terminal: vertical end-bar, half-length
const STADIUM_W = 120;             // tree sub-model attachment: stadium (pill) width
const STADIUM_H = 32;              // stadium height
const STADIUM_INSET = 3;           // inner stroke inset, for the "double stroke" look
const HALO_GAP = 4;                // selection halo: extra offset beyond the node's own edge
const LABEL_MAX = 14;              // edge-label truncation length (brief: "max 14 chars + ...")
const SELF_LOOP_SPREAD = Math.PI / 6;  // self-loop anchors sit +-30deg either side of top (-90deg)
const SELF_LOOP_HEIGHT = 2;            // bezier control-point offset above the anchors, as a
                                        // multiple of the node radius
const BASE_W = 900;                // default viewBox width at zoom 1
const BASE_H = 640;                // default viewBox height at zoom 1
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 2.5;

// ---------------------------------------------------------------------------------------------
// Pure geometry helpers (exported + tested).
// ---------------------------------------------------------------------------------------------

// edgePath(from, to, r) -> 'd' attribute string for a straight edge from one node's rim to the
// other's, i.e. the raw center-to-center line trimmed by `r` at BOTH ends (so it starts/ends on
// the node circles rather than at their centers, leaving room for an arrowhead marker drawn via
// marker-end on the <path> the caller builds from this string). from/to are [x, y] pairs.
export function edgePath(from, to, r) {
  const [x1, y1] = from;
  const [x2, y2] = to;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len === 0) return `M ${x1} ${y1} L ${x2} ${y2}`; // degenerate: coincident points
  const ux = dx / len;
  const uy = dy / len;
  const sx = x1 + ux * r;
  const sy = y1 + uy * r;
  const ex = x2 - ux * r;
  const ey = y2 - uy * r;
  return `M ${sx} ${sy} L ${ex} ${ey}`;
}

// selfLoopPath(xy, r) -> 'd' attribute string for a small loop that leaves the node's circle just
// above-left of top, arcs up and over, and re-enters just above-right of top — both anchor points
// lie exactly on the circle (distance r from the center), so the loop reads as attached to the
// node rather than floating near it. xy is the node's [x, y] center.
export function selfLoopPath(xy, r) {
  const [cx, cy] = xy;
  const a0 = -Math.PI / 2 - SELF_LOOP_SPREAD;
  const a1 = -Math.PI / 2 + SELF_LOOP_SPREAD;
  const x0 = cx + r * Math.cos(a0);
  const y0 = cy + r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy + r * Math.sin(a1);
  const h = r * SELF_LOOP_HEIGHT;
  return `M ${x0} ${y0} C ${x0} ${y0 - h} ${x1} ${y1 - h} ${x1} ${y1}`;
}

// Label anchor for a self-loop: above the loop's own apex, so it never overlaps the arc.
function selfLoopLabelPos(xy, r) {
  const [cx, cy] = xy;
  return [cx, cy - r - r * SELF_LOOP_HEIGHT - 6];
}

// edgeLabelPos(from, to) -> [x, y]: the midpoint of the two points, offset 10px perpendicular to
// the line — always to the same side (rotate the edge vector -90deg, i.e. (dy, -dx) normalized),
// so a label sits consistently "above" a left-to-right edge rather than flipping sides
// unpredictably depending on edge direction.
export function edgeLabelPos(from, to) {
  const [x1, y1] = from;
  const [x2, y2] = to;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const px = dy / len;
  const py = -dx / len;
  return [mx + px * 10, my + py * 10];
}

function truncateLabel(s, max = LABEL_MAX) {
  const str = String(s);
  return str.length > max ? `${str.slice(0, max - 1)}…` : str;
}

// The p-source text rule (brief, verbatim): 'rest' shown as 'rest', numbers as-is, expressions
// verbatim — then truncated. Used for both markov transition `p` and tree node `p`.
function pLabelText(p) {
  if (p === undefined || p === null) return '';
  return truncateLabel(typeof p === 'number' ? String(p) : p);
}

// ---------------------------------------------------------------------------------------------
// scopedStore(store, modelName) -> a store-shaped wrapper whose .get().model is
// store.get().model.models[modelName], and whose .applyOp maps fn over that same sub-model,
// splicing the result back into a fresh top-level model before handing it to the real store's
// applyOp (which does the actual serialize/reparse/commit/undo-snapshot work). Everything else
// (select/undo/redo/markSaved/subscribe) passes straight through — undo history, dirty state and
// selection are document-wide, not per-sub-model (constraints.md: "Undo/redo = document text
// snapshots"). Composable: scopedStore(scopedStore(store, 'a'), 'b') correctly reaches
// model.models.a.models.b, so entering nested sub-models is just chaining this wrapper once per
// currentModelPath segment (see scopedStoreFor below) — no separate multi-level-aware variant
// needed.
// ---------------------------------------------------------------------------------------------

export function scopedStore(store, modelName) {
  return {
    get() {
      const outer = store.get();
      const sub = outer.model && outer.model.models ? outer.model.models[modelName] : undefined;
      return { ...outer, model: sub ?? null };
    },
    applyOp(fn, opts) {
      store.applyOp((model) => {
        const sub = model.models && model.models[modelName];
        if (!sub) throw new Error(`scopedStore: sub-model '${modelName}' not found`);
        const newSub = fn(sub);
        return { ...model, models: { ...model.models, [modelName]: newSub } };
      }, opts);
    },
    select: (sel) => store.select(sel),
    undo: () => store.undo(),
    redo: () => store.redo(),
    markSaved: () => store.markSaved(),
    subscribe: (listener) => store.subscribe(listener),
  };
}

function scopedStoreFor(baseStore, path) {
  return path.reduce((s, name) => scopedStore(s, name), baseStore);
}

// ---------------------------------------------------------------------------------------------
// DOM-heavy renderer.
// ---------------------------------------------------------------------------------------------

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) e.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (c !== null && c !== undefined) e.appendChild(c);
  }
  return e;
}

function text(s) {
  return document.createTextNode(s);
}

function hasReward(entry) {
  return entry.cost !== undefined || entry.utility !== undefined;
}

function payoffSummary(payoffs) {
  const keys = Object.keys(payoffs || {});
  if (keys.length === 0) return '';
  return keys.map((k) => `${k} ${payoffs[k]}`).join('   ');
}

function treeNodeKind(node, path) {
  if (path.length === 1) return 'root';
  if (node.children.length === 0) return node.model ? 'submodel' : 'terminal';
  return 'chance';
}

export function createCanvas(svgEl, store, { layoutFor }) {
  const breadcrumbEl = typeof document !== 'undefined' ? document.getElementById('breadcrumb') : null;

  let tool = 'select';
  const currentModelPath = []; // names into .models, chained; [] = the top-level model itself

  const view = { x: 0, y: 0, w: BASE_W, h: BASE_H };
  let zoom = 1;
  let dragState = null;

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

  // -------- svg defs (arrowhead marker) --------

  function buildDefs() {
    const defs = el('defs');
    const marker = el('marker', {
      id: 'arrow',
      viewBox: '0 0 10 10',
      refX: '9',
      refY: '5',
      markerWidth: '7',
      markerHeight: '7',
      orient: 'auto-start-reverse',
    });
    marker.appendChild(el('path', { class: 'arrowhead', d: 'M0,0 L10,5 L0,10 Z' }));
    defs.appendChild(marker);
    return defs;
  }

  // -------- markov --------

  function renderMarkovEdge(activeStore, from, to, fromXY, toXY, label, reward, selected) {
    const g = el('g', { class: 'edge' });
    const isLoop = from === to;
    const d = isLoop ? selfLoopPath(fromXY, NODE_R) : edgePath(fromXY, toXY, NODE_R);
    if (selected) g.appendChild(el('path', { class: 'edge-halo', d }));
    g.appendChild(el('path', { class: 'edge-line', d, 'marker-end': 'url(#arrow)' }));
    if (label) {
      const [lx, ly] = isLoop ? selfLoopLabelPos(fromXY, NODE_R) : edgeLabelPos(fromXY, toXY);
      const labelText = reward ? `${label} ⊕` : label;
      g.appendChild(el('text', { class: 'edge-label', x: lx, y: ly, 'text-anchor': 'middle' }, text(labelText)));
    }
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      activeStore.select({ kind: 'edge', id: { from, to } });
    });
    return g;
  }

  function renderMarkovNode(activeStore, state, xy, isSelected) {
    const [cx, cy] = xy;
    const g = el('g', { class: 'node', 'data-kind': 'state', transform: `translate(${cx},${cy})` });
    if (isSelected) g.appendChild(el('circle', { class: 'halo', cx: 0, cy: 0, r: NODE_R + HALO_GAP }));
    g.appendChild(el('circle', { class: 'node-shape', cx: 0, cy: 0, r: NODE_R }));
    g.appendChild(el('text', {
      class: 'node-name', x: 0, y: 0, 'text-anchor': 'middle', 'dominant-baseline': 'central',
    }, text(state.name)));
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      activeStore.select({ kind: 'state', id: state.name });
    });
    return g;
  }

  function renderMarkov(activeStore, model, positions, selection, edgesG, nodesG) {
    for (const state of model.states) {
      const row = model.transitions[state.name];
      const fromXY = positions[state.name];
      if (!row || !fromXY) continue;

      if (row.type === 'multinomial') {
        const total = Object.values(row.counts).reduce((a, b) => a + b, 0);
        for (const [target, count] of Object.entries(row.counts)) {
          const toXY = positions[target];
          if (target !== state.name && !toXY) continue;
          const selected = selection?.kind === 'edge' && selection.id.from === state.name && selection.id.to === target;
          edgesG.appendChild(renderMarkovEdge(activeStore, state.name, target, fromXY, toXY, `${count}/${total}`, false, selected));
        }
      } else {
        for (const [target, entry] of Object.entries(row.to)) {
          const toXY = positions[target];
          if (target !== state.name && !toXY) continue;
          const selected = selection?.kind === 'edge' && selection.id.from === state.name && selection.id.to === target;
          edgesG.appendChild(renderMarkovEdge(activeStore, state.name, target, fromXY, toXY, pLabelText(entry.p), hasReward(entry), selected));
        }
      }
    }

    for (const state of model.states) {
      const xy = positions[state.name];
      if (!xy) continue;
      const isSelected = selection?.kind === 'state' && selection.id === state.name;
      nodesG.appendChild(renderMarkovNode(activeStore, state, xy, isSelected));
    }
  }

  // -------- tree --------

  function shapeForKind(kind) {
    switch (kind) {
      case 'root':
        return el('rect', {
          class: 'node-shape', x: -ROOT_HALF, y: -ROOT_HALF, width: ROOT_HALF * 2, height: ROOT_HALF * 2,
        });
      case 'chance':
        return el('circle', { class: 'node-shape', cx: 0, cy: 0, r: NODE_R });
      case 'terminal':
        return el('line', { class: 'terminal-bar', x1: 0, y1: -TERMINAL_HALF, x2: 0, y2: TERMINAL_HALF });
      case 'submodel': {
        const g = el('g', { class: 'submodel-shape' });
        g.appendChild(el('rect', {
          class: 'node-shape', x: -STADIUM_W / 2, y: -STADIUM_H / 2, width: STADIUM_W, height: STADIUM_H,
          rx: STADIUM_H / 2, ry: STADIUM_H / 2,
        }));
        const iw = STADIUM_W - 2 * STADIUM_INSET;
        const ih = STADIUM_H - 2 * STADIUM_INSET;
        g.appendChild(el('rect', {
          class: 'node-shape-inner', x: -iw / 2, y: -ih / 2, width: iw, height: ih, rx: ih / 2, ry: ih / 2,
        }));
        return g;
      }
      default:
        return el('circle', { class: 'node-shape', cx: 0, cy: 0, r: NODE_R });
    }
  }

  function haloForKind(kind) {
    switch (kind) {
      case 'root':
        return el('rect', {
          class: 'halo', x: -ROOT_HALF - HALO_GAP, y: -ROOT_HALF - HALO_GAP,
          width: (ROOT_HALF + HALO_GAP) * 2, height: (ROOT_HALF + HALO_GAP) * 2,
        });
      case 'terminal':
        return el('circle', { class: 'halo', cx: 0, cy: 0, r: TERMINAL_HALF + HALO_GAP });
      case 'submodel':
        return el('rect', {
          class: 'halo', x: -STADIUM_W / 2 - HALO_GAP, y: -STADIUM_H / 2 - HALO_GAP,
          width: STADIUM_W + 2 * HALO_GAP, height: STADIUM_H + 2 * HALO_GAP,
          rx: (STADIUM_H + 2 * HALO_GAP) / 2, ry: (STADIUM_H + 2 * HALO_GAP) / 2,
        });
      case 'chance':
      default:
        return el('circle', { class: 'halo', cx: 0, cy: 0, r: NODE_R + HALO_GAP });
    }
  }

  // Approximate per-shape trim radius so an edge's endpoint lands close to the node's own outline
  // rather than uniformly at NODE_R regardless of shape (a stadium is much wider than a circle;
  // a terminal bar has almost no footprint at all). edgePath's contract is a single `r` trimming
  // BOTH ends equally, so this is a deliberate approximation, not exact geometry — good enough for
  // the mostly-horizontal edges autoTree produces.
  function treeTrimRadius(kind) {
    if (kind === 'terminal') return 12;
    if (kind === 'submodel') return STADIUM_W / 2 - 10;
    return NODE_R; // root, chance
  }

  function renderTreeEdge(activeStore, parentXY, childXY, label, childPath, childKind) {
    const g = el('g', { class: 'edge' });
    const d = edgePath(parentXY, childXY, treeTrimRadius(childKind));
    g.appendChild(el('path', { class: 'edge-line', d }));
    if (label) {
      const [lx, ly] = edgeLabelPos(parentXY, childXY);
      g.appendChild(el('text', { class: 'edge-label', x: lx, y: ly, 'text-anchor': 'middle' }, text(label)));
    }
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      activeStore.select({ kind: 'node', id: childPath });
    });
    return g;
  }

  function renderTreeNode(activeStore, node, xy, kind, isSelected, path) {
    const [cx, cy] = xy;
    const g = el('g', { class: 'node', 'data-kind': kind, transform: `translate(${cx},${cy})` });
    if (isSelected) g.appendChild(haloForKind(kind));
    g.appendChild(shapeForKind(kind));

    const isBar = kind === 'terminal';
    const nameX = isBar ? TERMINAL_HALF - 4 : 0;
    const anchor = isBar ? 'start' : 'middle';
    const nameText = kind === 'submodel' ? `model: ${node.model}` : node.name;
    g.appendChild(el('text', {
      class: 'node-name', x: nameX, y: isBar ? -4 : 0, 'text-anchor': anchor,
      'dominant-baseline': isBar ? undefined : 'central',
    }, text(nameText)));

    const payoffText = payoffSummary(node.payoffs);
    if (payoffText) {
      g.appendChild(el('text', {
        class: 'node-payoff', x: nameX, y: isBar ? 12 : NODE_R + 14, 'text-anchor': anchor,
      }, text(payoffText)));
    }

    g.addEventListener('click', (e) => {
      e.stopPropagation();
      activeStore.select({ kind: 'node', id: path });
    });
    if (kind === 'submodel') {
      g.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        currentModelPath.push(node.model);
        render();
      });
    }
    return g;
  }

  function renderTree(activeStore, model, positions, selection, edgesG, nodesG) {
    let selectedNode = null;
    if (selection?.kind === 'node') {
      try {
        selectedNode = nodeAt(model, selection.id);
      } catch {
        selectedNode = null; // selection belongs to a different (sub-)model than the one shown
      }
    }

    function walk(node, path, parentXY) {
      const key = path.join('/');
      const xy = positions[key];
      if (!xy) return;

      const kind = treeNodeKind(node, path);

      if (parentXY) {
        const isRootChild = path.length === 2; // strategy branch: never carries 'p' (ops.js rule)
        const label = isRootChild ? '' : pLabelText(node.p);
        edgesG.appendChild(renderTreeEdge(activeStore, parentXY, xy, label, path, kind));
      }

      nodesG.appendChild(renderTreeNode(activeStore, node, xy, kind, node === selectedNode, path));

      for (const child of node.children) walk(child, [...path, child.name], xy);
    }

    walk(model.tree, [model.tree.name], null);
  }

  // -------- top-level render --------

  function buildSvg(model) {
    svgEl.replaceChildren();
    svgEl.appendChild(buildDefs());
    const edgesG = el('g', { class: 'edges' });
    const nodesG = el('g', { class: 'nodes' });
    svgEl.appendChild(edgesG);
    svgEl.appendChild(nodesG);

    const positions = layoutFor(model);
    const selection = store.get().selection;
    const activeStore = scopedStoreFor(store, currentModelPath);

    if (model.type === 'markov') renderMarkov(activeStore, model, positions, selection, edgesG, nodesG);
    else if (model.type === 'tree') renderTree(activeStore, model, positions, selection, edgesG, nodesG);
  }

  function render() {
    const top = store.get().model;
    if (!top) {
      svgEl.replaceChildren();
      renderBreadcrumb();
      return;
    }
    let model = resolveActiveModel();
    if (!model) {
      currentModelPath.length = 0; // the path no longer resolves (e.g. edited away) — bail to main
      model = top;
    }
    renderBreadcrumb();
    buildSvg(model);
  }

  // -------- pan (background drag, select tool only) + wheel zoom to cursor --------

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

  svgEl.addEventListener('pointerdown', (e) => {
    if (e.target !== svgEl) return; // only the bare background, not a node/edge
    dragState = { startX: e.clientX, startY: e.clientY, startViewX: view.x, startViewY: view.y, moved: false, pointerId: e.pointerId };
    try { svgEl.setPointerCapture(e.pointerId); } catch { /* not supported in this environment */ }
  });

  svgEl.addEventListener('pointermove', (e) => {
    if (!dragState) return;
    const dxPx = e.clientX - dragState.startX;
    const dyPx = e.clientY - dragState.startY;
    if (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3) dragState.moved = true;
    if (tool === 'select') {
      const rect = svgEl.getBoundingClientRect();
      const scaleX = rect.width ? view.w / rect.width : 1;
      const scaleY = rect.height ? view.h / rect.height : 1;
      view.x = dragState.startViewX - dxPx * scaleX;
      view.y = dragState.startViewY - dyPx * scaleY;
      applyViewBox();
    }
  });

  function endDrag(e) {
    if (!dragState) return;
    const wasClick = !dragState.moved;
    try { svgEl.releasePointerCapture(dragState.pointerId); } catch { /* already released */ }
    dragState = null;
    // click on empty canvas clears selection (routed through whichever scope is active)
    if (wasClick) scopedStoreFor(store, currentModelPath).select({ kind: null, id: null });
  }
  svgEl.addEventListener('pointerup', (e) => endDrag(e));
  svgEl.addEventListener('pointercancel', () => { dragState = null; });

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
    setTool(t) { tool = t; },
    currentModelPath,
  };
}
