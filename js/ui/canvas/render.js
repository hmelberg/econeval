// SVG renderer for econeval models — the DOM-heavy "build the tree of SVG elements" half of the
// old js/ui/canvas.js, split out in Task 4 (see js/ui/canvas/index.js's header for the other
// half and the split's rationale). Every function here only builds/wires DOM: no store access, no
// op-application. index.js resolves positions/selection from the store and drives buildSvg() with
// them already computed, passing in `handlers` so a pointerdown here calls back into index.js's
// gesture machinery instead of owning any of it (constraints.md: "DOM modules hold no business
// logic").
//
// buildSvg(svgEl, model, {positions, selection, handlers}) -> nodeIndex
//   handlers: {onNodePointerDown(e, id), onEdgePointerDown(e, target)}
//   Task 8 adds a third, onContextMenu, together with the listeners that call it — do NOT wire a
//   contextmenu listener here, so the whole context-menu feature lands in one reviewable diff
//   (controller ruling).
//   nodeIndex entry: {kind:'state'|'node', key?|path?, xy, hit, el, treeKind?, node?, state?}
//   `hit` comes from geometry.hitShapeFor(...) — a shape descriptor ({shape,r}|{shape,w,h}), NOT
//   the old scalar hitR (Task 4 replaced every hitR with this).

import {
  NODE_R, ROOT_HALF, TERMINAL_HALF, STADIUM_W, STADIUM_H, STADIUM_INSET, HALO_GAP,
  edgePath, selfLoopPath, selfLoopLabelPos, edgeLabelPos, hitShapeFor,
} from './geometry.js';
import { nodeAt } from '../ops.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

const LABEL_MAX = 14;              // edge-label truncation length (brief: "max 14 chars + ...")

export function truncateLabel(s, max = LABEL_MAX) {
  const str = String(s);
  return str.length > max ? `${str.slice(0, max - 1)}…` : str;
}

// The p-source text rule (brief, verbatim): 'rest' shown as 'rest', numbers as-is, expressions
// verbatim — then truncated. Used for both markov transition `p` and tree node `p`.
export function pLabelText(p) {
  if (p === undefined || p === null) return '';
  return truncateLabel(typeof p === 'number' ? String(p) : p);
}

export function el(tag, attrs = {}, ...children) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) e.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (c !== null && c !== undefined) e.appendChild(c);
  }
  return e;
}

export function text(s) {
  return document.createTextNode(s);
}

export function hasReward(entry) {
  return entry.cost !== undefined || entry.utility !== undefined;
}

export function payoffSummary(payoffs) {
  const keys = Object.keys(payoffs || {});
  if (keys.length === 0) return '';
  return keys.map((k) => `${k} ${payoffs[k]}`).join('   ');
}

export function treeNodeKind(node, path) {
  if (path.length === 1) return 'root';
  if (node.children.length === 0) return node.model ? 'submodel' : 'terminal';
  return 'chance';
}

// -------- svg defs (arrowhead marker) --------

export function buildDefs() {
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

export function renderMarkovEdge(from, to, fromXY, toXY, label, reward, selected, handlers) {
  const g = el('g', { class: 'edge' });
  const isLoop = from === to;
  const d = isLoop ? selfLoopPath(fromXY, NODE_R) : edgePath(fromXY, toXY, NODE_R);
  // Invisible wide twin of the visible line, drawn first (under the halo, the line and its
  // arrowhead marker). Edges are 1.5px strokes with no fill, so SVG's default visiblePainted
  // hit-testing otherwise limits clicks to the painted 1.5px stroke — css/canvas.css's .edge-hit
  // carries the real, forgiving hit area (Task 4: this is the fix for the reported "only reacts to
  // a specific area" complaint).
  g.appendChild(el('path', { class: 'edge-hit', d }));
  if (selected) g.appendChild(el('path', { class: 'edge-halo', d }));
  g.appendChild(el('path', { class: 'edge-line', d, 'marker-end': 'url(#arrow)' }));
  if (label) {
    const [lx, ly] = isLoop ? selfLoopLabelPos(fromXY, NODE_R) : edgeLabelPos(fromXY, toXY);
    const labelText = reward ? `${label} ⊕` : label;
    g.appendChild(el('text', { class: 'edge-label', x: lx, y: ly, 'text-anchor': 'middle' }, text(labelText)));
  }
  g.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    handlers.onEdgePointerDown(e, { variant: 'markov', from, to });
  });
  return g;
}

export function renderMarkovNode(state, xy, isSelected, handlers) {
  const [cx, cy] = xy;
  const g = el('g', { class: 'node', 'data-kind': 'state', transform: `translate(${cx},${cy})` });
  if (isSelected) g.appendChild(el('circle', { class: 'halo', cx: 0, cy: 0, r: NODE_R + HALO_GAP }));
  g.appendChild(el('circle', { class: 'node-shape', cx: 0, cy: 0, r: NODE_R }));
  g.appendChild(el('text', {
    class: 'node-name', x: 0, y: 0, 'text-anchor': 'middle', 'dominant-baseline': 'central',
  }, text(state.name)));
  g.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    handlers.onNodePointerDown(e, { kind: 'state', key: state.name });
  });
  return g;
}

export function renderMarkov(model, positions, selection, edgesG, nodesG, handlers, nodeIndex) {
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
        edgesG.appendChild(renderMarkovEdge(state.name, target, fromXY, toXY, `${count}/${total}`, false, selected, handlers));
      }
    } else {
      for (const [target, entry] of Object.entries(row.to)) {
        const toXY = positions[target];
        if (target !== state.name && !toXY) continue;
        const selected = selection?.kind === 'edge' && selection.id.from === state.name && selection.id.to === target;
        edgesG.appendChild(renderMarkovEdge(state.name, target, fromXY, toXY, pLabelText(entry.p), hasReward(entry), selected, handlers));
      }
    }
  }

  for (const state of model.states) {
    const xy = positions[state.name];
    if (!xy) continue;
    const isSelected = selection?.kind === 'state' && selection.id === state.name;
    const g = renderMarkovNode(state, xy, isSelected, handlers);
    nodesG.appendChild(g);
    nodeIndex.push({ kind: 'state', key: state.name, xy, hit: hitShapeFor('state'), el: g, state });
  }
}

// -------- tree --------

export function shapeForKind(kind) {
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

export function haloForKind(kind) {
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
// the mostly-horizontal edges autoTree produces. A different concern from hit-testing (which
// geometry.hitShapeFor/isInside/pickNode own) — do not fold the two together.
export function treeTrimRadius(kind) {
  if (kind === 'terminal') return 12;
  if (kind === 'submodel') return STADIUM_W / 2 - 10;
  return NODE_R; // root, chance — and, called with an undefined kind, markov states too
}

export function renderTreeEdge(parentXY, childXY, label, childPath, childKind, handlers) {
  const g = el('g', { class: 'edge' });
  const d = edgePath(parentXY, childXY, treeTrimRadius(childKind));
  // Same invisible-hit-twin treatment as renderMarkovEdge above — a tree "edge" IS how you select
  // its child node, so this was doubly bad before: only a 1.5px sliver was clickable, AND clicking
  // an edge is the only way to select the child.
  g.appendChild(el('path', { class: 'edge-hit', d }));
  g.appendChild(el('path', { class: 'edge-line', d }));
  if (label) {
    const [lx, ly] = edgeLabelPos(parentXY, childXY);
    g.appendChild(el('text', { class: 'edge-label', x: lx, y: ly, 'text-anchor': 'middle' }, text(label)));
  }
  g.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    handlers.onEdgePointerDown(e, { variant: 'tree', path: childPath });
  });
  return g;
}

export function renderTreeNode(node, xy, kind, isSelected, path, handlers) {
  const [cx, cy] = xy;
  const g = el('g', { class: 'node', 'data-kind': kind, transform: `translate(${cx},${cy})` });

  // A tree terminal is a bare <line> (shapeForKind('terminal')) — no fill, so (like edges above)
  // only its 1.5px stroke would otherwise be clickable, and its text label would be the only
  // realistic target. Every OTHER kind has a paper-filled shape that already carries its own hit
  // area, so only the terminal needs this transparent rect, sized from geometry.hitShapeFor and
  // appended first so it sits under the halo/shape/text (Task 4).
  if (kind === 'terminal') {
    const t = hitShapeFor('terminal');
    g.appendChild(el('rect', {
      class: 'node-hit', x: -t.w / 2, y: -t.h / 2, width: t.w, height: t.h,
    }));
  }

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

  g.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    handlers.onNodePointerDown(e, { kind: 'node', path });
  });
  return g;
}

export function renderTree(model, positions, selection, edgesG, nodesG, handlers, nodeIndex) {
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
      edgesG.appendChild(renderTreeEdge(parentXY, xy, label, path, kind, handlers));
    }

    const g = renderTreeNode(node, xy, kind, node === selectedNode, path, handlers);
    nodesG.appendChild(g);
    nodeIndex.push({ kind: 'node', path, xy, hit: hitShapeFor(kind), el: g, treeKind: kind, node });

    for (const child of node.children) walk(child, [...path, child.name], xy);
  }

  walk(model.tree, [model.tree.name], null);
}

// -------- top-level render --------

export function buildSvg(svgEl, model, { positions, selection, handlers }) {
  svgEl.replaceChildren();
  svgEl.setAttribute('data-model-type', model.type);
  svgEl.appendChild(buildDefs());
  const edgesG = el('g', { class: 'edges' });
  const nodesG = el('g', { class: 'nodes' });
  svgEl.appendChild(edgesG);
  svgEl.appendChild(nodesG);

  const nodeIndex = [];
  if (model.type === 'markov') renderMarkov(model, positions, selection, edgesG, nodesG, handlers, nodeIndex);
  else if (model.type === 'tree') renderTree(model, positions, selection, edgesG, nodesG, handlers, nodeIndex);
  return nodeIndex;
}
