// Auto-layout: pure, DOM-free. Produces canvas node positions for a Model when the model has no
// (or only partial) explicit `layout`. Layout keys are the binding rule from constraints.md:
// markov keys are state names; tree keys are '/'-joined node paths from the root, inclusive
// (e.g. 'Root/A/Win'). Every function here is a pure (model) -> positions map; nothing touches
// the DOM and nothing mutates the model.

const MARKOV_CENTER = { x: 360, y: 280 };

// States on a ring: center (360, 280), radius = max(140, 46*n/pi), in declaration order
// (model.states array order — the order states appear in the parsed YAML), first state at
// angle -90 deg (straight up from center). Positions are rounded to integers.
export function autoMarkov(model) {
  const states = model.states ?? [];
  const n = states.length;
  const positions = {};
  if (n === 0) return positions;

  const { x: cx, y: cy } = MARKOV_CENTER;
  const r = Math.max(140, (46 * n) / Math.PI);

  states.forEach((state, i) => {
    const angle = -Math.PI / 2 + i * ((2 * Math.PI) / n);
    positions[state.name] = [
      Math.round(cx + r * Math.cos(angle)),
      Math.round(cy + r * Math.sin(angle)),
    ];
  });

  return positions;
}

// Left-to-right tidy layout: x = 90 + depth*170 (root depth 0). Leaves get consecutive y slots
// starting at 60, step 74, assigned in depth-first declaration order (the order children appear
// in each node's `children` array, which mirrors YAML declaration order). An internal node's y
// is the rounded mean of its children's y. The root is included. Keys are '/'-joined paths from
// the root, inclusive.
export function autoTree(model) {
  const positions = {};
  const root = model.tree;
  if (!root) return positions;

  let nextLeafY = 60;
  const LEAF_STEP = 74;
  const X_STEP = 170;
  const X0 = 90;

  // Post-order visit: a node's y depends on its children's y (already-assigned), so children are
  // visited (and their y recorded) before the parent's own position is written.
  function visit(node, path, depth) {
    const key = path.join('/');
    const x = X0 + depth * X_STEP;
    let y;
    if (node.children.length === 0) {
      y = nextLeafY;
      nextLeafY += LEAF_STEP;
    } else {
      const childYs = node.children.map((child) => visit(child, [...path, child.name], depth + 1));
      y = Math.round(childYs.reduce((sum, v) => sum + v, 0) / childYs.length);
    }
    positions[key] = [x, y];
    return y;
  }

  visit(root, [root.name], 0);
  return positions;
}

// layoutFor(model) -> {key: [x, y]}. Merges explicit model.layout entries (they win) with
// auto-fill for every key the auto-layout produces but the explicit layout doesn't mention.
// Works on a sub-model object as-is: sub-models (model.models.*) are normalized through the same
// parser into the same Model shape (type/states/tree/layout all present), so this needs no
// special-casing for sub-models — callers just pass whichever model object is being rendered.
export function layoutFor(model) {
  const auto = model.type === 'markov' ? autoMarkov(model) : autoTree(model);
  return { ...auto, ...(model.layout ?? {}) };
}
