// js/ui/validation-path.js — pure path -> canvas-selection resolver for the Validation tab's
// click-through rows (Task 6). No DOM, no store: takes a check() finding's `path` string and the
// TOP-LEVEL model it was computed against, and returns either a ready-to-dispatch selection or
// null (the caller renders a plain, non-interactive row for null).
//
// resolveFindingPath(path, model) -> {kind:'state'|'node', id, modelPath} | null
//
// check.js's path vocabulary (see js/analysis/check.js):
//   states.<name>[...]            -- a markov state's own findings (payoff eval errors, ...)
//   transitions.<from>[...]       -- a markov row's findings (E_NO_ROW/E_ROWSUM/E_TWO_RESTS, an
//                                     edge's own p/cost/utility eval errors, ...) -- always keyed
//                                     by the SOURCE state, never a distinct edge selection (the
//                                     brief only wires state/node click-through, never 'edge')
//   tree[.<dot-joined descendant path, OMITTING the root node's own name>]
//                                  -- inspector.js's nodePathToCheckPath documents this same
//                                     divergence from ops.nodeAt's path convention (root name
//                                     included); this function re-attaches the scoped model's
//                                     actual tree root name to rebuild an ops.nodeAt-shaped array.
//                                     TRIMMED to the nearest ancestor node when the tail names a
//                                     FIELD rather than a child -- see resolveTreePath below (the
//                                     markov branch above already does the equivalent by simply
//                                     never looking past segs[1]; the tree branch needs an actual
//                                     walk since check.js's tree paths can run arbitrarily deep:
//                                     `tree.Surgery.cost` (a payoff eval error), `tree.Treat.Cure.
//                                     with.start` (an attach-override error) both still name the
//                                     Surgery/Cure NODE, just with extra trailing field segments)
//   models.<name>.<...>           -- chained recursively for a sub-model-scoped finding, exactly
//                                     mirroring checkModelContent's own recursive pathPrefix
//   params.<name>[...], settings...
//                                  -- never resolves to a canvas entity -- always null
//
// modelPath convention matches store.js/inspector.js/canvas.js's own scopedStore chain: [] for
// the top-level model, an array of `models:` registry names walked in declaration order for a
// nested finding. `id` for a 'state' selection is the bare state name (store.js's own
// selection.id shape); `id` for a 'node' selection is the FULL ops.nodeAt-style path array, root
// name included.
//
// Best-effort, per the task brief: an unresolvable prefix (an unknown sub-model name anywhere in
// the `models.` chain, a state/node name that doesn't actually exist on the resolved scoped
// model, or a path shape this function doesn't recognize at all -- chiefly params/settings) always
// returns null rather than guessing or throwing. A finding is only ever offered as clickable when
// it is GUARANTEED to resolve to a real entity on the model actually being checked -- resolution
// is against the real model object, never a bare string-shape match.

// Walks `contentSegs` (segs.slice(1) -- check.js's tree path with the actual root name OMITTED,
// per its own convention) as far as each segment matches a CHILD of the current node, then stops
// and returns the deepest node reached -- rather than requiring the WHOLE tail to be child names.
// This is what lets `tree.Surgery.cost` (Surgery's own payoff field) and `tree.Treat.Cure.with.
// start` (Cure's attach-override field) both still resolve to their owning node, instead of
// falling all the way through to a plain row just because the path's LAST segment happens to be a
// field name rather than another child. contentSegs === [] (a bare "tree" finding, e.g. the root's
// own payoff error) trivially resolves to the root itself -- nothing to walk.
//
// The one case this deliberately does NOT trim back to the root for: the very FIRST content
// segment failing to match any child of the actual root at all. That means the reference itself is
// broken (an unknown/typo'd child name), not a legitimate trailing field on an otherwise-valid
// node -- silently resolving every broken tree reference to "select the root" would be worse than
// a plain row (it would look clickable and correct, but select the wrong thing). Only a segment
// AFTER at least one successful child-hop is treated as "a field on this node, not a broken
// reference" and safely trimmed away.
//
// Per-segment matching (`node.children.find((c) => c.name === seg)`) intentionally mirrors
// ops.nodeAt's own rule exactly (see js/ui/ops.js) -- the returned path is therefore always a
// prefix of what a full, successful ops.nodeAt(scopedModel, fullPath) walk would have produced,
// which is what guarantees it round-trips through store.js's isSelectionValid (also just an
// ops.nodeAt call) without a separate cross-check needed here.
function resolveTreePath(scopedModel, contentSegs) {
  const root = scopedModel.tree;
  if (!root) return null;
  if (contentSegs.length === 0) return [root.name];

  let node = root;
  const resolved = [root.name];
  for (const seg of contentSegs) {
    const child = node.children.find((c) => c.name === seg);
    if (!child) break; // a trailing FIELD segment (payoff key, 'with', 'start', ...) -- stop here
    node = child;
    resolved.push(seg);
  }
  return resolved.length > 1 ? resolved : null; // the first content segment itself never matched
}

function resolveScopedModel(model, modelPath) {
  let m = model;
  for (const name of modelPath) {
    if (!m || !m.models || !(name in m.models)) return undefined;
    m = m.models[name];
  }
  return m;
}

const MODELS_PREFIX_RE = /^models\.([^.]+)\.(.*)$/;

export function resolveFindingPath(path, model) {
  if (typeof path !== 'string' || path === '' || !model) return null;

  const modelPath = [];
  let rest = path;
  let m = MODELS_PREFIX_RE.exec(rest);
  while (m) {
    modelPath.push(m[1]);
    rest = m[2];
    m = MODELS_PREFIX_RE.exec(rest);
  }

  const scoped = resolveScopedModel(model, modelPath);
  if (!scoped) return null; // an unknown sub-model anywhere in the chain -- ambiguous, plain row

  const segs = rest.split('.');

  if (segs[0] === 'states' || segs[0] === 'transitions') {
    const id = segs[1];
    if (id && scoped.type === 'markov' && scoped.states?.some((s) => s.name === id)) {
      return { kind: 'state', id, modelPath };
    }
    return null;
  }

  if (segs[0] === 'tree') {
    if (scoped.type !== 'tree' || !scoped.tree) return null;
    const resolved = resolveTreePath(scoped, segs.slice(1));
    if (!resolved) return null;
    return { kind: 'node', id: resolved, modelPath };
  }

  return null; // params.*, settings..., or any other shape -- never click-through
}
