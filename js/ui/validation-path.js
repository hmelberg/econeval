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
//                                     actual tree root name to rebuild an ops.nodeAt-shaped array
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

import { nodeAt } from './ops.js';

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
    const fullPath = segs.length > 1 ? [scoped.tree.name, ...segs.slice(1)] : [scoped.tree.name];
    try {
      nodeAt(scoped, fullPath);
      return { kind: 'node', id: fullPath, modelPath };
    } catch {
      return null;
    }
  }

  return null; // params.*, settings..., or any other shape -- never click-through
}
