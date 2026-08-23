// Pure model-editing operations: (model, ...) -> newModel.
// Every op deep-clones the model (structuredClone handles the params Map) before editing, so the
// input model is never mutated. Invalid input throws a plain Error with a clear message; the
// store surfaces these to the user. Part 1 covers markov editing (every op validates
// model.type === 'markov' and throws otherwise). Part 2 (below the markov section) covers tree
// editing (validates model.type === 'tree') plus params/settings ops, which work on BOTH model
// types and carry no type guard.

import { parseCycle } from '../core/model.js';

function clone(model) {
  return structuredClone(model);
}

function assertMarkov(model, opName) {
  if (!model || model.type !== 'markov') {
    throw new Error(`${opName}: model.type must be 'markov' (got '${model && model.type}')`);
  }
}

function renameKey(obj, oldKey, newKey) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) out[k === oldKey ? newKey : k] = v;
  return out;
}

function omitKey(obj, key) {
  if (!(key in obj)) return obj;
  const out = { ...obj };
  delete out[key];
  return out;
}

// Renames a state everywhere inside `transitions`: the row key itself (from), and — inside every
// row, including rows for other states — the target key (row.to) or the multinomial count key
// (row.counts).
function renameStateInTransitions(transitions, oldName, newName) {
  const out = {};
  for (const [from, row] of Object.entries(transitions)) {
    const newFrom = from === oldName ? newName : from;
    if (row.type === 'multinomial') {
      out[newFrom] = { type: 'multinomial', counts: renameKey(row.counts, oldName, newName) };
    } else {
      out[newFrom] = { type: 'p', to: renameKey(row.to, oldName, newName) };
    }
  }
  return out;
}

// Drops every reference to `name` inside `transitions`: the row itself, and — inside every
// remaining row — the target entry (row.to) or the multinomial count (row.counts).
function scrubStateInTransitions(transitions, name) {
  const out = {};
  for (const [from, row] of Object.entries(transitions)) {
    if (from === name) continue;
    if (row.type === 'multinomial') {
      out[from] = { type: 'multinomial', counts: omitKey(row.counts, name) };
    } else {
      out[from] = { type: 'p', to: omitKey(row.to, name) };
    }
  }
  return out;
}

function freeStateName(states) {
  const existing = new Set(states.map((s) => s.name));
  let i = 1;
  while (existing.has(`state${i}`)) i += 1;
  return `state${i}`;
}

export function addState(model, name) {
  assertMarkov(model, 'addState');
  const m = clone(model);

  let newName = name;
  if (newName === undefined || newName === null) {
    newName = freeStateName(m.states);
  } else {
    if (newName === '') throw new Error('addState: name must not be empty');
    if (m.states.some((s) => s.name === newName))
      throw new Error(`addState: state '${newName}' already exists`);
  }

  m.states.push({ name: newName, payoffs: { cost: 0, utility: 0 } });
  m.transitions[newName] = { type: 'p', to: { [newName]: { p: 'rest' } } };
  return m;
}

export function renameState(model, oldName, newName) {
  assertMarkov(model, 'renameState');
  if (newName === '' || newName === undefined || newName === null)
    throw new Error('renameState: newName must not be empty');

  const m = clone(model);
  if (!m.states.some((s) => s.name === oldName))
    throw new Error(`renameState: state '${oldName}' not found`);
  if (newName === oldName) return m; // no-op: renaming to the same name is not a collision
  if (m.states.some((s) => s.name === newName))
    throw new Error(`renameState: state '${newName}' already exists`);

  m.states = m.states.map((s) => (s.name === oldName ? { ...s, name: newName } : s));
  m.transitions = renameStateInTransitions(m.transitions, oldName, newName);

  if (m.settings.start && Object.prototype.hasOwnProperty.call(m.settings.start, oldName)) {
    m.settings.start = renameKey(m.settings.start, oldName, newName);
  }
  if (m.layout && Object.prototype.hasOwnProperty.call(m.layout, oldName)) {
    m.layout = renameKey(m.layout, oldName, newName);
  }

  return m;
}

export function deleteState(model, name) {
  assertMarkov(model, 'deleteState');
  const m = clone(model);
  if (!m.states.some((s) => s.name === name))
    throw new Error(`deleteState: state '${name}' not found`);

  m.states = m.states.filter((s) => s.name !== name);
  m.transitions = scrubStateInTransitions(m.transitions, name);

  if (m.settings.start && Object.prototype.hasOwnProperty.call(m.settings.start, name)) {
    m.settings.start = omitKey(m.settings.start, name);
  }
  if (m.layout && Object.prototype.hasOwnProperty.call(m.layout, name)) {
    m.layout = omitKey(m.layout, name);
  }

  return m;
}

export function addTransition(model, from, to) {
  assertMarkov(model, 'addTransition');
  const m = clone(model);

  if (!m.states.some((s) => s.name === from))
    throw new Error(`addTransition: state '${from}' not found`);
  if (!m.states.some((s) => s.name === to))
    throw new Error(`addTransition: state '${to}' not found`);

  const row = m.transitions[from];
  if (!row) throw new Error(`addTransition: no transitions row for state '${from}'`);
  if (row.type === 'multinomial')
    throw new Error(`addTransition: row '${from}' is a multinomial row; cannot add a probability target`);
  if (to in row.to)
    throw new Error(`addTransition: transition from '${from}' to '${to}' already exists`);

  const hasRest = Object.values(row.to).some((entry) => entry.p === 'rest');
  row.to[to] = { p: hasRest ? 0 : 'rest' };
  return m;
}

export function deleteTransition(model, from, to) {
  assertMarkov(model, 'deleteTransition');
  const m = clone(model);

  const row = m.transitions[from];
  if (!row) throw new Error(`deleteTransition: no transitions row for state '${from}'`);

  if (row.type === 'multinomial') {
    if (!(to in row.counts))
      throw new Error(`deleteTransition: no target '${to}' in row '${from}'`);
    delete row.counts[to];
  } else {
    if (!(to in row.to)) throw new Error(`deleteTransition: no target '${to}' in row '${from}'`);
    delete row.to[to];
  }

  return m;
}

const TRANSITION_ATTR_KEYS = new Set(['p', 'cost', 'utility']);

export function setTransitionAttr(model, from, to, key, value) {
  assertMarkov(model, 'setTransitionAttr');
  if (!TRANSITION_ATTR_KEYS.has(key))
    throw new Error(`setTransitionAttr: key must be one of p, cost, utility (got '${key}')`);

  const m = clone(model);
  const row = m.transitions[from];
  if (!row) throw new Error(`setTransitionAttr: no transitions row for state '${from}'`);
  if (row.type === 'multinomial')
    throw new Error(`setTransitionAttr: row '${from}' is a multinomial row`);

  const entry = row.to[to];
  if (!entry) throw new Error(`setTransitionAttr: no transition from '${from}' to '${to}'`);

  if (value === null) {
    if (key === 'p')
      throw new Error("setTransitionAttr: cannot remove 'p' — use deleteTransition");
    delete entry[key];
  } else {
    entry[key] = value;
  }

  return m;
}

export function setStatePayoff(model, name, key, value) {
  assertMarkov(model, 'setStatePayoff');
  const m = clone(model);
  const state = m.states.find((s) => s.name === name);
  if (!state) throw new Error(`setStatePayoff: state '${name}' not found`);

  if (value === null) delete state.payoffs[key];
  else state.payoffs[key] = value;

  return m;
}

// Task 10 fix: setLayout works on BOTH model types — its body only ever touches the generic
// m.layout map (never states/transitions/tree), and the layout-key rule (constraints.md) is
// itself generic: state names for markov, '/'-joined node paths for trees. The Select tool's
// node-drag gesture needs this to work identically for tree nodes as for markov states; there is
// no tree-specific variant of this op anywhere else in the file. (Originally written markov-only
// in Task 3, before Task 4 added tree editing — this was a gap, not a deliberate restriction; no
// test asserted the old assertMarkov guard.)
export function setLayout(model, key, xy) {
  const m = clone(model);

  if (xy === null) {
    if (m.layout && key in m.layout) m.layout = omitKey(m.layout, key);
    return m;
  }

  if (!Array.isArray(xy) || xy.length !== 2)
    throw new Error('setLayout: xy must be [x, y]');

  const rounded = [Math.round(xy[0]), Math.round(xy[1])];
  m.layout = { ...(m.layout ?? {}), [key]: rounded };
  return m;
}

// ============================================================================================
// Part 2: tree editing, plus params/settings ops (work on BOTH markov and tree models).
//
// Tree nodes are addressed by `path`: an array of names from the root inclusive, e.g.
// ['Treatment?', 'Surgery', 'Success'] — path[0] must equal the tree's root name. `layout` keys
// for a tree are the SAME path, '/'-joined ('Treatment?/Surgery/Success') — see the constraints
// doc's layout-key rule. Root children (path.length === 2) are strategy branches: entered
// unconditionally, never carry a 'p' (mirrors check.js's isRoot handling in walkTreeNode).
// ============================================================================================

function assertTree(model, opName) {
  if (!model || model.type !== 'tree') {
    throw new Error(`${opName}: model.type must be 'tree' (got '${model && model.type}')`);
  }
}

// Rewrites every layout key that names the subtree rooted at `oldPrefix` (the exact key, or any
// key nested under it via '/') to hang off `newPrefix` instead. Used by renameNode.
function rekeyLayoutSubtree(layout, oldPrefix, newPrefix) {
  if (!layout) return layout;
  const out = {};
  for (const [key, val] of Object.entries(layout)) {
    if (key === oldPrefix) out[newPrefix] = val;
    else if (key.startsWith(`${oldPrefix}/`)) out[newPrefix + key.slice(oldPrefix.length)] = val;
    else out[key] = val;
  }
  return out;
}

// Drops every layout key naming the subtree rooted at `prefix` (the exact key, or any key nested
// under it via '/'). Used by deleteNode.
function scrubLayoutSubtree(layout, prefix) {
  if (!layout) return layout;
  const out = {};
  for (const [key, val] of Object.entries(layout)) {
    if (key === prefix || key.startsWith(`${prefix}/`)) continue;
    out[key] = val;
  }
  return out;
}

function freeNodeName(children) {
  const existing = new Set(children.map((c) => c.name));
  let i = 1;
  while (existing.has(`branch${i}`)) i += 1;
  return `branch${i}`;
}

// nodeAt(model, path) -> Node. Exported: canvas/inspector address tree nodes by path, not by
// object reference. Read-only — does not clone; callers that mutate should call it on a model
// they already cloned (as every op below does), so the returned Node is part of that clone.
export function nodeAt(model, path) {
  assertTree(model, 'nodeAt');
  if (!Array.isArray(path) || path.length === 0)
    throw new Error('nodeAt: path must be a non-empty array of names');

  let node = model.tree;
  if (!node || node.name !== path[0])
    throw new Error(`nodeAt: path[0] '${path[0]}' does not match the tree root '${node && node.name}'`);

  for (let i = 1; i < path.length; i += 1) {
    const name = path[i];
    const child = node.children.find((c) => c.name === name);
    if (!child)
      throw new Error(`nodeAt: no child '${name}' under '${node.name}' (path: ${path.join('/')})`);
    node = child;
  }
  return node;
}

export function addChild(model, path, name) {
  assertTree(model, 'addChild');
  const m = clone(model);
  const parent = nodeAt(m, path);

  let newName = name;
  if (newName === undefined || newName === null) {
    newName = freeNodeName(parent.children);
  } else {
    if (newName === '') throw new Error('addChild: name must not be empty');
    if (parent.children.some((c) => c.name === newName))
      throw new Error(`addChild: sibling '${newName}' already exists`);
  }

  const child = { name: newName, payoffs: { utility: 0 }, children: [] };
  const isRootChild = path.length === 1;
  if (!isRootChild) {
    const hasRest = parent.children.some((c) => c.p === 'rest');
    child.p = hasRest ? 0 : 'rest';
  }
  parent.children.push(child);
  return m;
}

export function renameNode(model, path, newName) {
  assertTree(model, 'renameNode');
  if (newName === '' || newName === undefined || newName === null)
    throw new Error('renameNode: newName must not be empty');

  const m = clone(model);
  const node = nodeAt(m, path); // validates path exists
  const oldName = node.name;

  if (newName === oldName) return m; // no-op: renaming to the same name is not a collision

  if (path.length > 1) {
    const parent = nodeAt(m, path.slice(0, -1));
    if (parent.children.some((c) => c.name === newName))
      throw new Error(`renameNode: sibling '${newName}' already exists`);
  }

  node.name = newName;

  const oldPrefix = path.join('/');
  const newPrefix = [...path.slice(0, -1), newName].join('/');
  m.layout = rekeyLayoutSubtree(m.layout, oldPrefix, newPrefix);

  return m;
}

export function deleteNode(model, path) {
  assertTree(model, 'deleteNode');
  if (!Array.isArray(path) || path.length === 0)
    throw new Error('deleteNode: path must be a non-empty array of names');
  if (path.length === 1)
    throw new Error('deleteNode: the root node cannot be deleted');

  const m = clone(model);
  const node = nodeAt(m, path); // validates path exists
  const parent = nodeAt(m, path.slice(0, -1));
  parent.children.splice(parent.children.indexOf(node), 1);

  m.layout = scrubLayoutSubtree(m.layout, path.join('/'));
  return m;
}

const NODE_ATTR_KEYS = new Set(['p', 'delay', 'model', 'notes', 'source']);

export function setNodeAttr(model, path, key, value) {
  assertTree(model, 'setNodeAttr');
  if (!NODE_ATTR_KEYS.has(key))
    throw new Error(`setNodeAttr: key must be one of p, delay, model, notes, source (got '${key}')`);

  const m = clone(model);
  const node = nodeAt(m, path);

  // Root children are strategies, entered unconditionally (mirrors check.js's isRoot handling) —
  // they never carry a 'p', so setting or clearing one here is always a mistake to surface.
  if (key === 'p' && path.length === 2)
    throw new Error("setNodeAttr: 'p' is not valid on a root child (strategies are entered unconditionally)");

  if (value === null) {
    delete node[key];
  } else if (key === 'delay') {
    node.delay = parseCycle(value); // reuse model.js's unit table; parseCycle passes numbers through
  } else {
    node[key] = value;
  }

  return m;
}

export function setNodePayoff(model, path, key, value) {
  assertTree(model, 'setNodePayoff');
  const m = clone(model);
  const node = nodeAt(m, path);

  if (value === null) delete node.payoffs[key];
  else node.payoffs[key] = value;

  return m;
}

export function setWith(model, path, param, value) {
  assertTree(model, 'setWith');
  const m = clone(model);
  const node = nodeAt(m, path);

  if (value === null) {
    if (node.with) {
      delete node.with[param];
      if (Object.keys(node.with).length === 0) delete node.with;
    }
  } else {
    node.with = { ...(node.with ?? {}), [param]: value };
  }

  return m;
}

// --- params: work on BOTH markov and tree models — no type guard. ---

function freeParamName(params) {
  let i = 1;
  while (params.has(`param${i}`)) i += 1;
  return `param${i}`;
}

export function addParam(model, name, spec = { value: 0 }) {
  const m = clone(model);

  let newName = name;
  if (newName === undefined || newName === null) {
    newName = freeParamName(m.params);
  } else {
    if (newName === '') throw new Error('addParam: name must not be empty');
    if (m.params.has(newName)) throw new Error(`addParam: param '${newName}' already exists`);
  }

  m.params.set(newName, { ...spec });
  return m;
}

const PARAM_FIELDS = new Set(['value', 'low', 'high', 'dist', 'source', 'notes']);

export function setParam(model, name, field, value) {
  if (!PARAM_FIELDS.has(field))
    throw new Error(`setParam: field must be one of value, low, high, dist, source, notes (got '${field}')`);

  const m = clone(model);
  const spec = m.params.get(name);
  if (!spec) throw new Error(`setParam: param '${name}' not found`);

  if (value === null) {
    if (field === 'value' && spec.dist === undefined)
      throw new Error(`setParam: cannot remove 'value' from '${name}' — it has no 'dist' to fall back to`);
    delete spec[field];
  } else {
    spec[field] = value;
  }

  return m;
}

// Does NOT rewrite any expression that references the old name elsewhere in the model — by
// design (surprise-principle: a silent rewrite would be magic). check() flags the resulting
// orphaned reference as E_UNKNOWN_NAME, same as any other typo'd name.
export function renameParam(model, oldName, newName) {
  if (newName === '' || newName === undefined || newName === null)
    throw new Error('renameParam: newName must not be empty');

  const m = clone(model);
  if (!m.params.has(oldName))
    throw new Error(`renameParam: param '${oldName}' not found`);
  if (newName === oldName) return m; // no-op: renaming to the same name is not a collision
  if (m.params.has(newName))
    throw new Error(`renameParam: param '${newName}' already exists`);

  const spec = m.params.get(oldName);
  m.params.delete(oldName);
  m.params.set(newName, spec);
  return m;
}

export function deleteParam(model, name) {
  const m = clone(model);
  if (!m.params.has(name)) throw new Error(`deleteParam: param '${name}' not found`);
  m.params.delete(name);
  return m;
}

// --- settings: work on BOTH markov and tree models — no type guard. ---
//
// keyPath is dot-separated ('discount.cost', 'psa.n', or a bare top-level key like 'cycles').
// The one special case is 'cycle': it is not itself a Model settings field (parseModel folds the
// raw 'cycle' unit-string into settings.cycleYears at parse time) — re-parse it the same way,
// reusing model.js's parseCycle so the unit table never drifts between the parser and the editor.

export function setSetting(model, keyPath, value) {
  const m = clone(model);

  if (keyPath === 'cycle') {
    m.settings.cycleYears = parseCycle(value);
    return m;
  }

  const parts = keyPath.split('.');
  let obj = m.settings;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const k = parts[i];
    if (typeof obj[k] !== 'object' || obj[k] === null)
      throw new Error(`setSetting: '${keyPath}' is not a valid settings path (missing '${k}')`);
    obj = obj[k];
  }

  const lastKey = parts[parts.length - 1];
  if (value === null) delete obj[lastKey];
  else obj[lastKey] = value;

  return m;
}
