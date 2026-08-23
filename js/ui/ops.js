// Pure model-editing operations for markov models: (model, ...) -> newModel.
// Every op deep-clones the model (structuredClone handles the params Map) before editing, so the
// input model is never mutated. Invalid input throws a plain Error with a clear message; the
// store surfaces these to the user. Part 1 (this file) covers markov editing only — every op
// validates model.type === 'markov' and throws otherwise. Tree ops land in part 2.

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

export function setLayout(model, key, xy) {
  assertMarkov(model, 'setLayout');
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
