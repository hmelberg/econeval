import { load } from 'js-yaml';

export class ModelError extends Error {
  constructor(message, { line, hint, path } = {}) {
    super(message);
    this.line = line;
    this.hint = hint;
    this.path = path;
  }
}

const UNITS = { year: 1, month: 1 / 12, week: 7 / 365.25, day: 1 / 365.25 };

function parseCycle(str) {                    // "1 year" | "6 months" | number (years)
  if (str == null) return 1;
  if (typeof str === 'number') return str;
  const m = /^\s*([\d.]+)\s*(year|month|week|day)s?\s*$/.exec(String(str));
  if (!m) throw new ModelError(`settings.cycle: cannot parse '${str}' (use e.g. "1 year", "6 months")`, { path: 'settings.cycle' });
  return Number(m[1]) * UNITS[m[2]];
}

const PARAM_KEYS = new Set(['value', 'low', 'high', 'dist', 'source', 'notes']);
const unbalanced = (s) => typeof s === 'string' &&
  (s.split('(').length !== s.split(')').length);
const FLOW_HINT = 'Function calls like beta(a, b) inside {...} need quotes — or use block style (one field per line).';

function normParam(name, v) {
  if (typeof v === 'number' || typeof v === 'string') return { value: v };
  if (typeof v !== 'object' || v === null || Array.isArray(v))
    throw new ModelError(`params.${name}: must be a number, string, or mapping`, { path: `params.${name}` });
  const out = {};
  for (const k of Object.keys(v)) {
    if (!PARAM_KEYS.has(k)) {
      const hint = (unbalanced(k) || Object.values(v).some(unbalanced)) ? FLOW_HINT : undefined;
      throw new ModelError(`params.${name}: unknown field '${k}'`, { hint, path: `params.${name}` });
    }
    out[k] = v[k];
  }
  if (unbalanced(out.value) || unbalanced(out.dist))
    throw new ModelError(`params.${name}: unbalanced parentheses`, { hint: FLOW_HINT, path: `params.${name}` });
  return out;
}

// Every non-(source|notes) key on a state mapping is a tracked payoff (cost/utility/extras).
function normStates(raw) {
  return Object.entries(raw).map(([name, v]) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v))
      throw new ModelError(`states.${name}: must be a mapping of payoff fields`, { path: `states.${name}` });
    const payoffs = {};
    let source, notes;
    for (const [k, val] of Object.entries(v)) {
      if (k === 'source') source = val;
      else if (k === 'notes') notes = val;
      else payoffs[k] = val;
    }
    const state = { name, payoffs };
    if (source !== undefined) state.source = source;
    if (notes !== undefined) state.notes = notes;
    return state;
  });
}

const TRANSITION_KEYS = new Set(['p', 'cost', 'utility', 'source', 'notes']);

// Per row: a `multinomial` key -> {type:'multinomial', counts}; otherwise each target maps to
// {p, cost?, utility?, ...} — a scalar target value is probability shorthand -> {p: scalar}.
function normTransitions(raw) {
  const out = {};
  for (const [from, row] of Object.entries(raw)) {
    if (typeof row !== 'object' || row === null || Array.isArray(row))
      throw new ModelError(`transitions.${from}: must be a mapping of targets`, { path: `transitions.${from}` });

    if ('multinomial' in row) {
      const extraKeys = Object.keys(row).filter((k) => k !== 'multinomial');
      if (extraKeys.length > 0)
        throw new ModelError(
          `transitions.${from}: 'multinomial' cannot be combined with other targets (${extraKeys.join(', ')})`,
          { path: `transitions.${from}` }
        );
      const counts = row.multinomial;
      if (typeof counts !== 'object' || counts === null || Array.isArray(counts))
        throw new ModelError(`transitions.${from}.multinomial: must be a mapping of target counts`, { path: `transitions.${from}.multinomial` });
      out[from] = { type: 'multinomial', counts: { ...counts } };
      continue;
    }

    const to = {};
    for (const [target, v] of Object.entries(row)) {
      if (typeof v === 'number' || typeof v === 'string') {
        to[target] = { p: v };
      } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
        const entry = {};
        for (const k of Object.keys(v)) {
          if (!TRANSITION_KEYS.has(k))
            throw new ModelError(`transitions.${from}.${target}: unknown field '${k}'`, { path: `transitions.${from}.${target}` });
          entry[k] = v[k];
        }
        to[target] = entry;
      } else {
        throw new ModelError(`transitions.${from}.${target}: invalid value`, { path: `transitions.${from}.${target}` });
      }
    }
    out[from] = { type: 'p', to };
  }
  return out;
}

// tables: { name: { colName: number[] } } — passthrough with a shape check: every column in a
// table must be an array of numbers, and all columns in the same table must share a length.
function normTables(raw) {
  const out = {};
  if (!raw) return out;
  for (const [tname, cols] of Object.entries(raw)) {
    if (typeof cols !== 'object' || cols === null || Array.isArray(cols))
      throw new ModelError(`tables.${tname}: must be a mapping of columns`, { path: `tables.${tname}` });
    const colNames = Object.keys(cols);
    if (colNames.length === 0)
      throw new ModelError(`tables.${tname}: table has no columns`, { path: `tables.${tname}` });
    let len;
    for (const cname of colNames) {
      const col = cols[cname];
      if (!Array.isArray(col))
        throw new ModelError(`tables.${tname}.${cname}: column must be an array of numbers`, { path: `tables.${tname}.${cname}` });
      if (len === undefined) len = col.length;
      else if (col.length !== len)
        throw new ModelError(`tables.${tname}: all columns must have the same length`, { path: `tables.${tname}` });
      for (const x of col) {
        if (typeof x !== 'number')
          throw new ModelError(`tables.${tname}.${cname}: values must be numeric`, { path: `tables.${tname}.${cname}` });
      }
    }
    out[tname] = { ...cols };
  }
  return out;
}

const KNOWN_TYPES = new Set(['markov', 'tree']);
const CORRECTIONS = new Set(['half-cycle', 'life-table', 'none']);

// Top-level document keys this task normalizes into the Model shape. Anything else is an
// unrecognized/future key (e.g. `description`) and is preserved verbatim on round-trip.
const KNOWN_TOP = new Set([
  'econeval', 'type', 'name', 'meta', 'settings', 'params', 'tables', 'models',
  'states', 'transitions', 'strategies', 'tree', 'layout',
]);

export function normalizeModel(obj) {
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj))
    throw new ModelError('model must be a YAML mapping at the top level');

  if (obj.econeval === undefined)
    throw new ModelError("econeval: version key is required (e.g. 'econeval: 1')", { path: 'econeval' });
  const version = obj.econeval;

  if (typeof obj.name !== 'string' || obj.name.length === 0)
    throw new ModelError('name: a non-empty model name is required', { path: 'name' });
  const name = obj.name;

  const type = obj.type;
  if (!KNOWN_TYPES.has(type))
    throw new ModelError(`type: unknown type '${type}' (expected 'markov' or 'tree')`, { path: 'type' });

  const meta = obj.meta ?? {};

  // --- settings, with defaults ---
  const rawSettings = obj.settings ?? {};
  if (typeof rawSettings !== 'object' || rawSettings === null || Array.isArray(rawSettings))
    throw new ModelError('settings: must be a mapping', { path: 'settings' });

  if (type === 'markov') {
    const c = rawSettings.cycles;
    if (c === undefined || c === null)
      throw new ModelError('settings.cycles is required for markov models', { path: 'settings.cycles' });
    if (typeof c !== 'number' || !Number.isFinite(c) || c <= 0)
      throw new ModelError('settings.cycles must be a positive number', { path: 'settings.cycles' });
  }

  const correctionRaw = rawSettings.correction;
  const correction = correctionRaw === undefined ? 'half-cycle' : correctionRaw;
  if (!CORRECTIONS.has(correction))
    throw new ModelError(
      `settings.correction: unknown value '${correction}' (expected 'half-cycle', 'life-table', or 'none')`,
      { path: 'settings.correction' }
    );

  let start;
  const rawStart = rawSettings.start;
  if (rawStart === undefined || rawStart === null) start = null;
  else if (typeof rawStart === 'string') start = { [rawStart]: 1 };
  else start = rawStart;

  const settings = {
    cycles: rawSettings.cycles,
    cycleYears: parseCycle(rawSettings.cycle),
    discount: { cost: rawSettings.discount?.cost ?? 0, effect: rawSettings.discount?.effect ?? 0 },
    correction,
    wtp: rawSettings.wtp ?? null,
    psa: {
      n: rawSettings.psa?.n ?? 1000,
      seed: rawSettings.psa?.seed ?? 1,
      correlations: rawSettings.psa?.correlations ?? [],
    },
    start,
    age: rawSettings.age ?? null,
  };

  // --- params ---
  const rawParams = obj.params ?? {};
  if (typeof rawParams !== 'object' || rawParams === null || Array.isArray(rawParams))
    throw new ModelError('params: must be a mapping of param name to value', { path: 'params' });
  const params = new Map();
  for (const [pname, pval] of Object.entries(rawParams)) {
    params.set(pname, normParam(pname, pval));
  }

  // --- states / transitions (markov only; tree structure is Task 6's job) ---
  let states = [];
  let transitions = {};
  if (type === 'markov') {
    if (typeof obj.states !== 'object' || obj.states === null || Array.isArray(obj.states) || Object.keys(obj.states).length === 0)
      throw new ModelError('states: at least one state is required for markov models', { path: 'states' });
    states = normStates(obj.states);

    if (typeof obj.transitions !== 'object' || obj.transitions === null || Array.isArray(obj.transitions) || Object.keys(obj.transitions).length === 0)
      throw new ModelError('transitions: at least one transition row is required for markov models', { path: 'transitions' });
    transitions = normTransitions(obj.transitions);
  }

  // --- strategies (no strategies block -> a single implicit 'base' strategy) ---
  let strategies;
  if (obj.strategies === undefined) {
    strategies = { base: { overrides: {} } };
  } else {
    if (typeof obj.strategies !== 'object' || obj.strategies === null || Array.isArray(obj.strategies))
      throw new ModelError('strategies: must be a mapping of strategy name to overrides', { path: 'strategies' });
    strategies = {};
    for (const [sname, sval] of Object.entries(obj.strategies)) {
      if (sval !== undefined && (typeof sval !== 'object' || sval === null || Array.isArray(sval)))
        throw new ModelError(`strategies.${sname}: overrides must be a mapping of param to value`, { path: `strategies.${sname}` });
      strategies[sname] = { overrides: { ...(sval ?? {}) } };
    }
  }

  const tables = normTables(obj.tables);

  const model = {
    version, type, name, meta, settings, params, tables,
    models: {},   // filled in Task 6
    states, transitions, strategies,
    tree: null,   // filled in Task 6
    layout: obj.layout ?? null,
  };

  // Unknown top-level keys (e.g. `description`) are preserved verbatim on round-trip; flagging
  // them is check.js's job, not this parser's.
  for (const k of Object.keys(obj)) {
    if (!KNOWN_TOP.has(k)) model[k] = obj[k];
  }

  return model;
}

export function parseModel(text) {
  let obj;
  try {
    obj = load(text);
  } catch (e) {
    const line = e.mark ? e.mark.line + 1 : undefined;
    const srcLine = e.mark ? text.split('\n')[e.mark.line] ?? '' : '';
    const hint = srcLine.includes('(') ? FLOW_HINT : undefined;
    throw new ModelError(`YAML error: ${e.reason ?? e.message}`, { line, hint });
  }
  return normalizeModel(obj);
}
