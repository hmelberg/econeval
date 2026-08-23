import { load, dump } from 'js-yaml';

export class ModelError extends Error {
  constructor(message, { line, hint, path } = {}) {
    super(message);
    this.line = line;
    this.hint = hint;
    this.path = path;
  }
}

const UNITS = { year: 1, month: 1 / 12, week: 7 / 365.25, day: 1 / 365.25 };
const UNIT_ORDER = ['year', 'month', 'week', 'day']; // formatCycle's match priority

export function parseCycle(str) {              // "1 year" | "6 months" | number (years)
  if (str == null) return 1;
  if (typeof str === 'number') return str;
  const m = /^\s*([\d.]+)\s*(year|month|week|day)s?\s*$/.exec(String(str));
  if (!m) throw new ModelError(`settings.cycle: cannot parse '${str}' (use e.g. "1 year", "6 months")`, { path: 'settings.cycle' });
  return Number(m[1]) * UNITS[m[2]];
}

const CYCLE_TOL = 1e-9; // tolerance on the fraction match, per the brief

// formatCycle(years) -> the friendliest unit string that round-trips through parseCycle exactly:
// tries year, then month, then week, then day (UNIT_ORDER) for an integer count within
// CYCLE_TOL; a non-matching fraction falls back to decimal years (always singular 'year', never
// 'years' — e.g. 0.3 -> '0.3 year'). Shared by serializeModel (settings.cycle, tree node delay)
// and the inspector's Settings display, so the unit table can never drift between them.
export function formatCycle(years) {
  for (const unit of UNIT_ORDER) {
    const k = years / UNITS[unit];
    if (Math.abs(k - Math.round(k)) < CYCLE_TOL) {
      const n = Math.round(k);
      return `${n} ${unit}${n === 1 ? '' : 's'}`;
    }
  }
  return `${years} year`;
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
    // Same FLOW_HINT protection as normParam: a flow-style row like `{cost: 100, utility:
    // lookup(qol, age)}` has its comma-bearing call silently split by js-yaml into extra bogus
    // entries — surfaced here as an unbalanced-parens payoff value instead of an opaque later
    // failure (e.g. `compile(null)` resolving to an "unknown name: null" error far from the cause).
    for (const [k, val] of Object.entries(payoffs)) {
      if (unbalanced(val))
        throw new ModelError(`states.${name}.${k}: unbalanced parentheses`, { hint: FLOW_HINT, path: `states.${name}.${k}` });
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
      // Same FLOW_HINT protection as normParam/normStates: a flow-style row like `{alive: rest,
      // dead: lookup(mort, age)}` has its comma-bearing call silently split by js-yaml into extra
      // bogus entries — the checks below surface that as an unbalanced-parens value with a
      // targeted hint, at whichever entry the split first breaks (the plain scalar branch, the
      // per-target mapping's own fields, or the resulting bogus extra key/'invalid value' branch).
      if (typeof v === 'number' || typeof v === 'string') {
        if (unbalanced(v))
          throw new ModelError(`transitions.${from}.${target}: unbalanced parentheses`, { hint: FLOW_HINT, path: `transitions.${from}.${target}` });
        to[target] = { p: v };
      } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
        const entry = {};
        for (const k of Object.keys(v)) {
          if (!TRANSITION_KEYS.has(k)) {
            const hint = (unbalanced(k) || Object.values(v).some(unbalanced)) ? FLOW_HINT : undefined;
            throw new ModelError(`transitions.${from}.${target}: unknown field '${k}'`, { hint, path: `transitions.${from}.${target}` });
          }
          entry[k] = v[k];
        }
        if (unbalanced(entry.p) || unbalanced(entry.cost) || unbalanced(entry.utility))
          throw new ModelError(`transitions.${from}.${target}: unbalanced parentheses`, { hint: FLOW_HINT, path: `transitions.${from}.${target}` });
        to[target] = entry;
      } else {
        const hint = (unbalanced(target) || Object.values(row).some(unbalanced)) ? FLOW_HINT : undefined;
        throw new ModelError(`transitions.${from}.${target}: invalid value`, { hint, path: `transitions.${from}.${target}` });
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

// Same set, but named for the JS-side Model object fields (note `version` vs raw `econeval`) —
// used by serializeModel to spread back any unrecognized extra fields (e.g. `description`).
const MODEL_FIELD_KEYS = new Set([
  'version', 'type', 'name', 'meta', 'settings', 'params', 'tables', 'models',
  'states', 'transitions', 'strategies', 'tree', 'layout',
]);

// --- tree ---
//
// Reserved node keys: p, cost, utility, source, notes, children, model, with, delay, kind.
// `cost`/`utility` fold into the node's `payoffs` (same bucket as any non-reserved scalar
// extra, e.g. `relapses: 1`) — the Node shape has no separate cost/utility slots. `p`, `model`,
// `with`, `delay`, `source`, `notes` become direct Node fields. `children` is the explicit
// fallback for inline children (merged with any inline `Name: {...}` siblings; a name appearing
// in both is an error). `kind` is an internal-only parser override (root=decision, has-p=chance,
// leaf=terminal are always inferred) — consumed and discarded, never surfaced on the Node.
//
// Any other *mapping*-valued key is a child node; any other *scalar*-valued key is an extra
// tracked payoff.

// Reserved tree-node attribute keys — also used by the serializer: a CHILD whose name is one of
// these must be re-emitted under an explicit `children:` block, never inline, or re-parsing
// would reinterpret the name as a reserved attribute of the parent instead of a child.
export const TREE_RESERVED_KEYS = new Set([
  'p', 'cost', 'utility', 'source', 'notes', 'children', 'model', 'with', 'delay', 'kind',
]);

function normTreeWith(raw, path) {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw))
    throw new ModelError(`${path}.with: must be a mapping`, { path: `${path}.with` });
  return { ...raw };
}

function normTreeNode(name, raw, path) {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw))
    throw new ModelError(`${path}.${name}: tree node must be a mapping`, { path: `${path}.${name}` });

  const node = { name };
  const payoffs = {};
  const inlineChildren = [];
  let explicitChildrenRaw;

  for (const [k, v] of Object.entries(raw)) {
    switch (k) {
      case 'children':
        explicitChildrenRaw = v;
        break;
      case 'kind':
        break; // internal-only parser override; not surfaced on the Node
      case 'p':
        node.p = v;
        break;
      case 'model':
        if (typeof v !== 'string')
          throw new ModelError(`${path}.${name}.model: must be a string (sub-model name)`, { path: `${path}.${name}.model` });
        node.model = v;
        break;
      case 'with':
        node.with = normTreeWith(v, `${path}.${name}`);
        break;
      case 'delay':
        node.delay = parseCycle(v);
        break;
      case 'source':
        node.source = v;
        break;
      case 'notes':
        node.notes = v;
        break;
      case 'cost':
      case 'utility':
        payoffs[k] = v;
        break;
      default:
        if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
          inlineChildren.push([k, v]);
        } else if (typeof v === 'number' || typeof v === 'string') {
          payoffs[k] = v;
        } else {
          throw new ModelError(`${path}.${name}.${k}: invalid tree node value`, { path: `${path}.${name}.${k}` });
        }
    }
  }

  node.payoffs = payoffs;

  const childEntries = [...inlineChildren];
  if (explicitChildrenRaw !== undefined) {
    if (typeof explicitChildrenRaw !== 'object' || explicitChildrenRaw === null || Array.isArray(explicitChildrenRaw))
      throw new ModelError(`${path}.${name}.children: must be a mapping`, { path: `${path}.${name}.children` });
    for (const [cname, cval] of Object.entries(explicitChildrenRaw)) {
      if (childEntries.some(([existing]) => existing === cname))
        throw new ModelError(
          `${path}.${name}: child '${cname}' appears both inline and in 'children:' — ambiguous`,
          { path: `${path}.${name}` }
        );
      // An explicit children: entry can name-collide with a payoff extra declared directly on
      // this node (inline children can't — a single raw key can't be both scalar and mapping).
      // Serializing would otherwise silently drop the payoff (the child overwrites it in the
      // plain object) — surface it instead of losing data silently.
      if (Object.prototype.hasOwnProperty.call(payoffs, cname))
        throw new ModelError(
          `${path}.${name}: child '${cname}' (in 'children:') collides with a payoff of the same name — rename one`,
          { path: `${path}.${name}` }
        );
      childEntries.push([cname, cval]);
    }
  }

  node.children = childEntries.map(([cname, cval]) => normTreeNode(cname, cval, `${path}.${name}`));

  return node;
}

// tree: { rootName: {...} } — exactly one top-level key naming the root (decision) node.
function normTree(raw) {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw))
    throw new ModelError('tree: must be a mapping with a single root node', { path: 'tree' });
  const keys = Object.keys(raw);
  if (keys.length !== 1)
    throw new ModelError('tree: must have exactly one root node', { path: 'tree' });
  const [rootName] = keys;
  return normTreeNode(rootName, raw[rootName], 'tree');
}

// --- sub-models (models:) ---
//
// Each entry runs through the same normalizeModel, in sub-model context: `type` defaults to
// 'markov' if the entry has `states`, 'tree' if it has `tree` (only when `type` is omitted);
// `econeval`/`name` are not required (name falls back to the models: key); discount/wtp/psa in
// `settings` are top-level-only and become a ModelError if present in a sub-model.

function inferSubModelType(sub, name) {
  if (sub.type !== undefined) return sub.type;
  if (sub.states !== undefined) return 'markov';
  if (sub.tree !== undefined) return 'tree';
  throw new ModelError(
    `models.${name}: cannot infer 'type' — provide 'type:' or one of 'states:'/'tree:'`,
    { path: `models.${name}.type` }
  );
}

function normModels(raw) {
  const out = {};
  if (raw === undefined) return out;
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw))
    throw new ModelError('models: must be a mapping of name to sub-model', { path: 'models' });
  for (const [name, sub] of Object.entries(raw)) {
    if (typeof sub !== 'object' || sub === null || Array.isArray(sub))
      throw new ModelError(`models.${name}: must be a mapping`, { path: `models.${name}` });
    const type = inferSubModelType(sub, name);
    const subObj = sub.type === undefined ? { ...sub, type } : sub;
    try {
      out[name] = normalizeModel(subObj, { subModel: true, defaultName: name });
    } catch (e) {
      if (e instanceof ModelError) {
        throw new ModelError(`models.${name}: ${e.message}`, {
          line: e.line,
          hint: e.hint,
          path: e.path ? `models.${name}.${e.path}` : `models.${name}`,
        });
      }
      throw e;
    }
  }
  return out;
}

export function normalizeModel(obj, { subModel = false, defaultName } = {}) {
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj))
    throw new ModelError('model must be a YAML mapping at the top level');

  // `subModel: true` relaxes econeval/name requirements (a sub-model is embedded, not a
  // standalone document) and forbids the top-level-only settings (discount/wtp/psa).
  if (obj.econeval === undefined && !subModel)
    throw new ModelError("econeval: version key is required (e.g. 'econeval: 1')", { path: 'econeval' });
  const version = obj.econeval;

  let name;
  if (typeof obj.name === 'string' && obj.name.length > 0) name = obj.name;
  else if (subModel && defaultName) name = defaultName;
  else throw new ModelError('name: a non-empty model name is required', { path: 'name' });

  const type = obj.type;
  if (!KNOWN_TYPES.has(type))
    throw new ModelError(`type: unknown type '${type}' (expected 'markov' or 'tree')`, { path: 'type' });

  const meta = obj.meta ?? {};

  // --- settings, with defaults ---
  const rawSettings = obj.settings ?? {};
  if (typeof rawSettings !== 'object' || rawSettings === null || Array.isArray(rawSettings))
    throw new ModelError('settings: must be a mapping', { path: 'settings' });

  if (subModel) {
    for (const k of ['discount', 'wtp', 'psa']) {
      if (Object.prototype.hasOwnProperty.call(rawSettings, k))
        throw new ModelError(
          `settings.${k}: is a top-level-only setting (discount/wtp/psa apply once, at the top ` +
          `level) — not allowed inside a sub-model`,
          { path: `settings.${k}` }
        );
    }
  }

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

  // --- tree (tree type only) ---
  let tree = null;
  if (type === 'tree') {
    if (obj.tree === undefined)
      throw new ModelError('tree: required for tree models', { path: 'tree' });
    tree = normTree(obj.tree);
  }

  // --- sub-models: each entry runs through this same normalizer (subModel context) ---
  const models = normModels(obj.models);

  const model = {
    version, type, name, meta, settings, params, tables,
    models,
    states, transitions, strategies,
    tree,
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

// --- serializeModel: build a plain object mirroring the INPUT format from a normalized Model,
// then yaml.dump it. Every collapsing rule below is the exact inverse of a normalizeModel
// expansion, so parseModel(serializeModel(m)) always reproduces the same normalized Model —
// even where the plain object doesn't textually match what a human originally typed (e.g. a
// `start: well` shorthand and an explicit `start: {well: 1}` both normalize identically, so
// either serialization is round-trip-safe; we always emit the shorthand where one exists).

function collapseStart(startObj) {
  const keys = Object.keys(startObj);
  if (keys.length === 1 && startObj[keys[0]] === 1) return keys[0];
  return { ...startObj };
}

function settingsToPlain(s) {
  const out = {};
  if (s.cycles !== undefined) out.cycles = s.cycles;
  if (s.cycleYears !== 1) out.cycle = formatCycle(s.cycleYears);
  if (s.discount.cost !== 0 || s.discount.effect !== 0) out.discount = { ...s.discount };
  if (s.correction !== 'half-cycle') out.correction = s.correction;
  if (s.wtp !== null) out.wtp = s.wtp;
  if (s.psa.n !== 1000 || s.psa.seed !== 1 || (s.psa.correlations?.length ?? 0) > 0) {
    out.psa = { n: s.psa.n, seed: s.psa.seed };
    if ((s.psa.correlations?.length ?? 0) > 0) out.psa.correlations = s.psa.correlations;
  }
  if (s.start !== null) out.start = collapseStart(s.start);
  if (s.age !== null) out.age = s.age;
  return out;
}

// {value: x} alone -> bare x; anything with low/high/dist/source/notes stays a mapping.
function paramsToPlain(paramsMap) {
  const out = {};
  for (const [pname, p] of paramsMap) {
    const keys = Object.keys(p);
    out[pname] = (keys.length === 1 && keys[0] === 'value') ? p.value : { ...p };
  }
  return out;
}

function statesToPlain(states) {
  const out = {};
  for (const st of states) {
    const entry = { ...st.payoffs };
    if (st.source !== undefined) entry.source = st.source;
    if (st.notes !== undefined) entry.notes = st.notes;
    out[st.name] = entry;
  }
  return out;
}

function transitionsToPlain(transitions) {
  const out = {};
  for (const [from, row] of Object.entries(transitions)) {
    if (row.type === 'multinomial') {
      out[from] = { multinomial: { ...row.counts } };
      continue;
    }
    const rowOut = {};
    for (const [target, entry] of Object.entries(row.to)) {
      const keys = Object.keys(entry);
      rowOut[target] = (keys.length === 1 && keys[0] === 'p') ? entry.p : { ...entry };
    }
    out[from] = rowOut;
  }
  return out;
}

// {base: {overrides: {}}} alone (the implicit default) is omitted entirely.
function strategiesToPlain(strategies) {
  const names = Object.keys(strategies);
  if (names.length === 1 && names[0] === 'base' && Object.keys(strategies.base.overrides).length === 0)
    return undefined;
  const out = {};
  for (const [sname, s] of Object.entries(strategies)) out[sname] = { ...s.overrides };
  return out;
}

function treeNodeToPlain(node) {
  const out = {};
  if (node.p !== undefined) out.p = node.p;
  if (node.model !== undefined) out.model = node.model;
  if (node.with !== undefined) out.with = { ...node.with };
  if (node.delay !== undefined) out.delay = formatCycle(node.delay);
  if (node.source !== undefined) out.source = node.source;
  if (node.notes !== undefined) out.notes = node.notes;
  Object.assign(out, node.payoffs);

  // A child whose name collides with a reserved attribute key (e.g. a branch literally named
  // "cost" or "p") must go under an explicit `children:` block — emitting it inline would make
  // re-parsing treat the name as an attribute of THIS node instead of a child. Non-reserved
  // names keep emitting inline, as before.
  const explicitChildren = {};
  for (const child of node.children) {
    const plainChild = treeNodeToPlain(child);
    if (TREE_RESERVED_KEYS.has(child.name)) explicitChildren[child.name] = plainChild;
    else out[child.name] = plainChild;
  }
  if (Object.keys(explicitChildren).length > 0) out.children = explicitChildren;

  return out;
}

function modelToPlain(model) {
  const plain = {};
  if (model.version !== undefined) plain.econeval = model.version;
  plain.type = model.type;
  plain.name = model.name;
  if (model.meta && Object.keys(model.meta).length > 0) plain.meta = model.meta;

  const settingsPlain = settingsToPlain(model.settings);
  if (Object.keys(settingsPlain).length > 0) plain.settings = settingsPlain;

  if (model.params.size > 0) plain.params = paramsToPlain(model.params);
  if (model.tables && Object.keys(model.tables).length > 0) plain.tables = model.tables;

  if (model.models && Object.keys(model.models).length > 0) {
    const modelsPlain = {};
    for (const [mname, sub] of Object.entries(model.models)) modelsPlain[mname] = modelToPlain(sub);
    plain.models = modelsPlain;
  }

  if (model.type === 'markov') {
    if (model.states.length > 0) plain.states = statesToPlain(model.states);
    if (Object.keys(model.transitions).length > 0) plain.transitions = transitionsToPlain(model.transitions);
  }

  const strategiesPlain = strategiesToPlain(model.strategies);
  if (strategiesPlain !== undefined) plain.strategies = strategiesPlain;

  if (model.type === 'tree' && model.tree) plain.tree = { [model.tree.name]: treeNodeToPlain(model.tree) };

  if (model.layout !== null) plain.layout = model.layout;

  // Unrecognized extra fields (e.g. `description`, spread onto the model by normalizeModel).
  for (const k of Object.keys(model)) {
    if (!MODEL_FIELD_KEYS.has(k)) plain[k] = model[k];
  }

  return plain;
}

// yaml.dump with flowLevel:-1 guarantees block style everywhere, so a value containing '(' (a
// distribution call like `beta(202, 798)`) never lands inside a flow mapping `{...}` — the one
// place a bare comma inside the parens would be misparsed as a second field.
export function serializeModel(model) {
  return dump(modelToPlain(model), { flowLevel: -1, lineWidth: 120 });
}
