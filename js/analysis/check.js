import { compile, ExprError } from '../core/expr.js';
import { makeEnv } from '../engine/resolve.js';

const TOL = 1e-6;
const isRestSrc = (src) => typeof src === 'string' && src.trim() === 'rest';

// Sub-model lookup CHAINS lexically — own model's `models:` registry first, then ancestors up to
// the top level. This is a deliberate, hand-kept copy of js/engine/run.js's (unexported)
// lookupSubModel: controller ruling binds E_SUBMODEL_MISSING/E_SUBMODEL_CYCLE to the SAME
// chaining rule run.js actually uses at runtime, or check and run would silently disagree about
// which references are valid.
function lookupSubModel(registries, name) {
  for (const reg of registries) {
    if (reg && Object.prototype.hasOwnProperty.call(reg, name)) return reg[name];
  }
  return undefined;
}

// Evaluate one expression source (mean-mode env) purely to classify a failure into one of our
// finding codes. Returns the numeric value on success, or `undefined` on any failure (having
// already pushed a finding for the failures we recognize) — callers must treat `undefined` as
// "could not evaluate, skip downstream range/sum reasoning for this entry", never as 0.
function tryEval(src, env, path, findings) {
  if (src === undefined || src === null) return undefined;
  if (typeof src === 'number') return src;
  if (isRestSrc(src)) return undefined; // 'rest' is handled by the row/sibling-group logic, never compiled directly
  let expr;
  try {
    expr = compile(src);
  } catch {
    return undefined; // a syntax error here is not one of our ~13 codes — defensive no-op, never throw
  }
  try {
    return expr.eval(env);
  } catch (e) {
    if (e instanceof ExprError) {
      const um = /^unknown name: (.+)$/.exec(e.message);
      if (um) {
        const name = um[1];
        if (name === 'age') {
          findings.push({
            level: 'error', code: 'E_UNKNOWN_NAME', path,
            message: `'age' is referenced in '${src}' but settings.age is not set on this model — set settings.age to make 'age' available`,
          });
        } else {
          findings.push({
            level: 'error', code: 'E_UNKNOWN_NAME', path,
            message: `unknown name '${name}' referenced in expression '${src}'`,
          });
        }
        return undefined;
      }
      if (/^cycle in param resolution/.test(e.message)) {
        findings.push({ level: 'error', code: 'E_PARAM_CYCLE', path, message: e.message });
        return undefined;
      }
    }
    return undefined; // some other expr error (unknown function, bad lookup table, ...) — out of our code list
  }
}

// Shared row-sum / range / two-rest logic for BOTH a markov transition row and a tree sibling
// group. `entries`: [{target, src}]. `sumCode` distinguishes E_ROWSUM (markov) from E_TREE_PSUM
// (tree) — the one place the two contexts differ in vocabulary; everything else (two rests,
// out-of-range individual values, rest resolving out of [0,1]) is identical logic.
function evalProbGroup(entries, env, { levelPath, sumCode, findings }) {
  let sumKnown = 0;
  let restCount = 0;
  let anyEvalFailed = false;

  for (const { target, src } of entries) {
    if (isRestSrc(src)) { restCount++; continue; }
    if (src === undefined) { anyEvalFailed = true; continue; } // no dedicated code for "missing p"; skip
    const v = tryEval(src, env, `${levelPath}.${target}`, findings);
    if (v === undefined) { anyEvalFailed = true; continue; }
    if (v < -TOL || v > 1 + TOL) {
      findings.push({
        level: 'error', code: 'E_PRANGE', path: `${levelPath}.${target}`,
        message: `probability ${v} for '${target}' is outside [0,1]`,
      });
    }
    sumKnown += v;
  }

  if (restCount > 1) {
    findings.push({
      level: 'error', code: 'E_TWO_RESTS', path: levelPath,
      message: `more than one 'rest' among ${levelPath}`,
    });
    return;
  }
  if (anyEvalFailed) return; // can't reliably reason about a fill/sum when an entry didn't evaluate

  if (restCount === 1) {
    const restVal = 1 - sumKnown;
    if (restVal < -TOL || restVal > 1 + TOL) {
      findings.push({
        level: 'error', code: sumCode, path: levelPath,
        message: `'rest' resolves to ${restVal} in ${levelPath} (out of [0,1])`,
      });
    }
  } else if (Math.abs(sumKnown - 1) > TOL) {
    findings.push({
      level: 'error', code: sumCode, path: levelPath,
      message: `${levelPath}: probabilities sum to ${sumKnown}, expected 1`,
    });
  }
}

// Defensive re-check of table shape. model.js's normTables already enforces this at parse time
// (equal-length numeric-array columns) — this exists because check() is a general validator over
// any Model-shaped object, not only fresh parseModel() output (e.g. a Model mutated in place by a
// future editor UI, or hand-built in a test). Mirrors normTables' own rules exactly.
function checkTables(model, findings, pathPrefix) {
  for (const [tname, cols] of Object.entries(model.tables ?? {})) {
    const path = `${pathPrefix}tables.${tname}`;
    if (typeof cols !== 'object' || cols === null || Array.isArray(cols)) {
      findings.push({ level: 'error', code: 'E_TABLE_SHAPE', path, message: `table '${tname}' must be a mapping of columns` });
      continue;
    }
    const colNames = Object.keys(cols);
    let len;
    let bad = colNames.length === 0;
    for (const cname of colNames) {
      const col = cols[cname];
      if (!Array.isArray(col) || !col.every((x) => typeof x === 'number')) { bad = true; break; }
      if (len === undefined) len = col.length;
      else if (col.length !== len) { bad = true; break; }
    }
    if (bad) {
      findings.push({
        level: 'error', code: 'E_TABLE_SHAPE', path,
        message: `table '${tname}' columns must be equal-length numeric arrays`,
      });
    }
  }
}

// --- markov content ---

function checkMarkov(model, env, findings, pathPrefix) {
  const stateNames = new Set(model.states.map((s) => s.name));

  for (const s of model.states) {
    if (!(s.name in model.transitions)) {
      findings.push({
        level: 'error', code: 'E_NO_ROW', path: `${pathPrefix}transitions.${s.name}`,
        message: `state '${s.name}' has no transitions row`,
      });
    }
    for (const [k, v] of Object.entries(s.payoffs)) {
      tryEval(v, env, `${pathPrefix}states.${s.name}.${k}`, findings);
    }
  }

  for (const [from, row] of Object.entries(model.transitions)) {
    const fromPath = `${pathPrefix}transitions.${from}`;
    if (!stateNames.has(from)) {
      findings.push({
        level: 'error', code: 'E_UNKNOWN_STATE', path: fromPath,
        message: `transition source '${from}' is not a defined state`,
      });
    }

    if (row.type === 'multinomial') {
      for (const [target, src] of Object.entries(row.counts)) {
        const targetPath = `${fromPath}.${target}`;
        if (!stateNames.has(target)) {
          findings.push({
            level: 'error', code: 'E_UNKNOWN_STATE', path: targetPath,
            message: `transition target '${target}' is not a defined state`,
          });
        }
        tryEval(src, env, targetPath, findings);
      }
      continue;
    }

    for (const target of Object.keys(row.to)) {
      if (!stateNames.has(target)) {
        findings.push({
          level: 'error', code: 'E_UNKNOWN_STATE', path: `${fromPath}.${target}`,
          message: `transition target '${target}' is not a defined state`,
        });
      }
    }

    const entries = Object.entries(row.to).map(([target, e]) => ({ target, src: e.p }));
    evalProbGroup(entries, env, { levelPath: fromPath, sumCode: 'E_ROWSUM', findings });

    for (const [target, e] of Object.entries(row.to)) {
      if (e.cost !== undefined) tryEval(e.cost, env, `${fromPath}.${target}.cost`, findings);
      if (e.utility !== undefined) tryEval(e.utility, env, `${fromPath}.${target}.utility`, findings);
    }
  }

  if (model.settings.start) {
    for (const name of Object.keys(model.settings.start)) {
      if (!stateNames.has(name)) {
        findings.push({
          level: 'error', code: 'E_UNKNOWN_STATE', path: `${pathPrefix}settings.start`,
          message: `settings.start references unknown state '${name}'`,
        });
      }
    }

    // W_UNREACHABLE: BFS from the (state-valid) part of start over declared transition targets.
    const reachable = new Set(Object.keys(model.settings.start).filter((n) => stateNames.has(n)));
    const queue = [...reachable];
    while (queue.length > 0) {
      const cur = queue.shift();
      const row = model.transitions[cur];
      if (!row) continue;
      const targets = row.type === 'multinomial' ? Object.keys(row.counts) : Object.keys(row.to);
      for (const t of targets) {
        if (stateNames.has(t) && !reachable.has(t)) { reachable.add(t); queue.push(t); }
      }
    }
    for (const s of model.states) {
      if (!reachable.has(s.name)) {
        findings.push({
          level: 'warning', code: 'W_UNREACHABLE', path: `${pathPrefix}states.${s.name}`,
          message: `state '${s.name}' is unreachable from settings.start`,
        });
      }
    }
  }
  // No settings.start at all: reachability is undeterminable statically (an override could
  // supply it at run time) — deliberately not guessed at, matching the "no silent
  // renormalisation/invention" principle used elsewhere in this codebase.
}

// --- tree content (decision-shape + sibling-p sum + payoff unknown-names) ---

function walkTreeNode(node, isRoot, env, findings, path) {
  for (const [k, v] of Object.entries(node.payoffs)) {
    tryEval(v, env, `${path}.${k}`, findings);
  }
  if (node.children.length === 0) return;

  const missingP = node.children.some((c) => c.p === undefined);
  if (!isRoot && missingP) {
    findings.push({
      level: 'error', code: 'E_DECISION_BELOW_ROOT', path,
      message: `node '${node.name}' has children without 'p' — nested/sequential decisions are not supported in v1`,
    });
  } else if (!isRoot) {
    const entries = node.children.map((c) => ({ target: c.name, src: c.p }));
    evalProbGroup(entries, env, { levelPath: path, sumCode: 'E_TREE_PSUM', findings });
  }
  // isRoot: children are strategies, entered unconditionally — never sibling-p validated.

  for (const child of node.children) {
    walkTreeNode(child, false, env, findings, `${path}.${child.name}`);
  }
}

function checkTree(model, env, findings, pathPrefix) {
  if (!model.tree) return;
  walkTreeNode(model.tree, true, env, findings, `${pathPrefix}tree`);
}

// --- top-level recursion over the whole models: nesting (content validation, once per model) ---

function checkModelContent(model, findings, pathPrefix, parentEnv) {
  let env;
  try {
    env = makeEnv(model, { mode: 'mean', parent: parentEnv });
  } catch (e) {
    findings.push({
      level: 'error', code: 'E_UNKNOWN_NAME', path: pathPrefix || '(model)',
      message: `internal: could not build an evaluation environment: ${e.message}`,
    });
    return;
  }
  // Reserved words always known: t/time/state_time (sentinel cycle-1 values — check() evaluates
  // once, not across every cycle, so a value that only misbehaves at a later cycle is out of
  // scope here). `age` is known only when THIS model's own settings.age is set (never inherited
  // from a parent — matches markov.js, which always reads `model.settings.age` off the model
  // actually being run, never a parent's).
  env.vars.t = 1;
  env.vars.time = 0;
  env.vars.state_time = 1;
  if (model.settings?.age != null) env.vars.age = model.settings.age;

  for (const [pname, spec] of model.params) {
    if (spec.value !== undefined) tryEval(spec.value, env, `${pathPrefix}params.${pname}.value`, findings);
    if (spec.dist !== undefined) tryEval(spec.dist, env, `${pathPrefix}params.${pname}.dist`, findings);
  }

  checkTables(model, findings, pathPrefix);

  if (model.type === 'markov') checkMarkov(model, env, findings, pathPrefix);
  if (model.type === 'tree') checkTree(model, env, findings, pathPrefix);

  for (const [subName, sub] of Object.entries(model.models ?? {})) {
    checkModelContent(sub, findings, `${pathPrefix}models.${subName}.`, env);
  }
}

// --- attachment-graph-only pass: E_SUBMODEL_MISSING / E_SUBMODEL_CYCLE / with.start E_UNKNOWN_STATE.
// Walks `model:` edges starting from the top model's tree (and, recursively, into any resolved
// tree-type sub-model's own tree) — a SEPARATE traversal from checkModelContent's `models:`
// nesting walk above: this one follows attachment references, which may point sideways to a
// sibling sub-model or up to an ancestor's registry (see lookupSubModel), not just "down" into
// directly-nested sub-models. Content of every sub-model (attached or not) is already fully
// covered by checkModelContent; this pass never re-validates content, only reference resolution.

function checkAttachments(topModel, findings) {
  const registriesFor = (chain) => chain.map((m) => m.models ?? {});

  function walk(node, chain, attachPath, path) {
    if (node.model !== undefined) {
      const resolved = lookupSubModel(registriesFor(chain), node.model);
      if (!resolved) {
        findings.push({
          level: 'error', code: 'E_SUBMODEL_MISSING', path,
          message: `unknown sub-model '${node.model}' referenced at ${path}`,
        });
      } else if (attachPath.includes(resolved)) {
        findings.push({
          level: 'error', code: 'E_SUBMODEL_CYCLE', path,
          message: `sub-model reference cycle at '${path}': '${node.model}' revisits a model already on the attachment path`,
        });
      } else {
        const startOverride = node.with?.start;
        if (startOverride !== undefined && resolved.type === 'markov') {
          const known = new Set(resolved.states.map((s) => s.name));
          if (!known.has(startOverride)) {
            findings.push({
              level: 'error', code: 'E_UNKNOWN_STATE', path: `${path}.with.start`,
              message: `with.start '${startOverride}' is not a state of sub-model '${node.model}'`,
            });
          }
        }
        if (resolved.type === 'tree' && resolved.tree) {
          walk(resolved.tree, [resolved, ...chain], [...attachPath, resolved], `${path}[model:${node.model}]`);
        }
      }
    }
    for (const child of node.children) {
      walk(child, chain, attachPath, `${path}.${child.name}`);
    }
  }

  if (topModel.type === 'tree' && topModel.tree) {
    walk(topModel.tree, [topModel], [topModel], 'tree');
  }
}

// --- W_NO_DIST: whole-document warning (PSA is meaningless if nothing anywhere carries a dist) ---

function collectAllParams(model, out) {
  for (const spec of model.params.values()) out.push(spec);
  for (const sub of Object.values(model.models ?? {})) collectAllParams(sub, out);
}

/**
 * check(model) -> [{level:'error'|'warning', code, path, message}]
 *
 * Pure, read-only semantic validation over an already-normalized Model (js/core/model.js's
 * parseModel/normalizeModel output, or any equivalently-shaped object). Never throws — any
 * unexpected internal failure is reported as a finding, not propagated, per the module contract.
 */
export function check(model) {
  const findings = [];

  try {
    checkModelContent(model, findings, '', undefined);
  } catch (e) {
    findings.push({ level: 'error', code: 'E_UNKNOWN_NAME', path: '(model)', message: `internal error during content validation: ${e.message}` });
  }

  try {
    checkAttachments(model, findings);
  } catch (e) {
    findings.push({ level: 'error', code: 'E_SUBMODEL_MISSING', path: '(model)', message: `internal error during sub-model reference validation: ${e.message}` });
  }

  const allParams = [];
  try {
    collectAllParams(model, allParams);
  } catch {
    /* defensive; leave allParams as whatever was collected */
  }
  if (!allParams.some((p) => p.dist !== undefined)) {
    findings.push({
      level: 'warning', code: 'W_NO_DIST', path: 'params',
      message: 'no param anywhere in this model carries a dist — PSA would be meaningless',
    });
  }

  return findings;
}
