// Pure, DOM-free row-building, filtering, and findings-mapping for the outline sidebar.
// Every operation is declarative and fully tested (no imperative rendering, no state mutations).

// ================================================================================================
// ---------- Pure helpers (moved from inspector.js) ----------
// ================================================================================================

// scopePrefix(['a','b']) -> 'models.a.models.b.' — matches checkModelContent's own recursive
// pathPrefix building (js/analysis/check.js) exactly, since selection.modelPath segments are
// `models:` registry keys chained the same way (canvas.js's currentModelPath/scopedStore).
export function scopePrefix(modelPath) {
  return (modelPath ?? []).map((name) => `models.${name}.`).join('');
}

// check.js's tree-content paths start at 'tree' and OMIT the root node's own name (walkTreeNode is
// first called with path=`${pathPrefix}tree` for the root itself, then `${path}.${child.name}` for
// each descendant) — different from ops.nodeAt's path convention (root name included at path[0])
// and from the layout-key convention (root name included, '/'-joined). This is the one place those
// three conventions actually diverge; converts an ops-style node path to check.js's convention.
export function nodePathToCheckPath(path) {
  const rest = (path ?? []).slice(1);
  return rest.length ? `tree.${rest.join('.')}` : 'tree';
}

// ================================================================================================
// ---------- Row building ----------
// ================================================================================================

/**
 * buildOutline(model, modelPath = []) -> Row[]
 *
 * Row: {
 *   id,           // stable + unique: 'group:structure' | 'state:Well' | 'edge:Well>Sick'
 *                 //   | 'node:Root/A' | 'param:c_well' | 'submodel:post' | 'group:settings'
 *   kind,         // 'group' | 'state' | 'edge' | 'node' | 'param' | 'submodel'
 *   label,        // what the row shows
 *   detail,       // muted right-hand text: an edge's p, a node's payoff summary, '' if none
 *   depth,        // 0 = group header, then one per nesting level
 *   parentId,     // null for group headers — used for ancestor retention when filtering
 *   sel,          // the object handed to store.select, or null for a non-selectable row
 *   checkPaths,   // string[] this row owns findings for; [] when it owns none
 * }
 */
export function buildOutline(model, modelPath = []) {
  const rows = [];
  const prefix = scopePrefix(modelPath);

  // Helper to add a row
  function addRow(row) {
    rows.push(row);
  }

  // Helper to create group headers
  function addGroup(id, label, checkPaths = []) {
    addRow({
      id,
      kind: 'group',
      label,
      detail: '',
      depth: 0,
      parentId: null,
      sel: null,
      checkPaths,
    });
  }

  // ===== Structure group (states/edges or tree nodes) =====

  if (model.type === 'markov') {
    addGroup('group:structure', 'Structure');

    // Add states and their edges
    for (const state of model.states) {
      const stateName = state.name;
      addRow({
        id: `state:${stateName}`,
        kind: 'state',
        label: stateName,
        detail: '',
        depth: 1,
        parentId: 'group:structure',
        sel: { kind: 'state', id: stateName, modelPath },
        checkPaths: [
          `${prefix}states.${stateName}`,
          `${prefix}transitions.${stateName}`,
        ],
      });

      // Add edges from this state
      const row = model.transitions[stateName];
      if (row) {
        if (row.type === 'multinomial') {
          // Multinomial transitions: show count/total
          const total = Object.values(row.counts).reduce((a, b) => a + b, 0);
          for (const [target, count] of Object.entries(row.counts)) {
            addRow({
              id: `edge:${stateName}>${target}`,
              kind: 'edge',
              label: `→ ${target}`,
              detail: `${count}/${total}`,
              depth: 2,
              parentId: `state:${stateName}`,
              sel: { kind: 'edge', id: { from: stateName, to: target }, modelPath },
              checkPaths: [`${prefix}transitions.${stateName}.${target}`],
            });
          }
        } else {
          // Regular p-type transitions
          for (const [target, entry] of Object.entries(row.to)) {
            const pValue = entry.p;
            const detail = pValue === 'rest' ? 'rest' : (pValue === undefined || pValue === null ? '' : String(pValue));
            addRow({
              id: `edge:${stateName}>${target}`,
              kind: 'edge',
              label: `→ ${target}`,
              detail,
              depth: 2,
              parentId: `state:${stateName}`,
              sel: { kind: 'edge', id: { from: stateName, to: target }, modelPath },
              checkPaths: [`${prefix}transitions.${stateName}.${target}`],
            });
          }
        }
      }
    }
  } else if (model.type === 'tree') {
    addGroup('group:structure', 'Structure');

    // Recursively add tree nodes
    function addTreeNode(node, path) {
      const nodeId = path.join('/');
      const nodePath = path;
      addRow({
        id: `node:${nodeId}`,
        kind: 'node',
        label: node.name,
        detail: payoffSummary(node.payoffs),
        depth: path.length,
        parentId: path.length === 1 ? 'group:structure' : `node:${path.slice(0, -1).join('/')}`,
        sel: { kind: 'node', id: nodePath, modelPath },
        checkPaths: [`${prefix}${nodePathToCheckPath(nodePath)}`],
      });

      // Recursively add children
      for (const child of node.children) {
        addTreeNode(child, [...path, child.name]);
      }
    }

    if (model.tree) {
      addTreeNode(model.tree, [model.tree.name]);
    }
  }

  // ===== Submodels group (if any) =====
  if (model.models && Object.keys(model.models).length > 0) {
    addGroup('group:submodels', 'Sub-models');
    for (const [modelName] of Object.entries(model.models)) {
      addRow({
        id: `submodel:${modelName}`,
        kind: 'submodel',
        label: modelName,
        detail: '',
        depth: 1,
        parentId: 'group:submodels',
        sel: null, // sub-models not directly selectable from outline
        checkPaths: [`${prefix}models.${modelName}`],
      });
    }
  }

  // ===== Parameters group =====
  addGroup('group:parameters', 'Parameters');
  for (const [paramName] of model.params) {
    addRow({
      id: `param:${paramName}`,
      kind: 'param',
      label: paramName,
      detail: '',
      depth: 1,
      parentId: 'group:parameters',
      sel: null, // params not directly selectable from outline
      checkPaths: [`${prefix}params.${paramName}`],
    });
  }

  // ===== Settings group =====
  addGroup('group:settings', 'Settings', [`${prefix}settings`]);

  return rows;
}

// ================================================================================================
// ---------- Filtering (preserves ancestors) ----------
// ================================================================================================

/**
 * filterRows(rows, query) -> Row[]
 *
 * Case-insensitive substring match over label+detail.
 * A match keeps ALL its ancestors; a group header survives if any descendant matched.
 * Empty query returns rows unchanged.
 */
export function filterRows(rows, query) {
  if (!query || query.trim() === '') {
    return rows;
  }

  const lowerQuery = query.toLowerCase();

  // Mark which rows match the query (by checking label+detail)
  const matchedIds = new Set();
  const rowsById = new Map(rows.map((r) => [r.id, r]));

  for (const row of rows) {
    const text = (row.label + ' ' + row.detail).toLowerCase();
    if (text.includes(lowerQuery)) {
      matchedIds.add(row.id);
    }
  }

  // Mark rows whose descendants matched (keep ancestor chain alive)
  const surviveIds = new Set(matchedIds);
  for (const rowId of matchedIds) {
    // Walk up the parent chain and mark all ancestors
    let currentId = rowId;
    while (currentId) {
      surviveIds.add(currentId);
      const row = rowsById.get(currentId);
      currentId = row ? row.parentId : null;
    }
  }

  // Return rows that survived, preserving original order
  return rows.filter((r) => surviveIds.has(r.id));
}

// ================================================================================================
// ---------- Findings attachment (longest-match rule) ----------
// ================================================================================================

/**
 * attachFindings(rows, findings) -> {
 *   byRow,      // Map<rowId, finding[]>  — each finding goes to its LONGEST matching checkPath
 *   counts,     // Map<rowId, {errors, warnings}> — own findings PLUS every descendant's
 *   residual,   // finding[] matching no row at all; never swallowed
 * }
 */
export function attachFindings(rows, findings) {
  const byRow = new Map(); // rowId -> finding[]
  const residual = []; // findings that don't match any row

  // Build maps for fast lookup
  const rowsById = new Map(rows.map((r) => [r.id, r]));
  const parentMap = new Map(rows.map((r) => [r.id, r.parentId]));

  // For each finding, find the longest matching checkPath
  for (const finding of findings) {
    const path = finding.path;
    let bestMatch = null;
    let bestLength = -1;

    for (const row of rows) {
      for (const checkPath of row.checkPaths) {
        // Check if checkPath matches: exact match or prefix (path starts with checkPath + '.')
        if (path === checkPath) {
          if (checkPath.length > bestLength) {
            bestMatch = row.id;
            bestLength = checkPath.length;
          }
        } else if (path.startsWith(checkPath + '.')) {
          if (checkPath.length > bestLength) {
            bestMatch = row.id;
            bestLength = checkPath.length;
          }
        }
      }
    }

    if (bestMatch) {
      if (!byRow.has(bestMatch)) byRow.set(bestMatch, []);
      byRow.get(bestMatch).push(finding);
    } else {
      residual.push(finding);
    }
  }

  // Build counts: own findings + descendants
  const counts = new Map(); // rowId -> {errors, warnings}

  function getCounts(rowId) {
    if (counts.has(rowId)) return counts.get(rowId);

    let errors = 0;
    let warnings = 0;

    // Own findings
    if (byRow.has(rowId)) {
      for (const finding of byRow.get(rowId)) {
        if (finding.level === 'error') errors += 1;
        else if (finding.level === 'warning') warnings += 1;
      }
    }

    // Descendant findings
    for (const [otherId, otherParentId] of parentMap) {
      if (otherParentId === rowId) {
        const descendantCounts = getCounts(otherId);
        errors += descendantCounts.errors;
        warnings += descendantCounts.warnings;
      }
    }

    if (errors > 0 || warnings > 0) {
      counts.set(rowId, { errors, warnings });
    }

    return { errors, warnings };
  }

  // Compute counts for all rows
  for (const row of rows) {
    getCounts(row.id);
  }

  return { byRow, counts, residual };
}

// ================================================================================================
// ---------- Row selection (view helpers, Task 10) ----------
// ================================================================================================

function sameModelPath(a, b) {
  const x = a ?? [];
  const y = b ?? [];
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i += 1) if (x[i] !== y[i]) return false;
  return true;
}

// A Row's `sel.id` is either a bare string (state/param), an array of names (a tree node's
// root-inclusive path), or {from, to} (an edge) — never anything deeper. Plain shallow equality
// covers all three shapes without needing a general deep-equal.
function idsEqual(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => v === b[i]);
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    return a.from === b.from && a.to === b.to;
  }
  return false;
}

/**
 * rowForSelection(rows, selection) -> Row | null
 *
 * Finds the row a store selection ({kind, id, modelPath}) corresponds to. Matches row.sel by
 * (kind, id, modelPath) for canvas-facing rows (state/edge/node). Param rows carry sel:null
 * (params are not canvas entities — buildOutline never hands one to store.select), but the
 * outline still drives a param row's expansion through the very same store.select mechanism (the
 * DOM layer synthesizes {kind:'param', id: name, modelPath: []} on a param row click, reusing
 * store.js's existing 'param' selection-validity support rather than inventing a second, parallel
 * local-selection concept) — so a {kind:'param'} selection falls back to matching a param row by
 * name.
 */
export function rowForSelection(rows, selection) {
  if (!selection || selection.kind == null) return null;
  const modelPath = selection.modelPath ?? [];
  for (const row of rows) {
    if (!row.sel) continue;
    if (row.sel.kind !== selection.kind) continue;
    if (!sameModelPath(row.sel.modelPath, modelPath)) continue;
    if (idsEqual(row.sel.id, selection.id)) return row;
  }
  if (selection.kind === 'param' && modelPath.length === 0) {
    return rows.find((r) => r.kind === 'param' && r.label === selection.id) ?? null;
  }
  return null;
}

/**
 * collapseFilter(rows, collapsedIds) -> Row[]
 *
 * Drops every row whose ancestor chain (walking parentId) passes through a collapsed id.
 * collapsedIds only ever holds group-row ids in the DOM layer (only group headers are
 * individually collapsible there), but this walks the general parentId chain rather than
 * checking `kind`, so it composes correctly with any future collapsible row too.
 */
export function collapseFilter(rows, collapsedIds) {
  const byId = new Map(rows.map((r) => [r.id, r]));
  return rows.filter((row) => {
    let pid = row.parentId;
    while (pid) {
      if (collapsedIds.has(pid)) return false;
      pid = byId.get(pid)?.parentId ?? null;
    }
    return true;
  });
}

// ================================================================================================
// ---------- Helpers ----------
// ================================================================================================

// Summarize payoffs: "cost 100   utility 0.8"
function payoffSummary(payoffs) {
  if (!payoffs) return '';
  const keys = Object.keys(payoffs);
  if (keys.length === 0) return '';
  return keys.map((k) => `${k} ${payoffs[k]}`).join('   ');
}
