# econeval Editor Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mode-based canvas editor with a modeless one (decide-at-drop drag, Space to force an arrow, double-click to create), give every object a real hit area, and replace the three-tab inspector with a single filterable outline of the whole document.

**Architecture:** The model document stays the single source of truth; every edit is still a pure `(model) -> newModel` op dispatched through `store.applyOp`. `canvas.js` (1011 lines) and `inspector.js` (978 lines) split along the pure/DOM line the codebase already uses — pure geometry and outline-building get DOM-free modules with `node --test` coverage; the DOM modules become thin shells over them.

**Tech Stack:** Vanilla JS ES modules, no build step, no new npm deps. `node --test` for pure modules. SVG canvas. Netlify static hosting (manual deploy).

**Spec:** `docs/superpowers/specs/2026-08-24-econeval-editor-design.md` — binding. It supersedes the editor sections of `docs/superpowers/specs/2026-08-23-econeval-design.md`; everything else in that older spec (format, engines, analyses) still holds.

## Global Constraints

- ES modules, no framework, no build step, no new npm dependencies. UI under `js/ui/`, styles under `css/`.
- Tests run as `npm test` → `node --test test/*.test.js`. **Never `node --test <dir>`** — it fails on Node 26; the glob must be shell-expanded.
- All 392 existing tests stay green after every task. Run the full suite, not just the new file.
- Pure logic lives in DOM-free modules with tests. DOM modules hold no business logic.
- Every model mutation goes through `store.applyOp`; ops return new models via `structuredClone`-then-edit and never mutate their input.
- `flush()` is called before every op-producing gesture (pointerdown handlers, key handlers, rename commit) so a pending debounced YAML edit is committed first. This rule is unchanged and load-bearing.
- One undo entry per user gesture — a node drag is one entry, not one per pixel.
- Errors surfaced, never swallowed: op failures go to the `#canvas-toast` strip. No `window.confirm` / `window.alert` / `prompt` anywhere in canvas or inspector code (they block the extension-driven e2e run).
- English UI copy, sentence case, verbs on buttons. Every button a real `<button>` with `aria-label` + `title`. Visible keyboard focus everywhere.
- Only `--accent` marks selection, and only as a stroke — never a fill change (design tokens doc).
- Commit after every task. Push after every task (push-always applies to this repo).
- Deploy is **manual** and needs the site flag, because the CLI's global cache points at a different site:
  `netlify deploy --prod --dir . --site 4c526e64-937b-4c3a-a548-f701d9804a56`

## Existing interfaces these tasks build on (authoritative)

```js
// js/ui/ops.js — pure (model, ...) -> newModel, throw Error on invalid input
nodeAt(model, path)                    // path = names from root INCLUSIVE: ['Root','A','Win']
addChild(model, path, name?)           // new child p: sibling-has-rest ? 0 : 'rest'; root child gets NO p
deleteNode(model, path); renameNode(model, path, newName)
addState/renameState/deleteState/addTransition/deleteTransition/setTransitionAttr/setStatePayoff
setLayout(model, key, xy)              // xy null removes; works on BOTH model types
// module-private helpers already in ops.js, reused by Task 2:
//   rekeyLayoutSubtree(layout, oldPrefix, newPrefix), scrubLayoutSubtree(layout, prefix), omitKey(obj, key)

// js/core/model.js
parseModel(text) -> Model              // model.layout is `obj.layout ?? null` — ABSENT layout is null, not {}
serializeModel(model) -> yaml text     // emits `layout` only when model.layout !== null

// js/analysis/check.js
check(model) -> [{level:'error'|'warning', code, path, message}]
// paths: 'states.<name>.<payoff>', 'transitions.<from>', 'transitions.<from>.<to>[.cost|.utility]',
//        'params.<name>.value|.dist', 'settings.<key>', 'tree' / 'tree.<child>.<grandchild>' (root name OMITTED),
//        sub-models prefixed 'models.<name>.'

// js/ui/inspector.js — pure helpers already exported and tested (test/inspector-match.test.js)
scopePrefix(modelPath)                 // ['a','b'] -> 'models.a.models.b.'
nodePathToCheckPath(path)              // ['Root','A','Win'] -> 'tree.A.Win'  (root name dropped)
countByLevel(findings)
```

Layout-key rule (binding): `layout` keys are **state names** for markov, **`/`-joined root-inclusive node paths** for trees (`Root/A/Win`).

---

### Task 1: Extract `scopedStore` into its own module

`inspector.js` currently imports `scopedStore` from `canvas.js` — a backwards dependency between two peer UI modules that blocks the canvas split in Tasks 3-5. Pure move, no behaviour change.

**Files:**
- Create: `js/ui/scoped-store.js`
- Modify: `js/ui/canvas.js:139-170` (delete `scopedStore` + `scopedStoreFor`, import them instead), `js/ui/inspector.js:42` (import from the new module), `test/canvas-model.test.js:3` (repoint its `scopedStore` import)
- Test: `test/scoped-store.test.js`

**Interfaces:**
- Produces:

```js
export function scopedStore(store, modelName)   // store-shaped wrapper: .get().model is
                                                // store.get().model.models[modelName]; .applyOp maps the
                                                // fn over that sub-model and splices it back; .select
                                                // PREPENDS modelName to sel.modelPath; undo/redo/
                                                // markSaved/subscribe pass straight through
export function scopedStoreFor(baseStore, path) // path.reduce((s, name) => scopedStore(s, name), baseStore)
```

- [ ] **Step 1: Write the failing test**

```js
// test/scoped-store.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { scopedStore, scopedStoreFor } from '../js/ui/scoped-store.js';

// Minimal store-shaped fake: enough surface for the wrapper's contract.
function fakeStore(model) {
  const state = { model, selection: { kind: null, id: null } };
  const calls = [];
  return {
    calls,
    get: () => state,
    applyOp(fn) { state.model = fn(state.model); calls.push('applyOp'); },
    select(sel) { state.selection = sel; calls.push('select'); },
    undo() { calls.push('undo'); },
    redo() { calls.push('redo'); },
    markSaved() { calls.push('markSaved'); },
    subscribe() { calls.push('subscribe'); },
  };
}

const doc = () => ({ name: 'top', models: { inner: { name: 'inner', models: { deep: { name: 'deep' } } } } });

test('get() returns the named sub-model as .model', () => {
  const s = scopedStore(fakeStore(doc()), 'inner');
  assert.equal(s.get().model.name, 'inner');
});

test('get() returns null when the sub-model is absent', () => {
  const s = scopedStore(fakeStore({ name: 'top' }), 'missing');
  assert.equal(s.get().model, null);
});

test('applyOp splices the edited sub-model back into a fresh top-level model', () => {
  const base = fakeStore(doc());
  const s = scopedStore(base, 'inner');
  s.applyOp((m) => ({ ...m, name: 'edited' }));
  assert.equal(base.get().model.models.inner.name, 'edited');
  assert.equal(base.get().model.name, 'top');           // outer untouched
});

test('applyOp throws when the sub-model is gone', () => {
  const s = scopedStore(fakeStore({ name: 'top' }), 'missing');
  assert.throws(() => s.applyOp((m) => m), /missing/);
});

test('select prepends this wrapper name onto modelPath', () => {
  const base = fakeStore(doc());
  scopedStore(base, 'inner').select({ kind: 'state', id: 'Well' });
  assert.deepEqual(base.get().selection.modelPath, ['inner']);
});

test('chained wrappers reach nested models and stamp both names in order', () => {
  const base = fakeStore(doc());
  const s = scopedStoreFor(base, ['inner', 'deep']);
  assert.equal(s.get().model.name, 'deep');
  s.select({ kind: 'state', id: 'X' });
  assert.deepEqual(base.get().selection.modelPath, ['inner', 'deep']);
});

test('scopedStoreFor with an empty path returns the base store itself', () => {
  const base = fakeStore(doc());
  assert.equal(scopedStoreFor(base, []), base);
});

test('undo/redo/markSaved/subscribe pass through', () => {
  const base = fakeStore(doc());
  const s = scopedStore(base, 'inner');
  s.undo(); s.redo(); s.markSaved(); s.subscribe(() => {});
  assert.deepEqual(base.calls, ['undo', 'redo', 'markSaved', 'subscribe']);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/scoped-store.test.js`
Expected: FAIL — `Cannot find module '../js/ui/scoped-store.js'`

- [ ] **Step 3: Move the code**

Cut `scopedStore` (`canvas.js:139-166`) and `scopedStoreFor` (`canvas.js:168-170`) verbatim into `js/ui/scoped-store.js`, keeping their explanatory comments. In `canvas.js` replace them with `import { scopedStore, scopedStoreFor } from './scoped-store.js';` and drop the local `export` keyword on `scopedStore`.

Two other files import it from `canvas.js` and must be repointed in this same step, or the suite breaks: `inspector.js:42` (`import { scopedStore } from './canvas.js';` → `'./scoped-store.js'`), and `test/canvas-model.test.js:3`, which pulls `scopedStore` in alongside the geometry helpers — split that line into two imports, leaving `edgePath`/`selfLoopPath`/`edgeLabelPos`/`NODE_R` on `canvas.js` for now (Task 3 moves those). Its existing scopedStore tests, which drive a real `createStore`, stay exactly where they are — they complement the fake-store tests above rather than duplicating them.

- [ ] **Step 4: Run the full suite**

Run: `npm test`
Expected: PASS — 392 existing + 7 new.

- [ ] **Step 5: Commit and push**

```bash
git add js/ui/scoped-store.js js/ui/canvas.js js/ui/inspector.js test/scoped-store.test.js
git commit -m "refactor: extract scopedStore into js/ui/scoped-store.js"
git push
```

---

### Task 2: New model ops — `moveNode` and `clearLayout`

**Files:**
- Modify: `js/ui/ops.js` (append to the tree section, after `deleteNode`)
- Test: `test/ops-move-node.test.js`, `test/ops-clear-layout.test.js`

**Interfaces:**
- Produces:

```js
moveNode(model, path, newParentPath)  // tree only. Re-parents the node at `path` (with its whole
                                      // subtree) under the node at `newParentPath`.
                                      // p rules: newParent is the root (length 1) -> DELETE node.p
                                      //          else node.p === undefined -> set 'rest', or 0 if a
                                      //            destination sibling already has p === 'rest'
                                      //          else -> leave node.p untouched
                                      // Layout: rekeyLayoutSubtree(oldPath.join('/'), newPath.join('/'))
                                      // Throws: moving the root; newParentPath === path; newParentPath
                                      //         inside path's own subtree; a destination sibling with
                                      //         the same name (rejected, never silently renamed)
clearLayout(model, key?)              // key given: tree -> scrubLayoutSubtree(prefix); markov -> omitKey.
                                      // key omitted: drop everything. Result normalized to `null` when
                                      // empty, matching parseModel's `obj.layout ?? null` (so the model
                                      // still round-trips through serialize/parse).
```

- [ ] **Step 1: Write the failing tests**

```js
// test/ops-move-node.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';
import * as ops from '../js/ui/ops.js';

// Root children A, B are strategies (no p). A's children include a 'rest' sibling; B's do not.
// Both A and B have a child named 'Win' — that pair is the name-collision case.
const M = () => parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Win: {p: 0.5, utility: 10}
      Lose: {p: rest, utility: 0}
    B:
      Win: {p: 1, utility: 5}
layout:
  Root: [0, 0]
  Root/A: [10, 10]
  Root/A/Win: [20, 20]
  Root/A/Lose: [20, 40]
  Root/B: [10, 60]
  Root/B/Win: [20, 60]
`);

test('re-parents a node and keeps an existing p untouched', () => {
  const m2 = ops.moveNode(M(), ['Root', 'A', 'Lose'], ['Root', 'B']);
  assert.equal(ops.nodeAt(m2, ['Root', 'B', 'Lose']).p, 'rest');
  assert.deepEqual(ops.nodeAt(m2, ['Root', 'A']).children.map((c) => c.name), ['Win']);
  assert.deepEqual(ops.nodeAt(m2, ['Root', 'B']).children.map((c) => c.name), ['Win', 'Lose']);
});

test('re-parenting moves the whole subtree layout, not just the node', () => {
  const m2 = ops.moveNode(M(), ['Root', 'B'], ['Root', 'A']);
  assert.deepEqual(m2.layout['Root/A/B'], [10, 60]);
  assert.deepEqual(m2.layout['Root/A/B/Win'], [20, 60]);
  assert.ok(!('Root/B' in m2.layout));
  assert.ok(!('Root/B/Win' in m2.layout));
});

test('promoting to a root child removes p (strategies are unconditional)', () => {
  const m2 = ops.moveNode(M(), ['Root', 'A', 'Win'], ['Root']);
  assert.equal(ops.nodeAt(m2, ['Root', 'Win']).p, undefined);
  assert.deepEqual(m2.layout['Root/Win'], [20, 20]);
});

test('demoting a strategy gives it 0 when a destination sibling already has rest', () => {
  // A's children: Win (0.5), Lose (rest) -> B lands beside a 'rest' sibling
  const m2 = ops.moveNode(M(), ['Root', 'B'], ['Root', 'A']);
  assert.equal(ops.nodeAt(m2, ['Root', 'A', 'B']).p, 0);
});

test("demoting a strategy gives it 'rest' when no destination sibling has one", () => {
  // B's children: Win (1) -> no 'rest' present
  const m2 = ops.moveNode(M(), ['Root', 'A'], ['Root', 'B']);
  assert.equal(ops.nodeAt(m2, ['Root', 'B', 'A']).p, 'rest');
});

test('rejects moving the root', () => {
  assert.throws(() => ops.moveNode(M(), ['Root'], ['Root', 'A']), /root/i);
});

test('rejects dropping a node onto itself', () => {
  assert.throws(() => ops.moveNode(M(), ['Root', 'A'], ['Root', 'A']), /itself/i);
});

test('rejects dropping a node into its own subtree', () => {
  assert.throws(() => ops.moveNode(M(), ['Root', 'A'], ['Root', 'A', 'Win']), /descendant/i);
});

test('rejects a sibling name collision instead of renaming silently', () => {
  assert.throws(() => ops.moveNode(M(), ['Root', 'A', 'Win'], ['Root', 'B']), /already exists/i);
});

test('rejects a non-tree model and an unknown path', () => {
  const markov = parseModel('econeval: 1\ntype: markov\nname: m\nstates:\n  a: {cost: 0}\ntransitions:\n  a: {a: 1}\n');
  assert.throws(() => ops.moveNode(markov, ['Root'], ['Root']), /tree/);
  assert.throws(() => ops.moveNode(M(), ['Root', 'Nope'], ['Root', 'B']), /Nope/);
});

test('does not mutate its input', () => {
  const m = M();
  ops.moveNode(m, ['Root', 'A', 'Lose'], ['Root', 'B']);
  assert.deepEqual(ops.nodeAt(m, ['Root', 'A']).children.map((c) => c.name), ['Win', 'Lose']);
});

test('the result round-trips through serialize/parse', () => {
  const m2 = ops.moveNode(M(), ['Root', 'B'], ['Root', 'A']);
  assert.deepEqual(parseModel(serializeModel(m2)), m2);
});
```

```js
// test/ops-clear-layout.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel, serializeModel } from '../js/core/model.js';
import * as ops from '../js/ui/ops.js';

const TREE = () => parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Win: {p: rest, utility: 1}
    B:
      Win: {p: rest, utility: 2}
layout:
  Root: [0, 0]
  Root/A: [10, 10]
  Root/A/Win: [20, 20]
  Root/B: [10, 60]
`);

const MARKOV = () => parseModel(`
econeval: 1
type: markov
name: m
states:
  well: {cost: 1, utility: 1}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
layout:
  well: [10, 10]
  dead: [90, 10]
`);

test('clearLayout() with no key drops the whole layout as null', () => {
  const m2 = ops.clearLayout(TREE());
  assert.equal(m2.layout, null);
});

test('clearLayout(key) on a tree drops the node AND its subtree', () => {
  const m2 = ops.clearLayout(TREE(), 'Root/A');
  assert.ok(!('Root/A' in m2.layout));
  assert.ok(!('Root/A/Win' in m2.layout));
  assert.deepEqual(m2.layout['Root/B'], [10, 60]);
  assert.deepEqual(m2.layout.Root, [0, 0]);
});

test('clearLayout(key) on a markov model drops exactly that key', () => {
  const m2 = ops.clearLayout(MARKOV(), 'well');
  assert.ok(!('well' in m2.layout));
  assert.deepEqual(m2.layout.dead, [90, 10]);
});

test('clearing the last remaining key normalizes layout to null', () => {
  let m = ops.clearLayout(MARKOV(), 'well');
  m = ops.clearLayout(m, 'dead');
  assert.equal(m.layout, null);
});

test('clearing a key that is not there is a no-op, not an error', () => {
  const m2 = ops.clearLayout(MARKOV(), 'ghost');
  assert.deepEqual(m2.layout, MARKOV().layout);
});

test('a cleared model round-trips through serialize/parse', () => {
  const m2 = ops.clearLayout(TREE());
  assert.deepEqual(parseModel(serializeModel(m2)), m2);
});

test('does not mutate its input', () => {
  const m = TREE();
  ops.clearLayout(m, 'Root/A');
  assert.deepEqual(m.layout['Root/A'], [10, 10]);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `node --test test/ops-move-node.test.js test/ops-clear-layout.test.js`
Expected: FAIL — `ops.moveNode is not a function`, `ops.clearLayout is not a function`

- [ ] **Step 3: Implement both ops in `js/ui/ops.js`**

```js
// Re-parents the node at `path` (with its whole subtree) under the node at `newParentPath`. The
// tree reading of "connect these two": dragging A onto B makes A a child of B. Root-boundary `p`
// handling mirrors addChild exactly — a root child is a strategy, entered unconditionally, so it
// never carries a `p`; anywhere else a node needs one. An EXISTING p is never rewritten (the
// surprise principle: a silent renumbering would be magic; check() flags a bad row sum instead).
export function moveNode(model, path, newParentPath) {
  assertTree(model, 'moveNode');
  if (!Array.isArray(path) || path.length === 0)
    throw new Error('moveNode: path must be a non-empty array of names');
  if (!Array.isArray(newParentPath) || newParentPath.length === 0)
    throw new Error('moveNode: newParentPath must be a non-empty array of names');
  if (path.length === 1)
    throw new Error('moveNode: the root node cannot be moved');

  const samePath = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  if (samePath(path, newParentPath))
    throw new Error('moveNode: a node cannot be dropped onto itself');
  // newParentPath sits inside path's own subtree iff it is longer and shares path as a prefix.
  if (newParentPath.length > path.length && path.every((v, i) => v === newParentPath[i]))
    throw new Error('moveNode: a node cannot be moved into its own descendant');

  const m = clone(model);

  // Dropped onto the parent it already has: a no-op, not a collision. Without this the sibling-name
  // scan below would find the node itself (it has not been spliced out yet) and throw a false
  // "already exists". Same idiom renameState/renameNode use for renaming to the current name.
  if (path.slice(0, -1).join('/') === newParentPath.join('/')) return m;

  const node = nodeAt(m, path);                       // validates path
  const newParent = nodeAt(m, newParentPath);         // validates destination
  const oldParent = nodeAt(m, path.slice(0, -1));

  if (newParent.children.some((c) => c.name === node.name))
    throw new Error(`moveNode: '${newParent.name}' already has a child named '${node.name}'`);

  oldParent.children.splice(oldParent.children.indexOf(node), 1);

  if (newParentPath.length === 1) {
    delete node.p;                                    // promoted to a strategy
  } else if (node.p === undefined) {
    const hasRest = newParent.children.some((c) => c.p === 'rest');
    node.p = hasRest ? 0 : 'rest';
  }

  newParent.children.push(node);

  const oldPrefix = path.join('/');
  const newPrefix = [...newParentPath, node.name].join('/');
  m.layout = rekeyLayoutSubtree(m.layout, oldPrefix, newPrefix);

  return m;
}

// Tidy: drops explicit positions so layouts.js's autoMarkov/autoTree takes over again. Without a
// key, the whole layout goes. With one, a tree drops the named node's entire subtree (a child's
// pinned position is meaningless once its parent moves) while a markov model drops exactly the one
// state key. An empty result is normalized back to `null` — parseModel produces `obj.layout ?? null`
// for an absent layout, so anything else would fail to round-trip through serialize/parse.
export function clearLayout(model, key) {
  const m = clone(model);
  if (key === undefined || key === null) {
    m.layout = null;
    return m;
  }
  if (m.layout) {
    m.layout = m.type === 'tree' ? scrubLayoutSubtree(m.layout, key) : omitKey(m.layout, key);
    if (Object.keys(m.layout).length === 0) m.layout = null;
  }
  return m;
}
```

- [ ] **Step 4: Run the full suite**

Run: `npm test`
Expected: PASS — 392 existing + 7 (Task 1) + 19 new.

- [ ] **Step 5: Commit and push**

```bash
git add js/ui/ops.js test/ops-move-node.test.js test/ops-clear-layout.test.js
git commit -m "feat(ops): add moveNode (tree re-parent) and clearLayout (tidy)"
git push
```

---

### Task 3: `js/ui/canvas/geometry.js` — pure geometry, hit predicates, snap, fit

**Files:**
- Create: `js/ui/canvas/geometry.js`
- Modify: `js/ui/canvas.js:27-124` (delete the moved functions, import them instead), `test/canvas-model.test.js` (repoint imports; assertions unchanged)
- Test: `test/canvas-geometry.test.js`

**Interfaces:**
- Consumes: nothing.
- Produces (moved verbatim from `canvas.js`, plus four new functions):

```js
export const NODE_R = 26;
export const ROOT_HALF = 26; export const TERMINAL_HALF = 15;
export const STADIUM_W = 120; export const STADIUM_H = 32;
export const GRID = 12;                          // matches css/canvas.css's 12px dot grid
export const HIT_SLACK = 6;                      // extra forgiveness around every shape

edgePath(from, to, r)                            // moved, unchanged
selfLoopPath(xy, r)                              // moved, unchanged
edgeLabelPos(from, to)                           // moved, unchanged

// NEW. `hit` descriptors live on every nodeIndex entry (Task 4 builds them):
//   {shape:'circle', r} | {shape:'rect', w, h} | {shape:'stadium', w, h}
hitShapeFor(kind)                                // 'state'|'chance' -> circle r NODE_R
                                                 // 'root'           -> rect  52x52
                                                 // 'submodel'       -> stadium 120x32
                                                 // 'terminal'       -> rect  16x30
isInside(point, xy, hit, slack = HIT_SLACK)      // -> bool. circle: hypot <= r+slack.
                                                 // rect: |dx| <= w/2+slack && |dy| <= h/2+slack.
                                                 // stadium: rect test, then round the two end caps.
pickNode(point, nodeIndex, slack = HIT_SLACK)    // -> the LAST matching entry (topmost-rendered
                                                 // wins on overlap), or null
snapToGrid(xy, grid = GRID)                      // -> [x, y] rounded to the nearest multiple
fitBox(positions, pad = 60)                      // positions = [[x,y], ...] -> {x, y, w, h};
                                                 // empty input -> {x:0, y:0, w:900, h:640} (BASE_W/BASE_H)
```

- [ ] **Step 1: Write the failing test**

```js
// test/canvas-geometry.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  NODE_R, GRID, hitShapeFor, isInside, pickNode, snapToGrid, fitBox,
} from '../js/ui/canvas/geometry.js';

test('hitShapeFor maps every node kind to its real outline', () => {
  assert.deepEqual(hitShapeFor('state'), { shape: 'circle', r: NODE_R });
  assert.deepEqual(hitShapeFor('chance'), { shape: 'circle', r: NODE_R });
  assert.deepEqual(hitShapeFor('root'), { shape: 'rect', w: 52, h: 52 });
  assert.deepEqual(hitShapeFor('submodel'), { shape: 'stadium', w: 120, h: 32 });
  assert.deepEqual(hitShapeFor('terminal'), { shape: 'rect', w: 16, h: 30 });
});

test('circle hit: inside, on the rim, inside the slack, outside', () => {
  const c = hitShapeFor('state');   // r 26, slack 6
  assert.equal(isInside([100, 100], [100, 100], c), true);
  assert.equal(isInside([126, 100], [100, 100], c), true);   // exactly on the rim
  assert.equal(isInside([131, 100], [100, 100], c), true);   // within slack
  assert.equal(isInside([133, 100], [100, 100], c), false);  // past r + slack
});

test('rect hit respects width and height separately', () => {
  const r = hitShapeFor('terminal');  // 16 x 30, slack 6 -> +-14 x, +-21 y
  assert.equal(isInside([100, 100], [100, 100], r), true);
  assert.equal(isInside([113, 100], [100, 100], r), true);
  assert.equal(isInside([116, 100], [100, 100], r), false);
  assert.equal(isInside([100, 120], [100, 100], r), true);
  assert.equal(isInside([100, 123], [100, 100], r), false);
});

test('stadium hit rounds its end caps instead of squaring them', () => {
  const s = hitShapeFor('submodel');  // 120 x 32; caps are circles of r 16 at x = +-44
  assert.equal(isInside([100, 100], [100, 100], s, 0), true);
  assert.equal(isInside([158, 100], [100, 100], s, 0), true);   // on the cap's far edge
  assert.equal(isInside([161, 100], [100, 100], s, 0), false);
  // The corner of the bounding box is OUTSIDE the pill, which a rect test would wrongly accept.
  assert.equal(isInside([159, 115], [100, 100], s, 0), false);
});

test('pickNode returns the topmost (last) match when shapes overlap', () => {
  const index = [
    { key: 'under', xy: [100, 100], hit: hitShapeFor('state') },
    { key: 'over', xy: [110, 100], hit: hitShapeFor('state') },
  ];
  assert.equal(pickNode([105, 100], index).key, 'over');
  // 70 is 30 from 'under' (within r+slack 32) but 40 from 'over' (outside it). Probing at 80 would
  // land inside BOTH nodes' slack once the default forgiveness applies, and topmost-wins would
  // return 'over' — the opposite of what this assertion is checking.
  assert.equal(pickNode([70, 100], index).key, 'under');
  assert.equal(pickNode([400, 400], index), null);
});

test('pickNode is forgiving by HIT_SLACK by default', () => {
  // Probe BETWEEN the bare radius and radius + slack: a default of 0 would miss this, a default of
  // HIT_SLACK finds it. Probing inside the bare radius cannot tell the two defaults apart.
  const index = [{ key: 'only', xy: [100, 100], hit: hitShapeFor('state') }];
  assert.equal(pickNode([129, 100], index).key, 'only');   // 29 > r 26, <= r + slack 32
  assert.equal(pickNode([129, 100], index, 0), null);      // explicit 0 refuses it
});

test('snapToGrid rounds to the nearest 12px multiple', () => {
  assert.equal(GRID, 12);
  assert.deepEqual(snapToGrid([0, 0]), [0, 0]);
  assert.deepEqual(snapToGrid([17, 5]), [12, 0]);
  assert.deepEqual(snapToGrid([18, 19]), [24, 24]);
  // -5 / 12 rounds to -0; assert/strict tells -0 and 0 apart, so this pins the normalization.
  assert.deepEqual(snapToGrid([-17, -5]), [-12, 0]);
  assert.ok(Object.is(snapToGrid([-17, -5])[1], 0), 'negative zero must be normalized to +0');
});

test('fitBox wraps every position with padding', () => {
  const box = fitBox([[100, 100], [300, 200]], 50);
  assert.deepEqual(box, { x: 50, y: 50, w: 300, h: 200 });
});

test('fitBox falls back to the base viewBox when there is nothing to fit', () => {
  assert.deepEqual(fitBox([], 60), { x: 0, y: 0, w: 900, h: 640 });
});

test('fitBox never produces a zero-sized box for a single node', () => {
  const box = fitBox([[100, 100]], 60);
  assert.equal(box.w, 120);
  assert.equal(box.h, 120);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/canvas-geometry.test.js`
Expected: FAIL — `Cannot find module '../js/ui/canvas/geometry.js'`

- [ ] **Step 3: Create the module**

Move `NODE_R`, `ROOT_HALF`, `TERMINAL_HALF`, `STADIUM_W`, `STADIUM_H`, `STADIUM_INSET`, `HALO_GAP`, `SELF_LOOP_SPREAD`, `SELF_LOOP_HEIGHT`, `BASE_W`, `BASE_H`, `edgePath`, `selfLoopPath`, `selfLoopLabelPos`, `edgeLabelPos` out of `canvas.js` into the new module with their comments intact, exporting all of them. Then add the four new functions. The stadium test is the one with real geometry in it:

```js
// A stadium is a rect of (w - h) by h with a half-circle cap of radius h/2 at each end. Clamp the
// point's x into the straight section, then do one circle test against that clamped centre — this
// is the standard point-to-capsule distance, and it correctly rejects the bounding box's corners.
function insideStadium(dx, dy, w, h, slack) {
  const r = h / 2;
  const half = Math.max(0, w / 2 - r);
  const cx = Math.max(-half, Math.min(half, dx));
  return Math.hypot(dx - cx, dy) <= r + slack;
}

export function isInside(point, xy, hit, slack = HIT_SLACK) {
  const dx = point[0] - xy[0];
  const dy = point[1] - xy[1];
  if (hit.shape === 'circle') return Math.hypot(dx, dy) <= hit.r + slack;
  if (hit.shape === 'stadium') return insideStadium(dx, dy, hit.w, hit.h, slack);
  return Math.abs(dx) <= hit.w / 2 + slack && Math.abs(dy) <= hit.h / 2 + slack;
}

// Topmost-wins: nodeIndex is built in render order, so the LAST match is the one drawn on top —
// the one the user believes they are pointing at.
export function pickNode(point, nodeIndex, slack = HIT_SLACK) {
  for (let i = nodeIndex.length - 1; i >= 0; i -= 1) {
    if (isInside(point, nodeIndex[i].xy, nodeIndex[i].hit, slack)) return nodeIndex[i];
  }
  return null;
}

// The trailing `+ 0` is load-bearing, not decoration: Math.round(-5 / 12) is -0, and -0 * 12 stays
// -0. A -0 reaching setLayout serializes into the YAML as `-0`, which makes two otherwise identical
// models compare unequal — and node:assert/strict's deepEqual distinguishes it from 0, so the test
// below catches it. Adding 0 collapses -0 to 0 and leaves every other value alone.
export function snapToGrid(xy, grid = GRID) {
  return [Math.round(xy[0] / grid) * grid + 0, Math.round(xy[1] / grid) * grid + 0];
}

export function fitBox(positions, pad = 60) {
  if (!positions.length) return { x: 0, y: 0, w: BASE_W, h: BASE_H };
  const xs = positions.map((p) => p[0]);
  const ys = positions.map((p) => p[1]);
  const minX = Math.min(...xs); const maxX = Math.max(...xs);
  const minY = Math.min(...ys); const maxY = Math.max(...ys);
  return { x: minX - pad, y: minY - pad, w: (maxX - minX) + pad * 2, h: (maxY - minY) + pad * 2 };
}
```

In `canvas.js`, replace the deleted block with `import { ... } from './canvas/geometry.js';` and re-export nothing. Repoint `test/canvas-model.test.js`'s `edgePath`/`selfLoopPath`/`edgeLabelPos`/`NODE_R` import to `../js/ui/canvas/geometry.js` (its `scopedStore` import already points at `scoped-store.js` after Task 1). Every assertion in that file stays as it is — this is a move, not a rewrite.

- [ ] **Step 4: Run the full suite**

Run: `npm test`
Expected: PASS. `test/canvas-model.test.js` keeps passing unchanged apart from its imports.

- [ ] **Step 5: Commit and push**

```bash
git add js/ui/canvas/geometry.js js/ui/canvas.js test/canvas-geometry.test.js test/canvas-model.test.js
git commit -m "feat(canvas): pure geometry module with shape-aware hit predicates, snap and fit"
git push
```

---

### Task 4: Split the renderer and give every object a real hit area

The change users will feel first: edges stop being 1.5px targets.

**Files:**
- Create: `js/ui/canvas/render.js`, `js/ui/canvas/index.js`
- Delete: `js/ui/canvas.js` (its content moves into the two files above)
- Modify: `js/ui/app.js:13` (`import { createCanvas } from './canvas/index.js';`), `css/canvas.css`
- Test: no new pure tests — the geometry is already covered by Task 3. Verified in Chrome.

**Interfaces:**
- Consumes: everything from Task 3's `geometry.js`; `scopedStoreFor` from Task 1.
- Produces:

```js
// js/ui/canvas/render.js
buildSvg(svgEl, model, { positions, selection, handlers }) -> nodeIndex
// handlers: {onNodePointerDown(e, id), onEdgePointerDown(e, target)}
// Task 8 adds a third, onContextMenu, together with the listeners that call it — do NOT wire a
// contextmenu listener here, so the whole menu feature lands in one reviewable diff.
// nodeIndex entry: {kind:'state'|'node', key?|path?, xy, hit, el, treeKind?, node?, state?}
//   `hit` comes from geometry.hitShapeFor(...) — NOT the old scalar `hitR`.

// js/ui/canvas/index.js
createCanvas(svgEl, store, {layoutFor, flush}) -> {render, setTool, currentModelPath, openScope}
// Unchanged in this task — `setTool` and the toolbar survive it and are deleted in Task 5, so the
// app keeps working end-to-end at every commit.
```

- [ ] **Step 1: Move the code**

`render.js` takes `el`, `text`, `hasReward`, `payoffSummary`, `treeNodeKind`, `shapeForKind`, `haloForKind`, `treeTrimRadius`, `truncateLabel`, `pLabelText`, `buildDefs`, `renderMarkovEdge`, `renderMarkovNode`, `renderMarkov`, `renderTreeEdge`, `renderTreeNode`, `renderTree`, `buildSvg`. `index.js` takes everything else. No behaviour change in this step — get the suite green and the app rendering before touching anything.

- [ ] **Step 2: Add the edge hit layer**

In `renderMarkovEdge` and `renderTreeEdge`, prepend an invisible wide path with the same `d`, **before** the visible line so it never covers the arrowhead:

```js
g.appendChild(el('path', { class: 'edge-hit', d }));
```

and in `css/canvas.css`:

```css
/* Edges are 1.5px strokes with no fill, so SVG's default visiblePainted hit-testing gives them a
   1.5px-wide target — you had to hit an edge to the pixel. This invisible twin carries the hit
   area; it is drawn first so it sits under the visible line and the arrowhead marker. */
#canvas .edge-hit {
  fill: none;
  stroke: transparent;
  stroke-width: 14;
  stroke-linecap: round;
  pointer-events: stroke;
}
```

- [ ] **Step 3: Add the node hit layer for shapes with no fill**

`shapeForKind('terminal')` returns a bare `<line>`. Give the terminal (and only the terminal — every other kind has a paper fill that already carries its own hit area) a transparent rect sized from `geometry.hitShapeFor('terminal')`, appended first inside the node `<g>`:

```js
if (kind === 'terminal') {
  const t = hitShapeFor('terminal');
  g.appendChild(el('rect', {
    class: 'node-hit', x: -t.w / 2, y: -t.h / 2, width: t.w, height: t.h,
  }));
}
```

```css
#canvas .node-hit { fill: transparent; stroke: none; }
```

- [ ] **Step 4: Replace `hitR` with `hit` throughout**

Every `nodeIndex.push({...})` in `renderMarkov`/`renderTree` carries `hit: hitShapeFor(kind)` instead of `hitR: NODE_R` / `hitR: treeTrimRadius(kind)`. `treeTrimRadius` stays — it is edge-path trimming, a different concern from hit-testing, and keeps its current values. `index.js`'s local `hitTestNode` is deleted and every caller switches to `pickNode(point, nodeIndex)` from `geometry.js`. Grep for `hitR` afterwards; there must be no hits left.

- [ ] **Step 5: Verify in Chrome, then commit and push**

Run: `npm test` — PASS, all green.
Serve (`python3 -m http.server 8000`) and **hard-reload** (Chrome caches `js/` aggressively). Open the HIV example: clicking anywhere along a transition arrow selects it, not just dead-centre. Open the surgery example: clicking a terminal bar selects it. Both were near-impossible before.

```bash
git add js/ui/canvas/ js/ui/app.js css/canvas.css && git rm js/ui/canvas.js
git commit -m "feat(canvas): split the renderer out, give edges and terminals real hit areas"
git push
```

---

### Task 5: The modeless gesture set

**Files:**
- Create: `js/ui/canvas/gestures.js`
- Modify: `js/ui/canvas/index.js` (delete the toolbar, the `tool` variable, `setTool`, `updateToolButtons`, `TOOL_DEFS`, the V/A/C/D keydown arm, `addOnBackground`, `addOnNode`, `connectDrop`, and the whole pointer block), `css/canvas.css`
- Test: verified in Chrome. The decision logic that *can* be pure already is (`geometry.pickNode`).

**Interfaces:**
- Consumes: `pickNode`, `isInside`, `snapToGrid`, `edgePath`, `hitShapeFor` (geometry); `addState`, `addTransition`, `addChild`, `moveNode`, `setLayout` (ops); `scopedStoreFor`.
- Produces:

```js
createGestures(svgEl, {
  getNodeIndex, getModel, getActiveStore, clientToUser, flush, runOp,
  render, showToast, panBy, startRename, enterSubModel, selectTarget, openContextMenu,
}) -> {destroy}
// Owns pointerdown/move/up/cancel on svgEl and the Space keydown/keyup latch. Everything it needs
// from index.js arrives as a callback, so this module never reads the store or the DOM directly
// beyond svgEl. `openContextMenu` has no implementation until Task 8 — default it to a no-op here
// (`openContextMenu = () => {}`) so this task's app is complete on its own.
```

- [ ] **Step 1: Delete the toolbar**

Remove `TOOL_DEFS`, `toolButtons`, `updateToolButtons`, `setTool`, the `tool` variable, `svgEl.setAttribute('data-tool', ...)`, the `k === 'v'|'a'|'c'|'d'` arm of the keydown handler, and the `#canvas[data-tool=...]` cursor rules in `css/canvas.css`. `escapeAll()` keeps cancelling the rename and the gesture and now also clears the selection. Drop `setTool` from `createCanvas`'s return object.

- [ ] **Step 2: Implement the gesture state machine**

The pointer gesture carries `{target, startClientX, startClientY, startViewX, startViewY, moved, leftSource, pointerId, ghostNodeEl, ghostEdgeEl, grabDX, grabDY}`.

- **pointerdown on background** — `flush()`, start a pan gesture. On pointerup with `moved === false`: a plain click deselects; the hand-rolled double-click detector (reuse `lastDown`/`DOUBLE_CLICK_MS`, keyed on the string `'background'`) fires `createAt(cur)`.
- **pointerdown on a node** — `flush()`, re-resolve the entry from `getNodeIndex()` (it may have moved under us), record `grabDX/grabDY = xy - cursor` so the node keeps its grab offset, and clone the node's `<g>` into `ghostNodeEl` with class `drag-ghost`. Double-click still renames, or drills into a sub-model.
- **pointermove** — set `moved` past a 3px threshold. Set `leftSource` the first time `!isInside([cur.x, cur.y], source.xy, source.hit)`. Then decide which preview to show:

```js
// clientToUser returns {x, y}; geometry.js indexes points as [x, y] arrays. Convert ONCE, here —
// passing the object straight through makes every distance NaN and every hit silently miss, which
// is exactly the regression the Task 4 review caught. isInside now throws on a malformed point, so
// getting this wrong is loud rather than silent, but convert anyway.
const p = [cur.x, cur.y];
const forceArrow = spaceHeld;
const forceMove = e.altKey;
const over = pickNode(p, getNodeIndex());
const target = over && (over !== source || gesture.leftSource) ? over : null;
const connecting = !forceMove && (forceArrow || target !== null);
```

  When `connecting`, hide `ghostNodeEl`, show `ghostEdgeEl` from the source's rim to the cursor, and put `.drop-ring` on `target.el`. Otherwise show `ghostNodeEl` at `cur + grab offset` (through `snapToGrid` unless `e.metaKey`), hide the edge, clear the ring.
- **pointerup** — recompute the same three values, then commit. Every branch is a **single** `runOp` call, because the store snapshots one undo entry per `applyOp` and a drag must undo in one step:

```js
function endNodeGesture(g, cur, e) {
  const model = getModel();
  const source = g.target;
  const p = [cur.x, cur.y];              // clientToUser gives {x, y}; geometry wants [x, y]
  const over = pickNode(p, getNodeIndex());
  const target = over && (over !== source || g.leftSource) ? over : null;
  const connecting = !e.altKey && (spaceHeld || target !== null);
  const store = getActiveStore();

  if (!g.moved) { selectTarget(source); return; }          // a plain click still just selects

  if (!connecting) {
    const key = source.kind === 'state' ? source.key : source.path.join('/');
    const xy = [cur.x + g.grabDX, cur.y + g.grabDY];
    const at = e.metaKey ? [Math.round(xy[0]), Math.round(xy[1])] : snapToGrid(xy);
    runOp(store, (m) => setLayout(m, key, at));
    return;
  }

  if (model.type === 'markov') {
    if (target) {
      runOp(store, (m) => addTransition(m, source.key, target.key));
    } else {
      // One op: add the state, place it where the pointer was released, and connect to it. Read the
      // invented name back off the model addState returns — it always pushes the new state last.
      const at = e.metaKey ? [Math.round(cur.x), Math.round(cur.y)] : snapToGrid([cur.x, cur.y]);
      runOp(store, (m) => {
        const m1 = addState(m);
        const added = m1.states[m1.states.length - 1].name;
        return addTransition(setLayout(m1, added, at), source.key, added);
      });
    }
    return;
  }

  // tree
  if (target) {
    runOp(store, (m) => moveNode(m, source.path, target.path));
  } else {
    const at = e.metaKey ? [Math.round(cur.x), Math.round(cur.y)] : snapToGrid([cur.x, cur.y]);
    runOp(store, (m) => {
      const m1 = addChild(m, source.path);
      const parent = nodeAt(m1, source.path);
      const added = parent.children[parent.children.length - 1].name;
      return setLayout(m1, [...source.path, added].join('/'), at);
    });
  }
}
```

  `createAt(cur)` (the background double-click) follows the same shape: markov → one `runOp` doing `addState` + `setLayout`; tree → `addChild(selectedPath)` + `setLayout`, or `showToast('Select a parent node first.')` when the selection is absent, is not a `node`, or fails `sameModelPath(selection.modelPath, currentModelPath)`.

  `runOp` already catches and toasts — so a rejected `moveNode` (collision, descendant, root) surfaces its message and the canvas re-renders unchanged. No extra handling needed at the call site.

- **Space latch** — a `keydown`/`keyup` pair on `document` sets `spaceHeld`, guarded by the existing `isTypingTarget` check and the `dialog[open]` early-return, and only claimed while a node gesture is in flight (so a bare Space press elsewhere is untouched). `e.preventDefault()` on keydown stops the page scrolling. Reset `spaceHeld = false` on `blur` and on `pointercancel`, or a lost keyup leaves the latch stuck.

- [ ] **Step 3: Style the ghosts**

```css
/* The real node never moves during a drag — a translucent clone follows the cursor instead, so
   switching between the move reading and the connect reading is a crossfade between two overlays
   rather than the node snapping back to where it started. */
#canvas .drag-ghost { opacity: .45; pointer-events: none; }
#canvas .drop-ring {
  outline: none;
  stroke: var(--accent);
}
#canvas .drop-ring .node-shape { stroke: var(--accent); stroke-width: 3; }
```

- [ ] **Step 4: Verify every row of the gesture table in Chrome**

`npm test` — PASS. Then hard-reload and walk the table from the spec's §1: double-click empty creates; drag to empty moves (and snaps to the dot grid; ⌘ places freely); drag onto a node connects; drag out-and-back self-loops; Space-drag to empty creates-and-connects; ⌥-drag drops a node on top of another; the tree cases behave as the table says; the toast appears when a tree double-click has no parent selected.

- [ ] **Step 5: Commit and push**

```bash
git add js/ui/canvas/ css/canvas.css
git commit -m "feat(canvas): modeless editing — decide-at-drop drag, Space to force an arrow, double-click to create"
git push
```

---

### Task 6: View controls — pan, zoom, fit, tidy

**Files:**
- Modify: `js/ui/canvas/index.js`, `index.html:29` (nothing structural — `#canvas-toolbar` already exists), `css/app.css`
- Test: `fitBox`/`snapToGrid` already covered by Task 3. Verified in Chrome.

- [ ] **Step 1: Fix the wheel**

Today `wheel` zooms, so a two-finger trackpad scroll zooms the canvas. Split it:

```js
svgEl.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) {            // trackpad pinch arrives as ctrl+wheel
    zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.1 : 1 / 1.1);
    return;
  }
  const rect = svgEl.getBoundingClientRect();
  view.x += e.deltaX * (rect.width ? view.w / rect.width : 1);
  view.y += e.deltaY * (rect.height ? view.h / rect.height : 1);
  applyViewBox();
}, { passive: false });
```

- [ ] **Step 2: Add the four toolbar buttons**

Into `#canvas-toolbar`, alongside `panels.js`'s existing maximize span (append as siblings — never clear that node). Each a real `<button type="button">` with `aria-label` and `title`:

| id | glyph | label | action |
|---|---|---|---|
| `view-zoom-out` | `−` | Zoom out | `zoomAt(centre, 1/1.1)` |
| `view-zoom-in` | `+` | Zoom in | `zoomAt(centre, 1.1)` |
| `view-fit` | `⤢` | Fit to view | `fitToView()` |
| `view-tidy` | `⌗` | Tidy layout | `runOp(activeStore, (m) => clearLayout(m))` |

```js
function fitToView() {
  const box = fitBox(Object.values(layoutFor(resolveActiveModel() ?? {})), 60);
  view.x = box.x; view.y = box.y; view.w = box.w; view.h = box.h;
  zoom = BASE_W / box.w;                   // keep the wheel's clamp meaningful afterwards
  applyViewBox();
}
```

- [ ] **Step 3: Wire the keyboard**

In the existing document keydown handler, after the `dialog[open]` and `isTypingTarget` guards, and only when `e.metaKey || e.ctrlKey`: `0` → `fitToView()`, `=`/`+` → zoom in, `-` → zoom out, each with `preventDefault()`. Every other meta/ctrl chord still returns early untouched, so `⌘Z`/`⌘Y` undo/redo in `app.js` keep working.

- [ ] **Step 4: Verify in Chrome**

`npm test` — PASS. Hard-reload: two-finger scroll pans; pinch and ⌘-scroll zoom to the cursor; the four buttons work; ⌘0 frames the whole model after panning off into empty space; Tidy returns dragged nodes to the auto-layout.

- [ ] **Step 5: Commit and push**

```bash
git add js/ui/canvas/ css/app.css
git commit -m "feat(canvas): scroll pans, pinch zooms, plus fit/zoom/tidy controls"
git push
```

---

### Task 7: Keyboard editing

**Files:**
- Modify: `js/ui/canvas/index.js`
- Test: verified in Chrome.

- [ ] **Step 1: Extend the keydown handler**

Keep the existing order — `dialog[open]` early-return, then `isTypingTarget`, then Escape, then Delete/Backspace, then the meta/ctrl arm from Task 6 — and add below it, for the current selection only when `sameModelPath(selection.modelPath, currentModelPath)`:

- `Enter` → `startRename(entry)` for the selected node (no-op for an edge selection).
- `ArrowUp/Down/Left/Right` → `flush()`, then one `setLayout` op moving the node by `GRID` (`GRID * 4` with Shift). One op per keypress = one undo entry per nudge, which is the right granularity for a deliberate key press.
- `Escape` → already cancels the rename and the gesture; it now also clears the selection when there is nothing to cancel.

- [ ] **Step 2: Verify in Chrome**

`npm test` — PASS. Hard-reload: arrows nudge the selected node by one grid step, ⇧+arrow by four; Enter opens the inline rename; Escape backs out; none of it fires while typing in the YAML pane or a sidebar field; ⌘Z undoes a nudge.

- [ ] **Step 3: Commit and push**

```bash
git add js/ui/canvas/
git commit -m "feat(canvas): keyboard editing — arrows nudge, Enter renames, Escape deselects"
git push
```

---

### Task 8: Context menu

Replaces the deleted Delete tool.

**Files:**
- Create: `js/ui/canvas/context-menu.js`
- Modify: `js/ui/canvas/index.js`, `js/ui/canvas/render.js` (a `contextmenu` listener beside each `pointerdown` listener), `css/app.css`
- Test: verified in Chrome.

**Interfaces:**

```js
openContextMenu(x, y, items) -> void   // items: [{label, action, disabled?}] — a null entry is a separator
```

- [ ] **Step 1: Build the menu**

A plain absolutely-positioned `<div class="ctx-menu" role="menu">` appended to `document.body`, each item a real `<button role="menuitem">`. Closes on Escape, on any outside `pointerdown`, on scroll, and after any item runs. Flips left/up when it would overflow the viewport. **No `window.confirm` anywhere** — deleting from the menu is undoable, which is the confirmation.

- [ ] **Step 2: Wire the three targets**

`contextmenu` handlers call `e.preventDefault()`, select the target first (so the menu always acts on something visibly highlighted), then open:

- **node** — Delete · Rename · Tidy position (`clearLayout(m, key)`) · Enter sub-model (only when `treeKind === 'submodel'`) · Add child (`addChild(path)`) and Add sibling (`addChild(path.slice(0, -1))`, disabled on the root) in a tree.
- **edge** — Delete (`deleteTransition` for markov; `deleteNode(childPath)` for a tree edge, which *is* its child node).
- **background** — Add state (markov) / Add child of selection (tree) at the click point · Tidy layout · Fit to view.

There is deliberately no Duplicate: a collision-safe subtree copy is a third new op nobody asked for.

- [ ] **Step 3: Verify in Chrome**

`npm test` — PASS. Hard-reload: right-click each of the three targets; every item does what it says; the menu never escapes the viewport; Escape and an outside click both close it; ⌘Z undoes a menu delete.

- [ ] **Step 4: Commit and push**

```bash
git add js/ui/canvas/ css/app.css
git commit -m "feat(canvas): right-click context menu replacing the deleted Delete tool"
git push
```

---

### Task 9: `js/ui/outline/build.js` — rows, filter, findings

All of the sidebar's logic, DOM-free and fully tested. Task 10 is then only a renderer.

**Files:**
- Create: `js/ui/outline/build.js`
- Test: `test/outline-build.test.js`

**Interfaces:**
- Consumes: `scopePrefix`, `nodePathToCheckPath` (currently exported from `inspector.js`; move both into `build.js` and have `inspector.js` import them from there — `test/inspector-match.test.js` repoints its imports, assertions unchanged).
- Produces:

```js
// Row: {
//   id,           // stable + unique: 'group:structure' | 'state:Well' | 'edge:Well>Sick'
//                 //   | 'node:Root/A' | 'param:c_well' | 'submodel:post' | 'group:settings'
//   kind,         // 'group' | 'state' | 'edge' | 'node' | 'param' | 'submodel'
//   label,        // what the row shows
//   detail,       // muted right-hand text: an edge's p, a node's payoff summary, '' if none
//   depth,        // 0 = group header, then one per nesting level
//   parentId,     // null for group headers — used for ancestor retention when filtering
//   sel,          // the object handed to store.select, or null for a non-selectable row
//   checkPaths,   // string[] this row owns findings for; [] when it owns none
// }
buildOutline(model, modelPath = []) -> Row[]        // flat, in display order; depth encodes nesting
filterRows(rows, query) -> Row[]                    // case-insensitive substring over label+detail;
                                                    // a match keeps ALL its ancestors, a group header
                                                    // survives if any descendant matched; '' -> rows
attachFindings(rows, findings) -> {
  byRow,      // Map<rowId, finding[]>  — each finding goes to its LONGEST matching checkPath
  counts,     // Map<rowId, {errors, warnings}> — own findings PLUS every descendant's
  residual,   // finding[] matching no row at all; never swallowed
}
```

- [ ] **Step 1: Write the failing test**

```js
// test/outline-build.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseModel } from '../js/core/model.js';
import { buildOutline, filterRows, attachFindings } from '../js/ui/outline/build.js';

const MARKOV = () => parseModel(`
econeval: 1
type: markov
name: m
params:
  c_well: {value: 100}
states:
  well: {cost: c_well, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);

const TREE = () => parseModel(`
econeval: 1
type: tree
name: t
tree:
  Root:
    A:
      Win: {p: rest, utility: 10}
`);

const byId = (rows, id) => rows.find((r) => r.id === id);

test('markov: states are depth 1, their outgoing transitions depth 2', () => {
  const rows = buildOutline(MARKOV());
  assert.equal(byId(rows, 'group:structure').depth, 0);
  const well = byId(rows, 'state:well');
  assert.equal(well.depth, 1);
  assert.equal(well.parentId, 'group:structure');
  assert.deepEqual(well.sel, { kind: 'state', id: 'well', modelPath: [] });

  const edge = byId(rows, 'edge:well>dead');
  assert.equal(edge.depth, 2);
  assert.equal(edge.parentId, 'state:well');
  assert.equal(edge.label, '→ dead');
  assert.equal(edge.detail, '0.1');
  assert.deepEqual(edge.sel, { kind: 'edge', id: { from: 'well', to: 'dead' }, modelPath: [] });
});

test("markov: a 'rest' transition shows rest verbatim", () => {
  assert.equal(byId(buildOutline(MARKOV()), 'edge:well>well').detail, 'rest');
});

test('tree: nodes nest by depth with root-inclusive path ids', () => {
  const rows = buildOutline(TREE());
  assert.equal(byId(rows, 'node:Root').depth, 1);
  assert.equal(byId(rows, 'node:Root/A').depth, 2);
  const win = byId(rows, 'node:Root/A/Win');
  assert.equal(win.depth, 3);
  assert.equal(win.parentId, 'node:Root/A');
  assert.deepEqual(win.sel, { kind: 'node', id: ['Root', 'A', 'Win'], modelPath: [] });
});

test('parameters and settings get their own groups', () => {
  const rows = buildOutline(MARKOV());
  const p = byId(rows, 'param:c_well');
  assert.equal(p.parentId, 'group:parameters');
  assert.deepEqual(p.checkPaths, ['params.c_well']);
  assert.ok(byId(rows, 'group:settings'));
});

test('modelPath scopes both the selection and the check paths', () => {
  const rows = buildOutline(MARKOV(), ['post']);
  const well = byId(rows, 'state:well');
  assert.deepEqual(well.sel.modelPath, ['post']);
  assert.ok(well.checkPaths.every((p) => p.startsWith('models.post.')));
});

test('filter keeps matches and their ancestors, drops the rest', () => {
  const rows = buildOutline(MARKOV());
  const out = filterRows(rows, 'dead');
  const ids = out.map((r) => r.id);
  assert.ok(ids.includes('state:dead'));
  assert.ok(ids.includes('edge:well>dead'));
  assert.ok(ids.includes('state:well'), 'the matched edge keeps its parent state');
  assert.ok(ids.includes('group:structure'), 'the group header survives');
  assert.ok(!ids.includes('param:c_well'));
  assert.ok(!ids.includes('group:parameters'), 'a group with no surviving descendant is dropped');
});

test('filter is case-insensitive and matches detail text too', () => {
  const rows = buildOutline(MARKOV());
  assert.ok(filterRows(rows, 'DEAD').some((r) => r.id === 'state:dead'));
  assert.ok(filterRows(rows, '0.1').some((r) => r.id === 'edge:well>dead'));
});

test('an empty filter returns every row unchanged', () => {
  const rows = buildOutline(MARKOV());
  assert.deepEqual(filterRows(rows, ''), rows);
});

test('findings land on the most specific row that owns their path', () => {
  const rows = buildOutline(MARKOV());
  const findings = [
    { level: 'error', code: 'E_ROWSUM', path: 'transitions.well', message: 'row sums to 1.2' },
    { level: 'error', code: 'E_EXPR', path: 'transitions.well.dead', message: 'bad p' },
    { level: 'warning', code: 'W_X', path: 'states.well.cost', message: 'check cost' },
    { level: 'error', code: 'E_NOWHERE', path: 'meta.author', message: 'orphan' },
  ];
  const { byRow, counts, residual } = attachFindings(rows, findings);

  // 'transitions.well' is owned by the state row; 'transitions.well.dead' is longer, so it belongs
  // to the edge row rather than rolling up into the state's own bucket.
  assert.deepEqual(byRow.get('state:well').map((f) => f.code), ['E_ROWSUM', 'W_X']);
  assert.deepEqual(byRow.get('edge:well>dead').map((f) => f.code), ['E_EXPR']);
  assert.deepEqual(residual.map((f) => f.code), ['E_NOWHERE']);
});

test('counts roll descendants up into their ancestors', () => {
  const rows = buildOutline(MARKOV());
  const { counts } = attachFindings(rows, [
    { level: 'error', code: 'E_EXPR', path: 'transitions.well.dead', message: 'bad p' },
    { level: 'warning', code: 'W_X', path: 'states.well.cost', message: 'check cost' },
  ]);
  assert.deepEqual(counts.get('edge:well>dead'), { errors: 1, warnings: 0 });
  assert.deepEqual(counts.get('state:well'), { errors: 1, warnings: 1 });
  assert.deepEqual(counts.get('group:structure'), { errors: 1, warnings: 1 });
  assert.equal(counts.get('state:dead'), undefined);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/outline-build.test.js`
Expected: FAIL — `Cannot find module '../js/ui/outline/build.js'`

- [ ] **Step 3: Implement**

Group order is fixed: `structure`, `submodels` (omitted when `model.models` is empty), `parameters`, `settings`. Check-path ownership per row kind, all prefixed with `scopePrefix(modelPath)`:

| row | `checkPaths` |
|---|---|
| state `well` | `['states.well', 'transitions.well']` |
| edge `well→dead` | `['transitions.well.dead']` |
| tree node `['Root','A']` | `[nodePathToCheckPath(['Root','A'])]` → `'tree.A'` |
| param `c_well` | `['params.c_well']` |
| group settings | `['settings']` |

`attachFindings` sorts every row's `checkPaths` by descending length once, then for each finding takes the first row whose path is an exact match or a `path + '.'` prefix; `counts` is built by walking `parentId` upward from each owning row.

- [ ] **Step 4: Run the full suite**

Run: `npm test`
Expected: PASS, including the repointed `test/inspector-match.test.js`.

- [ ] **Step 5: Commit and push**

```bash
git add js/ui/outline/ test/outline-build.test.js js/ui/inspector.js test/inspector-match.test.js
git commit -m "feat(outline): pure row-building, filtering and findings-mapping module"
git push
```

---

### Task 10: The outline sidebar

Replaces the three-tab inspector. The largest task; it lands in one piece because a half-migrated sidebar leaves parameters or settings unreachable.

**Files:**
- Modify: `js/ui/inspector.js` (the tab strip, `activeTab`, `setActiveTab`, `TABS`, `renderTabStrip`, `onTabKeydown`, `focusTab`, `TAB_KEYS`, `renderSelectionTab`, `renderParametersTab`, `renderSettingsTab`, `renderSelectionPicker`, `appendScopeHint` all go; the field builders all stay), `js/ui/app.js:230-249`, `css/app.css:293-296,385-630`, `index.html:31`
- Test: verified in Chrome. The logic is Task 9's, already covered.

**Interfaces:**
- Consumes: `buildOutline`, `filterRows` (Task 9); `scopedStoreFor` (Task 1).
- Produces: `createInspector(rootEl, headEl, store, {flush, openScope}) -> {render, revealSelection}` — `setActiveTab` is gone, and `openScope` is new. A row click has to drill the canvas into the row's sub-model scope before selecting, but inspector.js holds no canvas reference — importing one would rebuild exactly the backwards peer dependency Task 1 exists to remove. So `app.js` injects `canvas.openScope` (it constructs the canvas at line 225, before the inspector at line 230). Default it to a no-op.

- [ ] **Step 1: Build the shell**

`#inspector-tabs` is the pane's `.panel-head` and `panels.js` writes its own `.panel-ctl` span into it: append as a sibling, never clear it. The head gains the static text `Model`, so the `#pane-inspector[data-min] .panel-head::before` rule at `css/app.css:293-296` — which existed only to fake a label in the minimized state — is deleted.

Into `#inspector-body`: a sticky filter bar (a `type="search"` input plus an `Only findings` toggle button), then the row list.

```js
// One row. Indent comes from `depth` via a custom property, so nesting needs no wrapper elements
// and the whole list stays a flat sequence of siblings — which is what keeps scroll restoration and
// findings patching simple.
function outlineRow(row, { expanded, hasChildren, collapsed }) {
  const btn = h('button', {
    type: 'button', class: 'otl-row', id: `otl-${row.id}`,
    'data-kind': row.kind, style: `--depth:${row.depth}`,
    'aria-expanded': hasChildren ? String(!collapsed) : undefined,
    'aria-current': expanded ? 'true' : undefined,
  });
  btn.append(
    h('span', { class: 'otl-twisty' }, hasChildren ? (collapsed ? '▸' : '▾') : ''),
    h('span', { class: 'otl-label' }, row.label),
    h('span', { class: 'otl-detail' }, row.detail),
    h('span', { class: 'otl-dot', hidden: '' }),      // Task 11 fills this in place, never rebuilds it
  );
  btn.addEventListener('click', () => {
    if (row.kind === 'group') { toggleCollapsed(row.id); return; }
    if (row.sel) { openScope(row.sel.modelPath ?? []); store.select(row.sel); }
  });
  return btn;
}
```

```css
.otl-row {
  display: flex; align-items: center; gap: var(--sp-1);
  width: 100%; text-align: left; background: none; border: 0;
  padding: 3px var(--sp-2) 3px calc(var(--sp-2) + var(--depth) * 14px);
  font-size: var(--fs-2); color: var(--ink); cursor: pointer;
}
.otl-row[data-kind="group"] {
  font-size: var(--fs-0); color: var(--muted);
  text-transform: uppercase; letter-spacing: .04em;
}
.otl-row[aria-current="true"] { background: var(--accent-soft); }
.otl-row:hover { background: var(--accent-soft); }
.otl-twisty { width: 10px; color: var(--muted); }
.otl-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.otl-detail { font-family: var(--font-data); font-size: var(--fs-0); color: var(--muted); }
.otl-fields { padding-left: calc(var(--sp-2) + (var(--depth) + 1) * 14px); }
```

- [ ] **Step 2: Expansion and field editors**

Exactly one row is expanded at a time: the selected one. Its editors render into a `<div class="otl-fields">` inserted directly after its row, indented one level further. The existing field builders are reused **unchanged** — `renderStateFields`, `renderEdgeFields`, `renderNodeFields`, `fieldRow`, `wireCommit`, `wireExprInput`, `makeCommitter`, `keyValueEditor`, `kvRow`, `appendNameField`, `appendNumberField`, `appendTextSettingField`. Only `paramRow` is rewritten: it was a `<tr>` of seven `<td>`s and becomes the same five fields (`value`, `low`, `high`, `dist`, `source`) as vertical `fieldRow`s plus the delete button. Group headers toggle collapse independently of selection.

Scope: STRUCTURE rows edit through `scopedStoreFor(store, row.sel.modelPath)`; PARAMETERS and SETTINGS always edit the top-level store.

- [ ] **Step 3: Selection sync and persistence**

`revealSelection()` — expand the selected row, expand any collapsed group above it, `scrollIntoView({block: 'nearest'})`. Clicking a row calls `openScope(row.sel.modelPath)` **before** `store.select(row.sel)` — the same ordering `app.js:242-249` already depends on, so the canvas halo's `sameModelPath` check sees the right scope. In `app.js`, `selectOnCanvas`'s third line becomes `inspector.revealSelection();`.

Persist the collapsed-group set and the last filter string through `panels.js`'s existing read-merge-write `saveLayout` pattern, replacing the now-dead `tab` key.

- [ ] **Step 4: Preserve render discipline**

`shouldSkipRender` stays exactly as it is — skip the structural rebuild while focus is on a real input inside `rootEl`, with the one exception that a selection deleted out from under the user forces an immediate reconcile. Extend `render()` to capture `rootEl.scrollTop` and the expanded/collapsed set before `replaceChildren()` and restore both after. Without this the panel visibly jumps on every keystroke that reaches the store.

- [ ] **Step 5: Verify in Chrome**

`npm test` — PASS. Hard-reload and check: every state, edge, tree node, parameter and sub-model appears; the filter narrows and keeps ancestors; clicking a row selects on canvas and vice versa; every field still commits (payoffs, `p`, cost/utility, delay, sub-model, `with`, all five parameter fields, every setting); the parameter name field is finally readable; typing in a field never gets interrupted by a rebuild; scroll position holds; a sub-model row drills the canvas in.

- [ ] **Step 6: Commit and push**

```bash
git add js/ui/inspector.js js/ui/app.js css/app.css index.html
git commit -m "feat(inspector): replace the three tabs with one filterable outline"
git push
```

---

### Task 11: Findings on outline rows

**Files:**
- Modify: `js/ui/inspector.js`, `css/app.css`
- Test: `attachFindings` already covered by Task 9. Verified in Chrome.

- [ ] **Step 1: Render the dots**

Keep the existing 300ms-debounced `check()` and the DOM-patch-not-rebuild discipline: `applyFindingsToDom` must never trigger a structural render, or it will steal focus from a field being typed into. For each row, `attachFindings(rows, latestFindings)` gives `byRow` (a dot on the row: `--danger` when any finding is an error, `--warn` otherwise, `title` = the joined messages) and `counts` (a count on group headers). Findings whose path matches a field currently rendered inside the expanded row still show inline beneath that field via the existing `fieldSlots` mechanism. `residual` renders in a `Model findings` list pinned at the bottom of the outline — nothing is swallowed.

- [ ] **Step 2: Wire the toggle**

The `Only findings` button in the filter bar composes with the text filter: apply `filterRows` first, then keep only rows that have findings themselves or an ancestor relationship to one (reuse `counts`, which already rolls descendants up). Its `aria-pressed` reflects the state.

- [ ] **Step 3: Verify in Chrome**

`npm test` — PASS. Hard-reload, break something (a bad expression in a `p`, a row that does not sum to 1, an unknown parameter name): the dot lands on the right row, the group header counts it, `Only findings` narrows to it, and clicking through gets you to the field. The Validation tab's click-through still lands on the right row via `revealSelection`.

- [ ] **Step 4: Commit and push**

```bash
git add js/ui/inspector.js css/app.css
git commit -m "feat(inspector): findings as row dots with group roll-up and an only-findings filter"
git push
```

---

### Task 12: Docs, full smoke, deploy

**Files:**
- Modify: `README.md` (the editor section), `docs/superpowers/plans/2026-08-24-editor-rework.md` (tick every box)

- [ ] **Step 1: Update the README**

Replace the four-tool description with the gesture table from the spec's §1, the new view controls, and the outline sidebar. Anything describing Select/Add/Connect/Delete or the Selection/Parameters/Settings tabs is now wrong — grep for `tool`, `Selection tab`, `Parameters tab` and fix every hit.

- [ ] **Step 2: Full-suite and cross-browser smoke**

`npm test` — every test green. Then in Chrome, on both `examples/hiv.yaml` (markov) and `examples/surgery.yaml` (tree): build a small model from scratch using only canvas gestures, run it, confirm the results drawer still works, undo back to the start, redo forward. Then check the YAML pane round-trips everything the new gestures produced, and that the Validation tab's click-through still selects and reveals. Repeat the pointer gestures once in Safari and once in Firefox — pointer capture and `contextmenu` differ between engines and this round leans on both.

- [ ] **Step 3: Deploy and confirm live**

```bash
netlify deploy --prod --dir . --site 4c526e64-937b-4c3a-a548-f701d9804a56
```

Hard-reload `https://econeval.netlify.app` and repeat the smoke walk against production — `js/` is cached aggressively, so a soft reload can show the old bundle and hide a broken deploy.

- [ ] **Step 4: Commit and push**

```bash
git add README.md docs/superpowers/plans/2026-08-24-editor-rework.md
git commit -m "docs: editor rework — gesture table, view controls, outline sidebar"
git push
```
