# econeval Phase 2 (Editor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The interactive editor: app shell with resizable/maximizable panels, SVG model canvas with drawing gestures, inspector (Selection/Parameters/Settings), two-way YAML sync, local storage with versions, file import/export, examples menu — deployed to Netlify so every push is live.

**Architecture:** The phase-1 model document stays the single source of truth. UI edits are pure model→model operations (`ops.js`) dispatched through one store (`store.js`); the YAML pane and the canvas are two live views. DOM code is a thin shell over pure, node-tested modules (ops, store, sync, layouts, panel geometry, files).

**Tech Stack:** Vanilla JS ES modules, no build step. Browser gets js-yaml via an import map pointing at a vendored ESM file. `node --test` for all pure modules. Netlify static hosting.

**Spec:** `docs/superpowers/specs/2026-08-23-econeval-design.md` — UI section + flexible-windows requirement + design paragraph are binding.

## Global Constraints

- ES modules; UI code under `js/ui/`, styles under `css/`, no framework, no new npm deps (js-yaml only, vendored for the browser).
- Tests via `npm test` (`node --test test/*.test.js`); phase-1's 103 tests must stay green after every task.
- Pure logic (ops, store, sync, layouts, splitter geometry, files/registry) lives in DOM-free modules with tests; DOM modules (canvas, inspector, panels wiring, app) hold no business logic.
- Every model mutation flows through `store.apply(...)`; no module mutates a model object in place (ops return new models via structuredClone-then-edit).
- Undo/redo = document text snapshots (spec). One undo entry per user gesture (a node drag = one entry, not per pixel).
- The app must load from plain static hosting over HTTP (dev: `python3 -m http.server` or `npx serve`; `file://` unsupported — module imports + fetch). Hard-reload when verifying (Chrome caches js/ aggressively).
- Errors surfaced, never swallowed: YAML parse errors show line + hint; expression errors show inline in the inspector; `check()` findings appear as a badge + per-field markers. No silent fallbacks.
- English UI copy, sentence case, verbs on buttons ("Save version", not "Submit").
- Accessibility floor: visible keyboard focus everywhere, all toolbar buttons real `<button>`s with `aria-label` + `title`, `prefers-reduced-motion` respected (no animation is fine), canvas selection operable via inspector even if pointer gestures fail.
- Commit after every task; push after every task.

## Phase-1 interfaces the UI consumes (authoritative)

```js
import { parseModel, serializeModel, ModelError } from '../core/model.js';
// parseModel(text) -> Model (throws ModelError {message, line?, hint?, path?})
// serializeModel(model) -> yaml text (block style for call rows; round-trip safe)
// Model: {version, type:'markov'|'tree', name, meta, settings, params:Map, tables, models,
//         states:[{name,payoffs}], transitions, strategies, tree:Node|null, layout}
// Node: {name, p?, payoffs, children:[], model?, with?, delay?}
// settings: {cycles?, cycleYears, discount:{cost,effect}, correction, wtp?, psa:{n,seed,correlations}, start, age}
import { compile, ExprError } from '../core/expr.js';   // compile(src).eval(env); use for field validation (compile only — don't eval)
import { check } from '../analysis/check.js';           // check(model) -> [{level,code,path,message}]
```

Layout-key rule (spec refinement, binding): `layout` keys are **state names** for markov and **node paths** (`root/child/grandchild`, '/'-joined names) for trees. Values `[x, y]` numbers.

## Design tokens (binding; from the phase-2 design pass)

Signature: **the canvas is a publication figure** — paper surface with a faint dot grid; nodes drawn as 1.5px ink strokes with paper fill; edge probability labels in the mono data face, like typeset annotations. Chrome stays quiet so the figure is the hero. One accent only.

```css
:root {
  /* light */
  --bg: #F3F4F6;         /* chrome behind panels */
  --surface: #FFFFFF;    /* panels, topbar, dialogs */
  --paper: #FDFDFB;      /* canvas ground */
  --dot: #E3E2DC;        /* canvas dot grid */
  --ink: #1A1D21;        /* text + node strokes */
  --muted: #6B7280;
  --line: #E2E4E8;       /* hairline borders */
  --accent: #0E7A6E;     /* deep teal: selection, primary buttons, links */
  --accent-soft: #E4F2F0;
  --danger: #B42334;
  --warn: #B45309;
  --radius: 6px; --radius-sm: 4px;
  --font-ui: system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, monospace;
  --fs-0: 11px; --fs-1: 12px; --fs-2: 13px; --fs-3: 15px; --fs-4: 18px;  /* base UI = --fs-2 */
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px;   /* 8px rhythm */
  --shadow-pop: 0 4px 16px rgba(0,0,0,.12);  /* dialogs/popovers only; panels are flat */
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#131518; --surface:#1B1E23; --paper:#1F2328; --dot:#2A2F36; --ink:#E7E9EC;
  --muted:#9AA1AA; --line:#2A2E34; --accent:#2FA795; --accent-soft:#16332F;
  --danger:#E5484D; --warn:#D97706; --shadow-pop: 0 4px 16px rgba(0,0,0,.5);
}}
:root[data-theme="dark"] { /* same dark values, verbatim */ }
```

Node vocabulary on canvas: markov state = circle (r 26); tree decision root = square; chance node = circle; terminal = short vertical end-bar; sub-model attachment = stadium (rounded rect) with a **double stroke**. Dot grid: `background-image: radial-gradient(var(--dot) 1px, transparent 1px); background-size: 12px 12px;` on the canvas ground. Selection = 3px `--accent` halo (outer stroke), never a fill change. Numbers (edge labels, params table values, YAML pane) always `--font-data`.

---

### Task 1: Shell, vendored js-yaml, Netlify

**Files:**
- Create: `index.html`, `netlify.toml`, `js/vendor/js-yaml.mjs` (copied), `js/ui/.gitkeep`, `css/.gitkeep`
- Modify: `README.md` (dev-server + live-URL section)
- Test: `test/vendor.test.js`

**Interfaces:**
- Produces: the import map every browser module relies on; the deployed site.

- [ ] **Step 1: Failing test** — `test/vendor.test.js`:

```js
import test from 'node:test';
import assert from 'node:assert/strict';

test('vendored js-yaml is importable and matches the npm package API', async () => {
  const vendored = await import('../js/vendor/js-yaml.mjs');
  assert.equal(typeof vendored.load, 'function');
  assert.equal(typeof vendored.dump, 'function');
  const npm = await import('js-yaml');
  assert.deepEqual(vendored.load('a: 1'), npm.load('a: 1'));
});
```

- [ ] **Step 2: Run** — FAIL (file missing). **Step 3:** `cp node_modules/js-yaml/dist/js-yaml.mjs js/vendor/js-yaml.mjs` (if the package ships no `.mjs`, report BLOCKED with the package's dist listing — do not hand-bundle). **Step 4: Run** — PASS.

- [ ] **Step 5: index.html** — semantic skeleton only (styling is Task 2):

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>econeval</title>
<link rel="stylesheet" href="css/tokens.css">
<link rel="stylesheet" href="css/app.css">
<link rel="stylesheet" href="css/canvas.css">
<script type="importmap">{"imports": {"js-yaml": "./js/vendor/js-yaml.mjs"}}</script>
<script type="module" src="js/ui/app.js"></script>
</head>
<body>
<header id="topbar">
  <span id="brand">econeval</span>
  <span id="model-name" contenteditable="false"></span><span id="type-badge"></span>
  <nav id="actions">
    <button id="btn-new">New</button> <button id="btn-open">Open</button>
    <button id="btn-save">Save version</button> <button id="btn-examples">Examples</button>
    <button id="btn-import">Import</button> <button id="btn-export">Export</button>
    <button id="btn-yaml" aria-pressed="false">YAML</button>
  </nav>
</header>
<main id="workspace">
  <aside id="pane-yaml" class="panel" hidden><div class="panel-head">YAML<span class="panel-ctl"></span></div><textarea id="yaml-text" spellcheck="false"></textarea><div id="yaml-error" hidden></div></aside>
  <div class="splitter" id="split-l" role="separator" aria-orientation="vertical" tabindex="0"></div>
  <section id="pane-canvas" class="panel"><div id="canvas-toolbar"></div><div id="breadcrumb" hidden></div><svg id="canvas"></svg></section>
  <div class="splitter" id="split-r" role="separator" aria-orientation="vertical" tabindex="0"></div>
  <aside id="pane-inspector" class="panel"><div class="panel-head" id="inspector-tabs"></div><div id="inspector-body"></div></aside>
</main>
<dialog id="dlg-open"></dialog>
</body>
</html>
```

`js/ui/app.js` for now: `console.log('econeval shell')` — real wiring lands in Task 12.

- [ ] **Step 6: netlify.toml** — `[build]\n  publish = "."\n` (no build command). Deploy: `netlify status` — if not logged in, finish everything else, commit/push, and report DONE_WITH_CONCERNS naming the exact blocked commands (`netlify sites:create --name econeval`, `netlify deploy --prod --dir .`) so the controller can surface them; if logged in, create the site, deploy, and put the live URL in README. Custom domain econeval.melberg.app is dashboard work — note it in README as "pending: custom domain", never attempt DNS changes.

- [ ] **Step 7:** README section "Run locally" (`python3 -m http.server 8000` → http://localhost:8000; hard-reload after JS changes) + "Live" (URL or pending note). Full suite. **Step 8: Commit and push.**

---

### Task 2: Design tokens and chrome CSS

**Files:**
- Create: `css/tokens.css`, `css/app.css`, `css/canvas.css` (ground + dot grid only; node styles land with Task 9)

**Interfaces:** Produces every CSS custom property listed in the Design tokens block (verbatim — reviewers diff against it) and the chrome layout: topbar 44px; `#workspace` as `display:grid; grid-template-columns: var(--w-yaml, 0px) 4px 1fr 4px var(--w-insp, 300px);` (splitters are the 4px tracks); panels flat `--surface` with `--line` hairlines; buttons quiet (transparent bg, `--ink` text, `--line` border, `--accent` primary for Save); focus ring `outline: 2px solid var(--accent); outline-offset: 1px` on every interactive element; textarea `#yaml-text` in `--font-data --fs-1`; `#yaml-error` strip `--danger` text on `--surface`; dialogs use `--shadow-pop`. Dark mode via the token blocks only — no dark-specific selectors elsewhere.

- [ ] **Step 1:** Write the three files exactly per the token block + chrome spec above. No test (visual); acceptance = Step 2.
- [ ] **Step 2:** Serve locally, load the page in both themes (OS toggle or `data-theme`), confirm: topbar/panel hairlines visible, focus rings on tab, YAML pane hidden by default, no horizontal scroll at 1280 and 900px widths. Record what you checked in the report.
- [ ] **Step 3: Commit and push.**

---

### Task 3: Model operations — markov (`ops.js` part 1)

**Files:**
- Create: `js/ui/ops.js`
- Test: `test/ops-markov.test.js`

**Interfaces:**
- Produces (all pure `(model, ...) -> newModel`, deep-cloning internally; throw `Error` with a clear message on invalid input — the store surfaces them):

```js
addState(model, name?)            // name default: first free of state1, state2, ...
                                  // new state: payoffs {cost: 0, utility: 0}; adds absorbing row {self: 1}? NO —
                                  // adds row {self: 'rest'} and NO other entries (row sums via rest to 1 alone)
renameState(model, oldName, newName)  // rewrites: states, every transitions row key and target key,
                                      // settings.start keys, layout key. Throws if newName exists or empty.
deleteState(model, name)          // removes state, its row, every target entry referencing it,
                                  // its start share, its layout entry
addTransition(model, from, to)    // adds target `to` to row `from` with p: if the row has no 'rest'
                                  // entry -> p:'rest'; else p:0. Throws if entry exists.
deleteTransition(model, from, to)
setTransitionAttr(model, from, to, key, value)   // key in {p, cost, utility}; value string|number;
                                                 // deleting: pass value === null to remove key (p:null removes whole entry? no — throws; use deleteTransition)
setStatePayoff(model, name, key, value)          // value null removes the key
setLayout(model, key, xy)         // xy [x,y] rounded to integers; null removes
```

- Naming note: models with a `type:'markov'` only; each op validates `model.type` and throws otherwise.

- [ ] **Step 1: Failing tests** — build models with `parseModel` fixtures; assert with `serializeModel` round-trips where useful:

```js
// test/ops-markov.test.js  (imports: parseModel from core, ops from js/ui/ops.js)
const M = () => parseModel(`
econeval: 1
type: markov
name: m
settings: {cycles: 3, start: well}
states:
  well: {cost: 100, utility: 0.8}
  dead: {cost: 0, utility: 0}
transitions:
  well: {well: rest, dead: 0.1}
  dead: {dead: 1}
`);

test('addState invents a free name and a rest self-row', () => {
  const m2 = ops.addState(M());
  assert.ok(m2.states.some(s => s.name === 'state1'));
  assert.deepEqual(m2.transitions.state1, { type: 'p', to: { state1: { p: 'rest' } } });
  assert.notDeepEqual(m2, M() && undefined); // original untouched:
  assert.equal(M().states.length, 2);
});

test('renameState rewrites rows, targets, start, layout', () => {
  let m = M(); m = ops.setLayout(m, 'well', [40, 40]);
  const m2 = ops.renameState(m, 'well', 'healthy');
  assert.ok(m2.states.some(s => s.name === 'healthy'));
  assert.ok(m2.transitions.healthy && !m2.transitions.well);
  assert.equal(m2.transitions.healthy.to.healthy.p, 'rest');
  assert.deepEqual(m2.settings.start, { healthy: 1 });
  assert.deepEqual(m2.layout.healthy, [40, 40]);
  assert.throws(() => ops.renameState(m2, 'healthy', 'dead'), /exists/);
});

test('deleteState scrubs every reference', () => {
  const m2 = ops.deleteState(M(), 'dead');
  assert.ok(!m2.transitions.dead);
  assert.ok(!('dead' in m2.transitions.well.to));
  assert.ok(!m2.states.some(s => s.name === 'dead'));
});

test('addTransition rest-default rule', () => {
  let m = ops.addState(M());                       // state1 row = {state1: rest}
  const a = ops.addTransition(m, 'state1', 'dead'); // row already has rest -> p 0
  assert.equal(a.transitions.state1.to.dead.p, 0);
  let noRest = ops.setTransitionAttr(M(), 'well', 'well', 'p', 0.9);
  const b = ops.addTransition(noRest, 'dead', 'well'); // dead row {dead:1} has no rest -> 'rest'
  assert.equal(b.transitions.dead.to.well.p, 'rest');
});

test('payoff and transition attr set/remove', () => {
  const m2 = ops.setStatePayoff(M(), 'well', 'c_drug', '2278');
  assert.equal(m2.states.find(s => s.name === 'well').payoffs.c_drug, '2278');
  const m3 = ops.setStatePayoff(m2, 'well', 'c_drug', null);
  assert.ok(!('c_drug' in m3.states.find(s => s.name === 'well').payoffs));
  const m4 = ops.setTransitionAttr(M(), 'well', 'dead', 'cost', 500);
  assert.equal(m4.transitions.well.to.dead.cost, 500);
});

test('every op round-trips through serialize/parse', () => {
  let m = ops.addState(M());
  m = ops.addTransition(m, 'state1', 'dead');
  m = ops.setLayout(m, 'state1', [120, 80]);
  assert.deepEqual(parseModel(serializeModel(m)), m);
});
```

- [ ] **Step 2: Run** — FAIL. **Step 3:** Implement with one internal `clone(model)` helper (structuredClone handles Maps). **Step 4: Run** — PASS, full suite green. **Step 5: Commit and push.**

---

### Task 4: Model operations — tree, params, settings (`ops.js` part 2)

**Files:**
- Modify: `js/ui/ops.js`
- Test: `test/ops-tree.test.js`

**Interfaces:**
- Produces (tree nodes addressed by **path**: array of names from root inclusive, e.g. `['Treatment?','Surgery','Success']`):

```js
nodeAt(model, path) -> Node                     // helper, exported (canvas/inspector use it)
addChild(model, path, name?)                    // default name branch1, branch2… free among siblings.
                                                // New child: {p: siblings-have-rest ? 0 : 'rest', payoffs:{utility:0}, children:[]}
                                                // EXCEPTION: path is root -> new strategy branch, NO p.
renameNode(model, path, newName)                // sibling-unique; rewrites layout path keys for the node AND its subtree
deleteNode(model, path)                         // root itself not deletable; scrubs subtree layout keys
setNodeAttr(model, path, key, value)            // key in {p, delay, model, notes, source}; null removes (p removable only on root children? p on root children is invalid — throw)
setNodePayoff(model, path, key, value)          // null removes
setWith(model, path, param, value)              // value null removes; empty with -> remove with
addParam(model, name?, spec = {value: 0})       // default name param1…
setParam(model, name, field, value)             // field in {value, low, high, dist, source, notes}; null removes field; removing 'value' while no dist -> throw
renameParam(model, oldName, newName)            // does NOT rewrite expressions (check() will flag orphans — surprise-principle: visible, not magic)
deleteParam(model, name)
setSetting(model, keyPath, value)               // keyPath like 'cycles', 'discount.cost', 'psa.n', 'cycle' (string -> re-parsed cycleYears via the same unit table — reuse model.js's parser by round-tripping through serialize/parse if not exported; if model.js needs a small export for this, add `parseCycle` to its exports as part of this task)
```

- [ ] **Step 1: Failing tests** — same pattern as Task 3; cover: addChild default-p rule (siblings with rest → 0; without → 'rest'; root child → no p), rename cascades layout keys `a/b` → subtree `a/b/c` keys, deleteNode scrubs subtree layout, params CRUD incl. renameParam leaving expressions untouched (assert the expression string still names the old param and `check(m)` reports E_UNKNOWN_NAME), `setSetting('discount.cost', 0.035)`, `setSetting('cycle', '1 month')` → cycleYears ≈ 1/12, and a final everything-round-trips test.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS + suite green. **Step 5: Commit and push.**

---

### Task 5: Store

**Files:**
- Create: `js/ui/store.js`
- Test: `test/store.test.js`

**Interfaces:**
- Produces:

```js
createStore(initialText) -> store
store.get() -> {text, model, parseError, selection, dirty, canUndo, canRedo}
// text: canonical document text. model: last GOOD parse (never null after a good initial text;
// stays at last-good when parseError is set). parseError: ModelError|null.
// selection: {kind:'state'|'edge'|'node'|'param'|null, id} — edge id {from,to}, node id = path array, others string.
store.setText(text)                 // YAML-pane origin: parse; good -> model swap + undo snapshot; bad -> parseError set, model kept
store.applyOp(fn, {label})          // canvas/inspector origin: fn(model)->model' ; serialize -> text; undo snapshot; clears parseError
store.select(sel) / store.undo() / store.redo()
store.markSaved()                   // dirty=false (Save version / autosave bookkeeping)
store.subscribe(listener) -> unsubscribe   // listener() called after every state change (no payload; pull via get())
```

- Undo: snapshot = text BEFORE the change; max 100 entries; redo cleared on new change; undo/redo restore text AND re-parse (always good — snapshots only taken from good states); selection preserved when still valid, else cleared.
- `applyOp` with a throwing `fn` leaves the store untouched and rethrows (caller shows the message).

- [ ] **Step 1: Failing tests** — cover: initial parse; setText good/bad (model kept on bad, parseError.line present); applyOp serializes + text changes + dirty; undo returns prior text and model; redo; redo cleared after new op; 100-cap (101 ops → oldest dropped); throwing op leaves state identical (deepEqual before/after); subscribe fires once per change and not after unsubscribe; selection cleared when the selected state is deleted via applyOp (use ops.deleteState).
- [ ] **Steps 2-4:** red → implement (~120 lines) → green + suite. **Step 5: Commit and push.**

---

### Task 6: YAML sync engine

**Files:**
- Create: `js/ui/sync.js`
- Test: `test/sync.test.js`

**Interfaces:**
- Produces a DOM-free state machine the app wires to the textarea:

```js
createSync(store, {debounceMs = 400, now = Date.now, setTimer = setTimeout, clearTimer = clearTimeout}) -> sync
sync.onUserInput(text)     // called on every textarea 'input'; debounced store.setText
sync.flush()               // immediate commit of pending input (blur / Run / Save call this)
sync.textForView() -> {text, dirtyFromModel}  // what the textarea should display:
// RULE: model-originated changes (store text changed NOT via onUserInput) always win and replace the view;
// while the user is mid-typing (pending debounce), the view is the user's text — never clobbered.
sync.dispose()
```

- Injected timers make it testable without real time.

- [ ] **Step 1: Failing tests** — fake-timer harness (capture the callback from `setTimer`, fire manually): input → no store change before debounce → fires after; rapid inputs coalesce to one setText; flush commits immediately and cancels the timer; a store change from `applyOp` while NOT typing updates `textForView`; a store change arriving while a debounce is pending does not clobber the pending user text (user text still wins until committed, then the parse error or success resolves it); dispose cancels timers.
- [ ] **Steps 2-4:** red → implement (~80 lines) → green + suite. **Step 5: Commit and push.**

---

### Task 7: Auto-layout

**Files:**
- Create: `js/ui/layouts.js`
- Test: `test/layouts.test.js`

**Interfaces:**
- Produces:

```js
layoutFor(model) -> {key: [x,y]}    // merged: explicit model.layout entries win; missing keys filled by auto-layout
autoMarkov(model) -> positions      // states on a ring: center (360, 280), radius max(140, 46*n/π), order = declaration order, first state at angle -90°
autoTree(model) -> positions        // left-to-right tidy layout: x = 90 + depth*170; leaves get consecutive y slots (start 60, step 74); an internal node's y = mean of its children's y. Keys are '/'-joined paths.
```

- [ ] **Step 1: Failing tests** — markov: 4 states → all on the ring (distance to center within 1e-6 of radius), first at top; tree (surgery example): leaves at y 60, 134, 208, 282; `Surgery` y = mean(60,134)=97; root x=90, depth-1 x=260, leaves x=430; `layoutFor` respects an explicit `layout: {well: [10, 20]}` while filling the rest; deterministic (two calls deepEqual).
- [ ] **Steps 2-4:** red → implement → green + suite. **Step 5: Commit and push.**

---

### Task 8: Panel manager — splitters, maximize, minimize, persistence

**Files:**
- Create: `js/ui/panels.js`
- Modify: `css/app.css` (splitter hover/active styles, `.panel-ctl` buttons, maximized/minimized states)
- Test: `test/panels.test.js`

**Interfaces:**
- Pure geometry (tested):

```js
clampPane(px, {min, max}) -> px
nextLayoutState(state, action) -> state
// state: {yaml: {w, open, min}, insp: {w, min}, maximized: null|'yaml'|'canvas'|'inspector'}
// actions: {type:'drag', pane:'yaml'|'insp', dx}, {type:'toggle-yaml'}, {type:'maximize', pane}, {type:'restore'},
//          {type:'minimize', pane}, — maximize when already maximized -> restore; minimize a maximized pane -> restore first.
serializeLayout(state) -> string / parseLayout(string) -> state|null   // localStorage 'econeval.layout.v1'; bad JSON -> null -> defaults
```

- DOM part (thin): applies state to CSS vars `--w-yaml`/`--w-insp` and `data-max`/`data-min` attributes on `#workspace`/panels; pointer-event drag on `.splitter` (pointercapture; also ArrowLeft/ArrowRight ±16px when a splitter has focus); YAML and inspector `.panel-head`s get ⤢ maximize and — minimize buttons (`aria-label="Maximize panel"` / `"Minimize panel"`); the canvas pane (no panel-head) gets its ⤢ button appended to `#canvas-toolbar` (canvas cannot minimize — only maximize/restore); maximized panel fills the workspace under the topbar (grid override), others `display:none`; minimized panel collapses to a 28px labeled strip that restores on click. State persisted on every change.
- Bounds: yaml 200–560px, inspector 240–480px.

- [ ] **Step 1: Failing tests** for the pure functions: drag clamps at both bounds; toggle-yaml opens at last width (default 300); maximize/restore round-trip; minimize on maximized restores first; serialize/parse round-trip; parseLayout('garbage') → null.
- [ ] **Steps 2-4:** red → implement pure module → green; then the DOM wiring + CSS; manual check (serve, drag both splitters, keyboard-drag, maximize each pane, minimize/restore, reload restores layout) — record checks in the report. **Step 5: Commit and push.**

---

### Task 9: Canvas — rendering

**Files:**
- Create: `js/ui/canvas.js`
- Modify: `css/canvas.css`
- Test: `test/canvas-model.test.js` (geometry helpers only)

**Interfaces:**
- Produces `createCanvas(svgEl, store, {layoutFor}) -> {render(), setTool(t), currentModelPath: []}` plus exported pure helpers (tested): `edgePath(from, to, r)` (line trimmed to node radii, with arrowhead ref), `selfLoopPath(xy, r)` (arc above the node), `edgeLabelPos(from, to)`.
- Render rules: viewBox pan/zoom via `viewBox` attr (wheel = zoom to cursor 0.5–2.5×, background drag in select mode = pan); markov: circle r 26, name centered (--fs-1), edges with arrowheads, label = the p source text (`rest` shown as `rest`, numbers as-is, expressions verbatim, `--font-data --fs-0`, max 14 chars + '…'); transition-reward edges get a small ⊕ marker; multinomial rows render each counts-target as an edge labeled `n/total`. Tree: shapes per the design vocabulary; each non-root child's edge labeled with its p; payoffs summary (cost/utility) as a second line under the name (--fs-0, muted). Sub-model terminal = stadium double-stroke labeled `model: name`; double-click enters it (`currentModelPath` pushed; breadcrumb `#breadcrumb` shows `main / name / …`, click to pop; canvas then renders `model.models[name]` read-only-false — same editing ops apply against the sub-model via a path-aware op wrapper: for v1, entering a sub-model re-roots the canvas AND inspector on a store whose applyOp maps `fn` over `model.models[name]`; implement as `scopedStore(store, modelName)` inside canvas.js, ~20 lines).
- Selection: halo per design; clicking empty space clears; everything re-renders from `store.subscribe`.

- [ ] **Step 1: Failing tests** for the three geometry helpers (numeric cases: horizontal pair trimmed by r each side; self-loop path starts/ends on the circle; label midpoint offset perpendicular by 10px).
- [ ] **Steps 2-4:** red → helpers green → full render implementation → serve and verify with both examples (hiv: 4-state ring, self-loops, rest labels; surgery: tree shapes, edge probabilities, payoff lines) in light+dark; record checks. Suite green. **Step 5: Commit and push.**

---

### Task 10: Canvas — editing gestures

**Files:**
- Modify: `js/ui/canvas.js`, `css/canvas.css`, `index.html` (toolbar buttons)

**Interfaces:**
- Toolbar (`#canvas-toolbar`): Select (V), Add (A), Connect (C), Delete (D) — buttons with `aria-pressed`, keyboard shortcuts, `title` tooltips naming the shortcut. Behavior by tool:
  - **Select:** node drag = move (layout op on pointerup — ONE undo entry, live preview via transform during drag); click = select; double-click name = inline rename (SVG `foreignObject` input; Enter commits via renameState/renameNode, Esc cancels; errors from ops shown in a transient toast strip); Delete/Backspace key = delete selection (with edges: deleteTransition).
  - **Add:** click empty canvas = addState at click point (markov: also setLayout) / on a tree: click a NODE = addChild of it (click empty space does nothing; cursor shows crosshair on nodes only).
  - **Connect:** drag from node A to node B = addTransition(A,B) (markov; A→A allowed = self-loop) / tree: drag from parent to empty space = addChild positioned by autolayout (tree connect to an EXISTING node is invalid — trees are trees; show toast).
  - **Delete:** click node/edge = delete it.
- Esc returns to Select from any tool. All ops go through `store.applyOp` (or the scoped store inside a sub-model).

- [ ] **Step 1:** Implement. **Step 2:** Manual verification script (serve): build the surgery tree from an empty tree model using only gestures (add root children, add chance branches, rename, set nothing in inspector yet), and a 3-state markov with a self-loop; undo ×5 / redo ×5 stays consistent with the YAML pane; record every check in the report. Suite green (no new node tests — gesture code is DOM; its ops are already covered).
- [ ] **Step 3: Commit and push.**

---

### Task 11: Inspector — Selection, Parameters, Settings

**Files:**
- Create: `js/ui/inspector.js`
- Modify: `css/app.css` (form styles, tab strip, finding badges)

**Interfaces:**
- Produces `createInspector(rootEl, tabsEl, store) -> {render()}`. Three tabs (buttons with `aria-selected`); active tab persisted in the layout localStorage blob.
  - **Selection:** empty-state text "Select a state, branch, or transition on the canvas." For a state/node: name field (rename op on change-commit), payoff rows (key/value inputs; value column `--font-data`; − remove buttons; "+ add payoff" appends key input), and for tree nodes p / delay / model / with editors (with = param/value rows). For an edge: p / cost / utility fields. Every expression field validates on input: `compile(value)` in try/catch → red border + message under the field on ExprError (compile only — no eval); commit on change/Enter via the matching op.
  - **Parameters:** table name | value | low | high | dist | source (+ delete col, "+ add parameter"); values in `--font-data`; same compile-validation per cell; edits via setParam/renameParam/addParam/deleteParam.
  - **Settings:** fields for name, type (read-only badge), cycles, cycle (text, e.g. "1 year"), discount.cost, discount.effect, correction (select: none / half-cycle / life-table), wtp, age, psa.n, psa.seed, start (select of state names; markov only). Edits via setSetting; invalid values surface the op's thrown message under the field.
  - **Findings:** on every store change run `check(model)` (debounced 300ms); tab strip shows a badge with the error count (danger) / warning count (warn); each finding with a `path` matching the currently-rendered fields shows inline under that field; the rest listed at the bottom of the active tab under "Model findings".
- [ ] **Step 1:** Implement. **Step 2:** Manual verification: load hiv example → Parameters shows p_AB row with dist; type `beta(1,` in a dist cell → inline ExprError; introduce a row-sum error via an edge p → badge shows 1 error and the message names the row; Settings edits round-trip into the YAML pane; record checks. Suite green. **Step 3: Commit and push.**

---

### Task 12: Files, examples, app wiring

**Files:**
- Create: `js/ui/files.js`, `js/ui/app.js` (real), `examples/index.json`
- Modify: `index.html` (dialog content), `README.md` (features list)
- Test: `test/files.test.js`

**Interfaces:**
- `files.js` (pure over an injected storage object, tested with a Map-backed shim):

```js
createRegistry(storage) -> reg     // storage: {getItem, setItem, removeItem}; key 'econeval.models.v1'
reg.list() -> [{id, name, updated, versionCount}]
reg.saveVersion(id|null, name, text, label?) -> id     // null id = new entry; prepends {ts, label, text}; caps 20 versions/model
reg.load(id, versionTs?) -> {text, name}               // default latest
reg.remove(id)
reg.autosave(text) / reg.readAutosave() -> text|null    // key 'econeval.autosave.v1'
```

- `app.js` wiring: boot = readAutosave() ?? blank markov template (inline string: 2 states, start, one rest row — valid per check); create store/sync/panels/canvas/inspector; topbar buttons → New (dialog: Markov / Tree → template), Open (dialog listing registry + versions, load/delete), Save version (prompt for optional label; `reg.saveVersion`; topbar shows model name + type badge + dirty dot), Examples (fetch `examples/index.json` → `[{file:"hiv.yaml", name:"HIV combination therapy (Chancellor 1997)"}, {file:"surgery.yaml", name:"Surgery vs medication (tree)"}]`, fetch chosen file → confirm-if-dirty → setText), Import (file input, .yaml/.yml, read → setText; parse errors surface in the YAML pane's strip), Export (Blob download `<name>.yaml` from current text), YAML toggle → panels toggle-yaml. Autosave on every store change (debounced 1s). `beforeunload` warning when dirty. Ctrl/Cmd+Z / Shift+Z / Y for undo/redo (when focus is not in a text field), Ctrl/Cmd+S = Save version.
- [ ] **Step 1: Failing tests** for `files.js` (registry CRUD, version cap at 20, autosave round-trip, corrupt JSON in storage → empty registry not a crash).
- [ ] **Steps 2-3:** red → implement files.js → green; then app.js wiring + dialogs.
- [ ] **Step 4: End-to-end manual pass** (serve; record every step in the report): boot → blank model on canvas; Examples → hiv → canvas + params populated; edit a probability in inspector → YAML pane updates; edit YAML text → canvas updates after debounce; break the YAML → error strip with line, canvas stays; fix → recovers; Save version, reload page → autosave restores; Open → load saved version; Export downloads; Import re-loads it; New → Tree → build 2 branches via gestures; undo/redo; maximize canvas; dark mode. Full suite green (103 + phase-2 tests).
- [ ] **Step 5: Commit and push.** If Netlify was set up in Task 1, confirm the deploy updated (`netlify deploy --prod --dir .` if manual) and note the live URL in the report.

---

## Self-review notes (already applied)

- Spec coverage: shell+deploy (T1), design language (T2, tokens verbatim from the design pass), document-first editing via ops/store (T3-5), two-way YAML sync (T6), auto-layout + layout-key rule (T7), flexible windows exactly as Hans specified — resize/maximize/minimize/persist (T8), publication-figure canvas with drawing gestures incl. sub-model drill-in and breadcrumb (T9-10), inspector with inline validation + live check() findings (T11), storage/versions/autosave/import/export/examples menu (T12). Phase 3 items deliberately absent: Run, results drawer, charts. Share links + AI = phase 4.
- Placeholder scan: none; prose-specified DOM tasks carry exact behavioral contracts and manual-verification scripts; all pure logic has real test code or an enumerated assertion list.
- Type consistency: ops signatures match store.applyOp usage; layout keys consistent between ops.setLayout/renameNode cascade, layouts.js, and canvas; `scopedStore` confined to canvas.js.
- Known risk, accepted: gesture/DOM code has no automated tests — mitigated by pure-op coverage underneath and mandatory recorded manual passes; controller verifies in-browser at the end.
