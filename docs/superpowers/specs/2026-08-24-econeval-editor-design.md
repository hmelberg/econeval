# econeval — modeless canvas editor + outline sidebar (design spec)

*2026-08-24. Approved section-by-section in conversation with Hans Melberg. Supersedes the editor
sections of `2026-08-23-econeval-design.md` (phase 2, "Task 10"); everything else in that spec —
the format, the engines, the analyses — is unchanged and still binding.*

## What and why

Phase 2 shipped a canvas editor built around a 4-tool toolbar: you pick Select / Add / Connect /
Delete, then click. Live use surfaced four problems:

1. **Modes are friction.** Every edit needs a tool decision first. The tools also grabbed the bare
   keys V/A/C/D globally, so typing `a` anywhere outside a text field silently switched tools.
2. **Objects are hard to hit.** `.edge-line` is `fill: none; stroke-width: 1.5`, and SVG's default
   `visiblePainted` means only the painted 1.5px stroke is clickable — you must hit an edge to the
   pixel. Tree terminal nodes are a 1.5px `<line>`, so only their text label is realistically
   clickable, and in tree mode clicking an edge is how you select the child node. Drop-target
   hit-testing (`hitTestNode`) treats every shape as a circle regardless of its real outline.
3. **The sidebar hides the model.** The Selection tab shows one object at a time and only ever the
   one the canvas selected; the Parameters tab crams seven columns (Name/Value/Low/High/Dist/
   Source/×) into a 300px pane, leaving ~35px per cell — the name column is unreadable.
4. **Layout is a one-way door.** Dragging a node writes an explicit `layout` entry that pins it
   forever; there is no way back to `autoMarkov`/`autoTree`.

This round replaces the mode-based editor with a modeless one, fixes hit-testing, and replaces the
three-tab inspector with a single filterable outline of the whole document.

**Non-goals for this round:** multi-select and marquee selection (deliberately declined — background
drag stays pan); a retained-mode/diffing renderer (full re-render is still fine at this scale);
multinomial transition-row editing; any change to the model format, the engines, or the analyses.

## 1. Interaction model

The toolbar's four tool buttons and the V/A/C/D shortcuts are **deleted**, along with the `tool`
state variable and the `data-tool` attribute they drove. `#canvas-toolbar` keeps panels.js's own
maximize button and gains the view controls from §2.

### Gesture table (normative)

| Gesture | Markov | Tree |
|---|---|---|
| Click empty | deselect | deselect |
| Double-click empty | new state at that point | new child of the selected node, at that point; toast if nothing is selected |
| Click object | select → halo on canvas, row expands in sidebar | same |
| Double-click node | inline rename; on a sub-model node, drill into it | same |
| Drag node → empty | move it | move it |
| Drag node → another node | transition A→B | A and its subtree become a child of B |
| Drag node → leave its own area and return | self-loop A→A | no-op |
| Space + drag → empty | new state there **and** an edge to it | new child there |
| Space + drag → node | edge (never a move) | re-parent |
| ⌥ + drag | force move — permits dropping a node on top of another | same |
| ⌘ + drag | ignore grid snap, place freely | same |
| Drag background | pan | pan |
| Right-click object | context menu (§1.3) | context menu (§1.3) |
| Delete / Backspace | delete selection | delete selection |
| Escape | cancel gesture / rename, else deselect | same |
| Enter | inline rename of the selection | same |
| Arrows (⇧ = larger step) | nudge the selected node | same |

### 1.1 Decide-at-drop, and how the preview stays honest

A drag from a node is resolved at **drop**, not at drag start: released over another node it is a
connect, released over empty space it is a move. The preview never lies about the model:

- The real node **never moves** during the drag. A translucent clone (`.drag-ghost`) follows the
  cursor with the grab offset preserved — fixing the current bug where the node's centre snaps to
  the cursor (`canvas.js:965`), so grabbing near a node's edge makes it jump.
- Over a valid drop target, the clone hides, a ghost arrow from the source's rim to the target's
  rim appears, and the target gets a `.drop-ring`. Off the target, the reverse. Because the real
  node stayed put, this is a crossfade between two overlays, never a snap-back.
- Holding **Space** forces the connect reading for the whole gesture: the clone never appears, the
  ghost arrow is drawn the entire time, and a drop on empty space creates a new node there and
  connects to it. Holding **⌥** forces the move reading.

**Self-loops.** A drag that starts and ends on the same node is ambiguous. Rule: the gesture tracks
a `leftSource` flag, set the first time the cursor leaves the source node's hit area. Dropping back
on the source with `leftSource` false is a zero-distance move (a no-op); with `leftSource` true it
is a self-loop. Markov only; in a tree it is a no-op.

### 1.2 Creation

Double-clicking empty space creates a node at that point in both model types. In a tree a node
cannot be an orphan, so it becomes a child of the **current selection**; the selection must be a
node in the scope the canvas is currently showing (`sameModelPath`), otherwise it does not count.

**Ordering, discovered in implementation:** a single click on empty space deselects, so by the time
the *second* click of a double-click runs, the selection the create needs is already gone. The first
click therefore snapshots the selection before clearing it, and the create reads that snapshot. Both
halves of this are required — dropping the deselect would leave no way to clear a selection by
clicking away, and dropping the snapshot would make tree double-click-create impossible.
With no usable selection the toast reads `Select a parent node first.` Space-drag from a node to
empty space is the direct alternative — it names the parent by where the drag started — and in a
tree the two produce the same result.

Double-click detection stays hand-rolled (the existing `lastDown` / `DOUBLE_CLICK_MS` machinery at
`canvas.js:523-547`): native `dblclick` synthesis interacts unpredictably with pointer capture, and
nothing in the canvas may depend on a native `click`/`dblclick` event.

### 1.3 Context menu

Right-click replaces the deleted Delete tool. On a node: **Delete · Rename · Tidy position**, plus
**Enter sub-model** on a sub-model attachment, plus **Add child · Add sibling** in a tree (both are
`addChild`, on this node's path and on its parent's respectively — no new op). On an edge:
**Delete**. On empty canvas: **Add state** (markov) / **Add child of selection** (tree) · **Tidy
layout** · **Fit to view**. There is deliberately no Duplicate: a collision-safe subtree copy is a
third new op nobody asked for. The menu is a plain absolutely-positioned `<div>`,
dismissed on Escape, on any outside pointerdown, and on scroll; it never uses a native
`window.confirm`/`alert` (they block the extension-driven e2e run).

### 1.4 Modifier hygiene

Space must not scroll the page and must not act while focus is in an input, textarea, select,
contenteditable, or the inline rename box — the existing `isTypingTarget` guard covers this and is
reused verbatim. Space is only claimed while the pointer is down on a node; a bare Space press
elsewhere is left alone. The existing `dialog[open]` early-return in the keydown handler stays.

## 2. Canvas mechanics

**Hit layers.** Every edge renders an invisible sibling path drawn *under* the visible line:
`stroke: transparent; stroke-width: 14; fill: none; pointer-events: stroke`. Terminal bars and any
other shape with no meaningful fill get a transparent hit rect sized to the shape's bounding box
plus padding. This is the single change that fixes "it only reacts to a specific area". Edge labels
stay inside the edge's `<g>` and remain clickable.

**Shape-aware drop targets.** `hitTestNode` stops approximating every node as a circle of `hitR`.
Each `nodeIndex` entry carries a `hit` descriptor — `{shape: 'circle', r}`, `{shape: 'rect', w, h}`,
`{shape: 'stadium', w, h}` — tested by a pure predicate with ~6px of slack. When shapes overlap, the
**last** match in `nodeIndex` order wins (the topmost-rendered node), not the first.

**Grid snap.** Dragged nodes snap to the existing 12px dot grid on drop and in the live preview;
⌘ held places freely. Snap applies only to nodes moved in this gesture — existing coordinates are
never re-snapped in bulk.

**Wheel and view.** The wheel currently zooms, which means a two-finger trackpad scroll zooms the
canvas. It flips to the platform standard: plain wheel / two-finger scroll **pans**, and
`ctrlKey || metaKey` + wheel **zooms to the cursor** (trackpad pinch arrives as ctrl+wheel). New
corner controls in `#canvas-toolbar`: zoom out, zoom in, **Fit to view**, **Tidy**. Keyboard: ⌘0
fit, ⌘+ / ⌘− zoom. Fit computes the bounding box of every node position plus padding and writes it
straight to the viewBox.

**Tidy.** Clears explicit `layout` entries so `autoMarkov`/`autoTree` takes over again — whole model
from the toolbar button or the background context menu, single node from a node's context menu.

## 3. Sidebar: one outline

The three-tab strip (Selection / Parameters / Settings) is removed. `#pane-inspector` becomes a
single scrolling outline: a filter input pinned at the top, then collapsible groups.

```
┌─ filter ────────────────┐
│ 🔍                      │
└─────────────────────────┘
▾ STRUCTURE            (3)
  ▾ Well                  ●
      Name
      [Well             ]
      Payoffs
        cost
        [1000           ] −
        utility
        [0.9            ] −
        + add payoff
    → Well          rest
    → Sick           0.2
    → Dead          0.05  ▲
  ▸ Sick
  ▸ Dead
▸ SUB-MODELS           (2)
▸ PARAMETERS           (8)
▸ SETTINGS
```

**Groups.**
- **STRUCTURE** — for a markov model, every state, with its outgoing transitions nested beneath it
  as `→ Target  p`; for a tree, the tree itself, indented by depth, with the payoff summary muted on
  the right. Shows the scope the canvas is currently in (`currentModelPath`).
- **SUB-MODELS** — one row per entry in the top-level `models:` registry, expandable to that
  sub-model's own structure. Clicking a sub-model row drills the canvas into it (`openScope`), so
  the outline is a map of the whole document, not just the visible scope.
  *v1 narrowing (accepted at final review):* a sub-model row **navigates rather than expands** — it
  has no child rows of its own. Its structure is reached by drilling in, at which point STRUCTURE
  shows it (see "Scope", below).
- **PARAMETERS** — one row per parameter; expanding shows value / low / high / dist / source as
  vertical label-above-input rows. This replaces the 7-column table outright.
- **SETTINGS** — always the top-level model's settings, as today's Settings tab renders them.

**Fields.** Selecting a row expands its editors indented beneath it: label above a full-width input,
which is what `.insp-row` / `.insp-row-label` already do. The existing field machinery —
`fieldRow`, `wireCommit`, `wireExprInput`, `makeCommitter`, `keyValueEditor`, `kvRow` — is reused
unchanged; only the container around it changes.

**Findings.** `check()` findings stop being a repeated "Model findings" list at the bottom of every
tab. Each outline row carries a check-path (built with the existing `scopePrefix` and
`nodePathToCheckPath` helpers), a finding shows as a dot on its row, and counts roll up to the group
header. A toggle beside the filter input restricts the outline to rows carrying findings, so "show
me what's broken" is one click. Findings that match no row still render in a residual list at the
bottom, so nothing is ever swallowed.

**Scope.** STRUCTURE follows the canvas scope and edits through the `scopedStore` chain; PARAMETERS
and SETTINGS remain top-level in this round, as ruled in the phase-2 spec, minus the muted hint that
used to explain the split (the outline shows both at once, so the split is visible rather than
described).

**Selection sync (both ways).** A canvas click selects, scrolls the matching row into view, and
expands it. A row click selects on canvas; for a row in another scope it calls `openScope` **before**
`store.select`, matching the ordering `app.js`'s `selectOnCanvas` already relies on so the halo's
`sameModelPath` check sees the right scope.

**What the tab strip leaves behind.** Three loose ends, each with a stated resolution:

- `#inspector-tabs` is the pane's `.panel-head`, and `panels.js` writes its own maximize/minimize
  `.panel-ctl` span into it. The outline must keep appending as a sibling and never clear that
  span. With the tab buttons gone the head has no text of its own, so it gets a static `Model`
  title — `css/app.css:295`'s `#pane-inspector[data-min] .panel-head::before` rule, which existed to
  supply a label in the minimized state, becomes redundant and is removed.
- The persisted `tab` key in `panels.js`'s layout blob (`loadLayout().tab`) is dead. It is replaced
  by a persisted set of collapsed group names plus the last filter string, written through the same
  read-merge-write `saveLayout` pattern.
- `createInspector` returns `{render, setActiveTab}` today, and `app.js`'s `selectOnCanvas`
  (`app.js:242-249`, the Validation tab's click-through) calls `setActiveTab('selection')` as its
  third step. It becomes `revealSelection()`: expand the selected row, scroll it into view, and
  expand any collapsed group containing it. `app.js`'s call site changes to match; the
  `openScope` → `store.select` ordering above it is unchanged and still load-bearing.

**Render discipline (carried over, extended).** The existing `shouldSkipRender` rule stands: skip
the structural rebuild whenever focus is on a real input inside the panel, regardless of what
triggered the store change, with the one exception that a selection deleted out from under the user
forces an immediate reconcile. Extended for the outline: scroll position and the set of expanded
groups/rows survive every rebuild.

## 4. New model ops

Both go in `js/ui/ops.js`, following its existing contract — pure `(model, ...) -> newModel`,
`structuredClone` first, plain `Error` with a clear message on invalid input, never a silent fixup.

**`moveNode(model, path, newParentPath)`** — tree only. Rejects, with a distinct message each:
moving the root; dropping a node onto itself; dropping it onto its own descendant; a sibling name
collision under the new parent (rejected, never silently renamed — matching `addChild`'s rule).
Re-keys the entire moved subtree's `layout` entries via the existing `rekeyLayoutSubtree`. Handles
the root boundary both ways: a node promoted to be a root child **loses** its `p` (strategies are
entered unconditionally — the rule `setNodeAttr` already enforces), and a node demoted below the
root **gains** `p: 'rest'`, or `0` if a sibling already carries `rest`, exactly as `addChild` does.

**`clearLayout(model, key?)`** — the Tidy op. With a key, drops that one entry (and, for a tree
node, its whole subtree, via the existing `scrubLayoutSubtree`); without one, drops `layout`
entirely.

## 5. Module split

`canvas.js` is 1011 lines and `inspector.js` is 978; this round would push them to roughly 1600 and
1300. They split along the pure/DOM line the codebase already uses:

```
js/ui/scoped-store.js       scopedStore + scopedStoreFor, moved out of canvas.js
js/ui/canvas/geometry.js    edgePath, selfLoopPath, edgeLabelPos, hit predicates,
                            snapToGrid, fitBox                                    (pure, tested)
js/ui/canvas/render.js      SVG building for markov + tree, including hit layers
js/ui/canvas/gestures.js    pointer state machine, modifiers, ghost overlays
js/ui/canvas/index.js       createCanvas wiring, scope/breadcrumb, view, context menu
js/ui/outline/build.js      model → row list, filter matching, findings mapping   (pure, tested)
js/ui/inspector.js          DOM shell: filter, rows, expand/collapse, field editors
```

`inspector.js` currently imports `scopedStore` *from* `canvas.js` — a backwards dependency between
two peer UI modules; `js/ui/scoped-store.js` fixes it, and both import from there.

Public surfaces: `createCanvas` returns `{render, setTool, currentModelPath, openScope}` today and
loses `setTool` — nothing outside the deleted toolbar called it. `createInspector` returns
`{render, setActiveTab}` and trades `setActiveTab` for `revealSelection` (§3). Those are the only
two call-site changes in `app.js`; `results.js`'s validation click-through goes through
`selectOnCanvas`, so it is untouched.

## 6. Testing

New pure tests, `node --test test/*.test.js` as always (never `node --test <dir>` — it fails on
Node 26):

- **`test/canvas-geometry.test.js`** — hit predicates per shape (inside, on the boundary, within
  slack, outside), topmost-wins on overlap, `snapToGrid` rounding, `fitBox` over a set of positions.
- **`test/ops-move-node.test.js`** — re-parent, both root-boundary `p` cases, self/descendant/root
  guards, sibling collision, subtree layout re-keying, and a serialize→parse round-trip of the
  result (the phase-1 YAML flow-mapping trap makes round-tripping a new op mandatory, not optional).
- **`test/ops-clear-layout.test.js`** — whole-model and single-key/subtree forms.
- **`test/outline-build.test.js`** — row lists for markov and tree, sub-model group, filter matching
  with ancestor retention, findings mapped to rows and rolled up to groups.

`test/canvas-model.test.js` follows the geometry move (imports repointed, assertions unchanged).
All 392 existing tests must stay green. Gesture behaviour itself is verified in real Chrome, as in
phase 3 — automated DOM tests are out of scope for this repo.

## Risks

- **Space modifier** must not scroll the page and must not fire while typing; it is claimed only
  while the pointer is down on a node.
- **Grid snap** applies only to nodes moved in the current gesture. A global re-snap would rewrite
  every existing model's coordinates on first drag.
- **`moveNode` round-trip.** The phase-1 trap — YAML flow mappings with commas inside calls
  (`{dist: beta(1, 2)}`) misparse *silently* — means any new op that restructures the document is
  tested through `serializeModel`/`parseModel`, not just in memory.
- **Focus preservation.** The outline rebuilds far more DOM than the old Selection tab did. The
  existing skip-while-focused discipline is necessary but not sufficient; scroll and expansion
  state must be restored explicitly or the panel will feel like it jumps.
- **Sub-model rows and scope.** STRUCTURE follows the canvas scope while PARAMETERS/SETTINGS stay
  top-level. If that reads as confusing in use, the fallback is to show the scope as a header above
  STRUCTURE rather than to change the ruling.
