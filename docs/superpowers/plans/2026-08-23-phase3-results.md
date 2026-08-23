# econeval Phase 3 (Results) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The results layer: a Run pipeline over the phase-1 engines, a resizable bottom results drawer with CEA table, Markov trace, tornado, PSA charts (CE-plane, CEAC, EVPI) and a Validation tab — plus the editor-polish bucket from the phase-2 final review. Deployed live.

**Architecture:** Pure analysis orchestration (`analyses.js`) and pure Plotly-spec builders (`charts.js`) under a thin drawer UI (`results.js`). The panel manager's pure reducer grows a results row. Charts read the design tokens at render time so light/dark theming is automatic.

**Tech Stack:** Plotly (plotly.js-dist-min, vendored UMD via script tag — the ONLY new dependency), everything else as phases 1–2.

**Spec:** `docs/superpowers/specs/2026-08-23-econeval-design.md` — results drawer + analyses sections binding.

## Global Constraints

- All phase-1/2 constraints hold (ES modules, `npm test` glob script, pure-logic-tested/DOM-thin, structured errors, English UI, a11y floor, commit+push per task).
- Phase-2's 278 tests stay green after every task.
- Plotly is loaded as a vendored UMD script (`window.Plotly`); chart modules take it as an injected argument so pure spec-builders stay testable without it.
- **Chart rules (from the dataviz pass, binding):** one axis per chart — never dual y-axes; a legend whenever ≥ 2 series (none for a single series); chart text (axis labels, values, legends) always wears ink/muted tokens, never a series color; series colors follow the STRATEGY (fixed declaration order), never rank, and never repaint when a filter changes; status colors (--danger/--warn) are reserved for findings, never used as series colors; hover tooltips on by default (Plotly hovermode); every chart tab offers a "Show data" table fallback; lines 2px, markers ≥ 8px with a 2px surface ring; grid/axis lines recessive (--line).
- **Chart palette (validated for CVD, chroma, lightness and contrast on both surfaces — byte-exact, pinned by test):**
  - light (`:root`): `--chart-1: #00806B; --chart-2: #3B5BD9; --chart-3: #C2611E; --chart-4: #8A4E9E; --chart-5: #6B7F1F; --chart-6: #C64B7E;`
  - dark (both dark blocks): `--chart-1: #2FA795; --chart-2: #7089E8; --chart-3: #D07C35; --chart-4: #AF7BCB; --chart-5: #879A2E; --chart-6: #C9688F;`
  - Strategy N gets `--chart-N` by declaration order; a 7th+ strategy folds into muted gray with a legend note (never a generated hue).
- Deploys: `netlify deploy --prod --dir . --site 4c526e64-937b-4c3a-a548-f701d9804a56` — ALWAYS the explicit --site flag.

## Module contracts (authoritative)

```js
// js/ui/analyses.js  (pure; consumes engine/run.js, analysis/{cea,dsa,psa,check}.js)
availability(model) -> {tornado: {ok, params: [names], reason?}, psa: {ok, reason?}, trace: bool /* markov only */}
gate(model) -> {ok: true} | {ok: false, errors: findings[]}        // check() errors block, warnings pass
runBase(model, {wtp}) -> {results, ceaRows, strategies: [names]}    // results = run(model); ceaRows = cea(results-shaped, {wtp}).rows
traceFor(results, strategy) -> {states: [names], series: {state: number[]}, cycles: n}  // from results.strategies[s].trace
runTornado(model, {a, b, wtp}) -> bars                              // = dsa.oneWay
runPsa(model, {n?, seed?}) -> psaResult                             // = psa()
psaDerived(psaResult, {comparator, wtpMax}) -> {plane: {strategy: [{dcost,dqaly}]}, wtps: number[], ceacCurves, evpi: number[]}
  // wtps = 26 points 0..wtpMax (wtpMax default 2.5*settings.wtp, fallback 100000)

// js/ui/charts.js  (pure spec builders; no DOM, no Plotly import)
readChartTheme(styleGetter) -> {colors: [6], ink, muted, line, surface, fontUI, fontData}
buildTraceSpec(traceData, theme)      -> {data, layout, config}
buildTornadoSpec(bars, theme)         -> {data, layout, config}   // two fixed hues: colors[1]=low, colors[2]=high; base zero-line
buildCEPlaneSpec(plane, theme, strategyIndex) -> {data, layout, config}
buildCEACSpec({wtps, curves}, theme, strategyIndex) -> {data, layout, config}
buildEVPISpec({wtps, evpi}, theme)    -> {data, layout, config}   // single series: colors[0], no legend
// strategyIndex: Map name -> palette slot (declaration order, stable across filters)
// every layout: paper_bgcolor & plot_bgcolor = surface, font.color = ink, gridcolor = line,
// displayModeBar false, margins tight; legend horizontal above plot when >= 2 series

// js/ui/results.js
createResults(paneEl, store, {flush, plotly, selectOnCanvas}) -> {render(), runBase(), openTab(t)}
// selectOnCanvas(sel) provided by app.js: store.select + inspector focus (Validation click-through)

// js/ui/panels.js (extended reducer)
// state.results {h, open}; actions: {type:'drag-results', dy}, {type:'toggle-results'},
// maximize/minimize accept 'results'; bounds h 160–600, default 300; CSS var --h-results;
// #split-b splitter hidden when closed/maximized-elsewhere; blob persists results state.
```

---

### Task 1: Plotly vendor, drawer shell, panels results-row

**Files:**
- Modify: `index.html` (script tag; `#split-b` + `#pane-results` markup; `#btn-run` in topbar), `js/ui/panels.js`, `css/app.css`, `css/tokens.css`, `test/panels.test.js`, `test/tokens.test.js`, `package.json` (dep)
- Create: `js/vendor/plotly.min.js` (copied from `node_modules/plotly.js-dist-min/plotly.min.js`)

**Interfaces:** Produces the drawer DOM (`#pane-results` with `.panel-head` "Results" + `#results-tabs` + `#results-body`), the grid row (`grid-template-rows: minmax(0,1fr) 4px var(--h-results, 0px)` on `#workspace`, columns unchanged, results row spanning all columns via `grid-column: 1 / -1`), the extended reducer per the contracts block, and the chart tokens.

- [ ] **Step 1: Failing tests** — extend `test/panels.test.js` (new cases: drag-results clamps 160/600; toggle-results reopens at last h (default 300); maximize 'results' / restore; minimize results on maximized restores first; serialize round-trip includes results; parseLayout backward-compat: a phase-2 blob WITHOUT results parses to defaults, not null) and `test/tokens.test.js` (pin all 18 --chart-N values byte-exact in :root and both dark blocks).
- [ ] **Step 2:** red → implement reducer + tokens → green.
- [ ] **Step 3:** `npm i plotly.js-dist-min`, copy the min file, add `<script defer src="js/vendor/plotly.min.js"></script>`, drawer markup, `#btn-run` (accent primary; `#btn-save` becomes quiet — Run is now THE primary action), CSS (drawer row, horizontal splitter cursor row-resize, tab strip reusing inspector tab styles, `--h-results`).
- [ ] **Step 4:** DOM wiring in panels.js for `#split-b` (vertical drag = dy, keyboard ArrowUp/Down ±16) + results panel-head ctl buttons; Run button does nothing yet (Task 4). Manual/DOM-shim check recorded: drawer opens/resizes/maximizes/persists; closed by default; splitter hidden when closed.
- [ ] **Step 5:** Full suite; commit and push.

---

### Task 2: Analysis orchestration (`analyses.js`)

**Files:**
- Create: `js/ui/analyses.js`
- Test: `test/analyses.test.js`

**Interfaces:** per the contracts block. Semantics: `gate` blocks only on `level==='error'`; `availability.tornado.ok` requires ≥ 2 strategies AND ≥ 1 param with both low and high (list them); `psa.ok` requires ≥ 1 dist-bearing param (top-level); `trace` true only for markov. `runBase` shapes `cea()` input from `run()` results (`{name: {cost, qaly}}`). `psaDerived` computes the wtp grid (26 points incl. 0), reuses `analysis/psa.js`'s `ceac`/`evpi`/`cePlane`.

- [ ] **Step 1: Failing tests** — using `examples/hiv.yaml` and `examples/surgery.yaml` via parseModel: hiv availability (tornado ok with p_AB listed; psa ok; trace true); hiv runBase ceaRows: combo ICER within 1e-2 of 5976.80 (the phase-1 golden value); traceFor returns 20 cycles and the state list; surgery: trace false, ceaRows ICER 28387.10 ± 1e-2; gate on a broken model (row-sum error fixture) returns ok:false with the finding; runTornado on hiv (a mono, b combo, wtp 30000) returns exactly one bar (p_AB) with base = independent dsa.oneWay result; psaDerived on a tiny seeded psa run: wtps length 26 starting 0, ceac curves sum to 1 at each wtp (± 1e-9), evpi non-negative.
- [ ] **Steps 2-4:** red → implement (~120 lines) → green + suite. **Step 5: Commit and push.**

---

### Task 3: Chart spec builders (`charts.js`)

**Files:**
- Create: `js/ui/charts.js`
- Test: `test/charts.test.js`

**Interfaces:** per the contracts block + the binding chart rules. Details: trace spec = one line per state (2px), x = cycle 0..n, hovermode 'x unified', legend when ≥ 2 states; tornado = horizontal overlaid bars per param (low bar colors[1], high bar colors[2], sorted by |range| — already sorted by dsa), base value as a vertical reference line, legend Low/High; CE-plane = scatter 8px markers with 2px surface-colored ring per strategy (colors by strategyIndex), quadrant zero-lines emphasized, x = ΔQALY y = ΔCost (label axes with the comparator name), legend when ≥ 2 series; CEAC = one 2px line per strategy vs wtp, y 0..1, legend; EVPI = single colors[0] line, no legend. All: axis/legend/hover text in ink/muted via layout.font — NEVER series colors; gridcolor line; surface backgrounds; displayModeBar false.

- [ ] **Step 1: Failing tests** — with a fake theme object (no DOM): every builder returns {data, layout, config}; trace: N series, line.width 2, layout.font.color === theme.ink, plot_bgcolor === theme.surface, legend present iff ≥ 2 series (single-state model → layout.showlegend false); tornado: exactly 2 trace objects colored theme.colors[1]/(2], orientation 'h'; CE-plane: marker.size ≥ 8 and marker.line.color === theme.surface with width 2, colors follow strategyIndex NOT array order (test with a shuffled subset — strategy 'combo' keeps its slot when 'mono' filtered out); CEAC y-axis range [0,1]; EVPI: 1 series, showlegend false, color theme.colors[0]; NO builder ever emits a second y-axis (assert layout.yaxis2 undefined for all).
- [ ] **Steps 2-4:** red → implement → green + suite. **Step 5: Commit and push.**

---

### Task 4: Run pipeline + CEA + Trace tabs (`results.js`)

**Files:**
- Create: `js/ui/results.js`
- Modify: `js/ui/app.js` (wire createResults, #btn-run, Ctrl/Cmd+Enter), `css/app.css` (CEA table, drawer header controls, stale banner, empty states)

**Interfaces:** `createResults(paneEl, store, {flush, plotly, selectOnCanvas})`. Run flow: flush() → gate(model) — on errors: open drawer + Validation tab + toast "Model has N errors — fix them to run" → runBase → drawer opens (toggle-results via panels if closed) → CEA tab renders. Drawer header: WTP input (numeric, default settings.wtp ?? 30000; change re-derives ceaRows + tornado without re-running engines where possible) and a muted run-stamp ("Run · mono, combo · 14:32"). Staleness: on every store change after a run, compare text to the run's text — differ ⇒ show the muted "Results are stale — Run again" banner (results stay visible). CEA tab: HTML table (strategy | cost | ΔC | QALY | ΔE | ICER | NMB | status) — numbers --font-data right-aligned, dominated/extended rows muted with a status chip, ICER null → "—", cost/NMB formatted 0-decimals, QALY 4; a footnote line naming wtp used. Trace tab (markov only; else hidden): strategy `<select>` + Plotly render via `plotly.react(el, spec.data, spec.layout, spec.config)`; "Show data" `<details>` table under every chart (cycles × states). Busy state on Run (disable button, aria-busy). Theme changes (matchMedia prefers-color-scheme + a data-theme MutationObserver) re-render the open tab's chart. Empty states for never-run.

- [ ] **Step 1:** Implement. **Step 2:** Recorded verification (real browser expected — chrome-devtools worked in T11/T12): hiv Run → CEA ICER ≈ 5977 rendered; trace shows 4 state lines with legend + unified hover; WTP change updates NMB column; edit model → stale banner; re-Run clears it; tree model → Trace tab hidden; gate-blocked model routes to Validation-tab placeholder (tab exists, content lands Task 6). **Step 3:** Full suite; commit and push.

---

### Task 5: Tornado + PSA tabs

**Files:**
- Modify: `js/ui/results.js`, `css/app.css`

**Interfaces:** Tornado tab: comparator selects a/b (defaults: first two strategies in declaration order), re-runs runTornado on change; unavailable → empty state naming WHY ("needs ≥ 2 strategies and ≥ 1 parameter with low and high — add bounds in Parameters"); chart + Show-data table. PSA tab: header shows n/seed from settings.psa with a "Run PSA (n=1000)" button (explicit — PSA is the only potentially-slow analysis; run synchronously with aria-busy; if it takes > ~2s on typical models flag in report); comparator select (default first strategy); after run: three stacked charts — CE-plane, CEAC, EVPI — each with Show-data; unavailable → empty state ("no parameter has a dist — add distributions in Parameters"); PSA results go stale with the same banner rule. Both tabs re-theme on mode change.

- [ ] **Step 1:** Implement. **Step 2:** Recorded verification: hiv tornado (p_AB single bar pair, base line correct sign); PSA run on hiv → CE-plane cloud + CEAC curves crossing + EVPI curve; comparator switch recolors nothing (colors follow strategy) but re-computes increments; dark-mode flip re-themes all three; Show-data tables match chart series lengths. **Step 3:** Full suite; commit and push.

---

### Task 6: Validation tab + run gating polish

**Files:**
- Modify: `js/ui/results.js`, `js/ui/app.js` (selectOnCanvas callback), `css/app.css`

**Interfaces:** Validation tab: all current findings (not debounce-stale: re-run check() on tab open), grouped Errors then Warnings, each row = level chip + code + message (--font-data for code), rows whose path resolves to a state (`states.X`, `transitions.X…`) or tree node are buttons — click → selectOnCanvas({kind, id, modelPath []}) + inspector Selection tab focus; unresolvable paths render as plain rows. Empty state: "No findings — the model is clean." Tab badge on the Results tab strip mirrors the inspector badge counts. Gate flow from Task 4 now lands here with content. `selectOnCanvas` in app.js: store.select + saveLayout tab 'selection' + inspector render (reuse existing helpers — do not duplicate logic).

- [ ] **Step 1:** Implement. **Step 2:** Recorded verification: break a row sum → Run → routed to Validation, click the finding → state selected on canvas + inspector shows it; fix → Run succeeds; warnings-only model runs fine and Validation lists the warnings. **Step 3:** Full suite; commit and push.

---

### Task 7: Editor-polish bucket (phase-2 final-review carryovers)

**Files:**
- Modify: `js/core/model.js`, `js/ui/inspector.js`, `js/ui/ops.js`, `index.html`, `test/model-tree.test.js` (or new `test/polish.test.js`), `test/ops-tree.test.js`, `test/store.test.js`

All items, each with a test where code changes:
1. **Cycle-unit round-trip formatting** (`serializeModel` + Settings display): cycleYears 1/12→`1 month`, k/12→`k months`, 7/365.25→`1 week`, k*7/365.25→`k weeks`, 1/365.25→`1 day`, k/365.25→`k days`, integer n→`n year`/`n years`; non-matching fractions keep decimal years (tolerance 1e-9 on the fraction match). Test: parse `cycle: 1 month` → serialize contains `cycle: 1 month`; `6 months`, `2 years`, `1 week` likewise; an odd 0.3 stays `0.3 year`. Inspector Settings shows the same formatted string.
2. **Model name field**: `contenteditable="plaintext-only"` + `aria-label="Model name"` + `role="textbox"` in index.html.
3. **Inspector tab strip**: preserve focus on the clicked tab across the rebuild; ArrowLeft/ArrowRight move between tabs (roving tabindex).
4. **Reserved payoff-key guard**: `setStatePayoff`/`setNodePayoff` throw on reserved keys (`cost`/`utility` allowed as themselves; forbidden as EXTRA names: `source`, `notes`, and for nodes also `p`, `kind`, `children`, `model`, `with`, `delay`) with a message naming the key. Tests both ops.
5. **Prune dead low/high finding registrations** in inspector.js.
6. **Test tidy-ups**: fix the tautological resetHistory assertion (snapshot before, compare after); add the prefix-collision layout fixture (siblings `A`/`AA` with layout keys, rename `A` → `A2`, assert `AA/...` keys untouched); add a root-rename test.

- [ ] **Step 1:** Failing tests for 1/4/6 → green. **Step 2:** Items 2/3/5 + recorded checks (tab arrows, name paste plain). **Step 3:** Full suite; commit and push.

---

### Task 8: E2E, deploy, README

**Files:**
- Modify: `README.md` (analyses section: what Run gives you, chart list, PSA note)

- [ ] **Step 1: Real-browser e2e** (record everything): hiv → Run → all four analysis tabs verified with values (CEA ICER ≈ 5977; trace 4 series; tornado p_AB; PSA n=1000 seeded → CEAC crossing, EVPI peak near the ICER region); WTP sweep changes NMB + CEAC vertical readout consistent; stale-banner cycle; Validation click-through; drawer resize/maximize/persist across reload; surgery tree → Run → CEA only + Trace hidden + tornado empty-state reason; dark mode: all charts re-themed, palette = the dark tokens; keyboard: Ctrl/Cmd+Enter runs, tab arrows work; `cycle: 1 month` model round-trips through a drag op with the unit intact.
- [ ] **Step 2:** Full suite; README update.
- [ ] **Step 3:** Commit, push, deploy (`--site 4c526e64-937b-4c3a-a548-f701d9804a56`), curl-verify live app.js + plotly vendor 200.

---

## Self-review notes (already applied)

- Spec coverage: run pipeline + results drawer with the spec's exact tab list (T1/T4-6), all phase-3 charts (T3/T4/T5), validation panel with click-through (T6), flexible-windows parity for the drawer (T1), polish bucket (T7), deploy (T8). Phase-4 items deliberately absent: AI, share links, help page.
- Chart decisions trace to the dataviz pass: palette validated by script on both real surfaces (values pinned by test); CEA is a table by form-choice; single-series EVPI carries no legend; tornado's two hues are palette slots, not status colors; no dual axes anywhere (tested).
- Type consistency: analyses.js output shapes match charts.js inputs (traceFor→buildTraceSpec, psaDerived→the three PSA builders); createResults options mirror createCanvas/createInspector patterns ({flush} + injected plotly); panels reducer extension follows the existing action vocabulary.
- Known risk, accepted: chart rendering itself is verified by recorded browser passes, not unit tests (Plotly is injected, spec-builders are the tested layer) — consistent with the phase-2 DOM convention.
