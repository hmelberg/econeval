# econeval — design spec

*2026-08-23. Approved section-by-section in conversation with Hans Melberg.*

## What and why

econeval is a browser-based app for building and analysing health economic evaluation models — decision trees and Markov (state-transition) models in v1, agent-based simulation later. It replaces the Anvil prototype `htaui`, whose concepts (draw-first editing, expression-valued attributes, a text outline as an alternate view, model versioning) it keeps, and whose code it treats as a spec, not a starting point.

The landscape research (2026-08-23) found the niche empty: no YAML/JSON standard for health-economic models exists (closest: heRoMod's JSON-shaped model object, heemod's R DSL, botech), and nothing browser-based does Markov + PSA + CEAC openly. SilverDecisions proves a serious client-side tree editor works but has no health-economics layer.

**Decisions fixed with Hans:**

- **Audience:** research-grade tool — usable for real evaluations/publications. Correctness and reproducibility first.
- **Stack:** static single-page app, vanilla JS ES modules, no build step. Compute engine in plain JS. Vendored deps only: js-yaml, Plotly.
- **Shape:** one app, one canvas; the model's `type` drives what the UI shows. No app split.
- **Storage:** local-first (localStorage + YAML file export/import). Anvil-endpoint sync is a later phase; the storage layer must not preclude it.
- **v1 model types:** decision trees + Markov. ABM is a later, separate project.
- **AI:** one thin BYOK feature in v1 ("Draft a model").
- **Name/repo:** `hmelberg/econeval` (old WIP sketch renamed to `hmelberg/oldeconeval`, done 2026-08-23). Deploy: econeval.melberg.app on Netlify, git-CD. Push-always applies. `htaui` stays untouched as reference.

## Architecture principle: the document is the app

The model document (YAML canonical, JSON isomorphic in memory) is the single source of truth. The canvas, the YAML pane, the engines, storage, share links, and the AI feature are all just readers/writers of that document. Every mutation goes through one dispatcher; undo/redo is document snapshots. This makes "one app vs several" a packaging question and makes AI = "an LLM that reads/writes the format".

## The model format

Format version key `econeval: 1`. Top-level keys: `econeval`, `type` (`tree` | `markov`, later `abm`), `name`, `description?`, `settings`, `params`, `tables?`, and per type either `states` + `transitions` + `strategies` (markov) or `tree` (tree). Optional `layout`. Unknown keys are preserved on round-trip but flagged by validation.

### Markov example

```yaml
econeval: 1
type: markov
name: HIV combination therapy

settings:
  cycles: 20
  cycle: 1 year
  discount: {cost: 0.06, effect: 0.015}
  correction: half-cycle        # none | half-cycle | life-table
  wtp: 30000
  psa: {n: 1000, seed: 42}
  start: A                      # or a distribution: {A: 0.8, B: 0.2}; default: first state

params:
  p_AB:   {value: 0.202, low: 0.15, high: 0.25, dist: beta(202, 798)}
  rr:     {value: 0.509, dist: lognormal(-0.675, 0.173), source: "Chancellor 1997"}
  c_drug: 2278                  # bare number = shorthand for {value: 2278}

states:
  A:     {cost: 2756 + c_drug, utility: 0.85}
  B:     {cost: 3052 + c_drug, utility: 0.71}
  death: {cost: 0, utility: 0}

transitions:
  A:     {A: rest, B: p_AB * rr, death: 0.01}
  B:     {B: rest, death: 0.15}
  death: {death: 1}

strategies:
  mono:  {}                     # baseline: params as declared
  combo: {c_drug: 5343}         # strategy = named parameter overrides
```

### Tree example

Nesting *is* the structure — the YAML doubles as the human-readable outline (replaces htaui's separate graph2string format).

```yaml
econeval: 1
type: tree
name: Surgery vs medication

params:
  p_success_surg: {value: 0.9, dist: beta(90, 10)}

tree:
  Treatment?:                   # root = decision node; its children are the strategies
    Surgery:
      cost: 5000
      Success: {p: p_success_surg, utility: 0.95}
      Failure: {p: rest, utility: 0.40, cost: 2000}
    Medication:
      cost: 800
      Success: {p: 0.60, utility: 0.90}
      Failure: {p: rest, utility: 0.50}
```

### Conventions (normative)

- **Everything is an expression.** Any value may be a number, arithmetic over param names, or a distribution call. One spec serves all runs: deterministic uses `value` (or the distribution mean if only `dist` is given), PSA samples `dist`, one-way DSA sweeps `low`–`high`. Params may reference other params (must form a DAG; cycles are validation errors).
- **Expression language** (in `core/expr.js`): numbers, `+ - * / ^`, parentheses, names, function calls, `min`, `max`, `if(cond, a, b)`, comparisons, `lookup(table, x)` (linear interpolation), reserved words `t` (cycle number, 1-based), `state_time` (cycles spent in current state), `rest`.
- **`rest`**: residual probability. In a Markov transition row: 1 minus the sum of the other outgoing probabilities. Among tree siblings: 1 minus the sum of sibling `p`s. At most one per row/sibling-group; validation enforces the remainder is in [0,1]. No silent renormalisation anywhere (surprise principle: errors are surfaced, never papered over).
- **Distributions** — heemod/field vocabulary, exactly these parameterisations: `beta(shape1, shape2)`, `gamma(mean, sd)`, `normal(mean, sd)`, `lognormal(meanlog, sdlog)`, `uniform(min, max)`, `triangular(min, mode, max)`. Deterministic value of a distribution = its mean.
- **Multinomial rows** (markov): a transition row may be written `A: {multinomial: {A: 721, B: 202, death: 10}}` — observed counts per target. Deterministic run normalises the counts; PSA draws the row from the implied Dirichlet. Row sums to 1 by construction; no `rest` allowed in such a row.
- **Tree node vocabulary**: reserved attribute keys are `p`, `cost`, `utility`, `kind`, `source`, `notes`, `children`; any other nested mapping key is a child node. `children:` is the unambiguous fallback (needed if a node must be named e.g. "cost"). `kind` is inferred (root = decision, has-`p` = chance branch, leaf = terminal) and only written when overriding. Costs/utilities accumulate along the path root→leaf.
- **Transition rewards** (markov): a transition value may be an object `{p: expr, cost: expr, utility: expr}` for one-time rewards on making that transition; plain value = probability shorthand.
- **Tables**: named columns of equal length, e.g. `mortality: {age: [40,50,60], rate: [0.002,0.005,0.012]}`; accessed via `lookup(mortality, age0 + t)` reading first column as x, second as y (multi-column lookup with an explicit column argument allowed: `lookup(mortality, x, rate)`).
- **`state_time`**: implemented by internal tunnel expansion of states whose expressions use it (up to `cycles` copies); invisible in the document.
- **Strategies** are named parameter-override maps (v1). Per-strategy structural overrides may come later; parameter dispatch covers the standard cases.
- **`layout:`** optional block, `nodeName: [x, y]`; semantic model stays clean and diffable. Missing/partial layout → auto-layout.
- **`source:`/`notes:`** allowed on params, states, transitions, tree nodes — the model carries its own evidence trail.

## Modules

```
core/model.js     parse, validate, serialize (YAML <-> JSON); schema errors with line refs
core/expr.js      expression parser/evaluator (Pratt), pure
core/dist.js      distributions (sample/mean/quantile) + seeded RNG (sfc32/mulberry32)
engine/markov.js  cohort trace, discounting, correction (none/half-cycle/life-table)
engine/tree.js    rollback expected values
analysis/cea.js   per-strategy totals, incremental table, dominance + extended dominance, ICER, NMB
analysis/psa.js   sampling loop -> CE-plane, CEAC (+CEAF), EVPI
analysis/dsa.js   one-way (tornado on NMB at settings.wtp, metric selectable), two-way grid
analysis/check.js validation: rows sum to 1, p in [0,1], unreachable states, dead ends, DAG params,
                  missing dist warnings for PSA
ui/               canvas (hand-rolled SVG), inspector panels, YAML pane, results, layout manager
ai/draft.js       BYOK draft-a-model (provider adapters: Anthropic, OpenAI)
```

Everything below `ui/` is pure functions, no DOM — tested with `node --test`.

**Canvas:** hand-rolled SVG (editor libraries are React-only or commercial; SilverDecisions precedent; Hans has draw/drawcast experience). SVG over `<canvas>` for DOM hit-testing, CSS styling, crisp zoom. Interactions: drag node to move; drag from node rim to create transition/child; double-click to rename; click to select → inspector; small explicit toolbar (select / add / connect / delete) instead of htaui's timing-based gestures.

## Analyses in v1

Base-case CEA (cost/QALY/LY per strategy, incremental table with dominance and extended dominance, ICER, NMB at WTP); Markov trace plot; discounting (separate cost/effect annual rates, per-cycle) and cycle correction; one-way DSA + tornado; two-way DSA; seeded PSA → CE-plane scatter, CEAC + CEAF, EVPI; live validation panel; `tables` lookup for age-dependent rates. Deferred: EVPPI, budget impact, subgroups.

**Golden tests:** reproduce published numbers — heemod's HIV example (mono vs combo), a Briggs textbook model, and a hand-computed tree. Correctness is claimed, not assumed.

## UI

Single workspace, no page navigation:

- **Top bar:** model name + type badge, Run, open/save, share, examples, settings, AI draft.
- **Center:** canvas (dominant).
- **Right inspector**, tabs: *Selection* (expression fields with inline validation), *Parameters* (table: name, value, low, high, dist, source), *Settings*.
- **Bottom results drawer**, tabs: CEA table, Trace, Tornado, PSA, Validation. Slides up on Run.
- **Left YAML pane**, toggleable, two-way live sync with the canvas.

**Flexible windows (per Hans):** all four panels (YAML, canvas, inspector, results) sit in a CSS-grid workspace with draggable splitters — horizontal resize for the side panels, vertical for the results drawer. Every panel has **maximize** (fills the workspace below the top bar; the top bar stays) and **minimize** (collapses to a labeled edge strip that restores on click). Panel sizes/states persist in localStorage. Implemented with pointer events on splitter divs; no library.

**Storage & sharing:** localStorage model registry with versions (save = new version, htaui-style); `.yaml` file export/import; share link = model compressed (lz-string) into the URL hash — zero backend, teaching-friendly. Examples menu ships the golden-test models.

## AI: Draft a model (BYOK)

Settings stores provider (Anthropic | OpenAI) + API key in localStorage only (never in the model document; both providers permit direct browser calls). Dialog: describe the decision problem in words → prompt = the format spec + few-shot examples → response validated by `core/model.js`, auto-retry on invalid (max 2) → opens on canvas with a persistent "AI draft — parameters unverified" banner and `source: "AI suggestion"` on every AI-provided value.

## Phases

1. **Core** — format + `core/` + engines + analyses, golden tests. No UI.
2. **Editor** — app shell, panel/layout manager, SVG canvas, inspector, YAML sync, storage, examples. Netlify site (econeval.melberg.app) set up at the start of this phase so every push is live.
3. **Results** — run pipeline, all charts, validation panel.
4. **AI + share + polish** — draft-a-model, share links, help page.
5. **Later, separate projects:** Anvil account sync, ABM, EVPPI/budget impact.
