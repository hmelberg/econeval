# econeval

Health economic evaluation in the browser: decision trees and Markov models with a
declarative YAML model format. Spec: `docs/superpowers/specs/2026-08-23-econeval-design.md`.

Status: phase 1 (compute core), phase 2 (editor), and phase 3 (results) are all complete,
covered by `npm test` (388 tests), including two golden examples cross-checked against
independent reference implementations (`test/golden.test.js`). The editor is a browser-based
canvas + YAML editor for the same model format the compute core runs:

- **Two-way YAML sync** — a live document store (undo/redo, selection) with a debounced YAML
  textarea; edits from either side (canvas, inspector, or hand-typed YAML) stay in sync, and
  a broken YAML edit shows an inline error (line + hint) without losing canvas state.
- **Publication-figure canvas** — auto-layout markov rings and tidy decision trees, drawn as
  ink-stroke nodes on a dot-grid paper ground; pan/zoom, drag-to-move, and Select/Add/
  Connect/Delete gesture tools, including sub-model drill-in with a breadcrumb trail.
- **Inspector** — Selection/Parameters/Settings tabs with inline expression validation and
  live `check()` findings (a tab-strip badge plus per-field and whole-model messages).
- **Results** — a Run button (or Ctrl/Cmd+Enter) computes CEA/Trace/Tornado/PSA/Validation
  into a drawer at the bottom of the workspace; see Analyses below for what each tab shows.
  A stale banner appears the moment the model changes underneath a computed result.
  Flexible panels (yaml/canvas/inspector/results) resize, maximize, and minimize, with layout
  persisted across reloads.
- **Files** — versioned local-storage model registry (up to 20 versions per model), autosave
  every second with restore-on-reload, an Examples menu (the two models below), and
  YAML Import/Export.
- Light/dark theme follows the OS preference (or an explicit `data-theme` override); charts
  re-theme in place when it changes.

Run `npm test` for the full suite (phase 1's compute core, phase 2's editor, and phase 3's
results layer — DOM-free logic and pure-function coverage for the DOM-heavy modules; chart
rendering itself is verified by recorded browser passes, not unit tests).

## Run locally

The app is plain static files — ES module imports and `fetch` need real HTTP, not
`file://`. From the repo root:

```sh
python3 -m http.server 8000
```

then open http://localhost:8000. (`npx serve` works too.) After editing anything under
`js/` or `css/`, hard-reload the page — Chrome caches those aggressively.

## Live

https://econeval.netlify.app

Pending: custom domain `econeval.melberg.app` (Netlify dashboard work — DNS is not touched
by this repo's tooling).

## Model format at a glance

A model is one YAML document: `econeval: 1`, a `type` (`markov` or `tree`), `params`, and
either `states`/`transitions`/`strategies` (Markov) or a nested `tree`. Every value —
costs, probabilities, utilities — is an expression: a number, arithmetic over param names,
or a distribution call (`beta(202, 798)` etc.). One spec drives every analysis: the base
case uses `value`, PSA samples `dist`, one-way DSA sweeps `low`–`high`.

### Markov example — `examples/hiv.yaml`

The Chancellor et al. 1997 HIV model (heemod's `c_homogeneous` vignette): a cohort moves
yearly through disease states A → B → C → death, comparing zidovudine monotherapy against
adding lamivudine (which multiplies every progression probability by a relative risk
`rr`). Effectiveness is life-years (utility 1 on every living state), costs are discounted
at 6%/year, and `rest` fills each row's diagonal so probabilities always sum to 1.

```yaml
econeval: 1
type: markov
name: HIV combination therapy

settings:
  cycles: 20
  cycle: 1 year
  discount: {cost: 0.06, effect: 0}
  correction: none
  start: A

params:
  # dist: beta(...) needs block style — a flow mapping {...} would split the call on its comma
  p_AB:
    value: 0.202
    dist: beta(202, 798)
    source: Chancellor 1997
  p_AC:
    value: 0.067
    dist: beta(67, 933)
    source: Chancellor 1997
  p_AD:
    value: 0.010
    dist: beta(10, 990)
    source: Chancellor 1997
  p_BC:
    value: 0.407
    dist: beta(407, 593)
    source: Chancellor 1997
  p_BD:
    value: 0.012
    dist: beta(12, 988)
    source: Chancellor 1997
  p_CD:
    value: 0.250
    dist: beta(250, 750)
    source: Chancellor 1997

  rr: 1                         # base (comparator, mono) params describe no treatment effect
  c_drug: 2278                  # zidovudine (AZT) monotherapy, annual drug cost

states:
  A:     {cost: 2756 + c_drug, utility: 1}
  B:     {cost: 3052 + c_drug, utility: 1}
  C:     {cost: 9007 + c_drug, utility: 1}
  death: {cost: 0, utility: 0}

transitions:
  A:     {A: rest, B: p_AB * rr, C: p_AC * rr, death: p_AD * rr}
  B:     {B: rest, C: p_BC * rr, death: p_BD * rr}
  C:     {C: rest, death: p_CD * rr}
  death: {death: 1}

strategies:
  mono: {}                      # comparator: zidovudine monotherapy, params as declared
  combo:                        # lamivudine added to zidovudine
    c_drug: 2278 + 2086         # + lamivudine annual drug cost
    rr: 0.509                   # combination therapy's relative risk of progression
```

### Tree example — `examples/surgery.yaml`

A decision tree, verbatim from the spec: nesting *is* the structure, so the YAML doubles as
the human-readable outline. The root's children (`Surgery`, `Medication`) are the
strategies; each chance node's `p` is an expression (here `p_success_surg`, a param with a
`beta(90, 10)` prior), and `rest` fills the remaining sibling probability. Costs and
utilities accumulate along the path from root to leaf.

```yaml
econeval: 1
type: tree
name: Surgery vs medication

params:
  p_success_surg:
    value: 0.9
    dist: beta(90, 10)

tree:
  Treatment?:
    Surgery:
      cost: 5000
      Success: {p: p_success_surg, utility: 0.95}
      Failure: {p: rest, utility: 0.40, cost: 2000}
    Medication:
      cost: 800
      Success: {p: 0.60, utility: 0.90}
      Failure: {p: rest, utility: 0.50}
```

### Analyses

Press **Run** to compute results into the drawer at the bottom of the workspace, five tabs:

- **CEA** — per-strategy cost/QALY totals and an incremental table with dominance and
  extended dominance, ICER, and net monetary benefit (NMB) at the willingness-to-pay
  threshold. The **WTP** field lives in the drawer's header and applies immediately to NMB
  and the CEAC/EVPI charts — no re-run needed, since it's a threshold on already-computed
  results, not a model input.
- **Trace** — the cohort's cycle-by-cycle state occupancy for a Markov model, one series per
  state. Not shown for decision trees, which have no cycles to trace.
- **Tornado** — one-way sensitivity on incremental NMB, swept from each param's `low`/`high`
  bounds (`dsa.oneWay`). Needs at least two strategies and at least one top-level param
  carrying both bounds; otherwise the tab explains what's missing instead of rendering.
- **PSA** — an explicit **Run PSA** button (separate from the main Run, since sampling is
  the expensive step) draws `settings.psa.n` iterations over every `dist`-bearing param,
  seeded by `settings.psa.seed` — the same seed always reproduces the same draws, so a PSA
  run is a deterministic, shareable result, not a one-off random sample. Produces the
  cost-effectiveness plane, the CEAC (probability each strategy is cost-effective, swept
  over WTP), and EVPI (expected value of perfect information), each with a "Show data"
  table. PSA also supports optional Gaussian-copula correlations between params; v1
  semantics: global params draw once per iteration and are shared everywhere, while a
  sub-model's own dist-bearing params draw independently per attachment (route through a
  global param via `with:` to share them).
- **Validation** — every `check()` finding, errors and warnings, each one clickable to
  select the offending state/branch/param on the canvas and focus it in the inspector.

A stale banner ("Results are stale — Run again.") appears the moment the model changes
underneath a computed result — CEA/Trace and PSA are tracked independently, so editing the
model after a PSA run flags PSA as stale even if you press Run again for CEA/Trace.

The compute core underneath also exposes two-way DSA grids (`js/analysis/dsa.js`'s
`twoWay`), not yet surfaced as a chart in the drawer.

See the full format spec at `docs/superpowers/specs/2026-08-23-econeval-design.md` for the
expression language, distributions, transition-reward and sub-model composition rules, and
validation checks.

`npm test` runs the suite.
