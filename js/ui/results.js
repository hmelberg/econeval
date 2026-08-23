// Results drawer: CEA + Trace + Tornado + PSA tabs (Validation content lands Task 6 — it is wired
// here only as an empty-state placeholder so the tab strip is complete now).
//
// Tornado (Task 5): pure function of the LIVE current model (store.get().model), never a frozen
// snapshot — it has no "stale" state of its own (unlike CEA/Trace/PSA, which all read from a
// frozen `lastRun`/`lastPsa` and participate in the text-comparison stale banner). It re-runs
// runTornado() on every tab-open, every a/b comparator change, and every WTP commit (ΔNMB depends
// on wtp) — cheap enough (2*nParams+1 run() calls) that there is no separate "Run Tornado" button,
// unlike PSA. `tornadoA`/`tornadoB` persist across tab switches (closure state), defaulting to the
// first two strategies in DECLARATION order and re-clamped to the live strategy list on every
// render (so switching to a model with different/fewer strategies never leaves a dangling
// selection). Strategy names for the two <select>s are derived locally (see `strategyNamesOf`
// below — a deliberate small duplicate of analyses.js's own private helper of the same name,
// avoiding both a public-contract change to analyses.js and an extra throwaway run() call just to
// learn names).
//
// PSA (Task 5): the ONE potentially-slow analysis in this app, so — unlike Tornado — it never runs
// implicitly. `lastPsa` (`{psaResult, text, ts, elapsedMs, n, seed}`) is a frozen snapshot,
// populated only by an explicit "Run PSA (n=N)" button click (see `runPsaNow`), and is NOT cleared
// or re-triggered by the global Run/`runNow()` — pressing the main Run button re-runs the base
// CEA, full stop; PSA keeps showing its last computed result until its own button is pressed again
// or the model changes underneath it. That means `lastPsa.text` can drift out of sync with
// `lastRun.text` (e.g.: run PSA, edit the model, press the global Run again — CEA's staleness
// clears but PSA's does not) — `renderStaleBanner()` checks BOTH snapshots independently and the
// one shared `.res-stale` banner fires if EITHER is out of sync with the live text, per the task
// brief ("a stale PSA shows the banner too"). Once `lastPsa` exists, everything else — the
// comparator select, the CE-plane/CEAC/EVPI specs, `wtpMax` — is recomputed FRESH on every render
// from the cached `psaResult` via the pure `psaDerived()` (no engine re-run): switching the
// comparator only ever calls `psaDerived()` again, never `runPsa()` again. `strategyIndex` for the
// PSA charts is built fresh each render from `psaResult.strategies` (declaration order, exactly
// like `buildStrategyIndex(lastRun.strategies)` elsewhere) — PSA has no dependency on `lastRun`
// ever having succeeded at all.
//
// createResults(paneEl, store, {flush, plotly, selectOnCanvas}) -> {render(), runBase(), openTab(t)}
//   paneEl: #pane-results — this module finds #results-tabs/#results-body inside it itself (the
//           only two ids index.html pre-declares; everything else under them is built here).
//   flush:  () => void, called before reading the model — mirrors canvas.js/inspector.js's own
//           `flush` param: a Run must never race a pending debounced YAML-pane edit.
//   plotly: injected `window.Plotly` (task-1) — only the Trace tab uses it in this task.
//   selectOnCanvas: (sel) => void, provided by app.js for the Validation tab's future click-
//           through (store.select + inspector focus) — accepted per the module contract but
//           UNUSED in this task (Validation has no real rows yet, only an error count).
//
// Run flow (public `runBase()`, the method app.js's #btn-run/Ctrl-Enter handler calls):
//   flush() -> store.get().parseError set? -> toast "Fix the YAML error before running.", stop
//   (lastRun untouched) -> gate(model): errors -> lastGateErrors set, Validation tab, toast, stop
//   (previous results, if any, are left exactly as they were — see "staleness" below); ok ->
//   runBase() (analyses.js) -> lastRun captured, CEA tab. Opening the drawer itself (toggle-results
//   via panels.js if closed) and the Run button's busy/disabled state are app.js's job, NOT this
//   module's — createResults's contract has no panels dispatcher, and #btn-run lives in the
//   topbar, outside paneEl.
//
// Review fix (Critical): the parseError check MUST read `store.get().parseError`, never
// `!store.get().model` — store.js keeps the LAST-GOOD model in place while a live YAML edit fails
// to parse (parseError set, model still the old non-null value). Guarding on `!model` alone misses
// that case entirely: Run would silently compute fresh results off the stale last-good model while
// the YAML pane is showing the user's broken, unparsed text, and lastRun.text would capture that
// broken text too — which means the "Results are stale" banner (a plain text-inequality check
// against lastRun.text) would never fire either, since the live (broken) text IS what got
// snapshotted. `model === null` only happens as a subset of `parseError` being set (the very first
// parse, at store construction, can leave model null) — checking `parseError` covers both cases in
// one guard and can never produce a false positive (parseError is only ever set together with, or
// instead of, a failed parse).
//
// Staleness: lastRun.text is a snapshot of store.get().text at the moment that run completed.
// Every store notification re-checks store.get().text !== lastRun.text and toggles a muted
// banner accordingly — cheap (just that one comparison + a hidden/textContent flip), so it runs on
// EVERY store change, unlike the tab body (see below). A re-run that succeeds replaces lastRun.text
// with the new current text, which is what "clears" the banner; a re-run that fails gate() leaves
// the OLD lastRun (and its now-stale text) untouched, so previously-good results stay visible,
// still correctly marked stale.
//
// Render discipline: the CEA/Trace tab BODY depends only on `lastRun` (frozen at run time) and the
// active tab/strategy selection — never on live store text — so it is only rebuilt on an actual
// tab switch, a run, a WTP edit, or a trace-strategy change. A store notification alone never
// rebuilds the body (it would be pointless work, and — more importantly — would being racy against
// nothing since no field here is store-bound); it only updates the stale banner and the Trace tab's
// hidden/visible state (markov-only), auto-hopping off Trace back to CEA if the live model just
// stopped being markov while Trace was the active tab.
//
// Chart re-render on theme change (extends to T5's Tornado/PSA/CE-plane/CEAC/EVPI charts):
// `activeChartRenderer` is a closure set by whichever tab body just mounted a Plotly chart (only
// Trace, in this task), cleared to null by every renderBody() before dispatching to a tab. A
// prefers-color-scheme match-media 'change' and a MutationObserver on <html data-theme> both just
// call `activeChartRenderer?.()` — T5 only needs to keep setting this same variable from its own
// chart-bearing tabs for theme re-rendering to keep working, no other plumbing to touch.
//
// Review fix (Important) — chart cleanup: every NEW DOM element handed to `plotly.react` MUST be
// mounted via the `mountChart(el, spec)` helper (defined near `purgeCharts()` below), not a bare
// `plotly.react(...)` call — it both draws the chart and adds `el` to `chartRegistry` at the same
// moment `activeChartRenderer` is set (see `renderTraceTab` for the pattern). `renderBody()` calls
// `purgeCharts()` (which calls `plotly.purge(el)` on every registered element, wrapped in try/catch
// so one bad purge can't block the rest, then clears the registry) BEFORE `bodyEl.replaceChildren()`
// — discarding a chart-bearing div without purging it first leaks the Plotly instance's internal
// listeners/WebGL context; over repeated tab/strategy switches this accumulates hidden
// `.js-plotly-plot` instances even though only one is ever visible in the DOM. `mountChart` also
// `.catch()`es the promise `plotly.react` returns — that promise can still be settling when a fast
// subsequent switch purges this same div before it resolves, and an unhandled rejection from
// Plotly's own internal continuation touching the now-purged div is a real, verified-live console
// error otherwise (10x rapid tab/strategy switching reproduced it on every cycle before the catch
// was added). `activeChartRenderer`'s own in-place `plotly.react(el, ...)` re-render (theme change)
// reuses the SAME div and must NOT call `mountChart` (that would double-add/double-purge it) — but
// still needs its own `.catch(logChartRejection)` for the same race, see `renderTraceTab`. T5's
// chart-bearing tabs must follow the same "new chart -> mountChart(el, spec); in-place re-render ->
// bare `plotly.react(...).catch(logChartRejection)`" split — `purgeCharts()`/`renderBody()` need no
// per-tab changes.
//
// Controller amendment (Task 5 review): every one of these `.catch(...)` handlers — the
// purge-raced-the-render rejection is expected/benign, per the above — MUST still log it via
// `logChartRejection` (`console.debug('plotly render rejected', err)`, defined just above
// `mountChart`) rather than swallowing it with a bare `() => {}`. A silent catch can also hide a
// GENUINE spec/runtime bug inside Plotly's own async render path (as opposed to a synchronous
// throw inside `buildXSpec`, which never reaches this catch at all) — logging costs nothing (it's
// `console.debug`, not a user-facing toast/error) and keeps the "expected race" story honest
// without reintroducing the original unhandled-rejection console error.
//
// Extending the tab registry (T6 — Validation is still the one placeholder left): `TABS` (id +
// label, tab STRIP only) and `TAB_RENDERERS` (id -> () => void, called with bodyEl already
// cleared) are the two lists to touch. Add a tab, or replace the validation placeholder entry with
// real content: `openTab`/`renderTabStrip`/`renderBody` need no changes for a same-shape addition.
// `renderTabStrip`'s per-button `hidden` rule is Trace-specific (markov-only) — a future
// conditionally-hidden tab would add its own branch there the same way.

import { gate, runBase, traceFor, availability, runTornado, runPsa, psaDerived } from './analyses.js';
import { cea } from '../analysis/cea.js';
import {
  readChartTheme, buildTraceSpec, buildTornadoSpec, buildCEPlaneSpec, buildCEACSpec, buildEVPISpec,
} from './charts.js';
import {
  formatMoney, format4, formatIcer, statusLabel, formatRunStamp, buildStrategyIndex, parseWtpInput,
  computeWtpMax, psaRunLabel, formatPsaStamp,
} from './results-format.js';

// ================================================================================================
// ---------- DOM micro-helper (mirrors app.js's/inspector.js's own h(); XSS: textContent only —
// children are always Nodes or String()-coerced text nodes, never innerHTML) ----------
// ================================================================================================

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const TOAST_MS = 3500; // matches canvas.js's own transient-toast duration

const TABS = [
  ['cea', 'CEA'],
  ['trace', 'Trace'],
  ['tornado', 'Tornado'],
  ['psa', 'PSA'],
  ['validation', 'Validation'],
];

export function createResults(paneEl, store, { flush = () => {}, plotly, selectOnCanvas = () => {} } = {}) {
  const tabsEl = paneEl.querySelector('#results-tabs');
  const bodyEl = paneEl.querySelector('#results-body');

  // `selectOnCanvas` is stored but unused this task — Task 6 wires the Validation tab's
  // click-through to it. Referencing it in a no-op keeps `eslint no-unused-vars`-style tools quiet
  // without pretending it's already load-bearing.
  void selectOnCanvas;

  let lastRun = null;          // {results, ceaRows, strategies, strategyIndex, wtp, text, ts}
  let lastGateErrors = null;   // Finding[] | null — from the most recent FAILED gate() call
  let lastPsa = null;          // {psaResult, text, ts, elapsedMs, n, seed} | null — see module doc
  let activeTab = 'cea';
  let selectedTraceStrategy = null;
  let tornadoA = null;         // Tornado's own a/b comparator selections — persist across renders
  let tornadoB = null;
  let psaComparator = null;    // PSA's own comparator selection — persists across renders
  let activeChartRenderer = null; // () => void, re-invoked on a theme change; see module doc above
  let toastTimer = null;
  const chartRegistry = new Set(); // live chart-bearing DOM elements; see purgeCharts()/module doc

  // Mirrors analyses.js's own PRIVATE strategyNamesOf(model) exactly (markov -> Object.keys
  // (model.strategies); tree -> its root's children names, NOT model.strategies — see that
  // module's comment). Duplicated here (not exported from analyses.js) so the Tornado tab's two
  // comparator <select>s can list strategy names without widening analyses.js's public contract
  // and without paying for a throwaway run() just to learn names (runBase() would otherwise be the
  // only way to get this list).
  function strategyNamesOf(model) {
    if (model.type === 'markov') return Object.keys(model.strategies);
    if (model.type === 'tree') return (model.tree?.children ?? []).map((c) => c.name);
    return [];
  }

  function isMarkovNow() {
    return store.get().model?.type === 'markov';
  }

  function readTheme() {
    return readChartTheme((prop) => getComputedStyle(document.documentElement).getPropertyValue(prop));
  }

  // ---------- header: tab strip / WTP input / run-stamp / stale banner / toast ----------

  const tabButtons = new Map();
  const tabStripEl = h('div', { class: 'res-tabstrip', role: 'tablist', 'aria-label': 'Results tabs' });
  for (const [id, label] of TABS) {
    const btn = h('button', { type: 'button', class: 'res-tab', role: 'tab', id: `res-tab-${id}` }, label);
    btn.addEventListener('click', () => openTab(id));
    tabButtons.set(id, btn);
    tabStripEl.appendChild(btn);
  }

  function initialWtp() {
    return store.get().model?.settings?.wtp ?? 30000;
  }

  // Review fix (Critical): the WTP `<input>`'s own last-known-GOOD numeric value, independent of
  // `lastRun` — updated by commitWtp/runNow every time the field parses to a real number (never on
  // a rejected empty/non-numeric edit). Kept separate from `lastRun.wtp` specifically so a value the
  // user has already typed-and-blurred survives a SECOND bad edit (e.g. clear the field twice in a
  // row) even before the very first Run happens — falling back to `lastRun?.wtp ?? initialWtp()`
  // instead would forget that first edit and revert all the way to the model's default.
  let lastGoodWtp = initialWtp();

  const wtpInput = h('input', {
    type: 'number', class: 'res-font-data res-wtp-input', min: '0', step: '1000',
    'aria-label': 'Willingness to pay', value: String(initialWtp()),
  });
  const stampEl = h('span', { class: 'res-stamp' });
  const headerEl = h('div', { class: 'res-header' },
    h('label', { class: 'res-wtp' }, 'WTP', wtpInput),
    stampEl,
  );

  const staleEl = h('div', { class: 'res-stale', role: 'status', hidden: '' }, 'Results are stale — Run again.');
  const toastEl = h('div', { class: 'res-toast', role: 'status', hidden: '' });

  tabsEl.replaceChildren(tabStripEl, headerEl, staleEl, toastEl);

  function showToast(message) {
    toastEl.textContent = message;
    toastEl.hidden = false;
    if (toastTimer !== null) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, TOAST_MS);
  }

  function commitWtp() {
    // Review fix (Critical): route through parseWtpInput rather than a raw `Number(...)` +
    // `isFinite` check — `Number('')` is `0`, a perfectly finite number, so the old check let
    // clearing the field silently commit wtp=0 (NMB would jump on the very next keystroke/blur
    // cycle, before the user had typed a replacement digit). A trimmed-empty or non-numeric field
    // now falls back to `lastGoodWtp` instead, and the field itself is written back to whatever
    // value actually took effect — so a rejected edit visibly snaps back rather than leaving stale
    // text in the input next to a silently-unchanged in-memory wtp.
    const n = parseWtpInput(wtpInput.value, lastGoodWtp);
    lastGoodWtp = n;
    wtpInput.value = String(n);
    if (lastRun) recomputeCea(n);
    // Tornado's ΔNMB depends on wtp too (see renderTornadoTab, which reads lastGoodWtp fresh on
    // every call) — unlike CEA, it needs no lastRun at all, so it re-renders here independent of
    // whether a base run has ever happened.
    if (activeTab === 'cea' || activeTab === 'tornado') renderBody();
  }
  wtpInput.addEventListener('change', commitWtp);
  wtpInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commitWtp(); }
  });

  // Re-derives ceaRows from the ALREADY-RUN engine results (lastRun.results) via cea() alone —
  // never re-runs run()/runBase(), per the brief ("re-derive ceaRows via cea() without re-running
  // engines"). Mirrors analyses.js's own runBase() reshaping of {name:{cost,qaly}} verbatim.
  function recomputeCea(wtp) {
    const ceaInput = {};
    for (const name of lastRun.strategies) {
      const r = lastRun.results.strategies[name];
      ceaInput[name] = { cost: r.cost, qaly: r.qaly };
    }
    lastRun.ceaRows = cea(ceaInput, { wtp }).rows;
    lastRun.wtp = wtp;
  }

  function renderStamp() {
    stampEl.textContent = lastRun ? formatRunStamp(lastRun.strategies, new Date(lastRun.ts)) : '';
  }

  function renderStaleBanner() {
    // Task 5: PSA carries its OWN snapshot (lastPsa.text), independent of lastRun.text — a global
    // Run does not touch/clear it (see module doc). The one shared banner fires if EITHER snapshot
    // is out of sync with the live text, so a stale PSA is flagged even while lastRun itself is
    // fresh (run PSA, edit the model, press Run again — CEA's staleness clears, PSA's does not).
    const text = store.get().text;
    const stale = (!!lastRun && text !== lastRun.text) || (!!lastPsa && text !== lastPsa.text);
    staleEl.hidden = !stale;
  }

  function renderTabStrip() {
    const markov = isMarkovNow();
    for (const [id, btn] of tabButtons) {
      if (id === 'trace') btn.hidden = !markov;
      btn.setAttribute('aria-selected', String(activeTab === id));
    }
  }

  // ---------- tab body ----------

  function renderPlaceholder(text) {
    bodyEl.appendChild(h('p', { class: 'res-empty' }, text));
  }

  function renderCeaTab() {
    if (!lastRun) { renderPlaceholder('Run the model to see cost-effectiveness results.'); return; }

    const table = h('table', { class: 'res-table' });
    table.appendChild(h('thead', {}, h('tr', {},
      h('th', {}, 'Strategy'), h('th', { class: 'res-num' }, 'Cost'), h('th', { class: 'res-num' }, 'ΔCost'),
      h('th', { class: 'res-num' }, 'QALY'), h('th', { class: 'res-num' }, 'ΔQALY'),
      h('th', { class: 'res-num' }, 'ICER'), h('th', { class: 'res-num' }, 'NMB'), h('th', {}, 'Status'),
    )));

    const tbody = h('tbody');
    for (const row of lastRun.ceaRows) {
      const muted = row.status === 'dominated' || row.status === 'extended';
      const tr = h('tr', { class: muted ? 'res-row-muted' : undefined });
      tr.append(
        h('td', {}, row.strategy),
        h('td', { class: 'res-num' }, formatMoney(row.cost)),
        h('td', { class: 'res-num' }, row.dcost === null ? '—' : formatMoney(row.dcost)),
        h('td', { class: 'res-num' }, format4(row.qaly)),
        h('td', { class: 'res-num' }, row.dqaly === null ? '—' : format4(row.dqaly)),
        h('td', { class: 'res-num' }, formatIcer(row.icer)),
        h('td', { class: 'res-num' }, row.nmb === undefined ? '—' : formatMoney(row.nmb)),
      );
      const statusTd = h('td', {});
      const label = statusLabel(row.status);
      if (label) statusTd.appendChild(h('span', { class: 'res-chip' }, label));
      tr.appendChild(statusTd);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    bodyEl.append(table, h('p', { class: 'res-foot' },
      `NMB uses a willingness-to-pay of ${formatMoney(lastRun.wtp)} per QALY.`));
  }

  function buildTraceDetails({ states, series, cycles }) {
    const det = h('details', { class: 'res-details' });
    det.appendChild(h('summary', {}, 'Show data'));
    const table = h('table', { class: 'res-table' });
    table.appendChild(h('thead', {}, h('tr', {},
      h('th', {}, 'Cycle'), ...states.map((s) => h('th', { class: 'res-num' }, s)),
    )));
    const tbody = h('tbody');
    for (let i = 0; i < cycles; i++) {
      const tr = h('tr', {}, h('td', {}, String(i + 1)));
      for (const s of states) tr.appendChild(h('td', { class: 'res-num' }, format4(series[s][i])));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    det.appendChild(table);
    return det;
  }

  function renderTraceTab() {
    if (!lastRun) { renderPlaceholder('Run the model to see the cohort trace.'); return; }
    const { strategies } = lastRun;
    if (!strategies.includes(selectedTraceStrategy)) selectedTraceStrategy = strategies[0] ?? null;
    if (!selectedTraceStrategy) { renderPlaceholder('No strategies to trace.'); return; }

    const wrap = h('div', { class: 'res-trace' });

    const select = h('select', { 'aria-label': 'Strategy' },
      ...strategies.map((s) => h('option', { value: s, selected: s === selectedTraceStrategy ? '' : undefined }, s)));
    select.addEventListener('change', () => {
      selectedTraceStrategy = select.value;
      renderBody(); // simplest correct rebuild: fresh chart + "Show data" table for the new strategy
    });
    wrap.appendChild(h('div', { class: 'res-trace-select' }, h('label', {}, 'Strategy ', select)));

    const traceData = traceFor(lastRun.results, selectedTraceStrategy);
    const chartEl = h('div', { class: 'res-chart' });
    wrap.appendChild(chartEl);

    const spec = buildTraceSpec(traceData, readTheme());
    mountChart(chartEl, spec);
    activeChartRenderer = () => {
      const s2 = buildTraceSpec(traceData, readTheme());
      // In-place update of the SAME div — not a re-mount, so no chartRegistry.add (would double-add
      // the same element; purgeCharts() would then try to purge an already-purged div on the second
      // pass, harmless since it's try/catch'd, but pointless). Same race-swallow as mountChart below:
      // this render can itself get purged (a tab switch fired right after a theme change) before
      // Plotly's own promise for it settles.
      plotly.react(chartEl, s2.data, s2.layout, s2.config).catch(logChartRejection);
    };

    wrap.appendChild(buildTraceDetails(traceData));
    bodyEl.appendChild(wrap);
  }

  // ---------- Tornado tab (Task 5) — see module doc for the "live model, no snapshot" design ----------

  function buildTornadoDetails(bars) {
    const det = h('details', { class: 'res-details' });
    det.appendChild(h('summary', {}, 'Show data'));
    const table = h('table', { class: 'res-table' });
    table.appendChild(h('thead', {}, h('tr', {},
      h('th', {}, 'Parameter'), h('th', { class: 'res-num' }, 'Low'), h('th', { class: 'res-num' }, 'High'),
      h('th', { class: 'res-num' }, 'ΔNMB @ low'), h('th', { class: 'res-num' }, 'ΔNMB @ high'),
    )));
    const tbody = h('tbody');
    for (const b of bars) {
      tbody.appendChild(h('tr', {},
        h('td', {}, b.param),
        h('td', { class: 'res-num' }, format4(b.low)),
        h('td', { class: 'res-num' }, format4(b.high)),
        h('td', { class: 'res-num' }, formatMoney(b.lowValue)),
        h('td', { class: 'res-num' }, formatMoney(b.highValue)),
      ));
    }
    table.appendChild(tbody);
    det.appendChild(table);
    return det;
  }

  // Controller amendment (Task 5 review): NEVER render a tornado with tornadoA === tornadoB — an
  // identical comparator pair makes every bar's ΔNMB exactly 0 (deltaNMB(a,a) cancels), a
  // misleading all-zero chart rather than a genuine "no difference" finding. This can happen two
  // ways: (1) manually — both <select>s list the same unfiltered strategy names, so nothing stops
  // a user picking the same one twice (the change handlers below fix this case via a swap); (2)
  // automatically — tornadoA/tornadoB are clamped to the live strategy list INDEPENDENTLY on every
  // render (each just needs to be `.includes()`d), so a model swap can land both on the same name
  // even though neither line individually did anything wrong. Called on every render as the single
  // place that enforces the invariant regardless of how it could have been broken.
  // `availability(model).tornado.ok` (checked by the caller before this ever runs) already
  // guarantees `strategies.length >= 2`, so `strategies.find(s => s !== tornadoA)` always succeeds
  // in practice — the `?? null` fallback is defensive only, never actually reached.
  function resolveTornadoAB(strategies) {
    if (!strategies.includes(tornadoA)) tornadoA = strategies[0] ?? null;
    if (!strategies.includes(tornadoB)) tornadoB = strategies.find((s) => s !== tornadoA) ?? strategies[0] ?? null;
    if (tornadoA !== null && tornadoA === tornadoB) {
      tornadoB = strategies.find((s) => s !== tornadoA) ?? null;
    }
  }

  function renderTornadoTab() {
    const { model, parseError } = store.get();
    if (parseError || !model) { renderPlaceholder('Fix the YAML error to see the Tornado analysis.'); return; }

    const avail = availability(model);
    if (!avail.tornado.ok) {
      // Verbatim per the task brief — the reviewer checks this exact wording.
      renderPlaceholder('needs ≥ 2 strategies and ≥ 1 parameter with low and high — add bounds in Parameters');
      return;
    }

    let strategies, bars;
    try {
      strategies = strategyNamesOf(model);
      resolveTornadoAB(strategies);
      bars = runTornado(model, { a: tornadoA, b: tornadoB, wtp: lastGoodWtp });
    } catch {
      // availability() only checks structural preconditions (>=2 strategies, >=1 bounded param) —
      // it does NOT run check()/gate(), so a model that's structurally eligible but otherwise
      // broken (e.g. a markov row-sum error) can still throw inside run(). Caught here rather than
      // pre-guarded with a separate gate() call, since this try/catch is needed regardless (gate()
      // does not cover every possible run()-time failure either).
      renderPlaceholder('Could not compute the Tornado analysis — check the model in Validation.');
      return;
    }

    const wrap = h('div', { class: 'res-tornado' });

    const selA = h('select', { class: 'res-select', 'aria-label': 'Comparator A' },
      ...strategies.map((s) => h('option', { value: s, selected: s === tornadoA ? '' : undefined }, s)));
    const selB = h('select', { class: 'res-select', 'aria-label': 'Comparator B' },
      ...strategies.map((s) => h('option', { value: s, selected: s === tornadoB ? '' : undefined }, s)));
    // Controller amendment: picking a value that collides with the OTHER select swaps them
    // (classic exchange) rather than leaving both selects on the same name — e.g. A=mono/B=combo,
    // user picks "combo" in A -> B becomes the old A value ("mono"), A becomes "combo": both
    // strategies stay represented, just on opposite sides. `resolveTornadoAB` (called from
    // renderTornadoTab on every render, including the one this triggers) is a second, independent
    // safety net for the case NEITHER handler runs (e.g. a model swap resets both selections).
    selA.addEventListener('change', () => {
      const next = selA.value;
      if (next === tornadoB) tornadoB = tornadoA;
      tornadoA = next;
      renderBody();
    });
    selB.addEventListener('change', () => {
      const next = selB.value;
      if (next === tornadoA) tornadoA = tornadoB;
      tornadoB = next;
      renderBody();
    });
    wrap.appendChild(h('div', { class: 'res-tornado-selects' },
      h('label', {}, 'A ', selA), h('label', {}, 'B ', selB)));

    const chartEl = h('div', { class: 'res-chart' });
    wrap.appendChild(chartEl);

    const spec = buildTornadoSpec(bars, readTheme());
    mountChart(chartEl, spec);
    activeChartRenderer = () => {
      const s2 = buildTornadoSpec(bars, readTheme());
      plotly.react(chartEl, s2.data, s2.layout, s2.config).catch(logChartRejection);
    };

    wrap.appendChild(buildTornadoDetails(bars));
    bodyEl.appendChild(wrap);
  }

  // ---------- PSA tab (Task 5) — see module doc for the lastPsa/staleness design ----------

  function buildPsaPlaneDetails(strategyNames, plane) {
    const det = h('details', { class: 'res-details' });
    det.appendChild(h('summary', {}, 'Show data'));
    const table = h('table', { class: 'res-table' });
    const headCells = [h('th', {}, '#')];
    for (const s of strategyNames) headCells.push(h('th', { class: 'res-num' }, `${s} ΔCost`), h('th', { class: 'res-num' }, `${s} ΔQALY`));
    table.appendChild(h('thead', {}, h('tr', {}, ...headCells)));
    const tbody = h('tbody');
    const n = strategyNames.length > 0 ? plane[strategyNames[0]].length : 0;
    for (let i = 0; i < n; i++) {
      const cells = [h('td', {}, String(i + 1))];
      for (const s of strategyNames) {
        const p = plane[s][i];
        cells.push(h('td', { class: 'res-num' }, formatMoney(p.dcost)), h('td', { class: 'res-num' }, format4(p.dqaly)));
      }
      tbody.appendChild(h('tr', {}, ...cells));
    }
    table.appendChild(tbody);
    det.appendChild(table);
    return det;
  }

  function buildCeacDetails(wtps, curves, strategyNames) {
    const det = h('details', { class: 'res-details' });
    det.appendChild(h('summary', {}, 'Show data'));
    const table = h('table', { class: 'res-table' });
    table.appendChild(h('thead', {}, h('tr', {},
      h('th', {}, 'WTP'), ...strategyNames.map((s) => h('th', { class: 'res-num' }, s)),
    )));
    const tbody = h('tbody');
    wtps.forEach((wtp, i) => {
      const cells = [h('td', { class: 'res-num' }, formatMoney(wtp))];
      for (const s of strategyNames) cells.push(h('td', { class: 'res-num' }, format4(curves[s][i])));
      tbody.appendChild(h('tr', {}, ...cells));
    });
    table.appendChild(tbody);
    det.appendChild(table);
    return det;
  }

  function buildEvpiDetails(wtps, evpiValues) {
    const det = h('details', { class: 'res-details' });
    det.appendChild(h('summary', {}, 'Show data'));
    const table = h('table', { class: 'res-table' });
    table.appendChild(h('thead', {}, h('tr', {}, h('th', {}, 'WTP'), h('th', { class: 'res-num' }, 'EVPI'))));
    const tbody = h('tbody');
    wtps.forEach((wtp, i) => {
      tbody.appendChild(h('tr', {},
        h('td', { class: 'res-num' }, formatMoney(wtp)), h('td', { class: 'res-num' }, formatMoney(evpiValues[i]))));
    });
    table.appendChild(tbody);
    det.appendChild(table);
    return det;
  }

  // Whether PSA could be (re-)run RIGHT NOW off the live model — gates the "Run PSA" button
  // (disabled, never hidden, when false) independent of whether lastPsa already holds an older
  // result. Wrapped in try/catch: availability()/gate() are both safe on a well-formed Model, but
  // this is called from render paths that must never throw.
  function psaLiveOk(model, parseError) {
    if (parseError || !model) return false;
    try { return gate(model).ok && availability(model).psa.ok; } catch { return false; }
  }

  function buildPsaHeader(model, liveOk) {
    const settings = model?.settings?.psa ?? { n: lastPsa?.n ?? 1000, seed: lastPsa?.seed ?? 1 };
    const { n, seed } = settings;
    const btn = h('button', { type: 'button', class: 'res-psa-run' }, psaRunLabel(n));
    btn.disabled = !liveOk;
    btn.addEventListener('click', runPsaNow);
    const children = [h('span', { class: 'res-psa-meta res-font-data' }, `n=${n}, seed=${seed}`), btn];
    if (lastPsa) {
      children.push(h('span', { class: 'res-stamp res-psa-stamp' }, formatPsaStamp(lastPsa.n, lastPsa.elapsedMs, new Date(lastPsa.ts))));
    }
    return h('div', { class: 'res-psa-head' }, ...children);
  }

  function renderPsaTab() {
    const { model, parseError } = store.get();
    const liveOk = psaLiveOk(model, parseError);

    if (!lastPsa) {
      if (parseError || !model) { renderPlaceholder('Fix the YAML error to see the PSA analysis.'); return; }
      if (!gate(model).ok) { renderPlaceholder('Model has errors — fix them in Validation before running PSA.'); return; }
      if (!availability(model).psa.ok) {
        // Verbatim per the task brief — the reviewer checks this exact wording.
        renderPlaceholder('no parameter has a dist — add distributions in Parameters');
        return;
      }
      bodyEl.appendChild(buildPsaHeader(model, liveOk));
      renderPlaceholder('Run PSA to see the cost-effectiveness plane, CEAC, and EVPI.');
      return;
    }

    bodyEl.appendChild(buildPsaHeader(model, liveOk));

    const { psaResult } = lastPsa;
    const strategies = psaResult.strategies;
    if (!strategies.includes(psaComparator)) psaComparator = strategies[0] ?? null;

    const wrap = h('div', { class: 'res-psa' });
    const sel = h('select', { class: 'res-select', 'aria-label': 'Comparator' },
      ...strategies.map((s) => h('option', { value: s, selected: s === psaComparator ? '' : undefined }, s)));
    sel.addEventListener('change', () => { psaComparator = sel.value; renderBody(); });
    wrap.appendChild(h('div', { class: 'res-psa-comparator' }, h('label', {}, 'Comparator ', sel)));

    const strategyIndex = buildStrategyIndex(strategies);
    const wtpMax = computeWtpMax(model?.settings?.wtp ?? null, lastGoodWtp);
    // Pure re-derivation from the CACHED psaResult — no engine re-run on a comparator switch (see
    // module doc). The comparator name rides on the plane object per T3's binding convention (the
    // 3-arg buildCEPlaneSpec contract has no room for a 4th argument).
    const derived = psaDerived(psaResult, { comparator: psaComparator, wtpMax });
    const plane = { ...derived.plane, comparator: psaComparator };

    const planeEl = h('div', { class: 'res-psa-chart' });
    const ceacEl = h('div', { class: 'res-psa-chart' });
    const evpiEl = h('div', { class: 'res-psa-chart' });

    function buildSpecs(theme) {
      return {
        plane: buildCEPlaneSpec(plane, theme, strategyIndex),
        ceac: buildCEACSpec({ wtps: derived.wtps, curves: derived.ceacCurves }, theme, strategyIndex),
        evpi: buildEVPISpec({ wtps: derived.wtps, evpi: derived.evpi }, theme),
      };
    }

    const s0 = buildSpecs(readTheme());
    mountChart(planeEl, s0.plane);
    mountChart(ceacEl, s0.ceac);
    mountChart(evpiEl, s0.evpi);
    // Task 5 requirement: a theme flip must re-render ALL THREE PSA charts, not just the last
    // mounted one — unlike every other chart-bearing tab (always exactly one chart), this closure
    // re-renders the full set in place.
    activeChartRenderer = () => {
      const s2 = buildSpecs(readTheme());
      plotly.react(planeEl, s2.plane.data, s2.plane.layout, s2.plane.config).catch(logChartRejection);
      plotly.react(ceacEl, s2.ceac.data, s2.ceac.layout, s2.ceac.config).catch(logChartRejection);
      plotly.react(evpiEl, s2.evpi.data, s2.evpi.layout, s2.evpi.config).catch(logChartRejection);
    };

    wrap.appendChild(h('div', { class: 'res-psa-section' },
      h('h3', {}, 'Cost-effectiveness plane'), planeEl,
      buildPsaPlaneDetails(Object.keys(derived.plane), derived.plane)));
    wrap.appendChild(h('div', { class: 'res-psa-section' },
      h('h3', {}, 'CEAC'), ceacEl, buildCeacDetails(derived.wtps, derived.ceacCurves, strategies)));
    wrap.appendChild(h('div', { class: 'res-psa-section' },
      h('h3', {}, 'EVPI'), evpiEl, buildEvpiDetails(derived.wtps, derived.evpi)));

    bodyEl.appendChild(wrap);
  }

  // Explicit, synchronous PSA run — the ONE potentially-slow analysis in this app (see module
  // doc). Mirrors app.js's own runModel() busy-state pattern: set disabled/aria-busy on the
  // button, defer the actual (synchronous, blocking) computation one tick via setTimeout(0) so
  // that busy state actually PAINTS before the main thread blocks. On success, busy state is
  // cleared implicitly — renderBody() replaces the whole tab body (fresh, un-busy button) rather
  // than un-disabling this same node; on a thrown error it's cleared explicitly on THIS button
  // before returning, since renderBody() is never reached on that path.
  function runPsaNow() {
    flush();
    const { model, parseError } = store.get();
    if (parseError || !model) { showToast('Fix the YAML error before running PSA.'); return; }
    if (!gate(model).ok) { showToast('Model has errors — fix them to run PSA.'); return; }
    if (!availability(model).psa.ok) return; // button is disabled in this case; defensive no-op

    const btn = bodyEl.querySelector('.res-psa-run');
    if (btn) { btn.disabled = true; btn.setAttribute('aria-busy', 'true'); }

    setTimeout(() => {
      const t0 = performance.now();
      let psaResult;
      try {
        psaResult = runPsa(model);
      } catch {
        if (btn) { btn.disabled = false; btn.removeAttribute('aria-busy'); }
        showToast('PSA failed to run — check the model.');
        return;
      }
      const elapsedMs = performance.now() - t0;
      lastPsa = {
        psaResult, text: store.get().text, ts: Date.now(), elapsedMs,
        n: model.settings.psa.n, seed: model.settings.psa.seed,
      };
      if (!psaResult.strategies.includes(psaComparator)) psaComparator = psaResult.strategies[0] ?? null;
      renderStaleBanner();
      renderBody();
    }, 0);
  }

  function renderValidationTab() {
    if (lastGateErrors && lastGateErrors.length) {
      const n = lastGateErrors.length;
      bodyEl.appendChild(h('p', { class: 'res-empty' }, `Model has ${n} error${n === 1 ? '' : 's'}.`));
    }
    bodyEl.appendChild(h('p', { class: 'res-empty' }, 'Validation details — content in Task 6.'));
  }

  const TAB_RENDERERS = {
    cea: renderCeaTab,
    trace: renderTraceTab,
    tornado: renderTornadoTab,
    psa: renderPsaTab,
    validation: renderValidationTab,
  };

  // Controller amendment (Task 5 review): the ONE handler every `plotly.react(...).catch(...)`
  // site in this module must pass — logs the (expected, benign — see below) purge-raced-the-render
  // rejection via `console.debug` instead of silently swallowing it, so a genuinely broken chart
  // spec/runtime error occurring inside Plotly's own async pipeline still surfaces somewhere.
  function logChartRejection(err) {
    console.debug('plotly render rejected', err);
  }

  // Review fix (Important): the two-step "mount a NEW chart div" pattern every chart-bearing tab
  // (Trace here; T5's Tornado/PSA/CE-plane/CEAC/EVPI) must use — `plotly.react(el, ...)` to draw it,
  // `chartRegistry.add(el)` so `purgeCharts()` can tear it down later. The `.catch(logChartRejection)`
  // on the react() call is NOT optional: `plotly.react`/`newPlot` return a promise that can still be
  // settling (Plotly's own internal layout/render pipeline) when a fast subsequent tab/strategy
  // switch calls `purgeCharts()` on this same div before that promise resolves — Plotly's internal
  // continuation then throws trying to read state off the now-purged div, surfacing as an "Uncaught
  // (in promise)" console error despite the purge itself being perfectly correct (verified live:
  // 10x rapid tab/strategy switching threw this on every cycle before the `.catch` was added, zero
  // after). An in-place re-render of an ALREADY-mounted div (see `activeChartRenderer`'s own
  // `plotly.react` call in `renderTraceTab` below) needs the same `.catch` but must NOT call
  // `mountChart`/re-add the element — it isn't a new chart.
  function mountChart(el, spec) {
    plotly.react(el, spec.data, spec.layout, spec.config).catch(logChartRejection);
    chartRegistry.add(el);
  }

  // Review fix (Important): purges every registered chart div BEFORE it's discarded. Wrapped
  // per-element in try/catch so one already-detached/broken chart can't stop the rest from being
  // purged (and can't throw out of renderBody(), which callers rely on being synchronous/non-
  // throwing). Always clears the registry after, regardless of any individual purge failure.
  function purgeCharts() {
    for (const el of chartRegistry) {
      try { plotly.purge(el); } catch { /* best-effort cleanup; a failed purge must not block rendering */ }
    }
    chartRegistry.clear();
  }

  function renderBody() {
    activeChartRenderer = null; // cleared unless the tab we're about to render sets it (Trace only, this task)
    purgeCharts(); // review fix (Important): every discarded chart div gets Plotly.purge'd first — see module doc
    bodyEl.replaceChildren();
    TAB_RENDERERS[activeTab]();
  }

  // ---------- public API ----------

  function openTab(t) {
    if (!tabButtons.has(t)) return;
    if (t === 'trace' && !isMarkovNow()) return; // a hidden tab can't be opened programmatically either
    activeTab = t;
    render();
  }

  function render() {
    renderTabStrip();
    renderStamp();
    renderStaleBanner();
    renderBody();
  }

  function runNow() {
    flush();
    const { model, text, parseError } = store.get();
    // Review fix (Critical): guard on `parseError`, never `!model` — store.js keeps the LAST-GOOD
    // model in place while a live YAML edit fails to parse, so `model` stays non-null even though
    // the YAML pane is showing broken, unparsed text. Checking `!model` alone let Run silently
    // compute fresh results off that stale model, AND captured the broken live text as
    // `lastRun.text` — which meant the stale-banner's text-inequality check could never fire
    // afterwards either (the "current" text WAS what got snapshotted). `parseError` is set in
    // exactly this case (and also covers the `model === null` boot-time case, a subset of it), so
    // checking it alone is both necessary and sufficient; `lastRun` is left completely untouched.
    if (parseError) { showToast('Fix the YAML error before running.'); return; }

    const g = gate(model);
    if (!g.ok) {
      lastGateErrors = g.errors;
      activeTab = 'validation';
      showToast(`Model has ${g.errors.length} error${g.errors.length === 1 ? '' : 's'} — fix them to run`);
      render();
      return;
    }

    lastGateErrors = null;
    // Review fix (Critical): same parseWtpInput guard as commitWtp — a cleared/garbage WTP field
    // must fall back to lastGoodWtp, never silently run with wtp=0. Also normalizes the field back
    // to whatever value actually took effect, same as commitWtp.
    const wtp = parseWtpInput(wtpInput.value, lastGoodWtp);
    lastGoodWtp = wtp;
    wtpInput.value = String(wtp);
    const { results, ceaRows, strategies } = runBase(model, { wtp });
    lastRun = { results, ceaRows, strategies, strategyIndex: buildStrategyIndex(strategies), wtp, text, ts: Date.now() };
    selectedTraceStrategy = strategies[0] ?? null;
    activeTab = 'cea';
    render();
  }

  // ---------- store subscription: staleness + trace-tab availability only (see module doc) ----------

  store.subscribe(() => {
    renderStaleBanner();
    renderTabStrip();
    if (activeTab === 'trace' && !isMarkovNow()) {
      activeTab = 'cea';
      renderTabStrip();
      renderBody();
    }
  });

  // ---------- theme re-render (extends to T5's chart tabs; see module doc) ----------

  const darkMq = window.matchMedia('(prefers-color-scheme: dark)');
  darkMq.addEventListener('change', () => activeChartRenderer?.());
  const themeObserver = new MutationObserver(() => activeChartRenderer?.());
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  render();

  return { render, runBase: runNow, openTab };
}
