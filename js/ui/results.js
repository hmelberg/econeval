// Results drawer: CEA + Trace tabs (Tornado/PSA content lands Task 5; Validation content lands
// Task 6 — both are wired here only as empty-state placeholders so the tab strip is complete now).
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
// still needs its own `.catch(() => {})` for the same race, see `renderTraceTab`. T5's chart-bearing
// tabs must follow the same "new chart -> mountChart(el, spec); in-place re-render -> bare
// `plotly.react(...).catch(() => {})`" split — `purgeCharts()`/`renderBody()` need no per-tab changes.
//
// Extending the tab registry (T5/T6): `TABS` (id + label, tab STRIP only) and `TAB_RENDERERS` (id
// -> () => void, called with bodyEl already cleared) are the two lists to touch. Add a tab: push
// onto TABS (grows the strip) and add its renderer to TAB_RENDERERS (replace the tornado/psa/
// validation placeholder entries here with real content) — `openTab`/`renderTabStrip`/`renderBody`
// need no changes for a same-shape addition. `renderTabStrip`'s per-button `hidden` rule is
// Trace-specific (markov-only) — a future conditionally-hidden tab would add its own branch there
// the same way.

import { gate, runBase, traceFor } from './analyses.js';
import { cea } from '../analysis/cea.js';
import { readChartTheme, buildTraceSpec } from './charts.js';
import {
  formatMoney, format4, formatIcer, statusLabel, formatRunStamp, buildStrategyIndex, parseWtpInput,
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
  let activeTab = 'cea';
  let selectedTraceStrategy = null;
  let activeChartRenderer = null; // () => void, re-invoked on a theme change; see module doc above
  let toastTimer = null;
  const chartRegistry = new Set(); // live chart-bearing DOM elements; see purgeCharts()/module doc

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
    if (!lastRun) return; // nothing to re-derive yet — the value is simply remembered for the next run
    recomputeCea(n);
    if (activeTab === 'cea') renderBody();
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
    const stale = !!lastRun && store.get().text !== lastRun.text;
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
      plotly.react(chartEl, s2.data, s2.layout, s2.config).catch(() => {});
    };

    wrap.appendChild(buildTraceDetails(traceData));
    bodyEl.appendChild(wrap);
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
    tornado: () => renderPlaceholder('Tornado sensitivity — content in Task 5.'),
    psa: () => renderPlaceholder('Probabilistic sensitivity analysis — content in Task 5.'),
    validation: renderValidationTab,
  };

  // Review fix (Important): the two-step "mount a NEW chart div" pattern every chart-bearing tab
  // (Trace here; T5's Tornado/PSA/CE-plane/CEAC/EVPI) must use — `plotly.react(el, ...)` to draw it,
  // `chartRegistry.add(el)` so `purgeCharts()` can tear it down later. The `.catch(() => {})` on the
  // react() call is NOT optional: `plotly.react`/`newPlot` return a promise that can still be
  // settling (Plotly's own internal layout/render pipeline) when a fast subsequent tab/strategy
  // switch calls `purgeCharts()` on this same div before that promise resolves — Plotly's internal
  // continuation then throws trying to read state off the now-purged div, surfacing as an "Uncaught
  // (in promise)" console error despite the purge itself being perfectly correct (verified live:
  // 10x rapid tab/strategy switching threw this on every cycle before the `.catch` was added, zero
  // after). An in-place re-render of an ALREADY-mounted div (see `activeChartRenderer`'s own
  // `plotly.react` call in `renderTraceTab` below) needs the same `.catch` but must NOT call
  // `mountChart`/re-add the element — it isn't a new chart.
  function mountChart(el, spec) {
    plotly.react(el, spec.data, spec.layout, spec.config).catch(() => {});
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
