// js/ui/results-format.js — pure formatting/derivation helpers for results.js. No DOM, no store,
// no Plotly: kept separate so number/run-stamp formatting is unit-testable
// (test/results-format.test.js), per constraints.md's pure-logic-tested/DOM-thin split. results.js
// itself stays DOM-thin and is covered by recorded browser verification instead.

const MONEY_FMT = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

// 0-decimal, thousands-separated — the CEA table's cost/ΔCost/ICER/NMB columns. Negative values
// keep Intl's default leading '-' sign, never accounting-style parentheses. Triage fix (c): a
// small negative value that ROUNDS to zero (e.g. -0.4) formats as the literal string "-0" —
// technically correct per IEEE-754/Intl, but reads as a bug ("negative zero dollars") in a money
// column; normalized to "0" here, the one string this formatter otherwise never produces on its
// own (real negative results always round to a nonzero magnitude, or they wouldn't be negative).
export function formatMoney(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) return '';
  const s = MONEY_FMT.format(n);
  return s === '-0' ? '0' : s;
}

// 4-decimal fixed — the CEA table's QALY/ΔQALY columns, and the Trace tab's "Show data" occupancy
// shares (both are naturally small values where 4 decimals reads as precise, not noisy).
export function format4(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) return '';
  return n.toFixed(4);
}

// ICER is `null` for the cheapest frontier row and for every dominated/extended row (cea.js's own
// contract) — rendered as an em dash so "no meaningful ICER" reads distinctly from a blank/zero
// cell, never silently as "0".
export function formatIcer(icer) {
  return icer === null || icer === undefined ? '—' : formatMoney(icer);
}

const STATUS_LABELS = { dominated: 'Dominated', extended: 'Extended dominated' };

// cea.js's row.status: '' (frontier, no chip) | 'dominated' | 'extended' -> a short chip label, or
// null when the row needs no chip at all.
export function statusLabel(status) {
  return STATUS_LABELS[status] ?? null;
}

// "Run · mono, combo · 14:32" — strategies in the declaration order runBase() returned them
// (never re-sorted), HH:MM 24-hour local time, zero-padded. `date` is an injected Date (real
// callers pass `new Date(ts)`; tests pass a fixed instance), keeping this pure/deterministic.
export function formatRunStamp(strategies, date) {
  const hh = String(date.getHours()).padStart(2, '0');
  const mm = String(date.getMinutes()).padStart(2, '0');
  return `Run · ${strategies.join(', ')} · ${hh}:${mm}`;
}

// Map name -> palette slot (0-based), in the given declaration-order array — the exact
// strategyIndex shape js/ui/charts.js's builders consume (readChartTheme's colors are indexed the
// same way). Stable across a psaDerived().plane subset, since it's keyed by name, not position.
export function buildStrategyIndex(strategies) {
  return new Map(strategies.map((name, i) => [name, i]));
}

// Review fix (Critical): the WTP `<input>`'s raw string value, parsed against a known-good
// fallback. `Number('')` and `Number('   ')` are both `0` — and `0` is a perfectly finite, legal
// WTP — so `Number.isFinite(Number(raw))` alone silently commits wtp=0 the instant a user clears
// the field (before they've typed a replacement digit). Trimmed-empty and non-numeric text both
// mean "the field doesn't hold a usable value right now" and fall back to `lastGood`; only an
// actually-parseable number is ever returned as-is (including a real, explicit `0` typed by the
// user — `'0'` is NOT empty). Callers (results.js's commitWtp/runNow) are expected to also write
// the returned value back into the input, so a rejected edit visibly reverts rather than leaving
// stale/incorrect text sitting in the field next to a silently-different in-memory wtp.
// Triage fix (d): a negative WTP is never a legal willingness-to-pay (NMB trades QALYs against
// cost at a non-negative rate), so it is rejected the same way empty/non-numeric text is —
// falling back to `lastGood` rather than silently accepting e.g. "-5000" and inverting the CEA
// frontier's NMB ranking.
export function parseWtpInput(raw, lastGood) {
  const trimmed = String(raw ?? '').trim();
  if (trimmed === '') return lastGood;
  const n = Number(trimmed);
  return (Number.isFinite(n) && n >= 0) ? n : lastGood;
}

// Task 5 (Tornado + PSA tabs): wtpMax fed into analyses.js's psaDerived() for the PSA tab's
// CEAC/EVPI x-axis range. Controller ruling: 2.5x whichever WTP is authoritative right now — the
// model's own `settings.wtp` (its author's stated willingness-to-pay) when set, else the CEA
// header's current WTP value (what the user is actually looking at), else a flat 100000 when
// neither is available (deliberately the same number psaDerived() itself falls back to, so the
// two stay in sync even though this function computes the value explicitly rather than relying on
// that default). This is results.js's job, not analyses.js's — psaDerived is pure over an
// already-computed psaResult and has no model in scope to read settings.wtp from.
export function computeWtpMax(settingsWtp, headerWtp) {
  const base =
    settingsWtp !== null && settingsWtp !== undefined ? settingsWtp
    : headerWtp !== null && headerWtp !== undefined ? headerWtp
    : null;
  return base !== null ? 2.5 * base : 100000;
}

// "Run PSA (n=1000)" — the PSA tab's own explicit run-button label. PSA is the only potentially
// slow analysis in this app (a full probabilistic sweep, `n` draws), so it never runs implicitly
// off the global Run button; the label always states exactly how many draws a click commits to.
export function psaRunLabel(n) {
  return `Run PSA (n=${n})`;
}

// "PSA · n=1000 · 842ms · 14:32" — mirrors formatRunStamp's own "Run · ... · HH:MM" shape, plus
// the draw count and wall-clock timing of the run that produced the currently-shown PSA charts
// (PSA is explicit/synchronous — see module doc in results.js — so this is worth surfacing).
// `elapsedMs` is rounded to the nearest whole millisecond for display; `date` is an injected Date
// (real callers pass `new Date(ts)`; tests pass a fixed instance), keeping this pure/deterministic.
export function formatPsaStamp(n, elapsedMs, date) {
  const hh = String(date.getHours()).padStart(2, '0');
  const mm = String(date.getMinutes()).padStart(2, '0');
  return `PSA · n=${n} · ${Math.round(elapsedMs)}ms · ${hh}:${mm}`;
}
