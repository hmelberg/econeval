// js/ui/results-format.js — pure formatting/derivation helpers for results.js. No DOM, no store,
// no Plotly: kept separate so number/run-stamp formatting is unit-testable
// (test/results-format.test.js), per constraints.md's pure-logic-tested/DOM-thin split. results.js
// itself stays DOM-thin and is covered by recorded browser verification instead.

const MONEY_FMT = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

// 0-decimal, thousands-separated — the CEA table's cost/ΔCost/ICER/NMB columns. Negative values
// keep Intl's default leading '-' sign, never accounting-style parentheses.
export function formatMoney(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) return '';
  return MONEY_FMT.format(n);
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
