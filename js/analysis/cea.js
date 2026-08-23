/**
 * cea(results, {wtp}) -> {rows}
 *
 * Pure incremental cost-effectiveness analysis over already-computed strategy results — no
 * engine dependencies. Takes `{name: {cost, qaly}}` and returns rows sorted by ascending cost
 * (dominated/extended rows stay in the list, in cost order), each tagged with a dominance
 * `status` and incremental cost/QALY/ICER figures computed against the previous *surviving*
 * (frontier) row.
 *
 * Two dominance passes:
 *   1. Strict (simple) dominance: a strategy is dominated if another strategy costs no more
 *      and yields no fewer QALYs, with at least one strict inequality. Checked against the
 *      full original set — this partial order has no cycles, so checking survivors-only would
 *      give the same result.
 *   2. Extended (weak) dominance: on the survivors of (1), in cost order, ICERs (vs. the
 *      previous survivor) must be non-decreasing along the frontier. While some ICER exceeds
 *      the next ICER, the earlier of the two strategies is removed (marked 'extended') and
 *      ICERs are recomputed, until the sequence is non-decreasing.
 *
 * `dcost`/`dqaly`/`icer` are only meaningful for rows left on the frontier (status ''); the
 * cheapest frontier row gets `icer: null` (no cheaper comparator). Dominated and
 * extended-dominated rows get `dcost`/`dqaly`/`icer: null` — they are not frontier points.
 * `nmb = wtp*qaly - cost` is computed for every row (not just frontier ones) when `wtp` is
 * given; omitted entirely otherwise.
 *
 * `icer` is `null` whenever `dqaly === 0` (never `NaN`/`Infinity`) — among survivors this can
 * only happen for an exact cost+qaly duplicate (anything with equal qaly and higher cost is
 * strictly dominated before reaching the ICER stage), and `null` already means "no meaningful
 * increment" everywhere else in this module. Controller ruling 2026-08-23.
 */
function icerOf(cur, prev) {
  const dqaly = cur.qaly - prev.qaly;
  return dqaly === 0 ? null : (cur.cost - prev.cost) / dqaly;
}

export function cea(results, { wtp } = {}) {
  const rows = Object.entries(results).map(([strategy, r]) => ({
    strategy, cost: r.cost, qaly: r.qaly, status: '',
  }));

  rows.sort((a, b) => a.cost - b.cost);

  // Pass 1: strict dominance, checked against the full set.
  for (const s of rows) {
    const dominated = rows.some((o) => o !== s &&
      o.cost <= s.cost && o.qaly >= s.qaly && (o.cost < s.cost || o.qaly > s.qaly));
    if (dominated) s.status = 'dominated';
  }

  // Pass 2: extended dominance over survivors, in cost order — ICERs vs. the previous
  // survivor must be non-decreasing; repeatedly drop the earlier strategy of any violating
  // pair and recompute until the sequence is clean.
  let survivors = rows.filter((s) => s.status === '');
  for (;;) {
    const icers = survivors.map((s, i) => (i === 0 ? null : icerOf(s, survivors[i - 1])));
    let violation = -1;
    for (let i = 1; i < icers.length - 1; i++) {
      if (icers[i] === null || icers[i + 1] === null) continue; // no meaningful comparison
      if (icers[i] > icers[i + 1]) { violation = i; break; }
    }
    if (violation === -1) break;
    survivors[violation].status = 'extended';
    survivors = survivors.filter((s) => s.status === '');
  }

  const survivorSet = new Set(survivors);
  let prev = null;
  for (const s of rows) {
    if (!survivorSet.has(s) || prev === null) {
      s.dcost = null; s.dqaly = null; s.icer = null;
    } else {
      s.dcost = s.cost - prev.cost;
      s.dqaly = s.qaly - prev.qaly;
      s.icer = icerOf(s, prev);
    }
    if (survivorSet.has(s)) prev = s;
  }

  if (wtp !== undefined) {
    for (const s of rows) s.nmb = wtp * s.qaly - s.cost;
  }

  return { rows };
}
