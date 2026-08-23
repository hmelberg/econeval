/**
 * js/ui/charts.js — pure Plotly spec builders. NO DOM, NO Plotly import: every builder returns a
 * plain {data, layout, config} object; results.js (a later task) feeds that straight into
 * `plotly.react(el, data, layout, config)` with an injected `window.Plotly`. Kept pure/DOM-free so
 * every builder is unit-testable with a fake theme object (see test/charts.test.js).
 *
 * readChartTheme(styleGetter) -> {colors:[6], ink, muted, line, surface, fontUI, fontData}
 *   `styleGetter` is an injected `prop => value` function (e.g.
 *   `getComputedStyle(document.documentElement).getPropertyValue`) so this stays DOM-free in
 *   tests. Values are trimmed (custom-property values often carry incidental whitespace).
 *
 * buildTraceSpec(traceData, theme)                        -> {data, layout, config}
 * buildTornadoSpec(bars, theme)                            -> {data, layout, config}
 * buildCEPlaneSpec(plane, theme, strategyIndex)            -> {data, layout, config}
 * buildCEACSpec({wtps, curves}, theme, strategyIndex)      -> {data, layout, config}
 * buildEVPISpec({wtps, evpi}, theme)                       -> {data, layout, config}
 *
 * strategyIndex: Map name -> palette slot (0-5), built by the CALLER from the model's own
 * strategy DECLARATION order (stable across filters/comparator changes — see analyses.js's
 * strategyNamesOf). Builders only ever CONSUME it via strategyIndex.get(name); they never derive
 * a slot from array/object position, which is what keeps a strategy's color pinned to it even
 * when a psaDerived().plane subset drops the comparator (see the "shuffled subset" test). Slot
 * >= 6 (a 7th+ strategy) folds into theme.muted ("muted gray") with a "(Other)" legend-name note,
 * per the chart-rules doc — never a generated hue.
 *
 * buildCEPlaneSpec's comparator name: the module contract pins the signature to exactly
 * (plane, theme, strategyIndex) — no 4th argument — so the comparator name (needed only for the
 * x-axis title, "Incremental QALYs vs <comparator>") travels as an OPTIONAL `plane.comparator`
 * string property, set by the caller alongside the strategy keys psaDerived().plane already
 * produces (results.js has the comparator in scope — it's the same value it passed into
 * psaDerived). Absent -> the title falls back to the comparator-free "Incremental QALYs". This
 * key is skipped when reading strategy series out of `plane` (see `strategyKeysOf`).
 *
 * Chart-rules compliance baked into every builder (never opt-in per chart):
 *   - one axis ever — no builder sets/returns xaxis2 or yaxis2.
 *   - legend iff >= 2 series (buildEVPISpec is always exactly 1 series -> always off; tornado is
 *     always exactly 2 (Low/High) -> always on).
 *   - text (axis titles, legend, hover) rides the single top-level `layout.font`/`hoverlabel`
 *     (ink) — never a series color. Hover backgrounds are pinned to `theme.surface` specifically
 *     because Plotly's own default hoverlabel background is the trace color, which this rules
 *     block forbids for text-bearing chrome.
 *   - series colors always come from `theme.colors`/strategyIndex, never rank/array order.
 *   - lines 2px; CE-plane markers 8px with a 2px `theme.surface` ring.
 *   - grid/zeroline recessive (`theme.line`) everywhere EXCEPT CE-plane, whose quadrant zero-lines
 *     are drawn explicitly via `layout.shapes` in `theme.ink` (still not a series color) with the
 *     native per-axis zeroline turned off so there's exactly one zero-line, not two overlapping.
 *   - paper/plot backgrounds = theme.surface; margins {t:24,r:16,b:44,l:56}; legend, when shown,
 *     horizontal and above the plot; config always {displayModeBar:false, responsive:true}.
 */

export function readChartTheme(styleGetter) {
  const get = (prop) => (styleGetter(prop) ?? '').toString().trim();
  return {
    colors: [1, 2, 3, 4, 5, 6].map((n) => get(`--chart-${n}`)),
    ink: get('--ink'),
    muted: get('--muted'),
    line: get('--line'),
    surface: get('--surface'),
    fontUI: get('--font-ui'),
    fontData: get('--font-data'),
  };
}

const OTHER_SUFFIX = ' (Other)';

// A slot >= theme.colors.length (a 7th+ strategy/state) folds to muted gray, never a generated hue.
function colorForSlot(slot, theme) {
  if (typeof slot === 'number' && slot >= 0 && slot < theme.colors.length) return theme.colors[slot];
  return theme.muted;
}

function labelForSlot(name, slot, theme) {
  const inPalette = typeof slot === 'number' && slot >= 0 && slot < theme.colors.length;
  return inPalette ? name : `${name}${OTHER_SUFFIX}`;
}

function baseConfig() {
  return { displayModeBar: false, responsive: true };
}

// Shared layout chrome: surface backgrounds, ink font, tight margins, hover styled off the series
// palette, and (when shown) a horizontal legend sitting above the plot area.
function baseLayout(theme, { showlegend }) {
  const layout = {
    paper_bgcolor: theme.surface,
    plot_bgcolor: theme.surface,
    font: { color: theme.ink, family: theme.fontUI, size: 12 },
    margin: { t: 24, r: 16, b: 44, l: 56 },
    showlegend,
    hoverlabel: { bgcolor: theme.surface, font: { color: theme.ink, family: theme.fontUI } },
  };
  if (showlegend) {
    layout.legend = { orientation: 'h', x: 0, y: 1.08, xanchor: 'left', yanchor: 'bottom' };
  }
  return layout;
}

// One axis config helper reused by every builder: recessive grid/zeroline (theme.line), values
// rendered in the monospace data font, title only when given. `extra` lets a specific builder
// override a field (CE-plane turns zeroline off; CEAC pins a [0,1] range; tornado's y-axis is
// categorical).
function axis(theme, { title, extra = {} } = {}) {
  const cfg = {
    gridcolor: theme.line,
    zerolinecolor: theme.line,
    zeroline: true,
    linecolor: theme.line,
    tickfont: { family: theme.fontData },
  };
  if (title) cfg.title = { text: title };
  return { ...cfg, ...extra };
}

/**
 * buildTraceSpec({states, series, cycles}, theme) -> {data, layout, config}
 * One 2px line per state; x = 1..cycles (traceFor's series has no cycle-0 entry — see
 * analyses.js). Legend iff >= 2 states. hovermode 'x unified'.
 */
export function buildTraceSpec(traceData, theme) {
  const { states, series, cycles } = traceData;
  const x = Array.from({ length: cycles }, (_, i) => i + 1);
  const showlegend = states.length >= 2;

  const data = states.map((state, i) => ({
    type: 'scatter',
    mode: 'lines',
    name: labelForSlot(state, i, theme),
    x,
    y: series[state],
    line: { width: 2, color: colorForSlot(i, theme) },
  }));

  const layout = {
    ...baseLayout(theme, { showlegend }),
    hovermode: 'x unified',
    xaxis: axis(theme, { title: 'Cycle' }),
    yaxis: axis(theme, { title: 'Share of cohort' }),
  };

  return { data, layout, config: baseConfig() };
}

/**
 * buildTornadoSpec(bars, theme) -> {data, layout, config}
 * bars = [{param, low, high, lowValue, highValue, base}], already sorted by the caller (dsa.js's
 * oneWay, by |range| descending) — order preserved verbatim, never re-sorted here. Two floating
 * horizontal bar traces (Plotly's `base` offset array) spanning base -> lowValue / base ->
 * highValue, fixed hues (Low = colors[1], High = colors[2]); the shared base value (identical on
 * every row — oneWay computes it once) is additionally drawn as a vertical `layout.shapes` line
 * with a muted-text annotation, so the base reads as a single reference line down the chart even
 * though it's also baked into each bar's floating offset.
 */
export function buildTornadoSpec(bars, theme) {
  const base = bars.length > 0 ? bars[0].base : 0;
  const y = bars.map((b) => b.param);
  const offsets = bars.map((b) => b.base);

  const low = {
    type: 'bar',
    orientation: 'h',
    name: 'Low',
    y,
    x: bars.map((b) => b.lowValue - b.base),
    base: offsets,
    marker: { color: theme.colors[1] },
  };
  const high = {
    type: 'bar',
    orientation: 'h',
    name: 'High',
    y,
    x: bars.map((b) => b.highValue - b.base),
    base: offsets,
    marker: { color: theme.colors[2] },
  };

  const layout = {
    ...baseLayout(theme, { showlegend: true }),
    barmode: 'overlay',
    xaxis: axis(theme, { title: 'Incremental NMB' }),
    // Categorical, and reversed so the largest-impact param (bars[0], sorted first by the caller)
    // renders at the top — the conventional tornado read — without touching bars' own order.
    yaxis: axis(theme, { extra: { type: 'category', autorange: 'reversed' } }),
    shapes: [
      {
        type: 'line', xref: 'x', yref: 'paper',
        x0: base, x1: base, y0: 0, y1: 1,
        line: { color: theme.muted, width: 1, dash: 'dot' },
      },
    ],
    annotations: [
      {
        x: base, y: 1, yref: 'paper', yanchor: 'bottom',
        text: 'Base', showarrow: false,
        font: { color: theme.muted, size: 11 },
      },
    ],
  };

  return { data: [low, high], layout, config: baseConfig() };
}

// plane's series keys, in the object's own key order, minus the optional `comparator` string
// property (see the module doc comment on buildCEPlaneSpec's comparator-name decision).
function strategyKeysOf(plane) {
  return Object.keys(plane).filter((k) => k !== 'comparator');
}

/**
 * buildCEPlaneSpec(plane, theme, strategyIndex) -> {data, layout, config}
 * plane = psaDerived().plane, i.e. {strategyName: [{dcost,dqaly}]} (never includes the
 * comparator strategy itself — analysis/psa.js's cePlane skips it), optionally carrying an extra
 * `plane.comparator` string the caller attaches for the x-axis title. 8px markers with a 2px
 * theme.surface ring; color = colorForSlot(strategyIndex.get(name)) — looked up per strategy
 * name, NEVER by plane's own key order or Map insertion order, so a strategy keeps its palette
 * slot regardless of which other strategies got filtered out of this particular plane. Quadrant
 * zero-lines are the one place this module deliberately does NOT keep the zeroline recessive:
 * native zeroline is turned off and replaced with explicit ink-colored `layout.shapes` lines.
 */
export function buildCEPlaneSpec(plane, theme, strategyIndex) {
  const names = strategyKeysOf(plane);
  const showlegend = names.length >= 2;

  const data = names.map((name) => {
    const slot = strategyIndex.get(name);
    const points = plane[name];
    return {
      type: 'scatter',
      mode: 'markers',
      name: labelForSlot(name, slot, theme),
      x: points.map((p) => p.dqaly),
      y: points.map((p) => p.dcost),
      marker: { size: 8, color: colorForSlot(slot, theme), line: { color: theme.surface, width: 2 } },
    };
  });

  const xTitle = plane.comparator ? `Incremental QALYs vs ${plane.comparator}` : 'Incremental QALYs';

  const layout = {
    ...baseLayout(theme, { showlegend }),
    xaxis: axis(theme, { title: xTitle, extra: { zeroline: false } }),
    yaxis: axis(theme, { title: 'Incremental cost', extra: { zeroline: false } }),
    shapes: [
      {
        type: 'line', xref: 'x', yref: 'paper',
        x0: 0, x1: 0, y0: 0, y1: 1,
        line: { color: theme.ink, width: 1 },
      },
      {
        type: 'line', xref: 'paper', yref: 'y',
        x0: 0, x1: 1, y0: 0, y1: 0,
        line: { color: theme.ink, width: 1 },
      },
    ],
  };

  return { data, layout, config: baseConfig() };
}

/**
 * buildCEACSpec({wtps, curves}, theme, strategyIndex) -> {data, layout, config}
 * One 2px line per strategy (curves' own keys — ceac() always includes every strategy, including
 * the comparator, unlike cePlane); color by strategyIndex.get(name), same rule as CE-plane.
 * y-axis pinned to [0,1] (a probability share).
 */
export function buildCEACSpec({ wtps, curves }, theme, strategyIndex) {
  const names = Object.keys(curves);
  const showlegend = names.length >= 2;

  const data = names.map((name) => {
    const slot = strategyIndex.get(name);
    return {
      type: 'scatter',
      mode: 'lines',
      name: labelForSlot(name, slot, theme),
      x: wtps,
      y: curves[name],
      line: { width: 2, color: colorForSlot(slot, theme) },
    };
  });

  const layout = {
    ...baseLayout(theme, { showlegend }),
    xaxis: axis(theme, { title: 'Willingness to pay' }),
    yaxis: axis(theme, { title: 'P(cost-effective)', extra: { range: [0, 1] } }),
  };

  return { data, layout, config: baseConfig() };
}

/**
 * buildEVPISpec({wtps, evpi}, theme) -> {data, layout, config}
 * Single series, always colors[0], always showlegend:false (there is only ever one EVPI curve —
 * no strategyIndex needed).
 */
export function buildEVPISpec({ wtps, evpi }, theme) {
  const data = [{
    type: 'scatter',
    mode: 'lines',
    name: 'EVPI',
    x: wtps,
    y: evpi,
    line: { width: 2, color: theme.colors[0] },
  }];

  const layout = {
    ...baseLayout(theme, { showlegend: false }),
    xaxis: axis(theme, { title: 'Willingness to pay' }),
    yaxis: axis(theme, { title: 'EVPI' }),
  };

  return { data, layout, config: baseConfig() };
}
