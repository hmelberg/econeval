import test from 'node:test';
import assert from 'node:assert/strict';
import {
  readChartTheme,
  buildTraceSpec,
  buildTornadoSpec,
  buildCEPlaneSpec,
  buildCEACSpec,
  buildEVPISpec,
} from '../js/ui/charts.js';

// A theme fixture matching the exact palette pinned in test/tokens.test.js (light values), so
// tests exercise realistic colors without going through the DOM/CSS at all.
const theme = {
  colors: ['#00806B', '#3B5BD9', '#C2611E', '#8A4E9E', '#6B7F1F', '#C64B7E'],
  ink: '#1A1D21',
  muted: '#6B7280',
  line: '#E2E4E8',
  surface: '#FFFFFF',
  fontUI: 'system-ui, -apple-system, "Segoe UI", sans-serif',
  fontData: 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, monospace',
};

// ================================================================================================
// readChartTheme
// ================================================================================================

test('readChartTheme: reads --chart-1..6, ink/muted/line/surface/font-ui/font-data via an injected getter, trimming whitespace', () => {
  const cssVars = {
    '--chart-1': '  #00806B  ',
    '--chart-2': '#3B5BD9',
    '--chart-3': '#C2611E',
    '--chart-4': '#8A4E9E',
    '--chart-5': '#6B7F1F',
    '--chart-6': '#C64B7E',
    '--ink': ' #1A1D21',
    '--muted': '#6B7280 ',
    '--line': '#E2E4E8',
    '--surface': '#FFFFFF',
    '--font-ui': ' system-ui, sans-serif ',
    '--font-data': 'ui-monospace, monospace',
  };
  const getter = (prop) => cssVars[prop] ?? '';
  const t = readChartTheme(getter);

  assert.deepEqual(t.colors, ['#00806B', '#3B5BD9', '#C2611E', '#8A4E9E', '#6B7F1F', '#C64B7E']);
  assert.equal(t.ink, '#1A1D21');
  assert.equal(t.muted, '#6B7280');
  assert.equal(t.line, '#E2E4E8');
  assert.equal(t.surface, '#FFFFFF');
  assert.equal(t.fontUI, 'system-ui, sans-serif');
  assert.equal(t.fontData, 'ui-monospace, monospace');
});

// ================================================================================================
// buildTraceSpec
// ================================================================================================

test('buildTraceSpec: returns {data, layout, config}; one line per state, width 2, x=1..cycles', () => {
  const traceData = {
    states: ['A', 'B', 'C', 'death'],
    series: {
      A: [0.9, 0.8, 0.7],
      B: [0.05, 0.1, 0.1],
      C: [0.03, 0.06, 0.1],
      death: [0.02, 0.04, 0.1],
    },
    cycles: 3,
  };
  const spec = buildTraceSpec(traceData, theme);

  assert.ok(spec.data && spec.layout && spec.config);
  assert.equal(spec.data.length, 4);
  for (const trace of spec.data) {
    assert.equal(trace.line.width, 2);
    assert.deepEqual(trace.x, [1, 2, 3]);
  }
  assert.deepEqual(spec.data.map((t) => t.y), [traceData.series.A, traceData.series.B, traceData.series.C, traceData.series.death]);
});

test('buildTraceSpec: legend shown for >=2 states; series colors follow state declaration order', () => {
  const traceData = { states: ['A', 'B'], series: { A: [1], B: [0] }, cycles: 1 };
  const spec = buildTraceSpec(traceData, theme);
  assert.equal(spec.layout.showlegend, true);
  assert.equal(spec.layout.legend.orientation, 'h');
  assert.equal(spec.data[0].line.color, theme.colors[0]);
  assert.equal(spec.data[1].line.color, theme.colors[1]);
});

test('buildTraceSpec: single-state model -> showlegend false', () => {
  const traceData = { states: ['A'], series: { A: [1, 1] }, cycles: 2 };
  const spec = buildTraceSpec(traceData, theme);
  assert.equal(spec.layout.showlegend, false);
});

test('buildTraceSpec: hovermode x unified, y title Share of cohort, ink font, surface bg', () => {
  const traceData = { states: ['A', 'B'], series: { A: [1], B: [0] }, cycles: 1 };
  const spec = buildTraceSpec(traceData, theme);
  assert.equal(spec.layout.hovermode, 'x unified');
  assert.equal(spec.layout.yaxis.title.text, 'Share of cohort');
  assert.equal(spec.layout.font.color, theme.ink);
  assert.equal(spec.layout.plot_bgcolor, theme.surface);
  assert.equal(spec.layout.paper_bgcolor, theme.surface);
});

test('buildTraceSpec: a 7th+ state folds to muted gray with an "(Other)" legend note', () => {
  const states = ['s0', 's1', 's2', 's3', 's4', 's5', 's6'];
  const series = Object.fromEntries(states.map((s) => [s, [1]]));
  const spec = buildTraceSpec({ states, series, cycles: 1 }, theme);
  assert.equal(spec.data[6].line.color, theme.muted);
  assert.match(spec.data[6].name, /Other/);
  assert.equal(spec.data[0].line.color, theme.colors[0]); // sanity: first 6 keep their hues
});

// ================================================================================================
// buildTornadoSpec
// ================================================================================================

const bars = [
  { param: 'p_AB', low: 0.15, high: 0.25, lowValue: -500, highValue: 800, base: 100 },
  { param: 'cost_x', low: 50, high: 150, lowValue: 300, highValue: -100, base: 100 },
];

test('buildTornadoSpec: returns {data, layout, config}; exactly 2 traces, colors[1]/colors[2], orientation h', () => {
  const spec = buildTornadoSpec(bars, theme);
  assert.ok(spec.data && spec.layout && spec.config);
  assert.equal(spec.data.length, 2);
  assert.equal(spec.data[0].orientation, 'h');
  assert.equal(spec.data[1].orientation, 'h');
  assert.equal(spec.data[0].marker.color, theme.colors[1]);
  assert.equal(spec.data[1].marker.color, theme.colors[2]);
});

test('buildTornadoSpec: bars retain the given order (no re-sort); low/high spans from base to lowValue/highValue', () => {
  const spec = buildTornadoSpec(bars, theme);
  assert.deepEqual(spec.data[0].y, ['p_AB', 'cost_x']);
  bars.forEach((b, i) => {
    assert.equal(spec.data[0].base[i] + spec.data[0].x[i], b.lowValue);
    assert.equal(spec.data[1].base[i] + spec.data[1].x[i], b.highValue);
  });
});

test('buildTornadoSpec: base drawn as a vertical reference line (layout.shapes) with a muted annotation', () => {
  const spec = buildTornadoSpec(bars, theme);
  assert.ok(Array.isArray(spec.layout.shapes) && spec.layout.shapes.length >= 1);
  const line = spec.layout.shapes.find((s) => s.type === 'line');
  assert.equal(line.x0, bars[0].base);
  assert.equal(line.x1, bars[0].base);
  assert.ok(Array.isArray(spec.layout.annotations) && spec.layout.annotations.length >= 1);
  assert.equal(spec.layout.annotations[0].font.color, theme.muted);
});

test('buildTornadoSpec: legend shown (Low/High), x title Incremental NMB', () => {
  const spec = buildTornadoSpec(bars, theme);
  assert.equal(spec.layout.showlegend, true);
  assert.equal(spec.data[0].name, 'Low');
  assert.equal(spec.data[1].name, 'High');
  assert.equal(spec.layout.xaxis.title.text, 'Incremental NMB');
});

// ================================================================================================
// buildCEPlaneSpec
// ================================================================================================

test('buildCEPlaneSpec: returns {data, layout, config}; marker size>=8 with 2px surface ring', () => {
  const strategyIndex = new Map([['mono', 0], ['combo', 1]]);
  const plane = { combo: [{ dcost: 1000, dqaly: 0.5 }, { dcost: 1200, dqaly: 0.6 }] };
  const spec = buildCEPlaneSpec(plane, theme, strategyIndex);

  assert.ok(spec.data && spec.layout && spec.config);
  assert.equal(spec.data.length, 1);
  const [trace] = spec.data;
  assert.ok(trace.marker.size >= 8);
  assert.equal(trace.marker.line.color, theme.surface);
  assert.equal(trace.marker.line.width, 2);
  assert.deepEqual(trace.x, [0.5, 0.6]);
  assert.deepEqual(trace.y, [1000, 1200]);
});

test('buildCEPlaneSpec: colors follow strategyIndex, not array/object order — combo keeps its slot when mono is filtered out (shuffled)', () => {
  // Declaration order: mono=0, combo=1, triple=2. `mono` is the comparator, so psaDerived's
  // plane never carries it — but the plane object's OWN key order is deliberately shuffled
  // (triple before combo) to prove color assignment reads strategyIndex.get(name), not
  // Object.keys(plane) position or Map insertion order.
  const strategyIndex = new Map([['mono', 0], ['combo', 1], ['triple', 2]]);
  const plane = {
    triple: [{ dcost: 10, dqaly: 1 }],
    combo: [{ dcost: 20, dqaly: 2 }],
  };
  const spec = buildCEPlaneSpec(plane, theme, strategyIndex);

  const tripleTrace = spec.data.find((t) => t.name === 'triple');
  const comboTrace = spec.data.find((t) => t.name === 'combo');
  assert.equal(comboTrace.marker.color, theme.colors[1]);
  assert.equal(tripleTrace.marker.color, theme.colors[2]);
});

test('buildCEPlaneSpec: legend iff >=2 series', () => {
  const strategyIndex = new Map([['mono', 0], ['combo', 1]]);
  const one = buildCEPlaneSpec({ combo: [{ dcost: 1, dqaly: 1 }] }, theme, strategyIndex);
  assert.equal(one.layout.showlegend, false);

  const strategyIndex2 = new Map([['mono', 0], ['combo', 1], ['triple', 2]]);
  const two = buildCEPlaneSpec({ combo: [{ dcost: 1, dqaly: 1 }], triple: [{ dcost: 2, dqaly: 2 }] }, theme, strategyIndex2);
  assert.equal(two.layout.showlegend, true);
});

test('buildCEPlaneSpec: comparator name (plane.comparator) labels the x-axis "Incremental QALYs vs <comparator>"; y is always "Incremental cost"', () => {
  const strategyIndex = new Map([['mono', 0], ['combo', 1]]);
  const withComparator = buildCEPlaneSpec(
    { comparator: 'mono', combo: [{ dcost: 1, dqaly: 1 }] }, theme, strategyIndex,
  );
  assert.equal(withComparator.layout.xaxis.title.text, 'Incremental QALYs vs mono');
  assert.equal(withComparator.layout.yaxis.title.text, 'Incremental cost');

  const withoutComparator = buildCEPlaneSpec({ combo: [{ dcost: 1, dqaly: 1 }] }, theme, strategyIndex);
  assert.equal(withoutComparator.layout.xaxis.title.text, 'Incremental QALYs');
});

test('buildCEPlaneSpec: zero-lines emphasized via layout.shapes (not series colors), native zeroline turned off', () => {
  const strategyIndex = new Map([['mono', 0], ['combo', 1]]);
  const spec = buildCEPlaneSpec({ combo: [{ dcost: 1, dqaly: 1 }] }, theme, strategyIndex);
  assert.equal(spec.layout.xaxis.zeroline, false);
  assert.equal(spec.layout.yaxis.zeroline, false);
  assert.ok(Array.isArray(spec.layout.shapes) && spec.layout.shapes.length === 2);
  for (const s of spec.layout.shapes) {
    assert.notEqual(s.line.color, theme.colors[0]);
    assert.notEqual(s.line.color, theme.colors[1]);
  }
});

// ================================================================================================
// buildCEACSpec
// ================================================================================================

test('buildCEACSpec: returns {data, layout, config}; y-axis range [0,1], y title P(cost-effective)', () => {
  const strategyIndex = new Map([['usual', 0], ['better', 1]]);
  const wtps = [0, 25000, 50000];
  const curves = { usual: [1, 0.6, 0.2], better: [0, 0.4, 0.8] };
  const spec = buildCEACSpec({ wtps, curves }, theme, strategyIndex);

  assert.ok(spec.data && spec.layout && spec.config);
  assert.deepEqual(spec.layout.yaxis.range, [0, 1]);
  assert.equal(spec.layout.yaxis.title.text, 'P(cost-effective)');
  assert.equal(spec.data.length, 2);
  for (const trace of spec.data) assert.equal(trace.line.width, 2);
});

test('buildCEACSpec: legend iff >=2 series; colors follow strategyIndex', () => {
  const strategyIndex = new Map([['usual', 0], ['better', 1], ['best', 2]]);
  const wtps = [0, 1];
  const spec = buildCEACSpec({ wtps, curves: { best: [1, 1], usual: [0, 0] } }, theme, strategyIndex);
  assert.equal(spec.layout.showlegend, true);
  const usual = spec.data.find((t) => t.name === 'usual');
  const best = spec.data.find((t) => t.name === 'best');
  assert.equal(usual.line.color, theme.colors[0]);
  assert.equal(best.line.color, theme.colors[2]);
});

// ================================================================================================
// buildEVPISpec
// ================================================================================================

test('buildEVPISpec: returns {data, layout, config}; single series, colors[0], showlegend false, y title EVPI', () => {
  const spec = buildEVPISpec({ wtps: [0, 1000, 2000], evpi: [0, 1.5, 3.2] }, theme);
  assert.ok(spec.data && spec.layout && spec.config);
  assert.equal(spec.data.length, 1);
  assert.equal(spec.data[0].line.color, theme.colors[0]);
  assert.equal(spec.data[0].line.width, 2);
  assert.equal(spec.layout.showlegend, false);
  assert.equal(spec.layout.yaxis.title.text, 'EVPI');
});

// ================================================================================================
// Cross-cutting: no builder ever emits a second axis; all share surface/ink/line/displayModeBar
// ================================================================================================

test('no builder ever emits yaxis2 or xaxis2; all share surface bg, ink font, line grid, displayModeBar false', () => {
  const strategyIndex = new Map([['a', 0], ['b', 1]]);
  const specs = [
    buildTraceSpec({ states: ['A', 'B'], series: { A: [1], B: [0] }, cycles: 1 }, theme),
    buildTornadoSpec(bars, theme),
    buildCEPlaneSpec({ a: [{ dcost: 1, dqaly: 1 }], b: [{ dcost: 2, dqaly: 2 }] }, theme, strategyIndex),
    buildCEACSpec({ wtps: [0, 1], curves: { a: [1, 0], b: [0, 1] } }, theme, strategyIndex),
    buildEVPISpec({ wtps: [0, 1], evpi: [0, 1] }, theme),
  ];
  for (const spec of specs) {
    assert.equal(spec.layout.yaxis2, undefined);
    assert.equal(spec.layout.xaxis2, undefined);
    assert.equal(spec.layout.paper_bgcolor, theme.surface);
    assert.equal(spec.layout.plot_bgcolor, theme.surface);
    assert.equal(spec.layout.font.color, theme.ink);
    assert.equal(spec.layout.xaxis.gridcolor, theme.line);
    assert.equal(spec.layout.yaxis.gridcolor, theme.line);
    assert.deepEqual(spec.layout.margin, { t: 24, r: 16, b: 44, l: 56 });
    assert.deepEqual(spec.config, { displayModeBar: false, responsive: true });
  }
});
