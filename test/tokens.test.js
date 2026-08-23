import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const tokensCss = readFileSync(path.join(__dirname, '../css/tokens.css'), 'utf8');
const appCss = readFileSync(path.join(__dirname, '../css/app.css'), 'utf8');

// Extract the content of the first {...} block that follows `startMarker`,
// respecting nested braces (needed for the @media-wrapped dark block).
function block(css, startMarker) {
  const start = css.indexOf(startMarker);
  assert.ok(start !== -1, `marker not found in css: ${startMarker}`);
  const braceStart = css.indexOf('{', start);
  assert.ok(braceStart !== -1, `no opening brace after marker: ${startMarker}`);
  let depth = 0;
  let i = braceStart;
  for (; i < css.length; i++) {
    if (css[i] === '{') depth++;
    else if (css[i] === '}') {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(depth === 0, `unbalanced braces after marker: ${startMarker}`);
  return css.slice(braceStart + 1, i);
}

// Parse `--name: value;` declarations out of a block's text into a map.
function customProps(blockText) {
  const map = {};
  const re = /--([a-zA-Z0-9-]+)\s*:\s*([^;]+);/g;
  let m;
  while ((m = re.exec(blockText))) {
    map[m[1]] = m[2].trim();
  }
  return map;
}

const LIGHT_EXPECTED = {
  bg: '#F3F4F6',
  surface: '#FFFFFF',
  paper: '#FDFDFB',
  dot: '#E3E2DC',
  ink: '#1A1D21',
  muted: '#6B7280',
  line: '#E2E4E8',
  accent: '#0E7A6E',
  'accent-soft': '#E4F2F0',
  danger: '#B42334',
  warn: '#B45309',
  radius: '6px',
  'radius-sm': '4px',
  'font-ui': 'system-ui, -apple-system, "Segoe UI", sans-serif',
  'font-data': 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, monospace',
  'fs-0': '11px',
  'fs-1': '12px',
  'fs-2': '13px',
  'fs-3': '15px',
  'fs-4': '18px',
  'sp-1': '4px',
  'sp-2': '8px',
  'sp-3': '12px',
  'sp-4': '16px',
  'sp-5': '24px',
  'shadow-pop': '0 4px 16px rgba(0,0,0,.12)',
};

const DARK_EXPECTED = {
  bg: '#131518',
  surface: '#1B1E23',
  paper: '#1F2328',
  dot: '#2A2F36',
  ink: '#E7E9EC',
  muted: '#9AA1AA',
  line: '#2A2E34',
  accent: '#2FA795',
  'accent-soft': '#16332F',
  danger: '#E5484D',
  warn: '#D97706',
  'shadow-pop': '0 4px 16px rgba(0,0,0,.5)',
};

test('tokens.css: light :root block has every binding token, byte-exact', () => {
  const light = customProps(block(tokensCss, ':root {'));
  for (const [name, expected] of Object.entries(LIGHT_EXPECTED)) {
    assert.equal(light[name], expected, `--${name} (light)`);
  }
});

test('tokens.css: @media (prefers-color-scheme: dark) block matches dark values, byte-exact', () => {
  // The @media rule's single child is `:root:not([data-theme="light"]) { ... }`;
  // pull that nested block out, then parse its declarations.
  const mediaOuter = block(tokensCss, '@media (prefers-color-scheme: dark)');
  const dark = customProps(block(mediaOuter, ':root:not([data-theme="light"])'));
  for (const [name, expected] of Object.entries(DARK_EXPECTED)) {
    assert.equal(dark[name], expected, `--${name} (@media dark)`);
  }
});

test('tokens.css: :root[data-theme="dark"] block repeats the same dark values, byte-exact', () => {
  const dark = customProps(block(tokensCss, ':root[data-theme="dark"]'));
  for (const [name, expected] of Object.entries(DARK_EXPECTED)) {
    assert.equal(dark[name], expected, `--${name} ([data-theme="dark"])`);
  }
});

test('app.css: #workspace grid-template-columns line matches the binding chrome spec', () => {
  assert.ok(
    appCss.includes('grid-template-columns: var(--w-yaml, 0px) 4px 1fr 4px var(--w-insp, 300px);'),
    'app.css must contain the exact grid-template-columns declaration from the plan'
  );
});

test('app.css: every "outline: none" is scoped with :not(:focus-visible), never an unconditional override', () => {
  // Regression: an unconditional `outline: none` on a selector with ID (or other
  // higher) specificity permanently beats the global `:focus-visible` ring rule,
  // silently defeating the "focus ring on every interactive element" requirement.
  // Walk each (selector { declarations }) rule and, for any block that sets
  // outline: none, require the selector itself to carry :not(:focus-visible).
  const ruleRe = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  let checked = 0;
  while ((m = ruleRe.exec(appCss))) {
    const selector = m[1].trim();
    const decls = m[2];
    if (/outline\s*:\s*none/i.test(decls)) {
      checked++;
      assert.ok(
        selector.includes(':not(:focus-visible)'),
        `unconditional "outline: none" on selector "${selector}" — must be scoped with :not(:focus-visible) so it never beats the global focus ring`
      );
    }
  }
  assert.ok(checked > 0, 'expected at least one outline: none declaration in app.css to check');
});
