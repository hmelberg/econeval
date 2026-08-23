import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatMoney, format4, formatIcer, statusLabel, formatRunStamp, buildStrategyIndex, parseWtpInput,
  computeWtpMax, psaRunLabel, formatPsaStamp,
} from '../js/ui/results-format.js';

// --- formatMoney ---

test('formatMoney: 0-decimal, thousands-separated, rounds to nearest integer', () => {
  assert.equal(formatMoney(5976.796881), '5,977');
  assert.equal(formatMoney(1234567), '1,234,567');
  assert.equal(formatMoney(0), '0');
});

test('formatMoney: negative values keep a leading minus, never parentheses', () => {
  assert.equal(formatMoney(-1234.6), '-1,235');
});

test('formatMoney: null/undefined/non-finite -> empty string', () => {
  assert.equal(formatMoney(null), '');
  assert.equal(formatMoney(undefined), '');
  assert.equal(formatMoney(NaN), '');
  assert.equal(formatMoney(Infinity), '');
});

// --- format4 ---

test('format4: fixed 4 decimals', () => {
  assert.equal(format4(0.123456), '0.1235');
  assert.equal(format4(1), '1.0000');
  assert.equal(format4(0), '0.0000');
});

test('format4: null/undefined/non-finite -> empty string', () => {
  assert.equal(format4(null), '');
  assert.equal(format4(undefined), '');
  assert.equal(format4(NaN), '');
});

// --- formatIcer ---

test('formatIcer: null -> em dash', () => {
  assert.equal(formatIcer(null), '—');
});

test('formatIcer: a number formats exactly like formatMoney', () => {
  assert.equal(formatIcer(5976.796881), '5,977');
});

// --- statusLabel ---

test('statusLabel: frontier (empty string) -> null (no chip)', () => {
  assert.equal(statusLabel(''), null);
});

test('statusLabel: dominated/extended -> short labels', () => {
  assert.equal(statusLabel('dominated'), 'Dominated');
  assert.equal(statusLabel('extended'), 'Extended dominated');
});

// --- formatRunStamp ---

test('formatRunStamp: "Run · <strategies> · HH:MM", zero-padded, declaration order preserved', () => {
  assert.equal(formatRunStamp(['mono', 'combo'], new Date(2026, 0, 1, 14, 32)), 'Run · mono, combo · 14:32');
  assert.equal(formatRunStamp(['Surgery'], new Date(2026, 0, 1, 9, 5)), 'Run · Surgery · 09:05');
});

// --- buildStrategyIndex ---

test('buildStrategyIndex: name -> declaration-order slot', () => {
  const idx = buildStrategyIndex(['mono', 'combo', 'triple']);
  assert.equal(idx.get('mono'), 0);
  assert.equal(idx.get('combo'), 1);
  assert.equal(idx.get('triple'), 2);
  assert.equal(idx.get('missing'), undefined);
});

// --- parseWtpInput ---

test('parseWtpInput: empty string -> lastGood (never silently 0)', () => {
  assert.equal(parseWtpInput('', 30000), 30000);
});

test('parseWtpInput: whitespace-only -> lastGood', () => {
  assert.equal(parseWtpInput('   ', 30000), 30000);
});

test('parseWtpInput: non-numeric text -> lastGood', () => {
  assert.equal(parseWtpInput('abc', 30000), 30000);
});

test('parseWtpInput: "0" is a real, explicit value -> 0, not lastGood', () => {
  assert.equal(parseWtpInput('0', 30000), 0);
});

test('parseWtpInput: a valid number string -> that number', () => {
  assert.equal(parseWtpInput('25000', 30000), 25000);
});

// --- computeWtpMax ---

test('computeWtpMax: settings.wtp present -> 2.5x settings.wtp, header ignored', () => {
  assert.equal(computeWtpMax(30000, 50000), 75000);
});

test('computeWtpMax: settings.wtp null/undefined -> 2.5x the header wtp', () => {
  assert.equal(computeWtpMax(null, 40000), 100000);
  assert.equal(computeWtpMax(undefined, 40000), 100000);
});

test('computeWtpMax: neither available -> flat 100000 fallback', () => {
  assert.equal(computeWtpMax(null, null), 100000);
  assert.equal(computeWtpMax(undefined, undefined), 100000);
});

test('computeWtpMax: settings.wtp = 0 is a real explicit value, not treated as absent', () => {
  assert.equal(computeWtpMax(0, 40000), 0);
});

// --- psaRunLabel ---

test('psaRunLabel: "Run PSA (n=<n>)"', () => {
  assert.equal(psaRunLabel(1000), 'Run PSA (n=1000)');
  assert.equal(psaRunLabel(500), 'Run PSA (n=500)');
});

// --- formatPsaStamp ---

test('formatPsaStamp: "PSA · n=<n> · <ms>ms · HH:MM", elapsedMs rounded', () => {
  assert.equal(formatPsaStamp(1000, 842.4, new Date(2026, 0, 1, 14, 32)), 'PSA · n=1000 · 842ms · 14:32');
  assert.equal(formatPsaStamp(60, 12.6, new Date(2026, 0, 1, 9, 5)), 'PSA · n=60 · 13ms · 09:05');
});
