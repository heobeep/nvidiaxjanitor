'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { isRetryableStatus, retryAfterMs } = require('../lib/retry-policy');

test('retries only the configured transient and key-specific status classes', () => {
  assert.equal(isRetryableStatus(null), true);
  assert.equal(isRetryableStatus(429), true);
  assert.equal(isRetryableStatus(503), true);
  assert.equal(isRetryableStatus(400), false);
  assert.equal(isRetryableStatus(422), false);
});

test('parses Retry-After delta-seconds and clamps extreme waits', () => {
  const now = 100000;
  assert.equal(retryAfterMs('15', now), 15000);
  assert.equal(retryAfterMs('0', now), 1000);
  assert.equal(retryAfterMs('99999', now), 3600000);
});

test('parses Retry-After HTTP dates and falls back for invalid values', () => {
  const now = Date.parse('2026-10-31T00:00:00Z');
  const date = new Date(now + 30000).toUTCString();
  assert.equal(retryAfterMs(date, now), 30000);
  assert.equal(retryAfterMs('not a date', now), 60000);
});
