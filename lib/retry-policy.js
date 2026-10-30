'use strict';

const RETRYABLE_STATUSES = new Set([401, 403, 404, 408, 425, 429, 500, 502, 503, 504]);

function isRetryableStatus(status) {
  return status == null || RETRYABLE_STATUSES.has(Number(status));
}

// Retry-After accepts either delta-seconds or an HTTP date. Clamp provider
// cooldowns so malformed headers cannot lock a credential out indefinitely.
function retryAfterMs(value, now = Date.now(), fallbackMs = 60000) {
  if (value == null || String(value).trim() === '') return fallbackMs;
  const raw = String(value).trim();
  const seconds = Number(raw);
  const target = Number.isFinite(seconds) ? now + Math.max(0, seconds) * 1000 : Date.parse(raw);
  if (!Number.isFinite(target)) return fallbackMs;
  return Math.min(60 * 60 * 1000, Math.max(1000, target - now));
}

module.exports = { isRetryableStatus, retryAfterMs };
