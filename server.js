// Lightweight OpenAI-compatible NVIDIA NIM gateway with an encrypted control plane.
'use strict';

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const crypto = require('crypto');
const fs = require('fs/promises');
const path = require('path');
const { applyModelPolicy, isNemotronLightning } = require('./lib/model-policy');
const { isRetryableStatus, retryAfterMs } = require('./lib/retry-policy');
const { StringDecoder } = require('string_decoder');
const { createThinkTagStripper } = require('./lib/stream-utils');

const app = express();
let modelRaceProgress = null;
app.set('trust proxy', 1);
app.use(cors());
app.use(express.json({ limit: '2mb' }));

const PORT = Number(process.env.PORT || 3000);
const NIM_API_BASE = (process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1').replace(/\/+$/, '');
const DATA_DIR = process.env.DATA_DIR || '/data';
const STATE_FILE = path.join(DATA_DIR, 'control-state.enc');
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || '';
const ENCRYPTION_KEY = Buffer.from(process.env.DATA_ENCRYPTION_KEY || '', 'hex');

const DEFAULT_SETTINGS = {
  temperature: 0.8,
  max_tokens: 4096,
  top_p: 1,
  presence_penalty: 0,
  frequency_penalty: 0,
  default_model: '',
  mistral_large4_enabled: true,
  mistral_large4_reasoning_effort: 'medium',
  mistral_large4_override_reasoning_effort: false,
  show_reasoning: true,
  thinking_mode: true,
  nemotron35_enabled: true,
  nemotron35_enable_thinking: true,
  nemotron35_reasoning_budget: 2048,
  nemotron35_repetition_penalty: 1.05,
  nemotron35_temperature: 1,
  nemotron35_top_p: 0.95
};

let state;
let saveQueue = Promise.resolve();
let roundRobinCursor = 0;
let modelCache = { at: 0, data: null };
const metrics = { startedAt: Date.now(), requests: 0, successes: 0, failures: 0, lastRequestAt: null, lastError: null };

function id(prefix) { return prefix + '_' + crypto.randomBytes(6).toString('hex'); }
function sha256(s) { return crypto.createHash('sha256').update(String(s)).digest('hex'); }
function safeEqual(a, b) {
  const aa = Buffer.from(String(a || ''));
  const bb = Buffer.from(String(b || ''));
  return aa.length === bb.length && aa.length > 0 && crypto.timingSafeEqual(aa, bb);
}
function signSession(expiry) {
  return crypto.createHmac('sha256', ENCRYPTION_KEY).update('admin:' + expiry).digest('base64url');
}
function sessionCookie(expiry) { return String(expiry) + '.' + signSession(expiry); }
function hasAdminSession(req) {
  const raw = (req.headers.cookie || '').split(';').map(x => x.trim()).find(x => x.startsWith('nj_admin='));
  if (!raw) return false;
  const token = decodeURIComponent(raw.slice('nj_admin='.length));
  const parts = token.split('.');
  if (parts.length !== 2 || !/^\d+$/.test(parts[0])) return false;
  const expiry = Number(parts[0]);
  return expiry > Date.now() && expiry < Date.now() + 9 * 60 * 60 * 1000 && safeEqual(parts[1], signSession(parts[0]));
}
function requireAdmin(req, res, next) {
  if (!hasAdminSession(req)) return res.status(401).json({ error: 'Sign in to manage this proxy.' });
  next();
}
function encryptState(value) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);
  const encrypted = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()]);
  return ['v1', iv.toString('base64'), cipher.getAuthTag().toString('base64'), encrypted.toString('base64')].join('.');
}
function decryptState(value) {
  const parts = value.split('.');
  if (parts.length !== 4 || parts[0] !== 'v1') throw new Error('Unknown encrypted state format; refusing to overwrite data.');
  const iv = Buffer.from(parts[1], 'base64');
  const tag = Buffer.from(parts[2], 'base64');
  const encrypted = Buffer.from(parts[3], 'base64');
  const decipher = crypto.createDecipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);
  decipher.setAuthTag(tag);
  return JSON.parse(Buffer.concat([decipher.update(encrypted), decipher.final()]).toString('utf8'));
}
async function persistState() {
  const snapshot = JSON.stringify(state);
  const write = async () => {
    await fs.mkdir(DATA_DIR, { recursive: true });
    const temp = STATE_FILE + '.tmp';
    await fs.writeFile(temp, encryptState(JSON.parse(snapshot)), { mode: 0o600 });
    await fs.rename(temp, STATE_FILE);
  };
  saveQueue = saveQueue.then(write, write);
  return saveQueue;
}
function emptyState() {
  return { version: 1, adminPasswordHash: sha256(ADMIN_PASSWORD), settings: Object.assign({}, DEFAULT_SETTINGS), providerKeys: [], clientKeys: [], createdAt: new Date().toISOString() };
}
async function initializeState() {
  if (ENCRYPTION_KEY.length !== 32) throw new Error('DATA_ENCRYPTION_KEY must be a 64-character hex string (32 bytes).');
  if (!ADMIN_PASSWORD) throw new Error('ADMIN_PASSWORD is required before the control panel can start.');
  await fs.mkdir(DATA_DIR, { recursive: true });
  let existing = false;
  try {
    const raw = await fs.readFile(STATE_FILE, 'utf8');
    state = decryptState(raw);
    existing = true;
  } catch (err) {
    if (err.code !== 'ENOENT') throw new Error('Could not decrypt existing proxy state. Check DATA_ENCRYPTION_KEY; data was not changed.');
  }
  if (!existing) {
    state = emptyState();
    if (process.env.NIM_API_KEY) {
      state.providerKeys.push({
        id: id('nim'), label: 'Primary NVIDIA key', secret: process.env.NIM_API_KEY,
        enabled: true, createdAt: new Date().toISOString(), lastStatus: null,
        lastError: '', lastSuccessAt: null, lastUsedAt: null, cooldownUntil: null, modelCount: 0
      });
    }
    if (process.env.PROXY_API_KEY) {
      const rawKey = process.env.PROXY_API_KEY;
      state.clientKeys.push({
        id: id('client'), label: 'Initial JanitorAI key', hash: sha256(rawKey),
        prefix: rawKey.slice(0, 10), createdAt: new Date().toISOString(), lastUsedAt: null
      });
    }
    await persistState();
  } else {
    state.settings = Object.assign({}, DEFAULT_SETTINGS, state.settings || {});
    state.providerKeys = Array.isArray(state.providerKeys) ? state.providerKeys : [];
    state.clientKeys = Array.isArray(state.clientKeys) ? state.clientKeys : [];
    if (!state.adminPasswordHash) {
      state.adminPasswordHash = sha256(ADMIN_PASSWORD);
      await persistState();
    }
  }
}
function clientKeyForRequest(req) {
  const auth = req.get('authorization') || '';
  const match = auth.match(/^Bearer\s+(.+)$/i);
  if (!match) return null;
  const digest = sha256(match[1].trim());
  return state.clientKeys.find(k => safeEqual(k.hash, digest)) || null;
}
function requireClientKey(req, res, next) {
  // Preserve uptime probes without exposing model data; HEAD responses have no body.
  if (req.method === 'HEAD' && req.path === '/models') return next();
  const key = clientKeyForRequest(req);
  if (!key) return res.status(401).json({
    error: { message: 'Invalid or missing proxy API key.', type: 'authentication_error', code: 401 }
  });
  key.lastUsedAt = new Date().toISOString();
  req.clientKeyRecord = key;
  next();
}
function cleanProviderError(error) {
  const status = error && error.response && error.response.status;
  const data = error && error.response && error.response.data;
  const detail = data && data.error && data.error.message ? data.error.message : (data && data.message ? data.message : '');
  return (status ? 'HTTP ' + status + (detail ? ': ' + String(detail).slice(0, 180) : '') : (error && error.code ? error.code : 'Connection error')).replace(/Bearer\s+\S+/ig, 'Bearer [redacted]');
}
function isRetryable(error) {
  const status = error && error.response && error.response.status;
  return isRetryableStatus(status);
}
function providerOrder(modelId) {
  const enabled = state.providerKeys.filter(k => k.enabled !== false);
  if (!enabled.length) return [];
  const now = Date.now();
  const rotate = list => {
    if (!list.length) return [];
    const offset = roundRobinCursor++ % list.length;
    return list.slice(offset).concat(list.slice(0, offset));
  };

  // Prefer keys whose last model-catalog response confirmed access to this
  // model, but retain the rest of the pool as failover candidates.
  const model = modelCache.data && Array.isArray(modelCache.data.models)
    ? modelCache.data.models.find(item => item.id === modelId)
    : null;
  if (model && Array.isArray(model.available_via) && model.available_via.length) {
    const preferred = enabled.filter(key => model.available_via.includes(key.label));
    const fallback = enabled.filter(key => !model.available_via.includes(key.label));
    const preferredReady = preferred.filter(key => !key.cooldownUntil || Date.parse(key.cooldownUntil) <= now);
    const fallbackReady = fallback.filter(key => !key.cooldownUntil || Date.parse(key.cooldownUntil) <= now);
    if (preferredReady.length) return rotate(preferredReady).concat(fallbackReady);
    if (fallbackReady.length) return rotate(fallbackReady);
  }

  const ready = enabled.filter(key => !key.cooldownUntil || Date.parse(key.cooldownUntil) <= now);
  const candidates = ready.length
    ? ready
    : enabled.slice().sort((a, b) => Date.parse(a.cooldownUntil || 0) - Date.parse(b.cooldownUntil || 0));
  return rotate(candidates);
}
async function requestWithFailover(method, endpoint, body, responseType) {
  const candidates = providerOrder(body && body.model);
  if (!candidates.length) {
    const e = new Error('Add an NVIDIA NIM API key in the control panel first.');
    e.status = 503;
    throw e;
  }
  let lastError;
  for (let i = 0; i < candidates.length; i++) {
    const provider = candidates[i];
    try {
      const response = await axios({
        method,
        url: NIM_API_BASE + endpoint,
        data: body,
        headers: { Authorization: 'Bearer ' + provider.secret, 'Content-Type': 'application/json' },
        responseType: responseType || 'json',
        timeout: responseType === 'stream' ? 0 : 60000,
        validateStatus: status => status >= 200 && status < 300
      });
      provider.lastStatus = response.status;
      provider.lastError = '';
      provider.lastSuccessAt = new Date().toISOString();
      provider.lastUsedAt = new Date().toISOString();
      provider.cooldownUntil = null;
      return { response, provider };
    } catch (error) {
      lastError = error;
      const status = error.response && error.response.status;
      provider.lastStatus = status || null;
      provider.lastError = cleanProviderError(error);
      if (status === 429) {
        const waitMs = retryAfterMs(error.response.headers && error.response.headers['retry-after']);
        provider.cooldownUntil = new Date(Date.now() + waitMs).toISOString();
      } else if (status === 401) {
        provider.cooldownUntil = new Date(Date.now() + 15 * 60 * 1000).toISOString();
      } else if (status === 403 || status === 503 || !status) {
        provider.cooldownUntil = new Date(Date.now() + 30000).toISOString();
      }
      await persistState();
      if (error.response && error.response.data && typeof error.response.data.destroy === 'function') error.response.data.destroy();
      if (!isRetryable(error) || i === candidates.length - 1) break;
    }
  }
  if (lastError && lastError.response) {
    lastError.status = lastError.response.status;
  } else if (lastError && !lastError.status) {
    lastError.status = 502;
  }
  throw lastError || new Error('All configured NVIDIA keys failed.');
}
async function fetchModelsForProvider(provider) {
  try {
    const response = await axios.get(NIM_API_BASE + '/models', {
      headers: { Authorization: 'Bearer ' + provider.secret },
      timeout: 15000,
      validateStatus: status => status >= 200 && status < 300
    });
    const models = Array.isArray(response.data && response.data.data) ? response.data.data : [];
    provider.lastStatus = response.status;
    provider.lastError = '';
    provider.lastSuccessAt = new Date().toISOString();
    provider.modelCount = models.length;
    provider.cooldownUntil = null;
    return { providerId: provider.id, label: provider.label, ok: true, models };
  } catch (error) {
    provider.lastStatus = error.response ? error.response.status : null;
    provider.lastError = cleanProviderError(error);
    return { providerId: provider.id, label: provider.label, ok: false, models: [], error: provider.lastError };
  }
}
async function refreshModels(force) {
  if (!force && modelCache.data && Date.now() - modelCache.at < 60000) return modelCache.data;
  const keys = state.providerKeys.filter(k => k.enabled !== false);
  const results = await Promise.all(keys.map(fetchModelsForProvider));
  const byId = new Map();
  results.forEach(result => result.models.forEach(model => {
    if (!model || !model.id) return;
    if (!byId.has(model.id)) byId.set(model.id, Object.assign({}, model, { available_via: [] }));
    const entry = byId.get(model.id);
    if (!entry.available_via.includes(result.label)) entry.available_via.push(result.label);
  }));
  const output = {
    models: Array.from(byId.values()).sort((a, b) => a.id.localeCompare(b.id)),
    providers: results.map(r => ({ providerId: r.providerId, label: r.label, ok: r.ok, count: r.models.length, error: r.error || '' })),
    fetchedAt: new Date().toISOString()
  };
  modelCache = { at: Date.now(), data: output };
  await persistState();
  if (!output.models.length && results.length && !results.some(r => r.ok)) {
    const e = new Error('Could not load models from any NVIDIA key. Check key status and quota.');
    e.status = 502;
    e.modelResults = output.providers;
    throw e;
  }
  return output;
}
function publicProvider(k) {
  return {
    id: k.id, label: k.label, enabled: k.enabled !== false,
    keyHint: '•••• ' + (k.secret ? k.secret.slice(-4) : '----'),
    createdAt: k.createdAt, lastStatus: k.lastStatus || null,
    lastError: k.lastError || '', lastSuccessAt: k.lastSuccessAt || null,
    lastUsedAt: k.lastUsedAt || null, cooldownUntil: k.cooldownUntil || null,
    modelCount: k.modelCount || 0
  };
}
function publicClientKey(k) {
  return { id: k.id, label: k.label, prefix: k.prefix, createdAt: k.createdAt, lastUsedAt: k.lastUsedAt || null };
}
function publicState(req) {
  return {
    status: 'online',
    baseUrl: (req.protocol + '://' + req.get('host') + '/v1'),
    settings: state.settings,
    providerKeys: state.providerKeys.map(publicProvider),
    clientKeys: state.clientKeys.map(publicClientKey),
    metrics: Object.assign({}, metrics, { uptimeSeconds: Math.floor((Date.now() - metrics.startedAt) / 1000) }),
    storage: 'encrypted persistent volume'
  };
}
function stripThinkTags(text, trim) {
  if (!text) return text;
  const result = String(text).replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '').replace(/<\/?think(?:ing)?>/gi, '');
  return trim === false ? result : result.trim();
}
function extractThinkTags(text) {
  const match = String(text || '').match(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/i);
  return match ? match[1].trim() : null;
}
function sendAdminHtml(req, res) {
  res.set('Cache-Control', 'no-store');
  res.sendFile(path.join(__dirname, 'public', 'admin.html'));
}

app.get('/', (req, res) => res.redirect('/admin'));
app.get('/admin', sendAdminHtml);
app.get('/health', (req, res) => res.json({
  status: 'ok', service: 'NVIDIA NIM Control', version: '2.0.0',
  persistence: 'encrypted', activeProviderKeys: state ? state.providerKeys.filter(k => k.enabled !== false).length : 0
}));

const loginAttempts = new Map();
app.post('/admin/api/login', (req, res) => {
  const ip = req.ip || 'unknown';
  const now = Date.now();
  let attempts = loginAttempts.get(ip);
  if (!attempts || now - attempts.startedAt > 15 * 60 * 1000) attempts = { startedAt: now, count: 0 };
  if (attempts.count >= 8) return res.status(429).json({ error: 'Too many sign-in attempts. Wait 15 minutes before trying again.' });
  const supplied = req.body && req.body.password;
  if (typeof supplied !== 'string' || !safeEqual(sha256(supplied), state.adminPasswordHash)) {
    attempts.count += 1;
    loginAttempts.set(ip, attempts);
    return res.status(401).json({ error: 'That admin password is not correct.' });
  }
  loginAttempts.delete(ip);
  const expiry = Date.now() + 8 * 60 * 60 * 1000;
  const secure = req.secure || req.get('x-forwarded-proto') === 'https';
  res.setHeader('Set-Cookie', 'nj_admin=' + encodeURIComponent(sessionCookie(expiry)) + '; HttpOnly; Path=/admin; SameSite=Strict; Max-Age=28800' + (secure ? '; Secure' : ''));
  res.json({ ok: true });
});
app.post('/admin/api/password', requireAdmin, async (req, res) => {
  const current = req.body && req.body.currentPassword;
  const next = req.body && req.body.newPassword;
  if (typeof current !== 'string' || !safeEqual(sha256(current), state.adminPasswordHash)) {
    return res.status(401).json({ error: 'Current admin password is incorrect.' });
  }
  if (typeof next !== 'string' || next.length < 12 || next.length > 128) {
    return res.status(400).json({ error: 'New password must be between 12 and 128 characters.' });
  }
  state.adminPasswordHash = sha256(next);
  await persistState();
  res.json({ ok: true });
});
app.post('/admin/api/logout', requireAdmin, (req, res) => {
  res.setHeader('Set-Cookie', 'nj_admin=; HttpOnly; Path=/admin; SameSite=Strict; Max-Age=0' + ((req.secure || req.get('x-forwarded-proto') === 'https') ? '; Secure' : ''));
  res.json({ ok: true });
});
app.use('/admin/api', requireAdmin);

app.get('/admin/api/overview', (req, res) => res.json(publicState(req)));
app.get('/admin/api/settings', (req, res) => res.json(state.settings));
app.put('/admin/api/settings', async (req, res) => {
  const input = req.body || {};
  const next = Object.assign({}, state.settings);
  const numeric = [
    ['temperature', 0, 2], ['max_tokens', 1, 131072],
    ['top_p', 0, 1], ['presence_penalty', -2, 2], ['frequency_penalty', -2, 2],
    ['nemotron35_reasoning_budget', -1, 32768],
    ['nemotron35_repetition_penalty', 0, 2],
    ['nemotron35_temperature', 0, 1],
    ['nemotron35_top_p', 0, 1]
  ];
  for (const item of numeric) {
    if (input[item[0]] !== undefined) {
      const n = Number(input[item[0]]);
      if (!Number.isFinite(n) || n < item[1] || n > item[2]) return res.status(400).json({ error: 'Invalid value for ' + item[0] + '.' });
      next[item[0]] = n;
    }
  }
  if (input.default_model !== undefined) next.default_model = String(input.default_model).slice(0, 200);
  if (input.mistral_large4_reasoning_effort !== undefined) {
    const effort = String(input.mistral_large4_reasoning_effort).toLowerCase();
    if (!['low', 'medium', 'high', 'none'].includes(effort)) return res.status(400).json({ error: 'Invalid Mistral Large 4 reasoning effort.' });
    next.mistral_large4_reasoning_effort = effort;
  }
  for (const name of ['show_reasoning', 'thinking_mode', 'nemotron35_enabled', 'nemotron35_enable_thinking', 'mistral_large4_enabled', 'mistral_large4_override_reasoning_effort']) {
    if (input[name] !== undefined) next[name] = Boolean(input[name]);
  }
  state.settings = next;
  await persistState();
  res.json({ ok: true, settings: state.settings });
});

app.get('/admin/api/models', async (req, res) => {
  try { res.json(await refreshModels(req.query.refresh === '1')); }
  catch (error) { res.status(error.status || 502).json({ error: error.message, providers: error.modelResults || [] }); }
});
app.post('/admin/api/models/refresh', async (req, res) => {
  try { res.json(await refreshModels(true)); }
  catch (error) { res.status(error.status || 502).json({ error: error.message, providers: error.modelResults || [] }); }
});
app.post('/admin/api/keys/providers', async (req, res) => {
  const label = String((req.body && req.body.label) || '').trim().slice(0, 60);
  const secret = String((req.body && req.body.key) || '').trim();
  if (!label || !secret || secret.length < 10) return res.status(400).json({ error: 'Give this NVIDIA key a label and paste a valid-looking API key.' });
  if (state.providerKeys.some(k => sha256(k.secret) === sha256(secret))) return res.status(409).json({ error: 'That NVIDIA key is already in the pool.' });
  const provider = {
    id: id('nim'), label, secret, enabled: true, createdAt: new Date().toISOString(),
    lastStatus: null, lastError: '', lastSuccessAt: null, lastUsedAt: null,
    cooldownUntil: null, modelCount: 0
  };
  state.providerKeys.push(provider);
  await persistState();
  modelCache = { at: 0, data: null };
  res.status(201).json({ ok: true, provider: publicProvider(provider) });
});
app.post('/admin/api/keys/providers/:id/test', async (req, res) => {
  const provider = state.providerKeys.find(k => k.id === req.params.id);
  if (!provider) return res.status(404).json({ error: 'NVIDIA key not found.' });
  const result = await fetchModelsForProvider(provider);
  await persistState();
  modelCache = { at: 0, data: null };
  res.status(result.ok ? 200 : 502).json({
    ok: result.ok, label: provider.label, count: result.models.length,
    models: result.models.map(m => ({ id: m.id, owned_by: m.owned_by || '' })),
    error: result.error || ''
  });
});
app.patch('/admin/api/keys/providers/:id', async (req, res) => {
  const provider = state.providerKeys.find(k => k.id === req.params.id);
  if (!provider) return res.status(404).json({ error: 'NVIDIA key not found.' });
  if (req.body && req.body.label !== undefined) provider.label = String(req.body.label).trim().slice(0, 60) || provider.label;
  if (req.body && req.body.enabled !== undefined) provider.enabled = Boolean(req.body.enabled);
  await persistState();
  modelCache = { at: 0, data: null };
  res.json({ ok: true, provider: publicProvider(provider) });
});
app.delete('/admin/api/keys/providers/:id', async (req, res) => {
  const before = state.providerKeys.length;
  state.providerKeys = state.providerKeys.filter(k => k.id !== req.params.id);
  if (state.providerKeys.length === before) return res.status(404).json({ error: 'NVIDIA key not found.' });
  await persistState();
  modelCache = { at: 0, data: null };
  res.json({ ok: true });
});
app.post('/admin/api/keys/clients', async (req, res) => {
  const label = String((req.body && req.body.label) || '').trim().slice(0, 60);
  if (!label) return res.status(400).json({ error: 'Give this client key a label, like JanitorAI.' });
  const secret = 'nj_' + crypto.randomBytes(32).toString('base64url');
  const record = {
    id: id('client'), label, hash: sha256(secret),
    prefix: secret.slice(0, 11), createdAt: new Date().toISOString(), lastUsedAt: null
  };
  state.clientKeys.push(record);
  await persistState();
  res.status(201).json({ ok: true, key: secret, record: publicClientKey(record) });
});
app.patch('/admin/api/keys/clients/:id', async (req, res) => {
  const client = state.clientKeys.find(k => k.id === req.params.id);
  if (!client) return res.status(404).json({ error: 'Client key not found.' });
  if (req.body && req.body.label !== undefined) client.label = String(req.body.label).trim().slice(0, 60) || client.label;
  await persistState();
  res.json({ ok: true, key: publicClientKey(client) });
});
app.delete('/admin/api/keys/clients/:id', async (req, res) => {
  const before = state.clientKeys.length;
  state.clientKeys = state.clientKeys.filter(k => k.id !== req.params.id);
  if (state.clientKeys.length === before) return res.status(404).json({ error: 'Client key not found.' });
  await persistState();
  res.json({ ok: true });
});

app.use('/v1', requireClientKey);

app.get('/v1/models', async (req, res) => {
  try {
    const result = await refreshModels(false);
    res.json({
      object: 'list',
      data: result.models.map(m => ({
        id: m.id, object: m.object || 'model',
        created: m.created || Math.floor(Date.now() / 1000),
        owned_by: m.owned_by || 'nvidia-nim',
        available_via: m.available_via
      }))
    });
  } catch (error) {
    res.status(error.status || 502).json({
      error: { message: error.message || 'Could not fetch NVIDIA models.', type: 'upstream_error', code: error.status || 502 }
    });
  }
});

app.post('/v1/chat/completions', async (req, res) => {
  metrics.requests++;
  metrics.lastRequestAt = new Date().toISOString();
  const body = req.body || {};
  const model = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : state.settings.default_model;
  if (!model) {
    metrics.failures++;
    return res.status(400).json({ error: { message: 'A model ID is required. Choose a model exposed by GET /v1/models.', type: 'invalid_request_error', code: 400 } });
  }
  if (!Array.isArray(body.messages)) {
    metrics.failures++;
    return res.status(400).json({ error: { message: 'messages must be an array.', type: 'invalid_request_error', code: 400 } });
  }
  const stream = Boolean(body.stream);
  // Apply model-aware defaults only when the client has not supplied its own.
  // This does not append system prompts or rewrite the conversation messages.
  const preparedBody = applyModelPolicy(body, model, state.settings);
  const extraBody = Object.assign({}, preparedBody.extra_body || {});
  if (state.settings.thinking_mode && model.startsWith('openai/gpt-oss')) {
    const chatTemplate = Object.assign({}, extraBody.chat_template_kwargs || {});
    if (chatTemplate.thinking === undefined) chatTemplate.thinking = true;
    extraBody.chat_template_kwargs = chatTemplate;
  }
  const nimRequest = Object.assign({}, preparedBody, {
    model,
    temperature: preparedBody.temperature ?? state.settings.temperature,
    max_tokens: preparedBody.max_tokens ?? state.settings.max_tokens,
    top_p: preparedBody.top_p ?? state.settings.top_p,
    presence_penalty: preparedBody.presence_penalty ?? state.settings.presence_penalty,
    frequency_penalty: preparedBody.frequency_penalty ?? state.settings.frequency_penalty,
    extra_body: Object.keys(extraBody).length ? extraBody : undefined,
    stream
  });
  try {
    const routed = await requestWithFailover('POST', '/chat/completions', nimRequest, stream ? 'stream' : 'json');
    const response = routed.response;
    metrics.successes++;
    metrics.lastError = null;
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      if (typeof res.flushHeaders === 'function') res.flushHeaders();
      let buffer = '';
      const decoder = new StringDecoder('utf8');
      const stripInlineThink = createThinkTagStripper();
      function forwardSseLine(line) {
        const normalized = line.replace(/\r$/, '');
        if (!normalized) { res.write('\n'); return; }
        if (!normalized.startsWith('data:')) { res.write(normalized + '\n'); return; }
        const rawData = normalized.slice(5).replace(/^\s+/, '');
        if (rawData === '[DONE]') { res.write('data: [DONE]\n\n'); return; }
        try {
          const data = JSON.parse(rawData);
          if (data.choices && data.choices[0] && data.choices[0].delta) {
            const delta = data.choices[0].delta;
            const raw = delta.content || '';
            if (raw) delta.content = stripInlineThink(raw);
            else delete delta.content;
            if (!state.settings.show_reasoning) delete delta.reasoning_content;
            if (raw && !delta.content) delete delta.content;
          }
          res.write('data: ' + JSON.stringify(data) + '\n\n');
        } catch (_) {
          // Keep non-JSON SSE payloads intact; never silently discard provider events.
          res.write(normalized + '\n\n');
        }
      }
      response.data.on('data', chunk => {
        buffer += decoder.write(chunk);
        let newline;
        while ((newline = buffer.search(/\r?\n/)) !== -1) {
          const separatorLength = buffer[newline] === '\r' ? 2 : 1;
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + separatorLength);
          forwardSseLine(line);
        }
      });
      response.data.on('end', () => {
        buffer += decoder.end();
        if (buffer.length) forwardSseLine(buffer);
        res.end();
      });
      response.data.on('error', err => { console.error('NIM stream error:', err.message); res.end(); });
      res.on('close', () => { if (!res.writableEnded && response.data && response.data.destroy) response.data.destroy(); });
    } else {
      const payload = response.data || {};
      const out = Object.assign({}, payload, {
        id: payload.id || ('chatcmpl-' + Date.now()),
        object: payload.object || 'chat.completion',
        created: payload.created || Math.floor(Date.now() / 1000),
        model,
        choices: (payload.choices || []).map((choice, index) => {
          const raw = choice.message && choice.message.content || '';
          const message = Object.assign({}, choice.message || {}, {
            role: choice.message && choice.message.role || 'assistant',
            content: stripThinkTags(raw)
          });
          const reasoning = (choice.message && choice.message.reasoning_content) || extractThinkTags(raw);
          if (state.settings.show_reasoning && reasoning) message.reasoning_content = reasoning;
          else if (!state.settings.show_reasoning) delete message.reasoning_content;
          return Object.assign({}, choice, {
            index: choice.index ?? index,
            message,
            finish_reason: choice.finish_reason ?? null
          });
        }),
        usage: payload.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
      res.json(out);
    }
  } catch (error) {
    metrics.failures++;
    metrics.lastError = cleanProviderError(error);
    console.error('Proxy request failed:', cleanProviderError(error));
    const status = error.status || (error.response && error.response.status) || 502;
    res.status(status).json({
      error: {
        message: (error.response && error.response.data && error.response.data.error && error.response.data.error.message) || error.message || 'Upstream NVIDIA request failed.',
        type: status === 401 ? 'authentication_error' : 'upstream_error',
        code: status
      }
    });
  }
});


app.get('/admin/api/models/race/status', (req, res) => {
  if (!modelRaceProgress) return res.json({ running:false, completed:0, total:0, results:[], events:[] });
  res.json(modelRaceProgress);
});

app.post('/admin/api/models/race', async (req, res) => {
  const startedAt = new Date().toISOString();
  const startedMs = Date.now();
  if (modelRaceProgress && modelRaceProgress.running) return res.status(409).json({ error:'A model race is already running.' });
  modelRaceProgress = { running:true, phase:'loading-catalog', startedAt, completed:0, total:0, activeModel:'', activeProfile:'', activeProvider:'', results:[], events:[], summary:{ total:0,alive:0,adaptive:0,reasoningOnly:0,empty:0,error:0 } };
  try {
    const catalogue = await refreshModels(true);
    const allModels = catalogue.models || [];
    const providers = state.providerKeys.filter(k => k.enabled !== false);
    if (!allModels.length) { modelRaceProgress.running=false; modelRaceProgress.phase='failed'; modelRaceProgress.error='No accessible models were returned by the enabled NVIDIA keys.'; return res.status(502).json({ error: modelRaceProgress.error, providers: catalogue.providers || [] }); }
    modelRaceProgress.phase='testing'; modelRaceProgress.total=allModels.length; modelRaceProgress.fetchedAt=catalogue.fetchedAt; modelRaceProgress.events.push({at:new Date().toISOString(),type:'start',message:'Catalog loaded: '+allModels.length+' models. Up to 5 concurrent probes.'});
    const profiles = [
      { name: 'baseline', max_tokens: 384, system: false, thinkingOff: false },
      { name: 'larger-budget', max_tokens: 1536, system: true, thinkingOff: false },
      { name: 'thinking-off', max_tokens: 1536, system: true, thinkingOff: true }
    ];
    function textContent(value) {
      if (typeof value === 'string') return value;
      if (Array.isArray(value)) return value.map(part => typeof part === 'string' ? part : (part && (part.text || part.content) || '')).join('');
      return '';
    }
    async function probe(modelId, provider, profile) {
      const userPrompt = profile.thinkingOff ? 'Say ALIVE_OK. Output visible answer text only.' : 'Reply with exactly ALIVE_OK and nothing else.';
      const messages = profile.system
        ? [{ role: 'system', content: 'You are a concise assistant. Do not output hidden reasoning. Return visible text only.' }, { role: 'user', content: userPrompt }]
        : [{ role: 'user', content: userPrompt }];
      let body = applyModelPolicy(
        { model: modelId, messages, temperature: 0, top_p: 1, max_tokens: profile.max_tokens, stream: false },
        modelId,
        state.settings
      );
      if (profile.thinkingOff) {
        const extra = Object.assign({}, body.extra_body || {});
        const template = Object.assign({}, extra.chat_template_kwargs || {});
        if (isNemotronLightning(modelId)) template.enable_thinking = false;
        else template.thinking = false;
        extra.chat_template_kwargs = template;
        body.extra_body = extra;
      }
      const t0 = Date.now();
      if (modelRaceProgress) { modelRaceProgress.activeModel=modelId; modelRaceProgress.activeProfile=profile.name; modelRaceProgress.activeProvider=provider.label; }
      try {
        const response = await axios.post(NIM_API_BASE + '/chat/completions', body, {
          headers: { Authorization: 'Bearer ' + provider.secret, 'Content-Type': 'application/json' },
          timeout: 30000, validateStatus: status => status >= 200 && status < 300
        });
        const payload = response.data || {};
        const choice = Array.isArray(payload.choices) ? payload.choices[0] : null;
        const message = choice && choice.message || {};
        const rawContent = textContent(message.content);
        const visibleContent = stripThinkTags(rawContent);
        const reasoning = textContent(message.reasoning_content) || extractThinkTags(rawContent) || '';
        return { profile: profile.name, httpStatus: response.status, latencyMs: Date.now() - t0, contentChars: visibleContent.length, rawContentChars: rawContent.length, reasoningChars: reasoning.length, finishReason: choice && choice.finish_reason || null, preview: visibleContent.slice(0, 180), reasoningPreview: reasoning.slice(0, 120), error: '' };
      } catch (error) {
        return { profile: profile.name, httpStatus: error.response && error.response.status || null, latencyMs: Date.now() - t0, contentChars: 0, rawContentChars: 0, reasoningChars: 0, finishReason: null, preview: '', reasoningPreview: '', error: cleanProviderError(error) };
      }
    }
    async function testModel(model) {
      const availableVia = Array.isArray(model.available_via) ? model.available_via : [];
      let candidates = providers.filter(p => availableVia.includes(p.label));
      if (!candidates.length) candidates = providers;
      const attempts = [];
      let firstError = '', reasoningPreview = '', totalLatency = 0;
      for (const provider of candidates) {
        for (const profile of profiles) {
          const a = await probe(model.id, provider, profile);
          attempts.push(Object.assign({ provider: provider.label }, a));
          totalLatency += a.latencyMs || 0;
          if (!firstError && a.error) firstError = a.error;
          if (a.reasoningPreview) reasoningPreview = a.reasoningPreview;
          if (modelRaceProgress) { modelRaceProgress.events.unshift({at:new Date().toISOString(),type:a.error?'error':a.contentChars?'success':'empty',model:model.id,profile:profile.name,provider:provider.label,latencyMs:a.latencyMs,httpStatus:a.httpStatus,contentChars:a.contentChars,message:a.error ? (a.httpStatus ? 'HTTP '+a.httpStatus+': '+a.error : a.error) : (a.contentChars ? 'Visible text returned ('+a.contentChars+' chars)' : 'HTTP success, but no visible text')}); modelRaceProgress.events=modelRaceProgress.events.slice(0,40); }
          if (!a.error && a.contentChars > 0) {
            return { model: model.id, status: profile.name === 'baseline' ? 'alive' : 'adaptive', workingProfile: profile.name, testedVia: provider.label, latencyMs: totalLatency, firstResponseMs: attempts[0] ? attempts[0].latencyMs : null, contentChars: a.contentChars, reasoningChars: a.reasoningChars, finishReason: a.finishReason, preview: a.preview, error: '', attempts: attempts.map(x => ({ profile:x.profile,provider:x.provider,httpStatus:x.httpStatus,latencyMs:x.latencyMs,contentChars:x.contentChars,reasoningChars:x.reasoningChars,finishReason:x.finishReason,error:x.error })) };
          }
          if (a.error && [401, 403, 404, 429].includes(a.httpStatus)) break;
        }
      }
      const reasoningOnly = attempts.some(a => a.reasoningChars > 0);
      const success = attempts.some(a => a.httpStatus >= 200 && a.httpStatus < 300);
      return { model: model.id, status: reasoningOnly ? 'reasoning-only' : success ? 'empty' : 'error', workingProfile: '', testedVia: attempts.length ? attempts[attempts.length - 1].provider : '', latencyMs: totalLatency, firstResponseMs: attempts[0] ? attempts[0].latencyMs : null, contentChars: 0, reasoningChars: attempts.reduce((n,a) => n + (a.reasoningChars || 0),0), finishReason: attempts.length ? attempts[attempts.length - 1].finishReason : null, preview: '', reasoningPreview, error: firstError, attempts: attempts.map(x => ({ profile:x.profile,provider:x.provider,httpStatus:x.httpStatus,latencyMs:x.latencyMs,contentChars:x.contentChars,reasoningChars:x.reasoningChars,finishReason:x.finishReason,error:x.error })) };
    }
    const results = new Array(allModels.length);
    let cursor = 0;
    await Promise.all(Array.from({ length: Math.min(5, allModels.length) }, async () => {
      while (true) {
        const index = cursor++;
        if (index >= allModels.length) break;
        if (modelRaceProgress) { modelRaceProgress.activeModel=allModels[index].id; modelRaceProgress.activeProfile='starting'; }
        try { results[index] = await testModel(allModels[index]); }
        catch (error) { results[index] = { model:allModels[index].id,status:'error',workingProfile:'',testedVia:'',latencyMs:0,firstResponseMs:null,contentChars:0,reasoningChars:0,finishReason:null,preview:'',error:cleanProviderError(error),attempts:[] }; }
        if (modelRaceProgress) { modelRaceProgress.results=results.filter(Boolean); modelRaceProgress.completed=modelRaceProgress.results.length; modelRaceProgress.summary={total:allModels.length,alive:0,adaptive:0,reasoningOnly:0,empty:0,error:0}; modelRaceProgress.results.forEach(r=>{if(r.status==='alive')modelRaceProgress.summary.alive++;else if(r.status==='adaptive')modelRaceProgress.summary.adaptive++;else if(r.status==='reasoning-only')modelRaceProgress.summary.reasoningOnly++;else if(r.status==='empty')modelRaceProgress.summary.empty++;else modelRaceProgress.summary.error++;}); modelRaceProgress.events.unshift({at:new Date().toISOString(),type:'model-done',model:results[index].model,status:results[index].status,latencyMs:results[index].latencyMs,message:results[index].status.toUpperCase().replace('-',' ') }); modelRaceProgress.events=modelRaceProgress.events.slice(0,40); }
      }
    }));
    const summary = { total:results.length,alive:0,adaptive:0,reasoningOnly:0,empty:0,error:0 };
    results.forEach(r => { if(r.status==='alive')summary.alive++;else if(r.status==='adaptive')summary.adaptive++;else if(r.status==='reasoning-only')summary.reasoningOnly++;else if(r.status==='empty')summary.empty++;else summary.error++; });
    const report={ startedAt,finishedAt:new Date().toISOString(),durationMs:Date.now()-startedMs,fetchedAt:catalogue.fetchedAt,summary,results };
    modelRaceProgress.running=false; modelRaceProgress.phase='complete'; modelRaceProgress.finishedAt=report.finishedAt; modelRaceProgress.durationMs=report.durationMs; modelRaceProgress.summary=summary; modelRaceProgress.results=results; modelRaceProgress.activeModel=''; modelRaceProgress.activeProfile=''; modelRaceProgress.events.unshift({at:report.finishedAt,type:'complete',message:'Race complete.'});
    res.json(report);
  } catch (error) { if(modelRaceProgress){modelRaceProgress.running=false;modelRaceProgress.phase='failed';modelRaceProgress.error=error.message||'Model race failed.';modelRaceProgress.events.unshift({at:new Date().toISOString(),type:'error',message:modelRaceProgress.error});} res.status(error.status || 502).json({ error:error.message || 'Model race failed.',providers:error.modelResults || [] }); }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: 'Endpoint ' + req.path + ' not found', type: 'invalid_request_error', code: 404 } });
});

initializeState().then(() => {
  app.listen(PORT, () => console.log('NVIDIA NIM Control running on port ' + PORT));
}).catch(error => {
  console.error('Startup failed:', error.message);
  process.exit(1);
});
