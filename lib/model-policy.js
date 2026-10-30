'use strict';

const NEMOTRON_LIGHTNING_PREFIX = 'nvidia/nemotron-3.5-lightning-30b-a3b';
const MISTRAL_LARGE4_PREFIX = 'mistralai/mistral-large-4';
const MISTRAL_REASONING_EFFORTS = new Set(['low', 'medium', 'high', 'none']);

function isNemotronLightning(model) {
  return typeof model === 'string' &&
    model.toLowerCase().startsWith(NEMOTRON_LIGHTNING_PREFIX);
}

function isMistralLarge4(model) {
  return typeof model === 'string' && model.toLowerCase().startsWith(MISTRAL_LARGE4_PREFIX);
}

function finiteNumber(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

/**
 * Add conservative, model-specific NVIDIA NIM defaults without rewriting
 * messages or overriding explicit client-supplied settings.
 *
 * Nemotron 3.5 exposes its thinking toggle through
 * extra_body.chat_template_kwargs.enable_thinking and its internal token
 * ceiling through the top-level reasoning_budget parameter.
 */
function applyModelPolicy(body, model, settings) {
  const next = Object.assign({}, body || {});
  const config = settings || {};

  if (isMistralLarge4(model) && config.mistral_large4_enabled !== false) {
    const requestedEffort = String(config.mistral_large4_reasoning_effort || 'medium').toLowerCase();
    const effort = MISTRAL_REASONING_EFFORTS.has(requestedEffort) ? requestedEffort : 'medium';
    if (config.mistral_large4_override_reasoning_effort === true || next.reasoning_effort === undefined) {
      next.reasoning_effort = effort;
    }
  }

  if (!isNemotronLightning(model) || config.nemotron35_enabled === false) {
    return next;
  }

  const extraBody = Object.assign({}, next.extra_body || {});
  const chatTemplate = Object.assign({}, extraBody.chat_template_kwargs || {});

  if (chatTemplate.enable_thinking === undefined) {
    chatTemplate.enable_thinking = config.nemotron35_enable_thinking !== false;
  }

  if (next.reasoning_budget === undefined && chatTemplate.reasoning_budget === undefined) {
    next.reasoning_budget = finiteNumber(config.nemotron35_reasoning_budget, 2048, -1, 32768);
  }


  if (next.temperature === undefined || next.temperature === null) {
    next.temperature = finiteNumber(config.nemotron35_temperature, 1, 0, 1);
  }

  if (next.top_p === undefined || next.top_p === null) {
    next.top_p = finiteNumber(config.nemotron35_top_p, 0.95, 0, 1);
  }

  extraBody.chat_template_kwargs = chatTemplate;
  next.extra_body = extraBody;
  return next;
}

module.exports = { isNemotronLightning, isMistralLarge4, applyModelPolicy };
