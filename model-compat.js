'use strict';

// Model-specific request normalization for NVIDIA NIM's OpenAI-compatible API.
// NIM is called through raw HTTP, so provider-specific fields must be top-level.

const LIGHTNING = 'nvidia/nemotron-3.5-lightning-30b-a3b';
const BOLT = 'nvidia/nemotron-3.5-bolt-e4b';
const NANO_OMNI = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning';
const MINIMAX_M31_FLASH = 'minimaxai/minimax-m3.1-flash';
const MISTRAL_LARGE_4 = 'mistralai/mistral-large-4-1t-a52b';

const ROLEPLAY_REDIRECTS = Object.freeze({
  // Mistral Large 4 repeatedly timed out on this user's NIM endpoint.
  // Preserve cached Janitor settings but route them to the live-tested model.
  [MISTRAL_LARGE_4]: LIGHTNING
});

const ROLEPLAY_FALLBACK_MODELS = Object.freeze([LIGHTNING, BOLT, MINIMAX_M31_FLASH]);

function resolveModel(requestedModel, defaultModel) {
  const requested = (typeof requestedModel === 'string' && requestedModel.trim())
    ? requestedModel.trim()
    : defaultModel;
  return {
    requestedModel: requested,
    model: ROLEPLAY_REDIRECTS[requested] || requested,
    redirected: Boolean(ROLEPLAY_REDIRECTS[requested])
  };
}

function buildNimRequest(body, model, settings) {
  const source = body && typeof body === 'object' ? body : {};
  const config = settings || {};
  const incomingExtras = source.extra_body && typeof source.extra_body === 'object'
    ? source.extra_body
    : {};
  const templateArgs = Object.assign(
    {},
    source.chat_template_kwargs && typeof source.chat_template_kwargs === 'object' ? source.chat_template_kwargs : {},
    incomingExtras.chat_template_kwargs && typeof incomingExtras.chat_template_kwargs === 'object' ? incomingExtras.chat_template_kwargs : {}
  );

  const isMistralLarge4 = model === MISTRAL_LARGE_4;
  const isMiniMaxFlash = model === MINIMAX_M31_FLASH;
  const isLightning = model === LIGHTNING;
  const isBolt = model === BOLT;
  const modelSampling = isMistralLarge4 || isMiniMaxFlash || isLightning || isBolt;

  const request = Object.assign({}, source, {
    model,
    temperature: source.temperature ?? (modelSampling ? 1.0 : config.temperature ?? 0.8),
    top_p: source.top_p ?? (modelSampling ? 0.95 : config.top_p ?? 1),
    max_tokens: source.max_tokens ?? config.max_tokens ?? 4096,
    stream: Boolean(source.stream)
  });

  // extra_body is an SDK convention, not a NIM REST field.
  delete request.extra_body;
  if (Object.keys(templateArgs).length) request.chat_template_kwargs = templateArgs;
  else delete request.chat_template_kwargs;

  if (isMistralLarge4) {
    request.reasoning_effort = 'low';
    delete request.chat_template_kwargs;
  } else if (isLightning) {
    // NVIDIA documents reasoning_budget as the control for this model.
    // A zero budget keeps hidden planning from consuming the RP reply budget.
    if (config.thinking_mode === true) {
      const supplied = Number(source.reasoning_budget);
      const budget = Number.isFinite(supplied) ? supplied : 256;
      request.reasoning_budget = Math.max(0, Math.min(512, budget));
    } else {
      request.reasoning_budget = 0;
    }
    delete request.reasoning_effort;
    delete request.chat_template_kwargs;
  } else if (model === NANO_OMNI) {
    request.chat_template_kwargs = Object.assign({}, templateArgs, {
      enable_thinking: config.thinking_mode === true
    });
    delete request.reasoning_effort;
  } else if (isMiniMaxFlash) {
    // Preserve the M3.1 model as an alternative, but disable agentic tool-planning
    // by default so it cannot spill its tool/control syntax into roleplay.
    request.chat_template_kwargs = Object.assign({}, templateArgs, {
      thinking_mode: config.thinking_mode === true ? 'enabled' : 'disabled'
    });
    delete request.reasoning_effort;
  } else if (isBolt) {
    delete request.reasoning_effort;
    // Bolt was clean on the roleplay test without a reasoning/template override.
  } else if (config.thinking_mode === true && model.startsWith('openai/gpt-oss')) {
    request.reasoning_effort = source.reasoning_effort || 'high';
  }

  return request;
}

module.exports = {
  LIGHTNING,
  BOLT,
  NANO_OMNI,
  MINIMAX_M31_FLASH,
  MISTRAL_LARGE_4,
  ROLEPLAY_REDIRECTS,
  ROLEPLAY_FALLBACK_MODELS,
  resolveModel,
  buildNimRequest
};
