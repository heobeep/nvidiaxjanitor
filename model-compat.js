'use strict';

// Model-specific request normalization for NVIDIA NIM's OpenAI-compatible API.
// This proxy calls NIM with raw HTTP requests, so provider parameters must be
// sent at the top level rather than inside an SDK-only extra_body wrapper.

const ROLEPLAY_REDIRECTS = Object.freeze({
  // MiniMax M3.1 Flash is optimized for agentic/coding workflows. JanitorAI
  // roleplay prompts can cause it to emit tool/UI/template control text. Route
  // old cached Janitor selections to the verified Mistral RP default.
  'minimaxai/minimax-m3.1-flash': 'mistralai/mistral-large-4-1t-a52b'
});

const MISTRAL_LARGE_4 = 'mistralai/mistral-large-4-1t-a52b';
const MINIMAX_M31_FLASH = 'minimaxai/minimax-m3.1-flash';

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
  const modelSampling = isMistralLarge4 || isMiniMaxFlash;

  const request = Object.assign({}, source, {
    model,
    temperature: source.temperature ?? (modelSampling ? 1.0 : config.temperature ?? 0.8),
    top_p: source.top_p ?? (modelSampling ? 0.95 : config.top_p ?? 1),
    max_tokens: source.max_tokens ?? config.max_tokens ?? 4096,
    stream: Boolean(source.stream)
  });

  // extra_body is a convenience used by some SDKs. It is not unwrapped by
  // this proxy's direct axios call to NIM; flatten supported fields instead.
  delete request.extra_body;

  if (Object.keys(templateArgs).length) request.chat_template_kwargs = templateArgs;
  else delete request.chat_template_kwargs;

  if (isMistralLarge4) {
    // NVIDIA documents only low/medium/high for this model. "none" is invalid;
    // it always reasons, and those tokens count toward max_tokens.
    request.reasoning_effort = 'low';
    request.max_tokens = Math.max(Number(request.max_tokens) || 0, 8192);
    // Mistral Large 4 does not honor chat_template_kwargs.enable_thinking.
    delete request.chat_template_kwargs;
  } else if (isMiniMaxFlash) {
    // If explicitly used outside the Janitor redirect, disable agentic thinking
    // using the model's documented chat-template control.
    request.chat_template_kwargs = Object.assign({}, templateArgs, { thinking_mode: 'disabled' });
    delete request.reasoning_effort;
  } else if (config.thinking_mode && model.startsWith('openai/gpt-oss')) {
    request.reasoning_effort = source.reasoning_effort || 'high';
  }

  return request;
}

module.exports = {
  ROLEPLAY_REDIRECTS,
  resolveModel,
  buildNimRequest
};
