'use strict';

// JanitorAI consumes OpenAI chat-completion content. Several NVIDIA NIM
// models return reasoning in separate fields, which JanitorAI may ignore.
// Convert that provider field into a visible <thinking> block only when the
// owner has enabled SHOW_REASONING. Keep the actual character reply separate.

function asText(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    return value.map(part => {
      if (typeof part === 'string') return part;
      if (part && typeof part.text === 'string') return part.text;
      if (part && typeof part.content === 'string') return part.content;
      return '';
    }).join('');
  }
  return '';
}

function extractReasoning(message) {
  const source = message && typeof message === 'object' ? message : {};
  for (const key of ['reasoning_content', 'reasoning', 'thinking', 'analysis']) {
    const text = asText(source[key]).trim();
    if (text) return text;
  }
  const content = asText(source.content);
  const match = content.match(/<(?:mm:think|thinking|think)>([\s\S]*?)<\/(?:mm:think|thinking|think)>/i);
  return match ? match[1].trim() : '';
}

function removeInlineThinking(content) {
  return asText(content)
    .replace(/<mm:think>[\s\S]*?<\/mm:think>/gi, '')
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .trim();
}

function toJanitorMessage(message, options) {
  const opts = options || {};
  const source = message && typeof message === 'object' ? message : {};
  const clean = opts.cleanAssistantMessage ? opts.cleanAssistantMessage(source) : Object.assign({}, source);
  if (!opts.showReasoning) return clean;

  const reasoning = extractReasoning(source);
  const safeReasoning = reasoning && opts.sanitizeAssistantText
    ? opts.sanitizeAssistantText(reasoning).text
    : reasoning;
  if (!safeReasoning) return clean;

  const originalContent = removeInlineThinking(source.content);
  const safeContent = opts.sanitizeAssistantText
    ? opts.sanitizeAssistantText(originalContent).text
    : originalContent;
  clean.content = '<thinking>\n' + safeReasoning + '\n</thinking>' +
    (safeContent ? '\n\n' + safeContent : '');
  return clean;
}

function createStreamReasoningState() {
  return { thinkingOpen: false };
}

function mapReasoningDelta(delta, state, options) {
  const opts = options || {};
  const next = Object.assign({}, delta || {});
  const reasoning = ['reasoning_content', 'reasoning', 'thinking', 'analysis']
    .map(key => asText(next[key]))
    .filter(Boolean)
    .join('');
  delete next.reasoning_content;
  delete next.reasoning;
  delete next.thinking;
  delete next.analysis;

  let visible = '';
  if (opts.showReasoning && reasoning) {
    const safeReasoning = opts.sanitizeAssistantText
      ? opts.sanitizeAssistantText(reasoning).text
      : reasoning;
    if (safeReasoning) {
      if (!state.thinkingOpen) visible += '<thinking>\n';
      visible += safeReasoning;
      state.thinkingOpen = true;
    }
  }

  const content = asText(next.content);
  if (content) {
    if (state.thinkingOpen) {
      visible += '\n</thinking>\n\n';
      state.thinkingOpen = false;
    }
    visible += opts.filterContent ? opts.filterContent(content) : content;
  }
  if (visible) next.content = visible;
  else if (Object.prototype.hasOwnProperty.call(next, 'content')) delete next.content;
  return next;
}

function closeStreamReasoning(state) {
  if (!state || !state.thinkingOpen) return '';
  state.thinkingOpen = false;
  return '\n</thinking>\n';
}

module.exports = {
  asText,
  extractReasoning,
  toJanitorMessage,
  createStreamReasoningState,
  mapReasoningDelta,
  closeStreamReasoning
};
