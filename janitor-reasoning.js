'use strict';

// JanitorAI can display a short thinking indicator, but raw provider reasoning
// is hidden. Model "reasoning" often contains internal scratchpads, prompt
// summaries or malformed templates; never forward those verbatim to chat.

const THINKING_STATUS = 'Reasoning received; composing the reply.';

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
  for (const key of ['reasoning_content', 'reasoning', 'thinking']) {
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
  const hasReasoning = Boolean(extractReasoning(source));
  if (!opts.showReasoning || !hasReasoning) return clean;

  const originalContent = removeInlineThinking(source.content);
  const safeContent = opts.sanitizeAssistantText
    ? opts.sanitizeAssistantText(originalContent).text
    : originalContent;
  clean.content = '<thinking>\n' + THINKING_STATUS + '\n</thinking>' +
    (safeContent ? '\n\n' + safeContent : '');
  return clean;
}

function createStreamReasoningState() {
  return { statusShown: false };
}

function mapReasoningDelta(delta, state, options) {
  const opts = options || {};
  const next = Object.assign({}, delta || {});
  const reasoning = ['reasoning_content', 'reasoning', 'thinking']
    .map(key => asText(next[key]))
    .filter(Boolean)
    .join('');
  delete next.reasoning_content;
  delete next.reasoning;
  delete next.thinking;
  delete next.analysis;

  let visible = '';
  if (opts.showReasoning && reasoning && !state.statusShown) {
    visible += '<thinking>\n' + THINKING_STATUS + '\n</thinking>\n\n';
    state.statusShown = true;
  }

  const content = asText(next.content);
  if (content) visible += opts.filterContent ? opts.filterContent(content) : content;
  if (visible) next.content = visible;
  else if (Object.prototype.hasOwnProperty.call(next, 'content')) delete next.content;
  return next;
}

function closeStreamReasoning() {
  return '';
}

module.exports = {
  asText,
  extractReasoning,
  toJanitorMessage,
  createStreamReasoningState,
  mapReasoningDelta,
  closeStreamReasoning,
  THINKING_STATUS
};
