'use strict';

// Convert explicitly structured provider reasoning to Janitor's optional
// <thinking> block. Reasoning formats are model-specific; never treat a generic
// "analysis" field as reasoning, and never let a runaway trace flood the chat.

const MAX_REASONING_CHARS = 800;

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

function cleanReasoningText(value, sanitizeAssistantText) {
  const raw = asText(value).trim();
  if (!raw || raw.length > MAX_REASONING_CHARS) return '';
  // This catches collapsed prompt dumps like the one that was flooding Janitor.
  if (raw.split(/\s+/).some(token => token.length > 40)) return '';
  const words = raw.split(/\s+/).filter(Boolean);
  if (words.length > 12) {
    const averageWordLength = words.reduce((sum, word) => sum + word.length, 0) / words.length;
    if (averageWordLength > 16) return '';
  }
  const cleaned = typeof sanitizeAssistantText === 'function'
    ? sanitizeAssistantText(raw)
    : { text: raw, hadControlArtifacts: false };
  if (!cleaned || cleaned.hadControlArtifacts) return '';
  const text = asText(cleaned.text).trim();
  if (!text || text.length > MAX_REASONING_CHARS) return '';
  return text;
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
  if (!opts.showReasoning) return clean;

  const reasoning = cleanReasoningText(extractReasoning(source), opts.sanitizeAssistantText);
  if (!reasoning) return clean;

  const originalContent = removeInlineThinking(source.content);
  const safeContent = opts.sanitizeAssistantText
    ? opts.sanitizeAssistantText(originalContent).text
    : originalContent;
  clean.content = '<thinking>\n' + reasoning + '\n</thinking>' +
    (safeContent ? '\n\n' + safeContent : '');
  return clean;
}

function createStreamReasoningState() {
  return { thinkingOpen: false, reasoningChars: 0 };
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
  if (opts.showReasoning && reasoning && state.reasoningChars < MAX_REASONING_CHARS) {
    let safeReasoning = cleanReasoningText(reasoning, opts.sanitizeAssistantText);
    const remaining = MAX_REASONING_CHARS - state.reasoningChars;
    if (safeReasoning.length > remaining) safeReasoning = safeReasoning.slice(0, remaining);
    if (safeReasoning) {
      if (!state.thinkingOpen) visible += '<thinking>\n';
      visible += safeReasoning;
      state.thinkingOpen = true;
      state.reasoningChars += safeReasoning.length;
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
  cleanReasoningText,
  extractReasoning,
  toJanitorMessage,
  createStreamReasoningState,
  mapReasoningDelta,
  closeStreamReasoning,
  MAX_REASONING_CHARS
};
