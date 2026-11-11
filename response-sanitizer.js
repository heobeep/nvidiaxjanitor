'use strict';

// Output hygiene for model-native reasoning/control artifacts. This is
// deliberately conservative: valid prose and structured OpenAI tool_calls
// are preserved; raw tool-call text and prompt-template markers are not.

const REASONING_KEYS = [
  'reasoning', 'reasoning_content', 'reasoning_details',
  'reasoning_summary', 'thinking', 'analysis', 'chain_of_thought'
];

function sanitizeAssistantText(value) {
  if (typeof value !== 'string' || !value) {
    return { text: value, hadControlArtifacts: false, severe: false };
  }

  const raw = value;
  const hadControlArtifacts =
    /<tool_call\b|<\/tool_call>|<mm:think>|<\/mm:think>|<\|(?:analysis|reasoning|thought|scratchpad|tool_call|tool_calls|im_start|im_end)\|>/i.test(raw) ||
    /\{\{(?:%|!|#|\{)/.test(raw) ||
    /%(?:USER|USERTEXT|ASSISTANT)%/i.test(raw) ||
    /Press button to send chat|Enter for linebreak|\[No tools\]\s*\[TO X POSTS/i.test(raw) ||
    /This document includes:\s*Text Generation via API/i.test(raw);

  let text = raw
    // Private reasoning may be inline for some NVIDIA model templates.
    .replace(/<mm:think>[\s\S]*?<\/mm:think>/gi, '')
    .replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '')
    .replace(/<\|(?:analysis|reasoning|thought|scratchpad)\|>[\s\S]*?(?=<\|(?:final|answer)\|>|$)/gi, '')
    .replace(/<\|(?:final|answer)\|>/gi, '')
    // A model occasionally serializes a tool call as visible text instead of
    // returning OpenAI's structured tool_calls field. Never execute it here.
    .replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/gi, '')
    .replace(/<tool_call\b[^>]*>[\s\S]*$/gi, '')
    // Remove broken template/control placeholders leaked by agentic models.
    .replace(/\{\{\{[\s\S]*?\}\}\}/g, '')
    .replace(/\{\{\{[\s\S]*$/g, '')
    .replace(/\{\{[%!#][\s\S]*?\}\}/g, '')
    .replace(/\{\{[%!#][\s\S]*$/g, '')
    .replace(/%(?:USER|USERTEXT|ASSISTANT)%[\s\S]*?(?:\}\}|\n|$)/gi, '')
    // Strip model token spellings, not ordinary angle-bracket prose.
    .replace(/<\|[a-z0-9_:-]+\|>/gi, '')
    .replace(/<\/?(?:s|bos|eos|pad)>/gi, '')
    // Janitor/UI chrome and leaked documentation are not chat prose.
    .replace(/(?:Press button to send chat|Enter for linebreak)[\s\S]*$/i, '')
    .replace(/\[No tools\]\s*\[TO X POSTS[\s\S]*$/i, '')
    .replace(/This document includes:\s*Text Generation via API[\s\S]*$/i, '')
    .replace(/\[The final refusal message contains system prompts\][\s\S]*$/i, '')
    .replace(/\[\s*\{\s*\"file_name\"[\s\S]*$/i, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const severe =
    /<tool_call\b|<\/tool_call>|<mm:think>|<\/mm:think>|<\|(?:analysis|reasoning|thought|scratchpad|tool_call|tool_calls|im_start|im_end)\|>/i.test(text) ||
    /\{\{(?:%|!|#|\{)/.test(text) ||
    /%(?:USER|USERTEXT|ASSISTANT)%/i.test(text) ||
    /Press button to send chat|Enter for linebreak|\[No tools\]\s*\[TO X POSTS/i.test(text);

  return { text, hadControlArtifacts, severe };
}

function cleanAssistantMessage(message) {
  const clean = Object.assign({}, message || {});
  for (const key of REASONING_KEYS) delete clean[key];

  if (typeof clean.content === 'string') {
    const result = sanitizeAssistantText(clean.content);
    clean.content = (result.hadControlArtifacts && result.text.length < 24)
      ? 'I could not produce a clean reply this turn. Please regenerate or choose another model.'
      : result.text;
  } else if (Array.isArray(clean.content)) {
    clean.content = clean.content.map(part => {
      if (!part || typeof part !== 'object') return part;
      const next = Object.assign({}, part);
      if (typeof next.text === 'string') next.text = sanitizeAssistantText(next.text).text;
      return next;
    });
  }
  return clean;
}

module.exports = { REASONING_KEYS, sanitizeAssistantText, cleanAssistantMessage };
