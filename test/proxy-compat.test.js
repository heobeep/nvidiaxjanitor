'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { resolveModel, buildNimRequest } = require('../model-compat');
const { sanitizeAssistantText, cleanAssistantMessage } = require('../response-sanitizer');
const { toJanitorMessage, createStreamReasoningState, mapReasoningDelta, closeStreamReasoning } = require('../janitor-reasoning');

test('legacy MiniMax M3.1 Janitor selection is redirected to Mistral Large 4.1', () => {
  const result = resolveModel('minimaxai/minimax-m3.1-flash', 'mistralai/mistral-large-4-1t-a52b');
  assert.equal(result.redirected, true);
  assert.equal(result.model, 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning');
});

test('Mistral Large uses supported low reasoning and enough output budget', () => {
  const body = buildNimRequest(
    { messages: [{ role: 'user', content: 'Continue the scene.' }], max_tokens: 512 },
    'mistralai/mistral-large-4-1t-a52b',
    { temperature: 0.8, top_p: 1, max_tokens: 4096 }
  );
  assert.equal(body.reasoning_effort, 'low');
  assert.equal(body.max_tokens, 512);
  assert.equal(body.temperature, 1.0);
  assert.equal(body.top_p, 0.95);
  assert.equal(body.chat_template_kwargs, undefined);
});

test('MiniMax M3.1 reasoning is explicitly disabled when directly invoked', () => {
  const body = buildNimRequest(
    { messages: [{ role: 'user', content: 'Hello' }] },
    'minimaxai/minimax-m3.1-flash',
    {}
  );
  assert.deepEqual(body.chat_template_kwargs, { thinking_mode: 'disabled' });
  assert.equal(body.reasoning_effort, undefined);
});

test('SDK extra_body fields are flattened for direct NIM REST calls', () => {
  const body = buildNimRequest({
    model: 'minimaxai/minimax-m3.1-flash',
    messages: [],
    extra_body: { chat_template_kwargs: { custom_flag: true } }
  }, 'minimaxai/minimax-m3.1-flash', {});
  assert.equal(body.extra_body, undefined);
  assert.equal(body.chat_template_kwargs.thinking_mode, 'disabled');
  assert.equal(body.chat_template_kwargs.custom_flag, true);
});

test('MiniMax thinking and raw tool-call/template artifacts never reach chat text', () => {
  const raw =
    '<mm:think>private plan</mm:think>' +
    'Kira\n{{%USERTEXT}}Obedient.%USER%/%Assistant%: text}}{{{content#make-text-long-and-detailed}}' +
    '<tool_call>{{!"identifier":"final"}}</tool_call>' +
    'The final refusal message contains system prompts';
  const result = sanitizeAssistantText(raw);
  assert.equal(result.hadControlArtifacts, true);
  const message = cleanAssistantMessage({ role: 'assistant', content: raw });
  assert.match(message.content, /could not produce a clean reply/i);
});

test('Janitor UI chrome and leaked documentation are removed', () => {
  const result = sanitizeAssistantText(
    'A character reply.\n\nThis document includes:\n\n• Text Generation via API\n\nWait\n\nPress button to send chat. Enter for linebreak'
  );
  assert.equal(result.text, 'A character reply.');
});


test('Nemotron Nano Reasoning receives its tested thinking parameter and keeps Janitor token budget', () => {
  const body = buildNimRequest(
    { messages: [{ role: 'user', content: 'Continue the scene.' }], max_tokens: 768 },
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
    { temperature: 0.8, top_p: 0.95, max_tokens: 2048, thinking_mode: true }
  );
  assert.deepEqual(body.chat_template_kwargs, { enable_thinking: true });
  assert.equal(body.max_tokens, 768);
});

test('structured NIM reasoning becomes visible Janitor <thinking> content', () => {
  const result = toJanitorMessage(
    { role: 'assistant', content: 'Hello there.', reasoning_content: 'I should answer warmly.' },
    { showReasoning: true, cleanAssistantMessage, sanitizeAssistantText }
  );
  assert.match(result.content, /^<thinking>\nI should answer warmly\.\n<\/thinking>/);
  assert.match(result.content, /Hello there\./);
  assert.equal(result.reasoning_content, undefined);
});

test('streamed reasoning opens and closes around the visible reply', () => {
  const state = createStreamReasoningState();
  const first = mapReasoningDelta(
    { reasoning_content: 'I should keep the scene moving.' },
    state,
    { showReasoning: true, sanitizeAssistantText, filterContent: value => value }
  );
  const second = mapReasoningDelta(
    { content: 'She smiles and steps inside.' },
    state,
    { showReasoning: true, sanitizeAssistantText, filterContent: value => value }
  );
  assert.equal(first.content, '<thinking>\nI should keep the scene moving.');
  assert.equal(second.content, '\n</thinking>\n\nShe smiles and steps inside.');
  assert.equal(closeStreamReasoning(state), '');
});

test('stream reasoning is closed cleanly if the model ends before visible content', () => {
  const state = createStreamReasoningState();
  mapReasoningDelta(
    { reasoning_content: 'Planning reply.' },
    state,
    { showReasoning: true, sanitizeAssistantText }
  );
  assert.equal(closeStreamReasoning(state), '\n</thinking>\n');
});

test('reasoning is never fabricated when upstream returned none', () => {
  const result = toJanitorMessage(
    { role: 'assistant', content: 'A normal reply.' },
    { showReasoning: true, cleanAssistantMessage, sanitizeAssistantText }
  );
  assert.equal(result.content, 'A normal reply.');
});
