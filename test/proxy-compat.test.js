'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { resolveModel, buildNimRequest } = require('../model-compat');
const { sanitizeAssistantText, cleanAssistantMessage } = require('../response-sanitizer');

test('legacy MiniMax M3.1 Janitor selection is redirected to Mistral Large 4.1', () => {
  const result = resolveModel('minimaxai/minimax-m3.1-flash', 'mistralai/mistral-large-4-1t-a52b');
  assert.equal(result.redirected, true);
  assert.equal(result.model, 'mistralai/mistral-large-4-1t-a52b');
});

test('Mistral Large uses supported low reasoning and enough output budget', () => {
  const body = buildNimRequest(
    { messages: [{ role: 'user', content: 'Continue the scene.' }], max_tokens: 512 },
    'mistralai/mistral-large-4-1t-a52b',
    { temperature: 0.8, top_p: 1, max_tokens: 4096 }
  );
  assert.equal(body.reasoning_effort, 'low');
  assert.equal(body.max_tokens, 8192);
  assert.equal(body.temperature, 0.8);
  assert.equal(body.top_p, 1);
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
