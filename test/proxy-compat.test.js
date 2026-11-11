'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { LIGHTNING, BOLT, MINIMAX_M31_FLASH, resolveModel, buildNimRequest } = require('../model-compat');
const { sanitizeAssistantText, cleanAssistantMessage } = require('../response-sanitizer');
const { toJanitorMessage, createStreamReasoningState, mapReasoningDelta, closeStreamReasoning, THINKING_STATUS } = require('../janitor-reasoning');

test('stale Mistral Large selection redirects to the live-tested Lightning model', () => {
  const result = resolveModel('mistralai/mistral-large-4-1t-a52b', 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning');
  assert.equal(result.redirected, true);
  assert.equal(result.model, LIGHTNING);
});

test('legacy MiniMax and Nano selections redirect to the tested roleplay model', () => {
  const mini = resolveModel(MINIMAX_M31_FLASH, LIGHTNING);
  const nano = resolveModel('nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', LIGHTNING);
  assert.equal(mini.redirected, true);
  assert.equal(mini.model, LIGHTNING);
  assert.equal(nano.redirected, true);
  assert.equal(nano.model, LIGHTNING);
});

test('Lightning uses the documented zero reasoning budget when thinking is off', () => {
  const body = buildNimRequest(
    { messages: [{ role: 'user', content: 'Continue the scene.' }], max_tokens: 512 },
    LIGHTNING,
    { temperature: 0.8, top_p: 1, max_tokens: 4096, thinking_mode: false }
  );
  assert.equal(body.reasoning_budget, 0);
  assert.equal(body.max_tokens, 512);
  assert.equal(body.temperature, 1.0);
  assert.equal(body.top_p, 0.95);
  assert.equal(body.chat_template_kwargs, undefined);
});

test('Lightning permits only a small reasoning budget when the owner enables thinking', () => {
  const body = buildNimRequest(
    { messages: [{ role: 'user', content: 'Continue the scene.' }] },
    LIGHTNING,
    { thinking_mode: true }
  );
  assert.equal(body.reasoning_budget, 256);
  const capped = buildNimRequest(
    { messages: [], reasoning_budget: 16000 },
    LIGHTNING,
    { thinking_mode: true }
  );
  assert.equal(capped.reasoning_budget, 512);
});

test('Bolt keeps ordinary fast chat sampling and no unverified reasoning overrides', () => {
  const body = buildNimRequest({ messages: [] }, BOLT, { thinking_mode: false });
  assert.equal(body.temperature, 1.0);
  assert.equal(body.top_p, 0.95);
  assert.equal(body.reasoning_effort, undefined);
});

test('MiniMax M3.1 thinking is disabled by default', () => {
  const body = buildNimRequest({ messages: [{ role: 'user', content: 'Hello' }] }, MINIMAX_M31_FLASH, { thinking_mode: false });
  assert.deepEqual(body.chat_template_kwargs, { thinking_mode: 'disabled' });
  assert.equal(body.reasoning_effort, undefined);
});

test('MiniMax M3.1 agentic thinking remains disabled even when global thinking is on', () => {
  const body = buildNimRequest({ messages: [] }, MINIMAX_M31_FLASH, { thinking_mode: true });
  assert.deepEqual(body.chat_template_kwargs, { thinking_mode: 'disabled' });
});

test('SDK extra_body fields are flattened for direct NIM REST calls', () => {
  const body = buildNimRequest({
    model: MINIMAX_M31_FLASH,
    messages: [],
    extra_body: { chat_template_kwargs: { custom_flag: true } }
  }, MINIMAX_M31_FLASH, {});
  assert.equal(body.extra_body, undefined);
  assert.equal(body.chat_template_kwargs.thinking_mode, 'disabled');
  assert.equal(body.chat_template_kwargs.custom_flag, true);
});

test('tool-call/template artefacts never reach chat text', () => {
  const raw = '<mm:think>private plan</mm:think>Kira\n{{%USERTEXT}}Obedient.%USER%/%Assistant%: text}}{{{content#make-text-long-and-detailed}}<tool_call>{{!"identifier":"final"}}</tool_call>';
  const result = sanitizeAssistantText(raw);
  assert.equal(result.hadControlArtifacts, true);
  const message = cleanAssistantMessage({ role: 'assistant', content: raw });
  assert.match(message.content, /could not produce a clean reply/i);
});

test('Janitor UI chrome and leaked documentation are removed', () => {
  const result = sanitizeAssistantText('A character reply.\n\nThis document includes:\n\n• Text Generation via API\n\nWait\n\nPress button to send chat. Enter for linebreak');
  assert.equal(result.text, 'A character reply.');
});

test('structured reasoning is represented by a short status, never raw model thoughts', () => {
  const secretLikeReasoning = 'Here is a thinking process: 1. AnalyzeUserInput 2. DetermineCharacter 3. FormulateResponse';
  const result = toJanitorMessage(
    { role: 'assistant', content: 'Hello there.', reasoning_content: secretLikeReasoning },
    { showReasoning: true, cleanAssistantMessage, sanitizeAssistantText }
  );
  assert.match(result.content, new RegExp(THINKING_STATUS));
  assert.match(result.content, /Hello there\./);
  assert.equal(result.content.includes(secretLikeReasoning), false);
  assert.equal(result.reasoning_content, undefined);
});

test('collapsed prompt dumps are not exposed even when provider has reasoning metadata', () => {
  const gibberish = 'NeedcontinuationfaithfulKKiperapersona:ContextRecap-'.repeat(60);
  const result = toJanitorMessage(
    { role: 'assistant', content: 'A clean reply.', reasoning_content: gibberish },
    { showReasoning: true, cleanAssistantMessage, sanitizeAssistantText }
  );
  assert.equal(result.content.includes(gibberish), false);
  assert.match(result.content, /Reasoning received/);
});

test('generic analysis fields are never treated as visible thinking', () => {
  const result = toJanitorMessage(
    { role: 'assistant', content: 'A normal reply.', analysis: 'This is not a supported reasoning field.' },
    { showReasoning: true, cleanAssistantMessage, sanitizeAssistantText }
  );
  assert.equal(result.content, 'A normal reply.');
});

test('streaming reasoning produces one status line and never streams raw thoughts', () => {
  const state = createStreamReasoningState();
  const first = mapReasoningDelta(
    { reasoning_content: 'I should continue the scene naturally.' },
    state,
    { showReasoning: true, sanitizeAssistantText, filterContent: value => value }
  );
  const second = mapReasoningDelta(
    { reasoning_content: 'AnalyzeUserInput: do not expose this long thought.' },
    state,
    { showReasoning: true, sanitizeAssistantText, filterContent: value => value }
  );
  const third = mapReasoningDelta(
    { content: 'She smiles and steps inside.' },
    state,
    { showReasoning: true, sanitizeAssistantText, filterContent: value => value }
  );
  assert.match(first.content, /Reasoning received/);
  assert.equal(second.content, undefined);
  assert.match(third.content, /She smiles/);
  assert.equal(third.content.includes('AnalyzeUserInput'), false);
  assert.equal(closeStreamReasoning(state), '');
});

test('reasoning is never fabricated when upstream returned none', () => {
  const result = toJanitorMessage(
    { role: 'assistant', content: 'A normal reply.' },
    { showReasoning: true, cleanAssistantMessage, sanitizeAssistantText }
  );
  assert.equal(result.content, 'A normal reply.');
});
