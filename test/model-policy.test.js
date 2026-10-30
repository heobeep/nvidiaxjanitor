'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { applyModelPolicy, isNemotronLightning } = require('../lib/model-policy');

const MODEL = 'nvidia/nemotron-3.5-lightning-30b-a3b';

test('recognizes the Nemotron 3.5 Lightning model ID case-insensitively', () => {
  assert.equal(isNemotronLightning(MODEL), true);
  assert.equal(isNemotronLightning(MODEL.toUpperCase()), true);
  assert.equal(isNemotronLightning('mistralai/mistral-large-4-1t-a52b'), false);
});

test('adds bounded Nemotron defaults while preserving the messages', () => {
  const messages = [{ role: 'user', content: 'Stay in character.' }];
  const input = { model: MODEL, messages };
  const output = applyModelPolicy(input, MODEL, {});

  assert.equal(output.reasoning_budget, 2048);
  assert.equal(output.temperature, 1);
  assert.equal(output.top_p, 0.95);
  assert.equal(output.extra_body.chat_template_kwargs.enable_thinking, true);
  assert.equal(output.extra_body.repetition_penalty, 1.05);
  assert.equal(output.messages, messages);
  assert.equal(input.extra_body, undefined);
});

test('explicit client values win over Nemotron defaults', () => {
  const input = {
    messages: [{ role: 'user', content: 'test' }],
    temperature: 0.7,
    top_p: 0.9,
    reasoning_budget: 1024,
    extra_body: {
      repetition_penalty: 1.12,
      chat_template_kwargs: { enable_thinking: false }
    }
  };
  const output = applyModelPolicy(input, MODEL, {});

  assert.equal(output.temperature, 0.7);
  assert.equal(output.top_p, 0.9);
  assert.equal(output.reasoning_budget, 1024);
  assert.equal(output.extra_body.repetition_penalty, 1.12);
  assert.equal(output.extra_body.chat_template_kwargs.enable_thinking, false);
  assert.equal(input.extra_body.chat_template_kwargs.enable_thinking, false);
});

test('does not apply Nemotron settings to other models', () => {
  const input = { model: 'mistralai/mistral-large-4-1t-a52b', messages: [] };
  assert.deepEqual(applyModelPolicy(input, input.model, {}), input);
});

test('can disable automatic Nemotron tuning without changing caller payload', () => {
  const input = { model: MODEL, messages: [{ role: 'user', content: 'test' }] };
  assert.deepEqual(applyModelPolicy(input, MODEL, { nemotron35_enabled: false }), input);
});

test('clamps invalid or excessive reasoning budgets to the supported range', () => {
  const output = applyModelPolicy({ messages: [] }, MODEL, { nemotron35_reasoning_budget: 999999 });
  assert.equal(output.reasoning_budget, 32768);
});
