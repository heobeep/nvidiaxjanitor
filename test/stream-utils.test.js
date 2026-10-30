'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createThinkTagStripper } = require('../lib/stream-utils');

test('strips inline think blocks when tags arrive in separate chunks', () => {
  const strip = createThinkTagStripper();
  const chunks = ['Hello ', '<thi', 'nk>private ', 'reasoning</th', 'ink> world'];
  const output = chunks.map(strip).join('');
  assert.equal(output, 'Hello  world');
});

test('handles thinking tags, case differences, and multiple blocks', () => {
  const strip = createThinkTagStripper();
  const chunks = ['A<THINKING>x</THINKING>B<think>y</think>C'];
  assert.equal(chunks.map(strip).join(''), 'ABC');
});

test('preserves normal text and a partial opening tag prefix until more data arrives', () => {
  const strip = createThinkTagStripper();
  assert.equal(strip('Normal <thi'), 'Normal ');
  assert.equal(strip('nk>hidden</think> visible'), ' visible');
});
