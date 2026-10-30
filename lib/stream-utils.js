'use strict';

const OPEN_TAGS = ['<think>', '<thinking>'];
const CLOSE_TAGS = ['</think>', '</thinking>'];

function longestSuffixPrefix(text, tags) {
  let best = 0;
  for (const tag of tags) {
    const limit = Math.min(text.length, tag.length - 1);
    for (let length = 1; length <= limit; length++) {
      if (text.endsWith(tag.slice(0, length))) best = Math.max(best, length);
    }
  }
  return best;
}

/**
 * Stateful inline <think> tag filtering for SSE token streams. Tags can be
 * split across arbitrary network chunks, so a per-chunk regular expression
 * is not sufficient.
 */
function createThinkTagStripper() {
  let pending = '';
  let insideThink = false;

  return function stripChunk(chunk) {
    pending += String(chunk || '');
    let visible = '';

    while (pending.length) {
      const lower = pending.toLowerCase();
      if (!insideThink) {
        let found = null;
        for (const tag of OPEN_TAGS) {
          const index = lower.indexOf(tag);
          if (index >= 0 && (!found || index < found.index)) found = { index, tag };
        }
        if (found) {
          visible += pending.slice(0, found.index);
          pending = pending.slice(found.index + found.tag.length);
          insideThink = true;
          continue;
        }
        const keep = longestSuffixPrefix(lower, OPEN_TAGS);
        visible += pending.slice(0, pending.length - keep);
        pending = keep ? pending.slice(-keep) : '';
        break;
      }

      let found = null;
      for (const tag of CLOSE_TAGS) {
        const index = lower.indexOf(tag);
        if (index >= 0 && (!found || index < found.index)) found = { index, tag };
      }
      if (found) {
        pending = pending.slice(found.index + found.tag.length);
        insideThink = false;
        continue;
      }
      const keep = longestSuffixPrefix(lower, CLOSE_TAGS);
      pending = keep ? pending.slice(-keep) : '';
      break;
    }

    return visible;
  };
}

module.exports = { createThinkTagStripper };
