// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const PROXY_API_KEY = process.env.PROXY_API_KEY;

// Protect OpenAI-compatible endpoints so only clients with the proxy key can use the NVIDIA quota.
app.use('/v1', (req, res, next) => {
  if (!PROXY_API_KEY) {
    return res.status(503).json({
      error: {
        message: 'Proxy authentication is not configured',
        type: 'server_error',
        code: 503
      }
    });
  }

  const authorization = req.get('authorization') || '';
  const suppliedKey = authorization.match(/^Bearer\s+(.+)$/i)?.[1];
  if (!suppliedKey || suppliedKey !== PROXY_API_KEY) {
    return res.status(401).json({
      error: {
        message: 'Invalid proxy API key',
        type: 'authentication_error',
        code: 401
      }
    });
  }
  next();
});

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// Strip any inline  thinking... response tags a model may emit inside content.
function stripThinkTags(text, trim = true) {
  if (!text) return text;
  const result = text
    .replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '')
    .replace(/<\/?think(?:ing)?>/gi, '');
  // Streaming chunks must preserve leading/trailing spaces; trimming each chunk
  // can concatenate adjacent words when JanitorAI joins the deltas.
  return trim ? result.trim() : result;
}

// Pull inline think-tag reasoning out of content when no reasoning_content field exists.
function extractThinkTags(text) {
  if (!text) return null;
  const m = text.match(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/i);
  return m ? m[1].trim() : null;
}




// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models exposed by NVIDIA NIM, preserving the provider's actual model IDs.
app.get('/v1/models', async (req, res) => {
  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}` }
    });
    res.json(response.data);
  } catch (error) {
    console.error('Models list error:', error.message);
    res.status(error.response?.status || 502).json(error.response?.data || {
      error: {
        message: 'Unable to retrieve models from NVIDIA NIM',
        type: 'upstream_error',
        code: error.response?.status || 502
      }
    });
  }
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Forward JanitorAI's selected model ID exactly as supplied.
    if (typeof model !== 'string' || !model.trim()) {
      return res.status(400).json({
        error: {
          message: 'A model ID is required. Choose a model exposed by GET /v1/models.',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }
    const nimModel = model;

    // Transform OpenAI request to NIM format; preserve JanitorAI parameters not explicitly adjusted here.
    const nimRequest = {
      ...req.body,
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      extra_body: (ENABLE_THINKING_MODE && nimModel.startsWith('openai/gpt-oss')) ? { chat_template_kwargs: { thinking: true } } : req.body.extra_body,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const d = data.choices[0].delta;
                const reasoning = d.reasoning_content;
                const rawContent = d.content || '';

                // Always keep reasoning out of the visible content channel.
                if (rawContent) {
                  d.content = stripThinkTags(rawContent, false);
                } else {
                  d.content = '';
                }

                if (SHOW_REASONING && reasoning) {
                  d.reasoning_content = reasoning;
                } else {
                  delete d.reasoning_content;
                }

                // Drop empty content deltas so consumers don't see stray blanks.
                if (!d.content) delete d.content;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          const rawContent = choice.message?.content || '';
          const content = stripThinkTags(rawContent);
          const reasoning =
            choice.message?.reasoning_content || extractThinkTags(rawContent);
          const msg = {
            role: choice.message.role,
            content: content
          };
          if (SHOW_REASONING && reasoning) {
            msg.reasoning_content = reasoning;
          }
          return {
            index: choice.index,
            message: msg,
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
