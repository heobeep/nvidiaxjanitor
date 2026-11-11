# NVIDIA NIM → JanitorAI Proxy

An OpenAI-compatible gateway for NVIDIA NIM, with encrypted provider/client key storage, model discovery, streaming chat completions, and optional visible model reasoning.

## JanitorAI connection

Use these values in JanitorAI's custom OpenAI-compatible proxy settings:

- **Base URL:** `https://nvidiaxjanitor-production.up.railway.app/v1`
- **API key:** a client key generated in this proxy's Admin → Client keys section (not the NVIDIA API key)
- **Model:** `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

Refresh JanitorAI's model list after changing the model. Older cached requests that still name `minimaxai/minimax-m3.1-flash` or `mistralai/mistral-large-4-1t-a52b` are redirected to the tested Nemotron Nano model.

## Endpoints

- `GET /health` — public, secret-free service and reasoning configuration status
- `GET /v1/models` — live NVIDIA model catalogue, filtered to models verified by the admin model-race tool; requires a proxy client key
- `POST /v1/chat/completions` — OpenAI-compatible chat completions; supports streaming and non-streaming; requires a proxy client key

## Thinking / reasoning

The default configuration enables model thinking and displays provider-returned reasoning in a visible `<thinking>...</thinking>` block before the character's reply. This depends on the selected model actually returning reasoning data; the proxy does not fabricate it.

The proxy uses model-specific request settings. For the tested Nemotron Nano Reasoning model, it sends top-level `chat_template_kwargs.enable_thinking`. SDK-style `extra_body.chat_template_kwargs` fields are flattened because this service calls NVIDIA's REST API directly.

The admin settings API supports `show_reasoning` and `thinking_mode`. New deployments migrate the previous forced-off values once, then preserve future admin choices.

## Security

- NVIDIA provider keys stay on the server and are encrypted in the Railway persistent volume.
- JanitorAI receives only a separately generated proxy client key.
- `/health` reveals status/configuration flags only; it never returns keys.
- Do not enable verbose logging of full request bodies in a public deployment.

## Local tests

Run `npm test` to run the Node.js regression tests.
