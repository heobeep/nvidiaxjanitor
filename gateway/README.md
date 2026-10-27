# Staging gateway migration

This directory is an isolated staging option. It does not replace the current Node proxy unless the Railway service is deliberately reconfigured to use this directory. The stable entry point is LiteLLM Proxy; our custom code is deliberately limited to the optional per-key roleplay formatting instruction.

## What this gives you

- LiteLLM's OpenAI-compatible /v1/chat/completions and /v1/models routes.
- Built-in Admin UI at /ui for provider credentials, model deployments, virtual keys, rate limits, and usage.
- Multiple NVIDIA NIM credentials and deployments under the same public model name. LiteLLM's router can retry/fail over from a deployment that is in cooldown.
- Optional roleplay formatting policy injected only for opted-in virtual keys.
- A separate no-janitor key profile with no injected roleplay instruction.

## Version and supply-chain caution

The image is pinned to ghcr.io/berriai/litellm:v1.106.4; do not change it to latest. LiteLLM reported a compromised v1.107.0 artifact on October 21, 2026, so this staging image and every upgrade must be verified before production rollout. For v1.106.4, use the official release's cosign key and command shown on the release page: https://github.com/BerriAI/litellm/releases/tag/v1.106.4. This repository contains no provider API keys, master key, database password, salt key, or user virtual keys.

## Railway resources and required services

The existing Node proxy currently has a 1 GB memory limit and used about 66 MB average / 101 MB peak in the past 24 hours. That is ample for Node, but not enough evidence to safely co-host LiteLLM and PostgreSQL. Current Railway templates report roughly 0.9 GB idle for LiteLLM alone. For staging, create a separate LiteLLM service with about 2 GB RAM and a PostgreSQL service; do not resize or replace the existing service until it passes the acceptance tests. A single LiteLLM replica can start without Redis; Redis is a later addition if you need shared cooldown/rate-limit state across replicas.

## Required secrets (set in Railway, never commit them)

- LITELLM_MASTER_KEY: a newly generated, high-entropy admin/root key.
- LITELLM_SALT_KEY: a high-entropy key used to encrypt provider credentials stored in the database. Preserve it securely; changing it can make stored provider credentials unreadable.
- DATABASE_URL: PostgreSQL connection URL, supplied by Railway's private service-reference variable.
- UI_USERNAME and UI_PASSWORD: a unique admin login with a strong password.
- RP_FORMATTING_ENABLED: optional global switch (true by default; set false to disable the roleplay instruction for every key).

Use the LiteLLM UI or management API to add NVIDIA NIM credentials. Never paste your NVIDIA key into JanitorAI; JanitorAI receives only a LiteLLM virtual key. Do not give the master key to JanitorAI.

## First-run setup in the UI

1. Open /ui and sign in with the configured UI username/password.
2. Add one NVIDIA NIM credential per provider key, named clearly (NIM free key 01, NIM free key 02, etc.). Keep the actual values in LiteLLM's encrypted credential store, not in Git or the prompt policy.
3. Add the first NVIDIA NIM model. For example, the provider model string for Nemotron 3.5 Lightning is nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b. Give the public model group a stable name such as nemotron-fast.
4. Add the same public model group again using the next NVIDIA NIM credential, and repeat for each key. Keeping the public model name identical lets the router treat the credential-backed entries as deployments in one pool. Use the UI's Test Connection before saving each entry.
5. Repeat the model group setup for other upstream models (for example a reasoning-focused Nemotron model or openai/gpt-oss-20b, if your NVIDIA account currently exposes them). Use honest names; do not label a Nemotron model as Claude or GPT-4.
6. Create a virtual key with alias janitor, restrict it to the model groups JanitorAI should use, and set metadata {"rp_formatting": true} if the UI offers the metadata field. If metadata is left empty, alias janitor opts in by default.
7. Create another key with alias no-janitor and metadata {"rp_formatting": false}. It receives no injected roleplay formatting text. Other aliases are off by default unless metadata explicitly opts them in.
8. Paste only the respective virtual key into the relevant client configuration. Use /v1 as the OpenAI-compatible base URL, according to the client's base-URL conventions.

## How provider key rotation works (and its limit)

For each public model group, add one deployment per NVIDIA NIM credential. LiteLLM handles routing, retries and deployment cooldowns; this avoids custom HTTP key-rotation code. The configured router uses two retries, cools down after one failure, and sets a 300-second cooldown. A 429 or other provider error can therefore move the request to another healthy deployment.

A cooldown is temporary, not permanent quota-exhaustion detection. If NVIDIA reports a key as out of credits for the whole day, use the Admin UI to disable that deployment until the provider quota resets. We should only add permanent auto-quarantine after testing the exact NVIDIA error shapes, because generic 429s may be short-lived rate limits rather than exhausted daily quota.

## Roleplay text formatting

prompts/roleplay_formatting.txt is the separately versioned prompt instruction. It asks for a matched pair of asterisks around a complete narration passage (for example, *She falls silent for a moment.*), never an asterisk before every word (for example, *She *falls *silent). It does not rewrite model output after generation.

The screenshot alone cannot prove whether the model emitted the literal asterisks or the client/parser introduced them. The current Node proxy only strips <think> tags; it does not insert asterisks. To diagnose conclusively, compare a sanitized upstream response with the gateway's downstream response using a test prompt, without recording private roleplay histories. The extra instruction is a mitigation, not proof of the root cause.

## Tests

Run the dependency-free policy tests with:

    cd gateway
    python -m unittest discover -s tests -v
    python -m py_compile rp_policy.py custom_callbacks.py

These tests cover profile selection, explicit metadata overrides, global toggle, prompt placement, idempotence, and preservation of the original request. They are unit tests for the custom policy only—not a substitute for live integration tests against the exact LiteLLM image, PostgreSQL configuration, NVIDIA keys, SSE streaming, and JanitorAI client.

## Production cutover checklist

- Verify the upstream image signature and exact version.
- Deploy this to a separate Railway service and database first; leave nvidiaxjanitor untouched.
- Confirm /health/liveliness, /v1/models, virtual-key authentication, the admin UI, provider credential encryption, and one real chat request.
- Test streaming output, final SSE chunk, 429/cooldown failover across at least two NIM keys, and no-janitor key behavior.
- Confirm the master key is never installed in JanitorAI and the UI is password-protected.
- Only then decide whether to migrate the client URL. Keep the old Node service available as rollback.
