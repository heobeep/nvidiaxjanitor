# Version is intentionally explicit; do not replace with latest.
# Verify the matching official image signature before production rollout.
FROM ghcr.io/berriai/litellm:v1.106.4

WORKDIR /app
COPY proxy_config.yaml /app/proxy_config.yaml
COPY custom_callbacks.py /app/custom_callbacks.py
COPY rp_policy.py /app/rp_policy.py
COPY prompts/roleplay_formatting.txt /app/prompts/roleplay_formatting.txt

ENV PORT=4000
EXPOSE 4000
CMD ["--config", "/app/proxy_config.yaml", "--port", "4000"]
