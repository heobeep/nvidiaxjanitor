"""LiteLLM Proxy callback for key-specific roleplay formatting instructions."""
from __future__ import annotations

import os
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger

from rp_policy import inject_roleplay_instruction, roleplay_enabled_for_key


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _load_instructions() -> str:
    path = Path(__file__).resolve().parent / "prompts" / "roleplay_formatting.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        # Fail closed: missing prompt file must not break otherwise-valid inference.
        print(f"[roleplay-policy] Formatting instructions unavailable: {exc}")
        return ""


class RoleplayFormattingHook(CustomLogger):
    """Adds RP formatting only for opted-in keys, using LiteLLM's public hook API."""

    def __init__(self) -> None:
        super().__init__()
        self.instructions = _load_instructions()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type not in {"completion", "text_completion"}:
            return data

        if not roleplay_enabled_for_key(
            user_api_key_dict,
            global_enabled=_env_enabled("RP_FORMATTING_ENABLED", default=True),
        ):
            return data

        return inject_roleplay_instruction(
            data,
            instructions=self.instructions,
            enabled=True,
        )


roleplay_hook = RoleplayFormattingHook()
