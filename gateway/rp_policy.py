"""Small, dependency-free policy helpers for key-specific roleplay formatting."""
from __future__ import annotations

import json
import re
from copy import copy
from typing import Any, Mapping

RP_MARKER = "[Gateway roleplay formatting policy v1]"
DEFAULT_RP_KEY_ALIASES = {"janitor", "janitor-rp"}


def _read_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _metadata_for(auth: Any) -> dict[str, Any]:
    value = _read_field(auth, "metadata", {})
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "on", "enabled"}:
            return True
        if cleaned in {"false", "0", "no", "off", "disabled"}:
            return False
    return None


def roleplay_enabled_for_key(auth: Any, global_enabled: bool = True) -> bool:
    """Use metadata override first, then the janitor key-alias convention.

    Set key metadata to {"rp_formatting": true/false} for an explicit per-key
    override. Without that field, only aliases janitor and janitor-rp opt in.
    The global switch can always disable the feature for every key.
    """
    if not global_enabled:
        return False

    metadata = _metadata_for(auth)
    override = _explicit_bool(metadata.get("rp_formatting"))
    if override is not None:
        return override

    alias = _read_field(auth, "key_alias", "")
    if not isinstance(alias, str):
        alias = ""
    normalized = re.sub(r"[^a-z0-9]+", "-", alias.strip().lower()).strip("-")
    return normalized in DEFAULT_RP_KEY_ALIASES


def inject_roleplay_instruction(
    request_data: Mapping[str, Any], instructions: str, enabled: bool = True
) -> dict[str, Any]:
    """Return a copied request with one formatting instruction in its system context.

    Existing user/assistant content is left unchanged. The instruction is added
    only once, and is placed at the end of the leading system/developer context
    so it can refine formatting without replacing the character prompt.
    """
    result = dict(request_data)
    if not enabled or not instructions.strip():
        return result

    original_messages = request_data.get("messages")
    if not isinstance(original_messages, list) or not original_messages:
        return result

    messages = [copy(message) if isinstance(message, dict) else message for message in original_messages]
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            if RP_MARKER in message["content"]:
                result["messages"] = messages
                return result

    directive = instructions.strip()
    leading_context_count = 0
    while leading_context_count < len(messages):
        message = messages[leading_context_count]
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            break
        leading_context_count += 1

    if leading_context_count:
        last_context_index = leading_context_count - 1
        last_context = messages[last_context_index]
        existing_content = last_context.get("content")
        if isinstance(existing_content, str):
            last_context["content"] = f"{existing_content.rstrip()}\n\n{directive}"
        else:
            messages.insert(leading_context_count, {"role": "system", "content": directive})
    else:
        messages.insert(0, {"role": "system", "content": directive})

    result["messages"] = messages
    return result
