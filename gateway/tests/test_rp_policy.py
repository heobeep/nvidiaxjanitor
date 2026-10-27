import unittest

from rp_policy import RP_MARKER, inject_roleplay_instruction, roleplay_enabled_for_key


PROMPT = RP_MARKER + "\nUse one matched pair of asterisks around a complete action passage."


class RoleplayPolicyTests(unittest.TestCase):
    def test_janitor_alias_defaults_on(self):
        self.assertTrue(roleplay_enabled_for_key({"key_alias": "janitor"}))
        self.assertTrue(roleplay_enabled_for_key({"key_alias": "Janitor RP"}))

    def test_plain_alias_defaults_off(self):
        self.assertFalse(roleplay_enabled_for_key({"key_alias": "no-janitor"}))
        self.assertFalse(roleplay_enabled_for_key({"key_alias": "general"}))

    def test_explicit_metadata_overrides_alias(self):
        self.assertFalse(roleplay_enabled_for_key({"key_alias": "janitor", "metadata": {"rp_formatting": False}}))
        self.assertTrue(roleplay_enabled_for_key({"key_alias": "no-janitor", "metadata": {"rp_formatting": True}}))

    def test_string_metadata_is_supported(self):
        self.assertTrue(roleplay_enabled_for_key({"key_alias": "no-janitor", "metadata": '{"rp_formatting": "true"}'}))
        self.assertFalse(roleplay_enabled_for_key({"key_alias": "janitor", "metadata": '{"rp_formatting": "false"}'}))

    def test_global_switch_overrides_every_key(self):
        self.assertFalse(roleplay_enabled_for_key({"key_alias": "janitor"}, global_enabled=False))
        self.assertFalse(roleplay_enabled_for_key({"key_alias": "plain", "metadata": {"rp_formatting": True}}, global_enabled=False))

    def test_injects_into_system_context_without_changing_chat(self):
        original = {"model": "rp-model", "messages": [
            {"role": "system", "content": "Character card."},
            {"role": "user", "content": "Hello"},
        ]}
        updated = inject_roleplay_instruction(original, PROMPT)
        self.assertIn(PROMPT, updated["messages"][0]["content"])
        self.assertEqual(updated["messages"][1], original["messages"][1])
        self.assertNotIn(PROMPT, original["messages"][0]["content"])

    def test_adds_system_message_when_no_initial_system_context(self):
        updated = inject_roleplay_instruction({"messages": [{"role": "user", "content": "Hello"}]}, PROMPT)
        self.assertEqual(updated["messages"][0], {"role": "system", "content": PROMPT})
        self.assertEqual(updated["messages"][1]["content"], "Hello")

    def test_does_not_modify_plain_key_request_when_disabled(self):
        original = {"messages": [{"role": "user", "content": "Hello"}]}
        updated = inject_roleplay_instruction(original, PROMPT, enabled=False)
        self.assertEqual(updated, original)

    def test_does_not_duplicate_prompt(self):
        original = {"messages": [
            {"role": "system", "content": "Character card.\n\n" + PROMPT},
            {"role": "user", "content": "Hello"},
        ]}
        updated = inject_roleplay_instruction(original, PROMPT)
        self.assertEqual(updated["messages"][0]["content"].count(RP_MARKER), 1)


if __name__ == "__main__":
    unittest.main()
