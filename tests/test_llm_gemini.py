import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm_gemini


class GeminiTokenBudgetTests(unittest.TestCase):
    def test_pooled_chat_falls_back_to_openai_compatible_provider(self):
        calls = []

        def limited(*_args, **_kwargs):
            calls.append("openrouter")
            error = RuntimeError("OpenRouter HTTP 402")
            error.status_code = 402
            raise error

        def fallback(*_args, **_kwargs):
            calls.append("fallback")
            return "fallback-result"

        with (
            patch.object(llm_gemini, "GEMINI_API_KEY", ""),
            patch.object(llm_gemini, "OPENROUTER_API_KEY", "or-test"),
            patch.object(llm_gemini, "FALLBACK_API_KEY", "fallback-test"),
            patch.object(llm_gemini, "_openrouter_chat_once", side_effect=limited),
            patch.object(llm_gemini, "_fallback_chat_once", side_effect=fallback),
        ):
            result = llm_gemini.pooled_llm_chat(
                [{"role": "user", "content": "test"}], timeout=1
            )

        self.assertEqual(result, "fallback-result")
        self.assertEqual(calls, ["openrouter", "fallback"])

    def test_llm_chat_falls_back_to_openrouter_after_gemini_failure(self):
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"choices": [{"message": {"content": "fallback"}}]},
        )
        with (
            patch.object(llm_gemini, "GEMINI_API_KEY", "AIza-test"),
            patch.object(llm_gemini, "OPENROUTER_API_KEY", "or-test"),
            patch.object(llm_gemini, "OPENROUTER_RETRIES", 1),
            patch.object(llm_gemini, "gemini_chat", side_effect=RuntimeError("HTTP 503")),
            patch.object(llm_gemini.requests, "post", return_value=response) as post,
        ):
            result = llm_gemini.llm_chat([{"role": "user", "content": "test"}])

        self.assertEqual(result, "fallback")
        self.assertEqual(post.call_args.kwargs["json"]["messages"][0]["content"], "test")

    def test_default_output_tokens_use_deployment_budget(self):
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "candidates": [{
                    "content": {"parts": [{"text": '{"ok": true}'}]},
                    "finishReason": "STOP",
                }]
            },
        )
        with (
            patch.object(llm_gemini, "GEMINI_API_KEY", "AIza-test"),
            patch.object(llm_gemini, "GEMINI_MODELS", ["test-model"]),
            patch.object(llm_gemini, "GEMINI_THINKING_LEVEL", ""),
            patch.object(llm_gemini, "GEMINI_MIN_OUTPUT_TOKENS", 0),
            patch.object(llm_gemini.requests, "post", return_value=response) as post,
        ):
            result = llm_gemini.gemini_chat(
                [{"role": "user", "content": "test"}], retries=1
            )

        self.assertEqual(result, '{"ok": true}')
        body = post.call_args.kwargs["json"]
        self.assertEqual(
            body["generationConfig"]["maxOutputTokens"],
            llm_gemini.DEFAULT_MAX_OUTPUT_TOKENS,
        )

    def test_fallback_requests_low_reasoning_effort(self):
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(llm_gemini, "FALLBACK_API_KEY", "fallback-test"), \
             patch.object(llm_gemini.requests, "post", return_value=response) as post:
            llm_gemini._fallback_chat_once(
                [{"role": "user", "content": "test"}],
                max_tokens=100,
                temperature=0.2,
                timeout=1,
            )
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")

    def test_explicit_minimum_can_raise_requested_budget(self):
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "candidates": [{
                    "content": {"parts": [{"text": "{}"}]},
                    "finishReason": "STOP",
                }]
            },
        )
        with (
            patch.object(llm_gemini, "GEMINI_API_KEY", "AIza-test"),
            patch.object(llm_gemini, "GEMINI_MODELS", ["test-model"]),
            patch.object(llm_gemini, "GEMINI_THINKING_LEVEL", ""),
            patch.object(llm_gemini, "GEMINI_MIN_OUTPUT_TOKENS", 8192),
            patch.object(llm_gemini.requests, "post", return_value=response) as post,
        ):
            llm_gemini.gemini_chat(
                [{"role": "user", "content": "test"}],
                max_tokens=4000,
                retries=1,
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["generationConfig"]["maxOutputTokens"],
            8192,
        )


if __name__ == "__main__":
    unittest.main()
