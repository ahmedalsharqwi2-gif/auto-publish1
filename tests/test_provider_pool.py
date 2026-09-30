import unittest
from unittest.mock import patch

from provider_pool import (
    CircuitBreaker,
    Provider,
    ProviderPool,
    ProviderPoolError,
    ProviderRateLimitError,
)


class ProviderPoolTests(unittest.TestCase):
    def test_429_fails_over_and_honors_bounded_attempts(self):
        calls = []

        def limited(**_kwargs):
            calls.append("gemini")
            raise ProviderRateLimitError("HTTP 429", retry_after=0)

        def backup(**_kwargs):
            calls.append("backup")
            return "ok"

        pool = ProviderPool([
            Provider("gemini", limited, max_attempts=2, rate_limit_per_second=0),
            Provider("openrouter", backup, max_attempts=1, rate_limit_per_second=0),
        ], backoff_base=0, random_jitter=0)
        self.assertEqual(pool.call(prompt="test"), "ok")
        self.assertEqual(calls, ["gemini", "gemini", "backup"])

    def test_circuit_breaker_skips_open_provider(self):
        calls = []

        def broken(**_kwargs):
            calls.append("broken")
            raise ProviderRateLimitError("429", retry_after=0)

        pool = ProviderPool([
            Provider(
                "broken", broken, max_attempts=1, rate_limit_per_second=0,
                breaker=CircuitBreaker(failure_threshold=1, recovery_seconds=3600),
            ),
        ], backoff_base=0, random_jitter=0)
        with self.assertRaises(ProviderPoolError):
            pool.call(prompt="first")
        with self.assertRaises(ProviderPoolError):
            pool.call(prompt="second")
        self.assertEqual(calls, ["broken"])

    def test_non_retryable_error_does_not_spin(self):
        calls = []

        def invalid(**_kwargs):
            calls.append(1)
            raise ValueError("invalid request")

        pool = ProviderPool([
            Provider("invalid", invalid, max_attempts=5, rate_limit_per_second=0),
        ], backoff_base=0, random_jitter=0)
        with self.assertRaises(ProviderPoolError):
            pool.call(prompt="test")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
