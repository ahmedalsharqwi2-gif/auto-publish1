import unittest
from unittest.mock import patch

from provider_pool import (
    CircuitBreaker,
    Provider,
    ProviderPool,
    ProviderPoolError,
    ProviderRateLimitError,
    TokenBucket,
    _retry_after_seconds,
    is_rate_limited,
    is_retryable,
)


class ProviderPoolTests(unittest.TestCase):
    def test_token_bucket_allows_burst_then_waits_for_refill(self):
        bucket = TokenBucket(rate=1, capacity=1)
        with patch("provider_pool.time.monotonic", side_effect=[0, 0, 1]), \
             patch("provider_pool.time.sleep") as sleep:
            bucket.updated_at = 0
            bucket.acquire()
            bucket.acquire()
        sleep.assert_called_once_with(1.0)

    def test_token_bucket_capacity_is_at_least_one(self):
        bucket = TokenBucket(rate=0.01, capacity=0)
        self.assertEqual(bucket.capacity, 1.0)
        self.assertEqual(bucket.tokens, 1.0)

    def test_circuit_breaker_requires_threshold_before_opening(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
        self.assertTrue(breaker.allow())
        breaker.failure()
        self.assertFalse(breaker.is_open)
        breaker.failure()
        self.assertTrue(breaker.is_open)
        self.assertFalse(breaker.allow())

    def test_circuit_breaker_half_open_probe_recovers(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.1)
        breaker.failure()
        self.assertFalse(breaker.allow())
        import time
        time.sleep(0.11)
        self.assertTrue(breaker.allow())
        breaker.success()
        self.assertTrue(breaker.allow())
        self.assertFalse(breaker.is_open)

    def test_circuit_breaker_reopens_after_failed_probe(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.1)
        breaker.failure()
        import time
        time.sleep(0.11)
        self.assertTrue(breaker.allow())
        breaker.failure()
        self.assertTrue(breaker.is_open)

    def test_retry_after_numeric_and_rate_classification(self):
        exc = ProviderRateLimitError("429", retry_after=7)
        self.assertEqual(_retry_after_seconds(exc), 7.0)
        self.assertTrue(is_rate_limited(exc))
        self.assertTrue(is_retryable(exc))

    def test_non_retryable_status_is_not_rate_limited(self):
        exc = ValueError("HTTP 400 invalid argument")
        self.assertFalse(is_rate_limited(exc))
        self.assertFalse(is_retryable(exc))

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
