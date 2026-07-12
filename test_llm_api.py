"""Tests for llm_api.py. Run with:  python3 -m unittest test_llm_api.py"""

import io
import json
import unittest
import urllib.error

import llm_api
from llm_api import ApiCallError, ModelAbort, build_request, call_api_model, parse_response

OPTIONS = {"request_timeout_s": 5, "max_retries": 3}


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def http_error(code, body=None, headers=None):
    return urllib.error.HTTPError(
        url="https://example",
        code=code,
        msg="err",
        hdrs=headers or {},
        fp=io.BytesIO(json.dumps(body or {"error": {"message": "boom"}}).encode()),
    )


class FakeUrlopen:
    """Yields the queued results (responses or exceptions) in order."""

    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class Silence:
    def __call__(self, *a, **k):
        pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        self._orig = llm_api._urlopen
        self.sleeps = []

    def tearDown(self):
        llm_api._urlopen = self._orig

    def call(self, model_id, results, settings=None):
        fake = FakeUrlopen(results)
        llm_api._urlopen = fake
        result = call_api_model(
            model_id,
            settings or {"api_key": "sk-test", "model": ""},
            "What is the diagnosis?",
            OPTIONS,
            log=Silence(),
            sleep=self.sleeps.append,
        )
        return result, fake

    # ----- request building -----

    def test_anthropic_request_shape(self):
        url, headers, body = build_request("claude", "sk-k", "claude-x", "Q?")
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "sk-k")
        self.assertIn("anthropic-version", headers)
        self.assertEqual(body["messages"][0]["content"], "Q?")
        self.assertIn("max_tokens", body)
        self.assertEqual(body["temperature"], 0)

    def test_openai_request_shape_no_temperature(self):
        url, headers, body = build_request("gpt", "sk-k", "gpt-x", "Q?")
        self.assertEqual(headers["Authorization"], "Bearer sk-k")
        self.assertNotIn("temperature", body)  # reasoning models reject it

    def test_gemini_request_shape(self):
        url, headers, body = build_request("gemini", "sk-k", "gemini-x", "Q?")
        self.assertIn("gemini-x", url)
        self.assertEqual(headers["x-goog-api-key"], "sk-k")
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "Q?")
        self.assertEqual(body["generationConfig"]["temperature"], 0)

    def test_grok_uses_openai_shape(self):
        url, headers, body = build_request("grok", "sk-k", "grok-x", "Q?")
        self.assertIn("api.x.ai", url)
        self.assertIn("temperature", body)

    # ----- response parsing -----

    def test_parse_all_styles(self):
        text, model = parse_response(
            "claude",
            {"model": "claude-x-2025", "content": [{"type": "text", "text": "Answer."}]},
        )
        self.assertEqual((text, model), ("Answer.", "claude-x-2025"))
        text, model = parse_response(
            "gpt",
            {"model": "gpt-x-2025", "choices": [{"message": {"content": "Answer."}}]},
        )
        self.assertEqual((text, model), ("Answer.", "gpt-x-2025"))
        text, model = parse_response(
            "gemini",
            {
                "modelVersion": "gemini-x-001",
                "candidates": [{"content": {"parts": [{"text": "Answer."}]}}],
            },
        )
        self.assertEqual((text, model), ("Answer.", "gemini-x-001"))

    def test_malformed_response_raises_call_error(self):
        with self.assertRaises(ApiCallError):
            parse_response("gpt", {"choices": []})

    # ----- call behavior -----

    def test_successful_call_records_versions(self):
        result, fake = self.call(
            "claude",
            [FakeResponse({"model": "claude-real", "content": [{"type": "text", "text": "A."}]})],
        )
        self.assertEqual(result["response_text"], "A.")
        self.assertEqual(result["model_requested"], "claude-sonnet-4-5")
        self.assertEqual(result["model_reported"], "claude-real")
        self.assertEqual(result["attempts"], 1)

    def test_settings_model_override(self):
        result, fake = self.call(
            "claude",
            [FakeResponse({"model": "x", "content": [{"type": "text", "text": "A."}]})],
            settings={"api_key": "sk", "model": "claude-custom"},
        )
        self.assertEqual(result["model_requested"], "claude-custom")
        body = json.loads(fake.requests[0].data)
        self.assertEqual(body["model"], "claude-custom")

    def test_rate_limit_retries_with_retry_after(self):
        ok = FakeResponse({"model": "m", "choices": [{"message": {"content": "A."}}]})
        result, _ = self.call("gpt", [http_error(429, headers={"Retry-After": "7"}), ok])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(self.sleeps, [7.0])

    def test_server_error_retries_then_fails(self):
        errors = [http_error(500)] * 4
        with self.assertRaises(ApiCallError):
            self.call("gpt", errors)
        self.assertEqual(len(self.sleeps), 3)  # max_retries waits, then gives up

    def test_401_aborts_model(self):
        with self.assertRaises(ModelAbort):
            self.call("gpt", [http_error(401)])
        self.assertEqual(self.sleeps, [])  # no retries on a bad key

    def test_404_aborts_model_mentioning_name(self):
        with self.assertRaises(ModelAbort) as ctx:
            self.call("grok", [http_error(404)])
        self.assertIn("grok-4", str(ctx.exception))

    def test_400_is_call_error_with_api_message(self):
        with self.assertRaises(ApiCallError) as ctx:
            self.call("claude", [http_error(400, {"error": {"message": "too long"}})])
        self.assertIn("too long", str(ctx.exception))

    def test_missing_key_aborts_before_any_request(self):
        fake = FakeUrlopen([])
        llm_api._urlopen = fake
        with self.assertRaises(ModelAbort):
            call_api_model("gemini", {"api_key": "", "model": ""}, "Q?", OPTIONS, log=Silence())
        self.assertEqual(fake.requests, [])

    def test_empty_answer_is_a_failure(self):
        with self.assertRaises(ApiCallError):
            self.call(
                "claude",
                [FakeResponse({"model": "m", "content": [{"type": "text", "text": "   "}]})],
            )

    def test_connection_error_retries(self):
        ok = FakeResponse({"model": "m", "choices": [{"message": {"content": "A."}}]})
        result, _ = self.call("gpt", [urllib.error.URLError("unreachable"), ok])
        self.assertEqual(result["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
