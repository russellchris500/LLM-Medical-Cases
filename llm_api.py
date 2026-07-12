#!/usr/bin/env python3
"""API clients for the frontier LLMs used by Program 3 (run_llms.py).

All four providers are called with a single JSON POST over urllib, so the
API path needs nothing installed beyond Python itself. Model *names* can be
overridden per model in Settings; endpoints and request shapes live here.

Requires only the Python 3 standard library.
"""

import json
import random
import ssl
import time
import urllib.error
import urllib.request

# Injectable for tests.
_urlopen = urllib.request.urlopen

# Every request is a single, self-contained question: no conversation
# history is ever sent, and no server-side storage is requested, so the
# models have NO MEMORY between cases. Deep thinking is asked for in each
# provider's own way (see build_request).
API_REGISTRY = {
    "claude": {
        "display_name": "Anthropic Claude",
        "style": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-sonnet-4-5",
        "supports_temperature": True,
        "thinking_budget": 10000,  # tokens of extended thinking when deep thinking is on
        "key_hint": "an Anthropic API key (console.anthropic.com)",
    },
    "gpt": {
        "display_name": "OpenAI GPT",
        "style": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-5",
        # OpenAI reasoning models reject the temperature parameter.
        "supports_temperature": False,
        "reasoning_effort": "high",  # sent when deep thinking is on
        "key_hint": "an OpenAI API key (platform.openai.com)",
    },
    "gemini": {
        "display_name": "Google Gemini",
        "style": "gemini",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "default_model": "gemini-2.5-pro",
        "supports_temperature": True,
        # -1 = dynamic thinking: the model thinks as long as it needs.
        "thinking_budget": -1,
        "key_hint": "a Google AI API key (aistudio.google.com)",
    },
    "grok": {
        "display_name": "xAI Grok",
        "style": "openai",
        "url": "https://api.x.ai/v1/chat/completions",
        "default_model": "grok-4",
        "supports_temperature": True,
        # grok-4 always reasons deeply and accepts no effort parameter.
        "key_hint": "an xAI API key (console.x.ai)",
    },
}

API_MODEL_IDS = list(API_REGISTRY)

MAX_TOKENS = 8192
# Claude's max_tokens must leave room for the thinking budget on top of
# the visible answer.
MAX_TOKENS_WITH_THINKING = 16384
RETRYABLE_HTTP = (429, 500, 502, 503, 529)


class ApiCallError(Exception):
    """A single call failed for good (e.g. HTTP 400); record it and move on."""


class ModelAbort(Exception):
    """The model is unusable for this whole run (bad key, unknown model name)."""


def resolve_model(model_id, settings_entry):
    override = (settings_entry.get("model") or "").strip()
    return override or API_REGISTRY[model_id]["default_model"]


def build_request(model_id, api_key, model, prompt_text, temperature=0, deep_thinking=True):
    """Return (url, headers, body_dict) for one question.

    The body always contains exactly one user message and asks for no
    server-side storage, so nothing carries over between cases. With
    deep_thinking on, each provider is asked to reason at length before
    answering, in its own dialect.
    """
    entry = API_REGISTRY[model_id]
    style = entry["style"]
    if style == "anthropic":
        url = entry["url"]
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if deep_thinking and entry.get("thinking_budget"):
            body["max_tokens"] = MAX_TOKENS_WITH_THINKING
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": entry["thinking_budget"],
            }
            # Anthropic requires the temperature to be left alone while
            # extended thinking is enabled.
        elif entry["supports_temperature"]:
            body["temperature"] = temperature
    elif style == "openai":
        url = entry["url"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            # Never store the exchange server-side: each case must be a
            # clean slate with no memory.
            "store": False,
        }
        if deep_thinking and entry.get("reasoning_effort"):
            body["reasoning_effort"] = entry["reasoning_effort"]
        if entry["supports_temperature"]:
            body["temperature"] = temperature
    elif style == "gemini":
        url = entry["url_template"].format(model=model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        body = {"contents": [{"parts": [{"text": prompt_text}]}]}
        generation_config = {}
        if entry["supports_temperature"]:
            generation_config["temperature"] = temperature
        if deep_thinking and entry.get("thinking_budget") is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": entry["thinking_budget"]
            }
        if generation_config:
            body["generationConfig"] = generation_config
    else:
        raise ValueError("Unknown API style: {}".format(style))
    return url, headers, body


def parse_response(model_id, data):
    """Return (response_text, model_reported) from a parsed response body."""
    style = API_REGISTRY[model_id]["style"]
    try:
        if style == "anthropic":
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            reported = data.get("model", "")
        elif style == "openai":
            text = data["choices"][0]["message"]["content"] or ""
            reported = data.get("model", "")
        else:  # gemini
            parts = data["candidates"][0]["content"]["parts"]
            # Parts flagged as "thought" are the model's internal reasoning,
            # not the answer.
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            reported = data.get("modelVersion", "")
    except (KeyError, IndexError, TypeError) as e:
        raise ApiCallError("The response had an unexpected shape ({}).".format(e))
    return text, reported


def _error_detail(body_bytes):
    try:
        data = json.loads(body_bytes.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))[:300]
    if isinstance(error, str):
        return error[:300]
    return ""


def _backoff_seconds(attempt, retry_after):
    if retry_after:
        try:
            return min(float(retry_after), 120)
        except ValueError:
            pass
    return min(2 * (2 ** attempt) + random.uniform(0, 1), 60)


def call_api_model(model_id, settings_entry, prompt_text, options, log=print, sleep=time.sleep):
    """Ask one question; returns a dict with text and metadata.

    Raises ModelAbort when the model should be dropped for the whole run
    (bad key, unknown model name) and ApiCallError when just this call
    failed (recorded as a failed answer).
    """
    entry = API_REGISTRY[model_id]
    api_key = (settings_entry.get("api_key") or "").strip()
    if not api_key:
        raise ModelAbort(
            "No API key is set for {} - add {} in Settings.".format(
                entry["display_name"], entry["key_hint"]
            )
        )
    model = resolve_model(model_id, settings_entry)
    url, headers, body = build_request(
        model_id,
        api_key,
        model,
        prompt_text,
        deep_thinking=options.get("deep_thinking", True),
    )
    payload = json.dumps(body).encode("utf-8")
    timeout = options.get("request_timeout_s", 180)
    max_retries = options.get("max_retries", 5)

    attempts = 0
    while True:
        attempts += 1
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with _urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text, reported = parse_response(model_id, data)
            if not text.strip():
                raise ApiCallError("The model returned an empty answer.")
            return {
                "response_text": text,
                "model_requested": model,
                "model_reported": reported or model,
                "attempts": attempts,
            }
        except urllib.error.HTTPError as e:
            detail = _error_detail(e.read())
            if e.code in (401, 403):
                raise ModelAbort(
                    "{} rejected the API key (HTTP {}). Re-enter the key in "
                    "Settings. {}".format(entry["display_name"], e.code, detail).strip()
                )
            if e.code == 404:
                raise ModelAbort(
                    "{} does not recognize the model name '{}' (HTTP 404). "
                    "Check the model name in Settings. {}".format(
                        entry["display_name"], model, detail
                    ).strip()
                )
            if e.code in RETRYABLE_HTTP and attempts <= max_retries:
                wait = _backoff_seconds(attempts, e.headers.get("Retry-After"))
                log(
                    "    {} is busy (HTTP {}), waiting {:.0f}s (attempt {}/{})...".format(
                        entry["display_name"], e.code, wait, attempts, max_retries
                    )
                )
                sleep(wait)
                continue
            raise ApiCallError(
                "HTTP {} from {}. {}".format(e.code, entry["display_name"], detail).strip()
            )
        except ssl.SSLCertVerificationError:
            raise ModelAbort(
                "Secure connection to {} failed (certificate problem). On a Mac, "
                "run 'Install Certificates.command' inside your Python folder and "
                "try again.".format(entry["display_name"])
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempts <= max_retries:
                wait = _backoff_seconds(attempts, None)
                log(
                    "    Connection problem reaching {} ({}); waiting {:.0f}s "
                    "(attempt {}/{})...".format(
                        entry["display_name"], getattr(e, "reason", e), wait, attempts, max_retries
                    )
                )
                sleep(wait)
                continue
            raise ApiCallError(
                "Could not reach {} after {} attempts ({}).".format(
                    entry["display_name"], attempts, getattr(e, "reason", e)
                )
            )
        except json.JSONDecodeError:
            raise ApiCallError(
                "{} sent back something that was not valid JSON.".format(entry["display_name"])
            )
