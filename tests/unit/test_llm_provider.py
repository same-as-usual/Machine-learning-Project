"""Free-provider path for the LLM labeler: presets, coercion, request shape."""

import argparse
import json

import pytest

from manipulens.labeling.labeling_functions import DIMENSIONS
from manipulens.labeling.llm_labeler import (
    OPENAI_COMPAT_PRESETS,
    _label_once_openai_compat,
    coerce_labels,
    resolve_provider,
)


def _args(**kw) -> argparse.Namespace:
    base = {"provider": "anthropic", "model": None, "base_url": None, "api_key_env": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_presets_are_wellformed():
    for name, p in OPENAI_COMPAT_PRESETS.items():
        assert p["base_url"].startswith("http"), name
        assert p["model"], name


def test_anthropic_is_default():
    assert resolve_provider(_args()) == {"kind": "anthropic"}


def test_preset_resolution_reads_key_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    p = resolve_provider(_args(provider="groq"))
    assert p["kind"] == "openai"
    assert p["api_key"] == "gsk-test"
    assert p["model"] == OPENAI_COMPAT_PRESETS["groq"]["model"]


def test_missing_key_env_exits_with_guidance(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        resolve_provider(_args(provider="gemini"))


def test_ollama_needs_no_key(monkeypatch):
    p = resolve_provider(_args(provider="ollama"))
    assert p["api_key"] is None
    assert "localhost" in p["base_url"]


def test_model_override_beats_preset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    p = resolve_provider(_args(provider="groq", model="qwen-2.5-32b"))
    assert p["model"] == "qwen-2.5-32b"


def test_coerce_clamps_and_defaults():
    raw = {DIMENSIONS[0]: 7, DIMENSIONS[1]: "1", DIMENSIONS[2]: None, "out_of_scope": 1}
    out = coerce_labels(raw)
    assert out[DIMENSIONS[0]] == 2  # clamped
    assert out[DIMENSIONS[1]] == 1  # string int ok
    assert out[DIMENSIONS[2]] == 0  # garbage -> conservative 0
    assert all(out[d] in (0, 1, 2) for d in DIMENSIONS)  # missing dims -> 0
    assert out["out_of_scope"] is True


def test_openai_compat_call_shape(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            content = json.dumps(dict.fromkeys(DIMENSIONS, 1) | {"out_of_scope": False})
            return {"choices": [{"message": {"content": content}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    labels, usd = _label_once_openai_compat(
        "https://api.groq.com/openai/v1/", "key123", "llama-3.3-70b-versatile",
        "system text", "Some headline",
    )
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer key123"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert usd == 0.0
    assert all(labels[d] == 1 for d in DIMENSIONS)
