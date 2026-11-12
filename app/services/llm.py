"""Every LLM call goes through ask_llm(): OpenRouter over plain HTTP, free models only."""
import hashlib
import json
import re

import requests
from sqlmodel import Session

from app import config, db
from app.models import LLMCache

TIMEOUT_SECONDS = 180
# Free models reason silently before answering. That made insights about 6 times slower, with no
# better scores in the evals (scripts/eval_*.py), so it is off. Saved answers are what keep a
# result stable: a low temperature did not clearly narrow the variation, and the same seed still
# gave different answers, so no seed is sent.
# The timeout only limits the gap between bytes, and free endpoints allow huge answers: one ran for
# 556s before being cut off. The longest real answer measured was about 1,400 tokens.
REQUEST_SETTINGS = {"reasoning": {"enabled": False}, "temperature": 0.2, "max_tokens": 8000}

# Account, phone and reference numbers. Amounts such as 100000.00 are kept.
LONG_NUMBER = re.compile(r"(?<!\d)\d{6,}(?!\d|\.\d)")
# UPI IDs and emails, such as name@okhdfc.
HANDLE = re.compile(r"[\w.+]+@[A-Za-z][\w.]*")


class LLMError(Exception):
    """The LLM could not give a usable answer. The message is safe to show the user."""


class LLMTimeout(LLMError):
    """The model did not answer in time. Free models sometimes stall, and a second try usually works."""


def mask(text: str) -> str:
    """Hide personal details, because free models may log what they receive."""
    return LONG_NUMBER.sub("[NUM]", HANDLE.sub("[ID]", text))


def parse_json(text: str) -> dict:
    """Read the JSON object from a model's answer, even if other text surrounds it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("The model did not return JSON.")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned invalid JSON.") from exc


def call_openrouter(
    system: str,
    user: str,
    schema: dict,
    models: list[str],
    options: dict | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> tuple[dict, str]:
    """Send one request. Returns the parsed answer and the model that served it.
    `options` replaces REQUEST_SETTINGS, which is how the evals compare settings."""
    if not config.OPENROUTER_API_KEY:
        raise LLMError("OPENROUTER_API_KEY is not set in .env.")
    # The key belongs to a paid account, so a paid model here would cost real money.
    if not models or any(not model.endswith(":free") for model in models):
        raise LLMError("LLM_MODELS must list at least one model, and only :free models.")

    payload = {
        "models": models,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": schema},
        },
        **(REQUEST_SETTINGS if options is None else options),
    }
    try:
        response = requests.post(
            config.OPENROUTER_URL,
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            json=payload,
            timeout=timeout,
        )
        body = response.json()
    except requests.Timeout as exc:
        raise LLMTimeout("The model took too long to answer.") from exc
    except requests.RequestException as exc:
        raise LLMError("Could not get a response from OpenRouter.") from exc

    # Errors can come back at the top level or on the choice, even with HTTP 200.
    choice = (body.get("choices") or [{}])[0]
    error = body.get("error") or choice.get("error")
    if error:
        message = error.get("message", "unknown error") if isinstance(error, dict) else error
        raise LLMError(f"OpenRouter error: {message}")
    if choice.get("finish_reason") == "length":
        raise LLMError("The model's answer was cut off because it was too long.")

    content = (choice.get("message") or {}).get("content") or ""
    return parse_json(content), body.get("model", "unknown")


def ask_llm(system: str, user: str, schema: dict, use_saved: bool = True, timeout: int | None = None) -> dict:
    """Ask the LLM for JSON matching `schema`, reusing the saved answer for a repeated prompt.
    With `use_saved` off, the prompt is always sent and the answer is left for the caller to keep.
    A `timeout` shorter than the default is for callers with a person waiting."""
    user = mask(user)
    prompt_hash = hashlib.sha256(json.dumps([system, user, schema], sort_keys=True).encode()).hexdigest()

    if use_saved:
        with Session(db.shared_engine) as session:
            saved = session.get(LLMCache, prompt_hash)
            if saved:
                return json.loads(saved.response)

    options = {} if timeout is None else {"timeout": timeout}
    answer, model = call_openrouter(system, user, schema, config.LLM_MODELS, **options)

    # Do not save an unusable answer, or every retry would get the same one back.
    missing = [field for field in schema.get("required", []) if field not in answer]
    if missing:
        raise LLMError(f"The model's answer is missing: {', '.join(missing)}.")

    if use_saved:
        with Session(db.shared_engine) as session:
            session.merge(LLMCache(prompt_hash=prompt_hash, response=json.dumps(answer), model=model))
            session.commit()
    return answer
