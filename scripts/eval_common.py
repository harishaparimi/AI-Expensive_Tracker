"""Shared by the eval scripts: the LLM settings to compare, and sending the requests.

Evals call OpenRouter directly and never use saved answers, so every run counts toward the daily cap.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.services.llm import LLMError, call_openrouter

# What each setting adds to the request. Free models reason silently first, and that is most of the wait.
SETTINGS = {
    "app": None,  # what the app really sends: REQUEST_SETTINGS in app/services/llm.py
    "default": {},  # the model's own default, with reasoning on
    "low": {"reasoning": {"effort": "low"}},
    "off": {"reasoning": {"enabled": False}},
    "off-seed": {"reasoning": {"enabled": False}, "temperature": 0.2, "seed": 7},
}
WORKERS = 3  # the free limit is 20 requests a minute


def parse_args(
    description: str, default_settings: list[str], default_runs: int, scenarios: list[str] | None = None
) -> argparse.Namespace:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console cannot otherwise print every character
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=default_settings)
    parser.add_argument("--runs", type=int, default=default_runs, help="real requests per setting")
    parser.add_argument("--model", default=None, help="a :free model; default is the first in LLM_MODELS")
    parser.add_argument("--out", type=Path, help="save the raw answers to this JSON file")
    parser.add_argument("--rescore", type=Path, help="score answers saved by --out, using no requests")
    if scenarios:
        parser.add_argument("--scenario", choices=scenarios, default=scenarios[0], help="what to change in the demo data")
    return parser.parse_args()


def send_all(jobs: list[dict], model: str | None) -> list[dict]:
    """Send each job ({setting, system, user, schema, ...}) and add its answer or error, and its time.

    Stops sending after a rate limit error, because failed attempts count toward the cap too.
    """
    model = model or config.LLM_MODELS[0]
    limited = threading.Event()

    def send(job: dict) -> dict:
        request = {key: job.pop(key) for key in ("system", "user", "schema")}
        result = {**job, "model": model, "seconds": None, "answer": None, "error": None}
        if limited.is_set():
            result["error"] = "not sent: rate limited earlier in this run"
            return result
        started = time.perf_counter()
        try:
            result["answer"], _ = call_openrouter(
                request["system"], request["user"], request["schema"], [model], SETTINGS[job["setting"]]
            )
        except LLMError as exc:
            result["error"] = str(exc)
            if "rate limit" in str(exc).lower():
                limited.set()
        result["seconds"] = round(time.perf_counter() - started, 1)
        print(f"  {job['setting']} run {job.get('run', 0) + 1}: {result['seconds']}s {result['error'] or 'answered'}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(send, jobs))


def save_or_load(args: argparse.Namespace, make_jobs) -> list[dict]:
    """Load saved answers with --rescore. Otherwise send the jobs, and save the answers with --out."""
    if args.rescore:
        return json.loads(args.rescore.read_text(encoding="utf-8"))
    results = send_all(make_jobs(), args.model)
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results
