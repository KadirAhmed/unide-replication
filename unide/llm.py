"""Thin wrapper around the OpenAI chat API (the paper used gpt-3.5-turbo-1106, retired 2026-09-28; default is its
successor snapshot 0125).

Rate limits are handled patiently: when OpenAI answers 429, the wrapper waits as long as the error message suggests
(requests-per-day caps refill continuously, e.g. one request every 8.64 s for 10,000/day) instead of giving up, so
long runs slow down rather than losing work. Running out of credit stops immediately with a clear message.
"""
import json
import os
import random
import re
import time


def suggested_wait(message: str) -> float | None:
    """Seconds to wait, parsed from '... Please try again in 8.64s' / '... in 896ms' / '... in 1m30s'."""
    m = re.search(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", message)
    if not m:
        return None
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def limit_kind(message: str) -> str:
    for code, name in (("(RPD)", "requests-per-day"), ("(RPM)", "requests-per-minute"),
                       ("(TPD)", "tokens-per-day"), ("(TPM)", "tokens-per-minute")):
        if code in message:
            return name
    return "rate"


class ChatLLM:
    def __init__(self, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.9,
                 max_retries: int = 6, max_rate_waits: int = 60):
        import openai  # imported lazily so the rest of the package works without it
        self._openai = openai
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries          # network / server errors
        self.max_rate_waits = max_rate_waits    # 429 rate-limit waits per request (~10+ minutes in total)

    def __call__(self, system: str, user: str, temperature: float | None = None) -> str:
        oa = self._openai
        delay, errors, waits = 2.0, 0, 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature if temperature is None else temperature,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content
            except oa.RateLimitError as e:
                msg = str(e)
                if "insufficient_quota" in msg:
                    raise RuntimeError("OpenAI reports insufficient_quota: the account is out of credit. "
                                       "Add credit under Settings > Billing, then rerun the same command.") from e
                waits += 1
                if waits > self.max_rate_waits:
                    raise
                wait = max(suggested_wait(msg) or delay, 1.0) + random.uniform(0.5, 3.0)
                if waits == 1 or waits % 10 == 0:
                    print(f"[llm] {limit_kind(msg)} limit reached; waiting {wait:.0f}s and retrying")
                time.sleep(wait)
                delay = min(delay * 2, 60)
            except (oa.APIConnectionError, oa.APITimeoutError, oa.InternalServerError) as e:
                errors += 1
                if errors > self.max_retries:
                    raise
                print(f"[llm] {type(e).__name__}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)


def parse_json(text: str):
    """Parse a JSON object from an LLM reply, tolerating code fences and surrounding prose."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise
