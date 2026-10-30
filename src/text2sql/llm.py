"""Prompt format (shared by training and eval) and one OpenAI-compatible client.

Hosted APIs, vLLM (base and fine-tuned) and a LiteLLM gateway all speak this protocol,
so every system under test goes through the same code path.
"""
import os
import re

from openai import OpenAI

from text2sql.db import SCHEMA

SYSTEM = f"Translate the question into one read-only SQLite query. Output only SQL.\nSchema:\n{SCHEMA}"


def messages(question: str, shots: tuple[tuple[str, str], ...] = ()) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM}]
    for q, s in shots:
        m += [{"role": "user", "content": q}, {"role": "assistant", "content": s}]
    return m + [{"role": "user", "content": question}]


def extract_sql(text: str) -> str:
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def complete(model: str, msgs: list[dict], seed: int = 42) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens)."""
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY", "none"))
    r = client.chat.completions.create(model=model, messages=msgs, temperature=0, seed=seed, max_tokens=400)
    return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
