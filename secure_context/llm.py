"""LLM backends. The pipeline only ever passes dehydrated, minimized text to these.

simulated   offline template answers (default, no key needed)
nvidia      hosted model on build.nvidia.com (default meta/llama-3.1-70b-instruct), needs NVIDIA_API_KEY
hf-local    open model on this machine (default Qwen/Qwen2.5-1.5B-Instruct), `pip install transformers torch`
hf-api      Hugging Face Inference Providers (default Qwen/Qwen2.5-7B-Instruct), needs HF_TOKEN
anthropic   needs ANTHROPIC_API_KEY and LLM_MODEL, plus `pip install anthropic`
openai      needs OPENAI_API_KEY and LLM_MODEL, plus `pip install openai`
"""
from __future__ import annotations

import os


class SimulatedLLM:
    name = "simulated"

    def complete(self, system: str, prompt: str, *, task=None, client=None, levels=None, session=None) -> str:
        from .tasks import simulated_answer
        return simulated_answer(task, client, levels, session)


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model or os.environ["LLM_MODEL"]
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, prompt: str, **_) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=900, system=system,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAILLM:
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model or os.environ["LLM_MODEL"]
        self.name = f"openai:{self.model}"

    def complete(self, system: str, prompt: str, **_) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content


class HFLocalLLM:
    """Open model run on this machine with transformers (Apple GPU via MPS when available)."""

    def __init__(self, model: str | None = None, max_new_tokens: int = 350):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model or "Qwen/Qwen2.5-1.5B-Instruct"
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.name = f"hf-local:{self.model_id}"

    def complete(self, system: str, prompt: str, **_) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


class HFInferenceLLM:
    """Hugging Face Inference Providers; needs HF_TOKEN."""

    def __init__(self, model: str | None = None):
        from huggingface_hub import InferenceClient
        self.model_id = model or "Qwen/Qwen2.5-7B-Instruct"
        self.client = InferenceClient(model=self.model_id, token=os.environ["HF_TOKEN"])
        self.name = f"hf-api:{self.model_id}"

    def complete(self, system: str, prompt: str, **_) -> str:
        r = self.client.chat_completion(messages=[{"role": "system", "content": system},
                                                  {"role": "user", "content": prompt}], max_tokens=500)
        return r.choices[0].message.content


class NvidiaLLM:
    """Hosted LLM on NVIDIA's API catalog (OpenAI-compatible). Needs NVIDIA_API_KEY."""
    URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, model: str | None = None, min_interval: float = 1.6):
        self.key = os.environ["NVIDIA_API_KEY"]
        self.model_id = model or os.environ.get("NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct")
        self.min_interval = min_interval
        self._last = 0.0
        self.name = f"nvidia:{self.model_id}"

    def complete(self, system: str, prompt: str, **_) -> str:
        import time
        import requests
        body = {"model": self.model_id, "temperature": 0.2, "max_tokens": 600,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = requests.post(self.URL, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("NVIDIA API kept failing (rate limit or server error)")


class GroqLLM:
    """Groq chat API with atomic, per-request checkpoints."""
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model=None, cache_dir=None):
        from pathlib import Path
        self.key = os.environ.get("GROQ_API_KEY")
        if not self.key:
            raise ValueError("Set GROQ_API_KEY in your Terminal before running the evaluation")
        self.model_id = model or os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
        self.name = f"groq:{self.model_id}"
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parents[1] / "data/cache/groq")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, system, prompt, *, json_mode=False, json_schema=None, **_):
        import hashlib
        import json
        import time
        import requests
        body = {"model": self.model_id, "temperature": 0.2, "max_completion_tokens": 4096,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / (digest + ".json")
        if path.exists():
            return json.loads(path.read_text())["content"]
        validation_failures = 0
        for attempt in range(8):
            try:
                r = requests.post(self.URL, json=body,
                                  headers={"Authorization": f"Bearer {self.key}"}, timeout=180)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 7:
                    raise RuntimeError("Groq connection failed; rerun to resume") from None
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == 7:
                    break
                try:
                    delay = float(r.headers.get("retry-after", 2 ** attempt))
                except ValueError:
                    delay = 2 ** attempt
                if delay > 300:
                    raise RuntimeError(f"Groq requests a {delay:.0f}s wait; rerun later to resume")
                time.sleep(max(0, delay))
                continue
            if not r.ok:
                try:
                    error = r.json().get("error", {})
                    code = str(error.get("code") or error.get("type") or "unknown")
                    message = str(error.get("message", "No error details returned"))
                except (ValueError, AttributeError):
                    code, message = "unknown", "Non-JSON error response"
                # Groq can still intermittently reject generated JSON in strict mode.
                # Retry only this generation error, never auth or invalid request settings.
                if (r.status_code == 400 and code == "json_validate_failed"
                        and (json_mode or json_schema is not None) and validation_failures < 2):
                    validation_failures += 1
                    time.sleep(2 ** (validation_failures - 1))
                    continue
                # Never include credentials or the API's failed_generation payload.
                detail = f"{code}: {message}".replace(self.key, "[redacted]")[:800]
                raise RuntimeError(f"Groq HTTP {r.status_code} ({self.model_id}): {detail}")
            choice = r.json()["choices"][0]
            content = choice["message"].get("content")
            if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
                raise RuntimeError("Groq returned an incomplete answer; it was not cached")
            if json_mode or json_schema is not None:
                json.loads(content)
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", dir=self.cache_dir, delete=False) as f:
                json.dump({"content": content}, f)
                temporary = f.name
            os.replace(temporary, path)
            return content
        raise RuntimeError("Groq rate limit or server errors persisted; rerun later to resume")


def get_backend(name: str = "simulated", model: str | None = None):
    if name == "groq":
        return GroqLLM(model)
    if name == "nvidia":
        return NvidiaLLM(model)
    if name == "anthropic":
        return AnthropicLLM(model)
    if name == "openai":
        return OpenAILLM(model)
    if name == "hf-local":
        return HFLocalLLM(model)
    if name == "hf-api":
        return HFInferenceLLM(model)
    if name == "simulated":
        return SimulatedLLM()
    raise ValueError(f"Unknown LLM backend: {name}")
