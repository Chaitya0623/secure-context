"""LLM backends. The pipeline only ever passes dehydrated, minimized text to these.

simulated   offline template answers (default, no key needed)
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


def get_backend(name: str = "simulated", model: str | None = None):
    if name == "anthropic":
        return AnthropicLLM(model)
    if name == "openai":
        return OpenAILLM(model)
    return SimulatedLLM()
