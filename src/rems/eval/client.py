"""A single non-streaming, no-automatic-retry transport for all benchmark calls."""
from __future__ import annotations

import time

from ..llm.metrics import LLMInvocationMetrics
from ..llm.provider import LLMProvider

ENGLISH_POLICY = (
    "Evaluation language and viewpoint: write all narrative summaries, object facts, "
    "portraits and search queries in English, keeping the required JSON keys and schema. "
    "The subject JShi observed the supplied dialogue; the named speakers are other people. "
    "Keep each speaker's first-person statements attributed to that speaker. "
    "Never turn a speaker's personal experiences into JShi's own biography. "
    "Use only the supplied material; do not invent facts or endings."
)


class EvaluationLLM(LLMProvider):
    def __init__(self, config, ledger, audit_path, max_output=4096):
        super().__init__(config)
        self._client.max_retries = 0
        self.ledger, self.audit_path, self.max_output = ledger, audit_path, max_output

    def complete(self, task_type, messages, *, temperature=None, model=None):
        import json
        messages = [dict(m) for m in messages]
        if task_type not in {"judge", "atoms", "core"}:
            # Evaluation-only language/viewpoint overlay; production prompts remain unchanged.
            messages.append({"role": "system", "content": ENGLISH_POLICY})
        resolved = model or self._get_model(task_type)
        if resolved != "deepseek-flash":
            raise ValueError("This price ledger is only valid for deepseek-flash")
        output_limit = max(self.max_output, 16384) if task_type == "event_enrichment" else self.max_output
        reservation = self.ledger.reserve(messages, output_limit, task_type)
        began = time.perf_counter()
        response = self._client.chat.completions.create(
            model=resolved, messages=messages, temperature=temperature if temperature is not None else 0,
            max_tokens=output_limit, extra_body={"thinking": {"type": "disabled"}}, timeout=60,
        )
        self.ledger.settle(reservation, response.usage)
        choice = response.choices[0]
        content = choice.message.content or ""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately no credentials or endpoint objects in the audit trail.
        with self.audit_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"task": task_type, "request_id": response.id,
                "response_model": response.model, "messages": messages, "output": content,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump()}, ensure_ascii=False) + "\n")
        self._invocations.append(LLMInvocationMetrics(
            task_type, resolved, (time.perf_counter()-began)*1000,
            response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens,
        ))
        if choice.finish_reason != "stop" or not content.strip():
            raise RuntimeError(f"Incomplete {task_type} output ({choice.finish_reason}); no score assigned")
        return content
