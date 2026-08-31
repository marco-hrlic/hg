"""vLLM backend.

Optional. Worth using when a task family needs long inputs or many items and
throughput dominates -- Phase 3's multi-task replication across 5-8 task
families is where per-item HF generation starts to hurt.

Constraints kept deliberately tight:

* greedy only for the primary dependence analysis (`temperature=0`), matching
  section 13's determinism requirement;
* `logprobs` requested so confidence is populated the same way as the other
  backends, otherwise routing strategies would silently behave differently
  depending on which backend produced the matrix;
* one model per process. vLLM does not reliably release GPU memory in-process,
  so `unload()` is honest about that and the runner is expected to run one
  family per container invocation when this backend is used.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from .base import GenerationSettings, InferenceItem, Prediction, register_backend
from .canonicalize import canonicalize

log = logging.getLogger(__name__)


@register_backend("vllm")
class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        *,
        checkpoint: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        adapter_path: str | None = None,
        synonyms: Mapping[str, str] | None = None,
        **_ignored: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = max_model_len
        self.adapter_path = adapter_path
        self.synonyms = dict(synonyms or {})
        self._llm = None
        self._lora_request = None

    def load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": self.checkpoint,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        if self.adapter_path:
            kwargs["enable_lora"] = True
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, self.adapter_path)
        self._llm = LLM(**kwargs)

    def unload(self) -> None:
        # vLLM holds the KV cache for the process lifetime. Dropping the handle
        # is not enough to free the device, so a multi-family panel on this
        # backend must run one family per process.
        self._llm = None
        self._lora_request = None
        log.warning(
            "vLLM does not release GPU memory in-process; run one model family "
            "per container invocation when using this backend"
        )

    def predict(
        self, items: Sequence[InferenceItem], settings: GenerationSettings
    ) -> list[Prediction]:
        self.load()
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_new_tokens,
            seed=settings.seed,
            stop=list(settings.stop) or None,
            logprobs=1,
        )
        kwargs: dict[str, Any] = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request

        outputs = self._llm.generate([i.prompt for i in items], params, **kwargs)

        out: list[Prediction] = []
        for item, result in zip(items, outputs):
            completion = result.outputs[0]
            text = completion.text
            parsed = canonicalize(text, item.labels, self.synonyms)
            lp = completion.cumulative_logprob
            n = max(len(completion.token_ids), 1)
            mean_lp = lp / n if lp is not None else None
            out.append(
                Prediction(
                    item_id=item.item_id,
                    label=parsed.label,
                    raw_output=text,
                    parse_ok=parsed.parse_ok,
                    confidence=(pow(2.718281828, mean_lp) if mean_lp is not None else None),
                    logprob=mean_lp,
                )
            )
        return out
