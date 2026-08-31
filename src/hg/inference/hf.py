"""HuggingFace transformers backend for the DGX Spark.

Two prediction paths, chosen by config:

* `scoring_mode: rank` (default) -- score each allowed label's continuation
  log-probability and take the argmax. This sidesteps canonicalization
  entirely, gives a genuine class distribution for every family, and is the
  right default for the dependence analysis: any disagreement it reports comes
  from the model, not from a parser (section 18).
* `scoring_mode: generate` -- greedy generation plus canonicalization. Needed
  for tasks whose answer is not a short fixed label, and kept so parse-failure
  behaviour is measurable rather than assumed away.

Loading is lazy and `unload()` actually frees the GPU: a five-family panel at
8-14B does not fit resident on one Spark, so the runner loads one family at a
time. Determinism follows section 13 -- greedy decoding, fixed seeds, and
torch's deterministic algorithms switched on.
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Any, Mapping, Sequence

from .base import (
    GenerationSettings,
    InferenceItem,
    Prediction,
    register_backend,
)
from .canonicalize import canonicalize

log = logging.getLogger(__name__)


@register_backend("transformers")
@register_backend("hf")
class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        *,
        checkpoint: str,
        dtype: str = "bfloat16",
        device_map: str | dict[str, Any] = "auto",
        scoring_mode: str = "rank",
        adapter_path: str | None = None,
        synonyms: Mapping[str, str] | None = None,
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        batch_size: int = 8,
        max_input_tokens: int = 2048,
        **_ignored: Any,
    ) -> None:
        if scoring_mode not in ("rank", "generate"):
            raise ValueError("scoring_mode must be 'rank' or 'generate'")
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.device_map = device_map
        self.scoring_mode = scoring_mode
        self.adapter_path = adapter_path
        self.synonyms = dict(synonyms or {})
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.batch_size = int(batch_size)
        self.max_input_tokens = int(max_input_tokens)
        self._model = None
        self._tokenizer = None

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.use_deterministic_algorithms(True, warn_only=True)

        dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding so the last position is the real final token for every
        # sequence in a batch -- otherwise ranked scores read off padding.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- prompting -------------------------------------------------------
    def _format(self, prompt: str) -> str:
        """Apply the family's own chat template to a family-neutral prompt.

        The instruction text is identical across families (see
        `hg.inference.prompts`); only the wrapper differs, because forcing one
        family's wrapper onto another is itself a format artifact.
        """
        tok = self._tokenizer
        if not self.use_chat_template or getattr(tok, "chat_template", None) is None:
            return prompt
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # -- prediction ------------------------------------------------------
    def predict(
        self, items: Sequence[InferenceItem], settings: GenerationSettings
    ) -> list[Prediction]:
        self.load()
        if self.scoring_mode == "rank":
            return self._predict_rank(items)
        return self._predict_generate(items, settings)

    def _predict_rank(self, items: Sequence[InferenceItem]) -> list[Prediction]:
        """Argmax over per-label continuation log-probabilities.

        Label log-probs are length-normalized by token count. Without that, a
        label that tokenizes into more pieces is penalized, and since
        tokenization differs by family that penalty would show up as
        cross-family disagreement.
        """
        import torch

        out: list[Prediction] = []
        for item in items:
            prompt = self._format(item.prompt)
            scores: dict[str, float] = {}
            for label in item.labels:
                scores[label] = self._label_logprob(prompt, label)
            best = max(scores, key=lambda l: scores[l])
            probs = _softmax(scores)
            out.append(
                Prediction(
                    item_id=item.item_id,
                    label=best,
                    raw_output=best,
                    parse_ok=True,
                    confidence=probs[best],
                    probs=probs,
                    logprob=scores[best],
                )
            )
        return out

    def _label_logprob(self, prompt: str, label: str) -> float:
        import torch

        tok = self._tokenizer
        prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        if prompt_ids.shape[1] > self.max_input_tokens:
            prompt_ids = prompt_ids[:, -self.max_input_tokens :]
        label_ids = tok(" " + label, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = torch.cat([prompt_ids, label_ids], dim=1).to(self._model.device)

        with torch.no_grad():
            logits = self._model(input_ids=input_ids).logits
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        n_label = label_ids.shape[1]
        targets = input_ids[:, 1:]
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        label_lp = picked[:, -n_label:].sum().item()
        return label_lp / max(n_label, 1)

    def _predict_generate(
        self, items: Sequence[InferenceItem], settings: GenerationSettings
    ) -> list[Prediction]:
        import torch

        tok = self._tokenizer
        out: list[Prediction] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            prompts = [self._format(i.prompt) for i in batch]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self._model.device)

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": settings.max_new_tokens,
                "do_sample": not settings.greedy,
                "pad_token_id": tok.pad_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if not settings.greedy:
                gen_kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
                torch.manual_seed(settings.seed or 0)

            with torch.no_grad():
                result = self._model.generate(**enc, **gen_kwargs)

            new_tokens = result.sequences[:, enc.input_ids.shape[1] :]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
            seq_lp = _sequence_logprobs(result, new_tokens, tok.pad_token_id)

            for item, text, lp in zip(batch, texts, seq_lp):
                parsed = canonicalize(text, item.labels, self.synonyms)
                out.append(
                    Prediction(
                        item_id=item.item_id,
                        label=parsed.label,
                        raw_output=text,
                        parse_ok=parsed.parse_ok,
                        confidence=(math.exp(lp) if lp is not None and lp < 0 else None),
                        logprob=lp,
                    )
                )
        return out


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _sequence_logprobs(result: Any, new_tokens: Any, pad_id: int | None) -> list[float | None]:
    """Mean per-token log-probability of each generated sequence."""
    try:
        import torch

        stacked = torch.stack(result.scores, dim=1)  # (batch, steps, vocab)
        logprobs = torch.log_softmax(stacked.float(), dim=-1)
        picked = logprobs.gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
        mask = new_tokens != pad_id if pad_id is not None else torch.ones_like(new_tokens, dtype=torch.bool)
        lengths = mask.sum(dim=1).clamp(min=1)
        summed = (picked * mask).sum(dim=1)
        return (summed / lengths).tolist()
    except Exception:  # scores unavailable on some model classes
        return [None] * new_tokens.shape[0]
