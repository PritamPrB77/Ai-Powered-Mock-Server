from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import Any

from app.config import Settings
from app.utils import extract_json_value


logger = logging.getLogger(__name__)


class Seq2SeqGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_error: Exception | None = None
        self._load_lock = Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.seq2seq_enabled)

    @property
    def available(self) -> bool:
        return self.enabled and self._load_error is None

    async def generate_json(self, prompt: str, temperature: float | None = None) -> Any | None:
        if not self.enabled:
            return None

        model_loaded = await asyncio.to_thread(self._ensure_model_loaded)
        if not model_loaded:
            return None

        generated_text = await asyncio.to_thread(self._generate_text, prompt, temperature)
        return extract_json_value(generated_text)

    async def warmup(self) -> bool:
        if not self.enabled:
            return False
        return await asyncio.to_thread(self._ensure_model_loaded)

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_error is not None:
            return False

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return True
            if self._load_error is not None:
                return False

            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except Exception as exc:  # noqa: BLE001
                self._load_error = exc
                logger.warning("Seq2Seq dependencies unavailable (%s). Falling back to OpenRouter only.", exc)
                return False

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.settings.seq2seq_model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.seq2seq_model_name)
                self._model.eval()
                self._torch = torch
            except Exception as exc:  # noqa: BLE001
                self._load_error = exc
                logger.warning(
                    "Failed to load Seq2Seq model '%s' (%s). Falling back to OpenRouter only.",
                    self.settings.seq2seq_model_name,
                    exc,
                )
                return False

        return True

    def _generate_text(self, prompt: str, temperature: float | None) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.seq2seq_max_input_tokens,
        )
        model_device = next(self._model.parameters()).device
        inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}

        do_sample = bool(self.settings.seq2seq_do_sample)
        selected_temperature = self.settings.seq2seq_temperature if temperature is None else float(temperature)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.seq2seq_max_new_tokens,
            "num_beams": self.settings.seq2seq_num_beams,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(0.01, selected_temperature)
            generation_kwargs["top_p"] = self.settings.seq2seq_top_p

        with self._torch.no_grad():
            output_tokens = self._model.generate(**inputs, **generation_kwargs)

        decoded = self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return decoded.strip()
