from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import Any

from sentence_transformers import SentenceTransformer

from app.utils import json_dumps


logger = logging.getLogger(__name__)


class SemanticValidator:
    def __init__(self, model_name: str, threshold: float) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model: SentenceTransformer | None = None
        self._load_error: Exception | None = None
        self._lock = Lock()

    def _get_model(self) -> SentenceTransformer | None:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                return None
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001
                self._load_error = exc
                logger.warning(
                    "Semantic model load failed (%s). Semantic checks will be skipped.",
                    exc,
                )
                return None
            return self._model

    async def score(self, source_payload: Any, candidate_payload: Any) -> float:
        model = self._get_model()
        if model is None:
            return 1.0

        source_text = json_dumps(source_payload)
        candidate_text = json_dumps(candidate_payload)

        embeddings = await asyncio.to_thread(
            model.encode,
            [source_text, candidate_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        similarity = float((embeddings[0] * embeddings[1]).sum())
        return similarity

    async def is_valid(
        self,
        source_payload: Any,
        candidate_payload: Any,
        threshold: float | None = None,
    ) -> bool:
        min_score = self.threshold if threshold is None else threshold
        similarity = await self.score(source_payload, candidate_payload)
        return similarity >= min_score

