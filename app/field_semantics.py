from __future__ import annotations

import logging
import re
from functools import lru_cache
from threading import Lock
from typing import Iterable


logger = logging.getLogger(__name__)


_POSITIVE_PROTOTYPES = (
    "A field that uniquely identifies one resource in a system.",
    "A field used as a lookup key for retrieving an entity.",
    "A field representing an index, order, sequence, or hierarchical locator.",
    "A field that references another entity as a foreign key.",
)

_NEGATIVE_PROTOTYPES = (
    "A field with descriptive natural-language content.",
    "A field containing display text, title, or explanation.",
    "A field that stores status wording or metadata description.",
    "A field for narrative commentary or message text.",
)

_POSITIVE_FALLBACK_STEMS = (
    "id",
    "identifier",
    "locator",
    "reference",
    "number",
    "key",
    "code",
    "index",
    "sequence",
    "serial",
    "ordinal",
    "position",
    "rank",
    "level",
    "parent",
    "child",
    "token",
    "uuid",
    "slug",
    "handle",
)

_NEGATIVE_FALLBACK_STEMS = (
    "name",
    "title",
    "description",
    "summary",
    "message",
    "content",
    "commentary",
    "translation",
    "text",
    "note",
    "status",
    "email",
    "address",
)


class IdentifierSemanticAnalyzer:
    _model = None
    _load_error: Exception | None = None
    _load_lock = Lock()
    _prototype_embeddings: tuple[list[list[float]], list[list[float]]] | None = None

    @classmethod
    def is_identifier_like(
        cls,
        field_name: str | None,
        context_keys: Iterable[str] | None = None,
    ) -> bool:
        if not field_name:
            return False

        field_phrase = cls._field_to_phrase(field_name)
        compact_field = cls._compact_key(field_name)
        if not field_phrase or not compact_field:
            return False

        if cls._matches_context_key(compact_field, context_keys):
            return True

        semantic_vote = cls._semantic_vote(field_phrase)
        if semantic_vote is not None:
            return semantic_vote

        return cls._fallback_vote(field_phrase)

    @classmethod
    def _semantic_vote(cls, field_phrase: str) -> bool | None:
        score = cls._semantic_margin(field_phrase)
        if score is None:
            return None

        if score >= 0.085:
            return True
        if score <= -0.08:
            return False
        return None

    @classmethod
    @lru_cache(maxsize=2048)
    def _semantic_margin(cls, field_phrase: str) -> float | None:
        model = cls._get_model()
        if model is None:
            return None

        prototype_embeddings = cls._get_prototype_embeddings(model)
        if prototype_embeddings is None:
            return None

        positive_embeddings, negative_embeddings = prototype_embeddings
        query_text = (
            "Field name in an API schema: "
            f"{field_phrase}"
        )
        encoded = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if len(encoded) != 1:
            return None

        query = encoded[0]
        positive_score = cls._mean_cosine_similarity(query, positive_embeddings)
        negative_score = cls._mean_cosine_similarity(query, negative_embeddings)
        return positive_score - negative_score

    @classmethod
    def _get_model(cls):
        if cls._model is not None:
            return cls._model
        if cls._load_error is not None:
            return None

        with cls._load_lock:
            if cls._model is not None:
                return cls._model
            if cls._load_error is not None:
                return None

            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # noqa: BLE001
                cls._load_error = exc
                logger.debug("Identifier semantic model unavailable: %s", exc)
                return None

            try:
                cls._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:  # noqa: BLE001
                cls._load_error = exc
                logger.debug("Failed to load identifier semantic model: %s", exc)
                return None

        return cls._model

    @classmethod
    def _get_prototype_embeddings(cls, model) -> tuple[list[list[float]], list[list[float]]] | None:
        if cls._prototype_embeddings is not None:
            return cls._prototype_embeddings

        with cls._load_lock:
            if cls._prototype_embeddings is not None:
                return cls._prototype_embeddings
            try:
                positive = model.encode(
                    list(_POSITIVE_PROTOTYPES),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                negative = model.encode(
                    list(_NEGATIVE_PROTOTYPES),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to encode identifier prototypes: %s", exc)
                return None

            cls._prototype_embeddings = (
                [vector for vector in positive],
                [vector for vector in negative],
            )
            return cls._prototype_embeddings

    @staticmethod
    def _mean_cosine_similarity(query_vector, prototype_vectors: list[list[float]]) -> float:
        if not prototype_vectors:
            return 0.0
        score = 0.0
        for vector in prototype_vectors:
            score += float((query_vector * vector).sum())
        return score / float(len(prototype_vectors))

    @classmethod
    def _fallback_vote(cls, field_phrase: str) -> bool:
        tokens = cls._tokenize(field_phrase)
        if not tokens:
            return False

        score = 0.0
        for token in tokens:
            positive_similarity = max(cls._token_similarity(token, stem) for stem in _POSITIVE_FALLBACK_STEMS)
            negative_similarity = max(cls._token_similarity(token, stem) for stem in _NEGATIVE_FALLBACK_STEMS)
            if positive_similarity >= 0.9 and positive_similarity > (negative_similarity + 0.1):
                score += 0.42
                continue
            if negative_similarity >= 0.9 and negative_similarity > (positive_similarity + 0.08):
                score -= 0.52
                continue
            score += (positive_similarity - negative_similarity) * 0.4

        compact_field = cls._compact_key(field_phrase)
        if compact_field.isdigit():
            score += 0.12
        if any(char.isdigit() for char in compact_field) and len(tokens) <= 3:
            score += 0.06

        return score >= 0.34

    @staticmethod
    def _token_similarity(token: str, reference: str) -> float:
        if not token or not reference:
            return 0.0
        if token == reference:
            return 1.0
        if token.startswith(reference) or reference.startswith(token):
            return 0.92
        longest = 0
        token_len = len(token)
        ref_len = len(reference)
        for i in range(token_len):
            for j in range(i + 1, token_len + 1):
                fragment = token[i:j]
                if fragment and fragment in reference:
                    longest = max(longest, len(fragment))
        return float(longest) / float(max(token_len, ref_len))

    @classmethod
    def _matches_context_key(
        cls,
        compact_field: str,
        context_keys: Iterable[str] | None,
    ) -> bool:
        if context_keys is None:
            return False

        for key in context_keys:
            candidate = cls._compact_key(key)
            if not candidate:
                continue
            if compact_field == candidate:
                return True
            if compact_field.endswith(candidate) or candidate.endswith(compact_field):
                return True
        return False

    @classmethod
    def _tokenize(cls, field_phrase: str) -> list[str]:
        tokens: list[str] = []
        for token in field_phrase.split():
            canonical = cls._canonicalize_token(token)
            if canonical:
                tokens.append(canonical)
        return tokens

    @staticmethod
    def _canonicalize_token(token: str) -> str:
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        return token

    @staticmethod
    def _compact_key(value: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @staticmethod
    def _field_to_phrase(value: str | None) -> str:
        text = str(value or "")
        if not text:
            return ""
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
        return " ".join(part for part in text.split() if part)
