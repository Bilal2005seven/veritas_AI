"""
VeritasAI V1 — Transformer NLI Service
=======================================
Performs three-class Natural Language Inference (NLI) between a *claim* and
a single *evidence* passage using a pre-trained RoBERTa-based cross-encoder.

Model
-----
``cross-encoder/nli-MiniLM2-L6-H768``

Why this model:
* Trained on MultiNLI + SNLI for three-class NLI (ENTAILMENT / NEUTRAL /
  CONTRADICTION) — exactly the task required.
* Cross-encoder architecture: the claim and evidence are fed together as a
  single sequence, so attention flows freely between them (unlike bi-encoders,
  which encode them separately).
* Compact (~66 M parameters, ~240 MB) and fast on CPU (~50 ms per pair on a
  modern laptop) — suitable for the Windows/CPU-only development environment.
* Distinct from ``sentence-transformers/all-MiniLM-L6-v2`` used for
  embeddings; it has a classification head instead of a pooling head.
* Actively maintained on HuggingFace Hub with known-good label ordering:
  index 0 = contradiction, index 1 = entailment, index 2 = neutral.

Design decisions
----------------
* **Lazy loading** — tokenizer and model are instantiated only on the first
  call to ``analyze()``.  Importing this module is instant.
* **Synchronous inference** — ``torch`` inference is CPU-bound; running it in
  an async wrapper would provide no benefit without a thread pool.  The public
  ``analyze()`` is therefore a plain function, consistent with how callers
  (Evidence Engine) will use it.
* **Single pair per call** — NLI is applied to one (claim, evidence) pair at
  a time.  Batching across multiple pieces of evidence is the caller's
  responsibility so that each result can be inspected independently.
* **No embeddings** — this service is entirely separate from
  ``app/rag/embeddings.py``.

Label mapping (model-specific)
-------------------------------
The ``cross-encoder/nli-MiniLM2-L6-H768`` model stores its id2label in the
config.  We read it at load time and fall back to a hard-coded default when
the config does not provide it.  Labels are normalised to upper-case so
downstream code can use string equality reliably.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from app.models.schemas import TransformerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "cross-encoder/nli-MiniLM2-L6-H768"

#: Maximum combined token length for (claim + evidence) pair.
#: The model's context window is 512 tokens; we leave a small margin.
MAX_LENGTH: int = 512

#: Fallback label order when the model config does not declare id2label.
#: Verified empirically for cross-encoder/nli-MiniLM2-L6-H768.
_FALLBACK_LABELS: dict[int, str] = {
    0: "CONTRADICTION",
    1: "ENTAILMENT",
    2: "NEUTRAL",
}

# Canonical labels used throughout the pipeline.
LABEL_ENTAILMENT: str = "ENTAILMENT"
LABEL_CONTRADICTION: str = "CONTRADICTION"
LABEL_NEUTRAL: str = "NEUTRAL"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TransformerService:
    """
    Wraps ``cross-encoder/nli-MiniLM2-L6-H768`` for claim–evidence NLI.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to
        ``cross-encoder/nli-MiniLM2-L6-H768``.

    Attributes
    ----------
    model_name : str
        The model identifier this service was initialised with.

    Examples
    --------
    >>> svc = TransformerService()
    >>> result = svc.analyze(
    ...     claim="The suspect was arrested by police.",
    ...     evidence="Authorities confirmed the arrest of a suspect last night.",
    ... )
    >>> result.label
    'ENTAILMENT'
    >>> result.entailment_score  # float in [0, 1]
    0.97...
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name: str = model_name
        self._tokenizer: Optional[object] = None   # loaded lazily
        self._model: Optional[object] = None       # loaded lazily
        self._id2label: dict[int, str] = _FALLBACK_LABELS.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load the tokenizer and model into memory (idempotent).

        Uses ``AutoTokenizer`` and ``AutoModelForSequenceClassification`` from
        the ``transformers`` library.  Model is placed on CPU and set to
        ``eval()`` mode.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        RuntimeError
            If the model cannot be loaded from the Hub or local cache.
        """
        if self._model is not None:
            return  # already loaded

        try:
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Run: pip install transformers"
            ) from exc

        logger.info("Loading NLI tokenizer '%s' …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info("Loading NLI model '%s' …", self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.eval()  # type: ignore[union-attr]

        # Read label mapping from model config if available.
        config_id2label: dict = getattr(
            getattr(self._model, "config", None), "id2label", {}
        )
        if config_id2label:
            self._id2label = {
                int(k): str(v).upper() for k, v in config_id2label.items()
            }
            logger.debug("NLI label mapping from config: %s", self._id2label)
        else:
            logger.debug(
                "No id2label in model config; using fallback: %s",
                self._id2label,
            )

        logger.info("NLI model loaded (labels: %s).", list(self._id2label.values()))

    @staticmethod
    def _validate_input(claim: str, evidence: str) -> None:
        """
        Raise ``TypeError`` or ``ValueError`` for obviously bad inputs.

        Parameters
        ----------
        claim : str
            Claim string.
        evidence : str
            Evidence string.

        Raises
        ------
        TypeError
            If either argument is not a string.
        ValueError
            If either argument is empty or whitespace-only.
        """
        if not isinstance(claim, str):
            raise TypeError(
                f"claim must be a str, got {type(claim).__name__!r}"
            )
        if not isinstance(evidence, str):
            raise TypeError(
                f"evidence must be a str, got {type(evidence).__name__!r}"
            )
        if not claim.strip():
            raise ValueError("claim must not be empty or whitespace-only.")
        if not evidence.strip():
            raise ValueError("evidence must not be empty or whitespace-only.")

    def _scores_to_result(self, logits: torch.Tensor) -> TransformerResult:
        """
        Convert raw logits to a :class:`~app.models.schemas.TransformerResult`.

        Applies softmax to produce a probability distribution, then maps each
        index to the canonical label via ``self._id2label``.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(1, num_labels)`` or ``(num_labels,)`` raw logits from the
            model.

        Returns
        -------
        TransformerResult
            Populated with label, dominant score, and all three class scores.
        """
        if logits.dim() == 2:
            logits = logits[0]  # (num_labels,)

        probs: torch.Tensor = F.softmax(logits.float(), dim=-1)  # (num_labels,)
        probs_list: list[float] = probs.tolist()

        # Map index → (label, probability).
        label_scores: dict[str, float] = {}
        for idx, prob in enumerate(probs_list):
            label = self._id2label.get(idx, f"CLASS_{idx}").upper()
            label_scores[label] = float(prob)

        # Dominant label = argmax.
        dominant_label: str = max(label_scores, key=label_scores.__getitem__)
        dominant_score: float = label_scores[dominant_label]

        return TransformerResult(
            label=dominant_label,
            score=round(dominant_score, 6),
            entailment_score=round(label_scores.get(LABEL_ENTAILMENT, 0.0), 6),
            contradiction_score=round(
                label_scores.get(LABEL_CONTRADICTION, 0.0), 6
            ),
            neutral_score=round(label_scores.get(LABEL_NEUTRAL, 0.0), 6),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, claim: str, evidence: str) -> TransformerResult:
        """
        Run NLI inference on a single (claim, evidence) pair.

        Parameters
        ----------
        claim : str
            The news claim or statement to check.  Must be non-empty.
        evidence : str
            A single evidence passage retrieved from the web or a corpus.
            Must be non-empty.

        Returns
        -------
        TransformerResult
            Contains:
            * ``label`` — dominant class (``ENTAILMENT``, ``CONTRADICTION``,
              or ``NEUTRAL``).
            * ``score`` — softmax probability of the dominant label.
            * ``entailment_score`` — probability for ENTAILMENT.
            * ``contradiction_score`` — probability for CONTRADICTION.
            * ``neutral_score`` — probability for NEUTRAL.

        Raises
        ------
        TypeError
            If *claim* or *evidence* is not a ``str``.
        ValueError
            If *claim* or *evidence* is empty or whitespace-only.
        RuntimeError
            If the model cannot be loaded.

        Notes
        -----
        Inference runs on CPU with ``torch.no_grad()``.  For batched
        evaluation across multiple evidence passages, call this method once
        per passage and aggregate results in the Evidence Engine.
        """
        self._validate_input(claim, evidence)
        self._load()

        # Tokenize as a sentence-pair sequence.
        inputs = self._tokenizer(  # type: ignore[operator]
            claim,
            evidence,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[operator]

        logits: torch.Tensor = outputs.logits
        return self._scores_to_result(logits)

    def analyze_batch(
        self,
        claim: str,
        evidence_list: list[str],
    ) -> list[TransformerResult]:
        """
        Run NLI inference for one claim against multiple evidence passages.

        This is a convenience wrapper around :meth:`analyze` that applies it
        to each item in *evidence_list* independently.  Results are returned
        in the same order as the input list.

        Parameters
        ----------
        claim : str
            The claim to check.
        evidence_list : list[str]
            One or more evidence passages.  Empty strings are skipped and
            replaced with a ``NEUTRAL`` result with zero confidence.

        Returns
        -------
        list[TransformerResult]
            One result per evidence passage, same order as input.

        Raises
        ------
        TypeError
            If *claim* is not a ``str`` or *evidence_list* is not a list.
        ValueError
            If *claim* is empty or whitespace-only.
        """
        if not isinstance(claim, str):
            raise TypeError(
                f"claim must be a str, got {type(claim).__name__!r}"
            )
        if not isinstance(evidence_list, list):
            raise TypeError(
                f"evidence_list must be a list, got "
                f"{type(evidence_list).__name__!r}"
            )
        if not claim.strip():
            raise ValueError("claim must not be empty or whitespace-only.")

        results: list[TransformerResult] = []
        for evidence in evidence_list:
            if not isinstance(evidence, str) or not evidence.strip():
                logger.warning(
                    "Skipping empty/invalid evidence in batch; "
                    "returning NEUTRAL placeholder."
                )
                results.append(
                    TransformerResult(
                        label=LABEL_NEUTRAL,
                        score=0.0,
                        entailment_score=0.0,
                        contradiction_score=0.0,
                        neutral_score=0.0,
                    )
                )
                continue
            results.append(self.analyze(claim, evidence))
        return results

    # ------------------------------------------------------------------
    # Convenience properties (for introspection / logging)
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model has been loaded into memory."""
        return self._model is not None

    @property
    def label_map(self) -> dict[int, str]:
        """Return a copy of the current id → label mapping."""
        return self._id2label.copy()
