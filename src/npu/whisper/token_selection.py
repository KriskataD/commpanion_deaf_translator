from __future__ import annotations

import numpy as np


class WhisperTokenSelectionMixin:
    def _ngram_max_repeat(self, toks: list[int], n: int, window: int = 80) -> int:
        if len(toks) < n:
            return 0
        seq = toks[-window:] if len(toks) > window else toks
        counts: dict[tuple[int, ...], int] = {}
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i : i + n])
            counts[g] = counts.get(g, 0) + 1
        return max(counts.values()) if counts else 0

    # ---------------------------
    # Logit filters (Whisper-like)
    # ---------------------------

    def _apply_suppression_to_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Suppress Whisper control/special tokens, but NEVER suppress EOT/eos.
        """
        eot = int(getattr(self.tokenizer, "eos_token_id", -1))
        if getattr(self, "suppress_tokens", None):
            for tid in self.suppress_tokens:
                tid = int(tid)
                if tid == eot:
                    continue
                if 0 <= tid < scores.shape[0]:
                    scores[tid] = -1e9
        return scores

    def _blank_token_ids(self) -> list[int]:
        """
        Whisper-style suppress-at-begin tokens:
        Prefer config-provided begin_suppress_tokens if present.
        Otherwise fallback to tokenizer.encode(" ") + EOT.
        """
        bst = getattr(self, "begin_suppress_tokens", None)
        if bst:
            # If it's a set/list already, just normalize to ints
            return [int(t) for t in bst]

        ids: list[int] = []
        try:
            ids = list(self.tokenizer.encode(" ", add_special_tokens=False))
        except TypeError:
            try:
                ids = list(self.tokenizer.encode(" "))
            except Exception:
                ids = []

        eot = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eot >= 0:
            ids.append(eot)

        # dedupe
        out: list[int] = []
        seen: set[int] = set()
        for x in ids:
            xi = int(x)
            if xi not in seen:
                seen.add(xi)
                out.append(xi)
        return out

    def _apply_suppress_blank(self, scores: np.ndarray) -> np.ndarray:
        if not getattr(self, "suppress_blank", True):
            return scores
        for tid in self._blank_token_ids():
            if 0 <= int(tid) < scores.shape[0]:
                scores[int(tid)] = -1e9
        return scores

    def _scores_from_logits(self, logits: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.squeeze(logits)).reshape(-1)

        if x.dtype == np.uint16:
            # quantized path (your existing constants)
            scale = np.float32(0.0012925398768857121)
            zp = np.float32(17867.0)
            return (x.astype(np.float32) - zp) * scale

        return x.astype(np.float32)

    def _filtered_scores(self, logits: np.ndarray, *, at_sample_begin: bool) -> np.ndarray:
        """
        Apply logit filtering:
          - SuppressBlank only at the first generated token
          - SuppressTokens always
        """
        scores = self._scores_from_logits(logits).astype(np.float32, copy=True)

        if at_sample_begin:
            scores = self._apply_suppress_blank(scores)

        scores = self._apply_suppression_to_scores(scores)
        return scores

    # ---------------------------
    # Token selection
    # ---------------------------

    def _select_next_token_from_logits(
        self,
        logits: np.ndarray,
        generated: list[int] | None = None,
        *,
        at_sample_begin: bool = False,
    ) -> int:
        scores = self._filtered_scores(logits, at_sample_begin=at_sample_begin)
        if getattr(self, "debug", False):
            self.logger.info("scores stats: min=%.4f max=%.4f", float(scores.min()), float(scores.max()))
        return int(np.argmax(scores))

    def _topk_from_logits(self, logits: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
        scores = self._scores_from_logits(logits)
        top = np.argsort(scores)[-k:][::-1]
        return [int(i) for i in top], [float(scores[int(i)]) for i in top]

    # Backward compatibility
    def _logits_to_scores(self, logits: np.ndarray) -> np.ndarray:
        return self._scores_from_logits(logits)