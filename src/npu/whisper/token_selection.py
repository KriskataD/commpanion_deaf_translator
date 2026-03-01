from __future__ import annotations

import numpy as np


class WhisperTokenSelectionMixin:
    def _ngram_max_repeat(self, toks: list[int], n: int, window: int = 80) -> int:
        if len(toks) < n:
            return 0
        seq = toks[-window:] if len(toks) > window else toks
        counts: dict[tuple[int, ...], int] = {}
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i:i + n])
            counts[g] = counts.get(g, 0) + 1
        return max(counts.values()) if counts else 0

    def _apply_suppression_to_scores(self, scores: np.ndarray) -> np.ndarray:
        # Suppress Whisper control/special tokens, but NEVER suppress EOT/eos.
        eot = int(getattr(self.tokenizer, "eos_token_id", -1))
        if getattr(self, "suppress_tokens", None):
            for tid in self.suppress_tokens:
                if tid == eot:
                    continue
                if 0 <= tid < scores.shape[0]:
                    scores[tid] = -1e9
        return scores

    def _select_next_token_from_logits(self, logits: np.ndarray, generated: list[int] | None = None) -> int:
        scores = self._logits_to_scores(logits)
        self.logger.info("scores stats: min=%.4f max=%.4f", float(scores.min()), float(scores.max()))
        scores = self._apply_suppression_to_scores(scores)

        # Match Whisper reference behavior for deterministic decoding: plain greedy.
        # Keep generated arg for backward compatibility with existing call sites.
        return int(np.argmax(scores))

    def _topk_from_logits(self, logits: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
        scores = self._logits_to_scores(logits)
        top = np.argsort(scores)[-k:][::-1]
        return [int(i) for i in top], [float(scores[int(i)]) for i in top]

    def _logits_to_scores(self, logits: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.squeeze(logits)).reshape(-1)

        if x.dtype == np.uint16:
            # From your decoder graph inspection:
            scale = np.float32(0.0012925398768857121)
            zp = np.float32(17867.0)

            scores = (x.astype(np.float32) - zp) * scale
            return scores

        return x.astype(np.float32)
