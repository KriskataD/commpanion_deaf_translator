from __future__ import annotations

from collections import Counter

import numpy as np


class WhisperTokenSelectionMixin:
    def _has_consecutive_repeat(self, toks: list[int], run: int = 4) -> bool:
        if len(toks) < run:
            return False
        last = toks[-1]
        return all(t == last for t in toks[-run:])

    def _has_abab_loop(self, toks: list[int], pairs: int = 4) -> bool:
        # Detect ABABABAB at the end (pairs=4 => 8 tokens)
        need = 2 * pairs
        if len(toks) < need:
            return False
        a = toks[-need]
        b = toks[-need + 1]
        for i in range(need):
            expected = a if (i % 2 == 0) else b
            if toks[-need + i] != expected:
                return False
        return True

    def _ngram_max_repeat(self, toks: list[int], n: int, window: int = 80) -> int:
        if len(toks) < n:
            return 0
        seq = toks[-window:] if len(toks) > window else toks
        counts: dict[tuple[int, ...], int] = {}
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i:i + n])
            counts[g] = counts.get(g, 0) + 1
        return max(counts.values()) if counts else 0

    def _no_repeat_ngram_bans(self, generated: list[int], n: int) -> set[int]:
        """
        Standard no-repeat-ngram constraint:
        If we've already seen an n-gram with prefix = last (n-1) tokens,
        ban the token(s) that would recreate those n-grams.
        """
        if n <= 1:
            return set()
        if len(generated) < (n - 1):
            return set()

        prefix = tuple(generated[-(n - 1):])
        banned: set[int] = set()

        # Collect all seen n-grams, but only those whose first (n-1) match current prefix
        # so we can ban the next token.
        for i in range(len(generated) - n + 1):
            gram = tuple(generated[i:i + n])
            if gram[:-1] == prefix:
                banned.add(int(gram[-1]))

        return banned

    def _apply_frequency_penalty(self, scores: np.ndarray, generated: list[int], alpha: float = 0.15) -> np.ndarray:
        """
        Mild deterministic penalty: decrease scores for tokens we've already used.
        This helps reduce 'engineering engineering engineering...' without randomness.

        alpha ~ 0.10–0.30 is a reasonable range for float scores.
        """
        if not generated:
            return scores

        # Use a tail window so we don't over-penalize legitimate earlier words.
        tail = generated[-200:] if len(generated) > 200 else generated
        counts = Counter(tail)

        for tid, c in counts.items():
            if 0 <= tid < scores.shape[0]:
                scores[tid] -= float(alpha) * float(c)

        return scores

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

    def _pick_token_with_loop_blocking(self, scores: np.ndarray, generated: list[int]) -> int:
        """
        Deterministic: greedy argmax, but with:
        - no-repeat-ngram bans (n=4 by default)
        - mild frequency penalty
        - iterative banning if we still try to extend a loop
        """
        # Work on a copy so callers can reuse original if needed
        work = scores.astype(np.float32, copy=True)

        # 1) mild frequency penalty (deterministic)
        work = self._apply_frequency_penalty(work, generated, alpha=0.15)

        # 2) no-repeat ngram bans (phrase repetition killer)
        no_repeat_n = 3  # try 3 if still too repetitive; try 5 if too strict
        banned = self._no_repeat_ngram_bans(generated, n=no_repeat_n)
        for tid in banned:
            if 0 <= tid < work.shape[0]:
                work[tid] = -1e9

        # 3) iterative banning for obvious token-level loops
        # Try a few times: if best candidate would cause a loop, ban it and retry.
        for _ in range(8):
            t0 = int(np.argmax(work))

            tmp = generated + [t0]
            bad = (
                self._has_consecutive_repeat(tmp, run=3)
                or self._has_abab_loop(tmp, pairs=2)
                or (self._ngram_max_repeat(tmp, n=3, window=80) >= 3)
            )

            if not bad:
                return t0

            # ban and retry
            work[t0] = -1e9

        # Fallback: if everything got banned, return the original greedy token
        return int(np.argmax(scores))

    def _select_next_token_from_logits(self, logits: np.ndarray, generated: list[int] | None = None) -> int:
        scores = self._logits_to_scores(logits)
        self.logger.info("scores stats: min=%.4f max=%.4f", float(scores.min()), float(scores.max()))
        scores = self._apply_suppression_to_scores(scores)

        #eot = int(getattr(self.tokenizer, "eos_token_id", -1))
        #if getattr(self, "_block_eot_steps", 0) > 0 and 0 <= eot < scores.shape[0]:
        #    scores[eot] = -1e9
        #    self._block_eot_steps -= 1

        # If no history provided, fall back to plain greedy
        if not generated:
            return int(np.argmax(scores))

        # Deterministic loop-blocking greedy
        return int(self._pick_token_with_loop_blocking(scores, generated))

    def _topk_from_logits(self, logits: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
        scores = self._logits_to_scores(logits)
        top = np.argsort(scores)[-k:][::-1]
        return [int(i) for i in top], [float(scores[int(i)]) for i in top]

    def _logits_to_scores(self, logits: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.squeeze(logits)).reshape(-1)

        if x.dtype == np.uint16:
            # From your decoder graph inspection:
            scale = np.float32(0.0012925398768857121)
            zp    = np.float32(17867.0)

            scores = (x.astype(np.float32) - zp) * scale
            return scores

        return x.astype(np.float32)
