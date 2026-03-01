from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import zlib

from dataclasses import dataclass


@dataclass
class _DecodeAttempt:
    # Full token ids including prompt + generated (no EOT inc:contentReference[oaicite:5]{index=5}se to append it)
    token_ids: list[int]

    # Whisper-style metrics used for fallback gating
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    # Metadata
    temperature: float

    # Internal bookkeeping (useful for ranking best_of samples)
    sum_logprob: float
    gen_len: int


class WhisperDecoderRuntimeMixin:
    def transcribe_wav(self, wav_path: Path, language: str | None = None) -> str:
        self.logger.info("Starting QNN transcription: %s (language=%s)", wav_path, language or "auto")

        if self.debug:
            self._debug_logits_fallback_warned = False

        audio = self._load_and_log_audio(Path(wav_path))
        features = self._extract_and_pack_features(audio)
        enc_cross_cache = self._run_encoder(features)

        prompt_ids = self._build_prompt_ids(language)
        if self.debug:
            self.logger.info("Prompt ids: %s", prompt_ids)
            self.logger.info("Prompt tokens: %s", self.tokenizer.decode(prompt_ids, skip_special_tokens=False))

        if not prompt_ids:
            raise RuntimeError("Prompt ids empty.")

        # Whisper-style fallback loop (no beam for now; temp=0 greedy, temp>0 sampling + best_of)
        attempt = self._decode_with_fallback(
            prompt_ids,
            enc_cross_cache,
            temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            best_of=5,
            seed=0,
        )

        return self._final_decode_and_log(attempt.token_ids)

    def _run_encoder(self, features: np.ndarray) -> dict[str, np.ndarray]:
        # ----- encoder -> cross-cache outputs -----
        enc_inputs = {self.encoder_input_name: features}
        t0 = time.perf_counter()
        enc_out = self.encoder_session.run(self.encoder_cross_cache_names, enc_inputs)
        self.logger.info("Encoder run returned in %.3fs", time.perf_counter() - t0)
        return {n: v for n, v in zip(self.encoder_cross_cache_names, enc_out)}

    # ---------------------------
    # Whisper heuristics helpers
    # ---------------------------

    def _compression_ratio(self, text: str) -> float:
        # Matches Whisper's zlib-based compression_ratio() utility :contentReference[oaicite:6]{index=6}
        b = text.encode("utf-8")
        if not b:
            return 0.0
        return len(b) / max(1, len(zlib.compress(b)))

    def _get_no_speech_token_id(self) -> int | None:
        # Whisper uses "<|nospeech|>"
        try:
            tid = self.tokenizer.convert_tokens_to_ids("<|nospeech|>")
        except Exception:
            tid = None
        if tid is None:
            return None
        tid = int(tid)
        return tid if tid >= 0 else None

    def _softmax_prob_from_scores(self, scores: np.ndarray, token_id: int) -> float:
        # stable softmax prob for one token
        s = scores.astype(np.float32, copy=False)
        m = float(np.max(s))
        p = np.exp(s - m)
        denom = float(np.sum(p))
        if denom <= 0.0:
            return 0.0
        return float(p[int(token_id)] / denom)

    def _no_speech_prob_from_logits(self, logits: np.ndarray) -> float:
        # IMPORTANT: no_speech_prob is measured on *raw* scores (no suppression filters),
        # analogous to Whisper's approach in its decoding pipeline :contentReference[oaicite:7]{index=7}
        tid = self._get_no_speech_token_id()
        if tid is None:
            return float("nan")
        scores = self._logits_to_scores(logits).astype(np.float32, copy=False)
        if not (0 <= tid < scores.shape[0]):
            return float("nan")
        return self._softmax_prob_from_scores(scores, tid)

    def _token_logprob(self, logits: np.ndarray, token_id: int, *, at_sample_begin: bool) -> float:
        # logprob under the same filtered distribution used for selection
        scores = self._filtered_scores(logits, at_sample_begin=at_sample_begin)

        m = float(np.max(scores))
        lse = m + float(np.log(np.sum(np.exp(scores - m))))
        return float(scores[int(token_id)] - lse)

    def _sample_token(self, logits: np.ndarray, temperature: float, rng: np.random.Generator, *, at_sample_begin: bool) -> int:
        scores = self._filtered_scores(logits, at_sample_begin=at_sample_begin)

        # temperature scaling
        scores = scores / float(temperature)

        # softmax
        m = float(np.max(scores))
        p = np.exp(scores - m)
        p = p / float(np.sum(p))

        return int(rng.choice(p.size, p=p))

    # ---------------------------
    # Decoder (prefill/generate)
    # ---------------------------

    def _decoder_prefill(
        self,
        prompt_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray], int, list[int], float]:
        """
        Prefill each prompt token sequentially (because input_ids is [1,1]).
        Captures no_speech_prob from the logits after the first token (SOT).
        """
        kv_cache = self._initialize_kv_cache() if self.has_kv_cache else {}

        logits = None
        pos = 0
        prev_cache_out: dict[str, np.ndarray] | None = None

        no_speech_prob = float("nan")

        for t, tok in enumerate(prompt_ids):
            if self.debug_kv and prev_cache_out is not None:
                wiring_ok = self._cache_dicts_equal(kv_cache, prev_cache_out)
                self.logger.info("PREFILL wiring: cache_in(t+1)==cache_out(t): %s", wiring_ok)

            cache_in = kv_cache
            logits, kv_cache = self._decoder_step(
                token_id=int(tok),
                pos=pos,
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )

            # Capture no_speech_prob from the SOT-step logits (t==0) using raw logits.
            # This aligns with the intent of Whisper's no_speech gating heuristic. :contentReference[oaicite:8]{index=8}
            if t == 0 and logits is not None:
                no_speech_prob = self._no_speech_prob_from_logits(logits)

            if self.debug_kv:
                layers_match, common_top_idx, global_max = self._cache_delta_summary(cache_in, kv_cache, pos)
                tok_str = self.tokenizer.decode([int(tok)], skip_special_tokens=False)
                self.logger.info(
                    "PREFILL t=%d token=%d/%r pos=%d cache_delta: layers_match_pos=%d/%d common_top_idx=%d global_max=%s",
                    t,
                    int(tok),
                    tok_str,
                    pos,
                    layers_match,
                    len(cache_in),
                    common_top_idx,
                    global_max,
                )

            prev_cache_out = kv_cache
            pos += 1
            if pos >= self.self_cache_len:
                break

        input_ids: list[int] = prompt_ids.copy()
        return logits, kv_cache, pos, input_ids, float(no_speech_prob)

    def _decoder_generate(
        self,
        logits: np.ndarray | None,
        kv_cache: dict[str, np.ndarray],
        pos: int,
        input_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> list[int]:
        """
        (Optional legacy path) Deterministic greedy generation.
        NOTE: This is *not* used by the Whisper-style fallback path below.
        """
        eot_token = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eot_token < 0:
            raise RuntimeError("Tokenizer eos_token_id missing.")

        remaining_positions = max(0, (self.self_cache_len - pos))
        max_new_tokens = min(200, remaining_positions)
        prompt_len = len(input_ids)

        self.logger.info(
            "Decoder prefill done. pos=%d self_cache_len=%d attn_max_len=%d -> max_new_tokens=%d",
            pos, self.self_cache_len, self.attn_max_len, max_new_tokens
        )

        for _ in range(max_new_tokens):
            if logits is None:
                last_tok = input_ids[-1]
                logits, kv_cache = self._decoder_step(
                    token_id=int(last_tok),
                    pos=max(0, pos - 1),
                    kv_cache=kv_cache,
                    enc_cross_cache=enc_cross_cache,
                )

            at_sample_begin = (len(input_ids) == prompt_len)
            next_token = int(self._select_next_token_from_logits(logits, generated=None, at_sample_begin=at_sample_begin))

            if next_token == eot_token:
                break

            input_ids.append(next_token)

            logits, kv_cache = self._decoder_step(
                token_id=next_token,
                pos=pos,
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )

            pos += 1
            if pos >= self.self_cache_len:
                break

        return input_ids

    def _final_decode_and_log(self, input_ids: list[int]) -> str:
        decoded_raw = self.tokenizer.decode(input_ids, skip_special_tokens=False).strip()
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()

        self.logger.info("DECODE (no-skip)='%s'", decoded_raw[:200])
        self.logger.info("DECODE (skip)='%s'", decoded[:200])
        self.logger.info("Completed QNN transcription (chars=%d).", len(decoded))
        return decoded

    def _decoder_step(
        self,
        token_id: int,
        pos: int,
        kv_cache: dict[str, np.ndarray],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run ONE decoder call. The model expects input_ids [1,1], attention_mask [1,1,1,200], position_ids [1]."""
        decoder_inputs: dict[str, Any] = {}

        input_ids_dtype = self._dtype_for_input(self.decoder_input_ids_name, fallback=np.int32)
        decoder_inputs[self.decoder_input_ids_name] = np.array([[token_id]], dtype=input_ids_dtype)

        count = min(pos + 1, self.self_cache_len)

        if self.profile.name == "large-v3-turbo":
            attn = np.full((1, 1, 1, self.attn_max_len), np.float16(-65504.0), dtype=np.float16)
            attn[0, 0, 0, :count] = np.float16(0.0)
            decoder_inputs[self.decoder_attention_mask_name] = attn
            if self.debug:
                self.logger.info("attn_mask float16 min=%.1f max=%.1f", float(attn.min()), float(attn.max()))
        else:
            attn = np.zeros((1, 1, 1, self.attn_max_len), dtype=np.uint16)
            attn[0, 0, 0, -count:] = np.uint16(65535)
            decoder_inputs[self.decoder_attention_mask_name] = attn
            self.logger.info("attn_mask uint16 min=%d max=%d", int(attn.min()), int(attn.max()))

        pid_dtype = self._dtype_for_input(self.decoder_position_ids_name, fallback=np.int32)
        if self.profile.name == "large-v3-turbo":
            pid = count - 1   # forward positions
        else:
            pid = self.self_cache_len - count  # reverse positions
        decoder_inputs[self.decoder_position_ids_name] = np.array([pid], dtype=pid_dtype)

        if self.has_kv_cache:
            decoder_inputs.update(kv_cache)

        if self.decoder_uses_cross_cache:
            decoder_inputs.update(enc_cross_cache)

        self.logger.info("pos=%d count=%d pid=%d", pos, count, pid)

        t0 = time.perf_counter()
        outputs = self.decoder_session.run(None, decoder_inputs)

        dt = time.perf_counter() - t0
        if self.debug_kv:
            self.logger.info("Decoder session.run() returned in %.3fs", dt)

        output_map = {node.name: value for node, value in zip(self.decoder_io.outputs, outputs)}
        logits = output_map[self.decoder_logits_name]

        if self.has_kv_cache:
            new_cache = {name.replace("_out", "_in"): output_map[name] for name in self.kv_self_out_names}
        else:
            new_cache = {}

        return logits, new_cache

    def _build_prompt_ids(self, language: str | None) -> list[int]:
        lang = (language or "en").lower()

        start = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if start is None or int(start) < 0:
            raise RuntimeError("Cannot resolve WhisperTokenizer SOT token id.")

        rest: list[int] = []
        if hasattr(self.tokenizer, "get_decoder_prompt_ids"):
            items = self.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
            rest = [int(tid) for _, tid in items]

        return [int(start)] + rest

    def _initialize_kv_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for node in self.decoder_io.inputs:
            if node.name in self.kv_self_in_names:
                shape = tuple(int(d) for d in node.shape)
                dtype = self._numpy_dtype_from_ort(node.type)
                if dtype == np.uint8:
                    cache[node.name] = np.full(shape, 128, dtype=np.uint8)  # common zero-point
                else:
                    cache[node.name] = np.zeros(shape, dtype=dtype)
        return cache

    # ---------------------------
    # Whisper-style single decode
    # ---------------------------

    def _decode_once(
        self,
        prompt_ids,
        enc_cross_cache,
        temperature: float,
        seed: int,
        max_new_tokens_cap: int = 96,
    ) -> _DecodeAttempt:
        logits, kv_cache, pos, input_ids, no_speech_prob = self._decoder_prefill(prompt_ids, enc_cross_cache)
        prompt_len = len(input_ids)

        rng = np.random.default_rng(seed)
        eot = int(getattr(self.tokenizer, "eos_token_id", -1))

        sum_lp = 0.0
        gen_tokens: list[int] = []

        remaining_positions = max(0, (self.self_cache_len - pos))
        max_new_tokens = min(max_new_tokens_cap, 200, remaining_positions)

        for _ in range(max_new_tokens):
            at_sample_begin = (len(input_ids) == prompt_len)

            if logits is None:
                # safety fallback: compute logits from last token
                last_tok = input_ids[-1]
                logits, kv_cache = self._decoder_step(
                    token_id=int(last_tok),
                    pos=max(0, pos - 1),
                    kv_cache=kv_cache,
                    enc_cross_cache=enc_cross_cache,
                )

            if temperature == 0.0:
                next_token = int(self._select_next_token_from_logits(logits, generated=None, at_sample_begin=at_sample_begin))
            else:
                next_token = int(self._sample_token(logits, temperature, rng, at_sample_begin=at_sample_begin))

            lp = self._token_logprob(logits, next_token, at_sample_begin=at_sample_begin)

            if next_token == eot:
                break

            input_ids.append(next_token)
            gen_tokens.append(next_token)
            sum_lp += float(lp)

            logits, kv_cache = self._decoder_step(
                token_id=next_token,
                pos=pos,
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )
            pos += 1
            if pos >= self.self_cache_len:
                break

        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        cr = float(self._compression_ratio(text))

        gen_len = len(gen_tokens)

        # avg_logprob as used by the original implementation (sum_logprobs / len(tokens)) :contentReference[oaicite:9]{index=9}
        avg_lp = float(sum_lp / max(1, gen_len))

        return _DecodeAttempt(
            token_ids=prompt_ids + gen_tokens,
            avg_logprob=avg_lp,
            compression_ratio=cr,
            no_speech_prob=float(no_speech_prob),
            temperature=float(temperature),
            sum_logprob=float(sum_lp),
            gen_len=int(gen_len),
        )

    # ---------------------------
    # Whisper-style fallback loop
    # ---------------------------

    def _decode_with_fallback(
        self,
        prompt_ids,
        enc_cross_cache,
        temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        best_of=5,
        seed=0,
        max_new_tokens_cap: int = 96,
    ) -> _DecodeAttempt:
        """
        Mirrors Whisper's decode_with_fallback logic:
          - try temps in order
          - fail if too repetitive (compression ratio) or too low-confidence (avg_logprob)
          - BUT if no_speech_prob is high AND avg_logprob is low, treat as silence (no fallback)
        :contentReference[oaicite:10]{index=10}
        """
        decode_result: _DecodeAttempt | None = None

        for t in temperatures:
            t = float(t)

            if t == 0.0:
                decode_result = self._decode_once(
                    prompt_ids,
                    enc_cross_cache,
                    temperature=0.0,
                    seed=seed,
                    max_new_tokens_cap=max_new_tokens_cap,
                )
            else:
                cands = [
                    self._decode_once(
                        prompt_ids,
                        enc_cross_cache,
                        temperature=t,
                        seed=seed + 1000 + i,
                        max_new_tokens_cap=max_new_tokens_cap,
                    )
                    for i in range(int(best_of))
                ]

                # pick best candidate at this temperature (max avg_logprob)
                decode_result = max(cands, key=lambda a: (a.sum_logprob / max(1, a.gen_len)))

            # --- gating ---
            needs_fallback = False

            if compression_ratio_threshold is not None and decode_result.compression_ratio > float(compression_ratio_threshold):
                needs_fallback = True

            if logprob_threshold is not None and decode_result.avg_logprob < float(logprob_threshold):
                needs_fallback = True

            # Silence override (exact condition described in transcribe.py) :contentReference[oaicite:11]{index=11}
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > float(no_speech_threshold)
                and logprob_threshold is not None
                and decode_result.avg_logprob < float(logprob_threshold)
            ):
                needs_fallback = False

            txt = self.tokenizer.decode(decode_result.token_ids, skip_special_tokens=True).strip()
            self.logger.info(
                "fallback attempt: temp=%.1f avg_logprob=%.3f comp_ratio=%.3f no_speech_prob=%.3f needs_fallback=%s text=%r",
                t,
                float(decode_result.avg_logprob),
                float(decode_result.compression_ratio),
                float(decode_result.no_speech_prob),
                bool(needs_fallback),
                txt[:120],
            )

            if not needs_fallback:
                break

        if decode_result is None:
            raise RuntimeError("decode_with_fallback: no attempts produced")

        return decode_result