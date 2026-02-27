from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np


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

        logits, kv_cache, pos, input_ids = self._decoder_prefill(prompt_ids, enc_cross_cache)
        input_ids = self._decoder_generate(logits, kv_cache, pos, input_ids, enc_cross_cache)

        return self._final_decode_and_log(input_ids)

    def _run_encoder(self, features: np.ndarray) -> dict[str, np.ndarray]:
        # ----- encoder -> cross-cache outputs -----
        enc_inputs = {self.encoder_input_name: features}
        t0 = time.perf_counter()
        enc_out = self.encoder_session.run(self.encoder_cross_cache_names, enc_inputs)
        self.logger.info("Encoder run returned in %.3fs", time.perf_counter() - t0)
        return {n: v for n, v in zip(self.encoder_cross_cache_names, enc_out)}

    def _decoder_prefill(
        self,
        prompt_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray], int, list[int]]:
        # KV cache init
        kv_cache = self._initialize_kv_cache() if self.has_kv_cache else {}

        # Prefill each prompt token sequentially (because input_ids is [1,1])
        # This warms the self-cache and produces logits for the next token.
        logits = None
        pos = 0
        prev_cache_out: dict[str, np.ndarray] | None = None
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

        # Now generate new tokens
        input_ids: list[int] = prompt_ids.copy()
        return logits, kv_cache, pos, input_ids

    def _decoder_generate(
        self,
        logits: np.ndarray | None,
        kv_cache: dict[str, np.ndarray],
        pos: int,
        input_ids: list[int],
        enc_cross_cache: dict[str, np.ndarray],
    ) -> list[int]:
        eot_token = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eot_token < 0:
            raise RuntimeError("Tokenizer eos_token_id missing.")

        # pos currently == len(prompt_ids)  (next position index to be generated)
        remaining_positions = max(0, (self.self_cache_len - pos))
        max_new_tokens = min(200, remaining_positions)

        prompt_len = len(input_ids)

        self.logger.info(
            "Decoder prefill done. pos=%d self_cache_len=%d attn_max_len=%d -> max_new_tokens=%d",
            pos, self.self_cache_len, self.attn_max_len, max_new_tokens
        )

        if self.debug:
            eot = int(getattr(self.tokenizer, "eos_token_id", -1))
            self.logger.info("eot_token_id=%d suppressed=%s", eot, (eot in self.suppress_tokens))

        # We already have logits from the last prefill step (unless prompt was empty)
        for step in range(max_new_tokens):
            if self.debug_kv and step % 10 == 0:
                self.logger.info("Decoder gen step %d/%d", step + 1, max_new_tokens)

            if logits is None:
                # Safety: compute logits for "next token" from last known token at previous position
                last_tok = input_ids[-1]
                logits, kv_cache = self._decoder_step(
                    token_id=int(last_tok),
                    pos=max(0, pos - 1),
                    kv_cache=kv_cache,
                    enc_cross_cache=enc_cross_cache,
                )

            generated = input_ids[prompt_len:]  # only tokens generated AFTER prompt
            next_token = int(self._select_next_token_from_logits(logits, generated=generated))

            if self.debug_kv and step == 0:
                top5_ids, top5_scores = self._topk_from_logits(logits, k=5)
                top5_toks = [self.tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in top5_ids]
                chosen_tok = self.tokenizer.decode([next_token], skip_special_tokens=False)
                self.logger.info(
                    "GEN-START from last-prompt logits: top5_ids=%s top5_toks=%s chosen_id=%d chosen_tok=%r",
                    top5_ids,
                    top5_toks,
                    next_token,
                    chosen_tok,
                )
                self.logger.info(
                    "GEN-STEP0 inputs: pos=%d input_id=%d/%r (from last prompt logits)",
                    pos,
                    next_token,
                    chosen_tok,
                )

            # If model ends immediately, stop.
            if next_token == eot_token:
                input_ids.append(next_token)
                break

            # IMPORTANT: next_token belongs to CURRENT pos
            input_ids.append(next_token)

            # HARD ABORT: if we're clearly stuck in repetition, stop early rather than output garbage.
            generated_now = input_ids[prompt_len:]
            if len(generated_now) >= 60:
                uniq_ratio = len(set(generated_now[-60:])) / 60.0
                if uniq_ratio < 0.25 or self._ngram_max_repeat(generated_now, n=4, window=120) >= 3:
                    if self.debug:
                        tail_txt = self.tokenizer.decode(generated_now[-30:], skip_special_tokens=False)
                        self.logger.info("Abort decode due to repetition (uniq_ratio=%.3f). Tail=%r", uniq_ratio, tail_txt)
                    break

            logits, kv_cache = self._decoder_step(
                token_id=next_token,
                pos=pos,  # <-- correct position for this token
                kv_cache=kv_cache,
                enc_cross_cache=enc_cross_cache,
            )

            if self.debug_kv and step == 0:
                top5_ids, top5_scores = self._topk_from_logits(logits, k=5)
                top5_toks = [self.tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in top5_ids]
                self.logger.info(
                    "GEN-STEP1 logits: top5_ids=%s top5_toks=%s top5_scores=%s",
                    top5_ids,
                    top5_toks,
                    [float(s) for s in top5_scores],
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
            # attention_mask: [1,1,1,200] uint16 (plain values, not packed-fp16)
            attn = np.zeros((1, 1, 1, self.attn_max_len), dtype=np.uint16)
            attn[0, 0, 0, -count:] = np.uint16(65535)
            decoder_inputs[self.decoder_attention_mask_name] = attn
            self.logger.info("attn_mask uint16 min=%d max=%d", int(attn.min()), int(attn.max()))

        # position_ids: [1] int32 (NOT [1,1])
        pid_dtype = self._dtype_for_input(self.decoder_position_ids_name, fallback=np.int32)
        if self.profile.name == "large-v3-turbo":
            # IMPORTANT: turbo expects FORWARD positions
            pid = count - 1   # == pos unless clipped
        else:
            # small-quantized expects REVERSE positions
            pid = self.self_cache_len - count
        decoder_inputs[self.decoder_position_ids_name] = np.array([pid], dtype=pid_dtype)


        # KV cache in (self)
        if self.has_kv_cache:
            decoder_inputs.update(kv_cache)

        # Cross cache in
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

        # Update KV cache from *_out
        if self.has_kv_cache:
            new_cache = {name.replace("_out", "_in"): output_map[name] for name in self.kv_self_out_names}
        else:
            new_cache = {}

        return logits, new_cache

    def _build_prompt_ids(self, language: str | None) -> list[int]:
        lang = (language or "en").lower()

        # Pull SOT directly from WhisperTokenizer rather than relying on a numeric id.
        start = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if start is None or int(start) < 0:
            raise RuntimeError("Cannot resolve WhisperTokenizer SOT token id.")

        # Get language/task prompt ids from tokenizer (e.g. <|en|><|transcribe|><|notimestamps|>)
        rest: list[int] = []
        if hasattr(self.tokenizer, "get_decoder_prompt_ids"):
            items = self.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
            rest = [int(tid) for _, tid in items]

        return [int(start)] + rest

    def _initialize_kv_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for node in self.decoder_io.inputs:
            if node.name in self.kv_self_in_names:
                shape = tuple(int(d) for d in node.shape)  # fully static
                #cache[node.name] = np.zeros(shape, dtype=self._numpy_dtype_from_ort(node.type))
                dtype = self._numpy_dtype_from_ort(node.type)
                if dtype == np.uint8:
                    cache[node.name] = np.full(shape, 128, dtype=np.uint8)  # ✅ common zero-point
                else:
                    cache[node.name] = np.zeros(shape, dtype=dtype)
        return cache
