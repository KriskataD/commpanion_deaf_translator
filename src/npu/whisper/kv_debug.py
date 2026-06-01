from __future__ import annotations

from collections import Counter

import numpy as np


class WhisperKvDebugMixin:
    def _infer_seq_axis(self, tensor: np.ndarray, self_cache_len: int) -> int:
        matches = [i for i, d in enumerate(tensor.shape) if int(d) == int(self_cache_len)]
        if not matches:
            raise RuntimeError(f"Unable to infer seq axis for cache tensor shape={tensor.shape}")
        return matches[-1]

    def _cache_delta_summary(
        self,
        cache_in: dict[str, np.ndarray],
        cache_out: dict[str, np.ndarray],
        position: int,
    ) -> tuple[int, int, float]:
        top_idxs: list[int] = []
        global_max = 0.0

        for name, in_tensor in cache_in.items():
            out_tensor = cache_out.get(name)
            if not isinstance(in_tensor, np.ndarray) or not isinstance(out_tensor, np.ndarray):
                continue

            seq_axis = self._infer_seq_axis(in_tensor, self.self_cache_len)
            if in_tensor.dtype.kind in {"u", "i"} or out_tensor.dtype.kind in {"u", "i"}:
                delta = np.abs(out_tensor.astype(np.int32) - in_tensor.astype(np.int32))
            else:
                delta = np.abs(out_tensor.astype(np.float32) - in_tensor.astype(np.float32))

            reduce_axes = tuple(ax for ax in range(delta.ndim) if ax != seq_axis)
            delta_per_pos = delta if not reduce_axes else np.max(delta, axis=reduce_axes)
            delta_per_pos = np.asarray(delta_per_pos).reshape(-1)
            top_idx = int(np.argmax(delta_per_pos))
            top_idxs.append(top_idx)
            global_max = max(global_max, float(np.max(delta_per_pos)))

        if not top_idxs:
            return 0, -1, global_max

        common_top_idx = Counter(top_idxs).most_common(1)[0][0]
        layers_match_pos = int(sum(1 for idx in top_idxs if idx == int(position)))
        return layers_match_pos, int(common_top_idx), global_max

    def _cache_dicts_equal(self, lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray]) -> bool:
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        for k in lhs:
            if not np.array_equal(lhs[k], rhs[k]):
                return False
        return True
