# core/adversarial/samplers.py

from __future__ import annotations

import math
import random
from typing import Iterator, List, Tuple, Optional

import torch
from torch.utils.data import Sampler

BatchIndices = List[Tuple[str, int]]  # [("base" | "adv", local_index)]


class MixedBatchSampler(Sampler[BatchIndices]):
    """Batch sampler that enforces a fixed per-batch adversarial composition.

    This sampler yields lists of tagged indices, where each element is a tuple:
      ("base" | "adv", local_index)

    Design goals:
      - Enforce exact per-batch counts: adv == round(B * adv_ratio)
      - Keep iteration strictly finite (no reuse/reshuffle loops)
      - Deterministic when a torch.Generator is provided
    """

    def __init__(
        self,
        base_len: int,
        adv_len: int,
        batch_size: int,
        adv_ratio: float,
        drop_last: bool = False,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Clamp ratio to [0, 1] and compute exact counts per batch
        r = float(max(0.0, min(1.0, adv_ratio)))
        adv_per_batch = int(round(batch_size * r))
        base_per_batch = batch_size - adv_per_batch
        if adv_per_batch < 0 or base_per_batch < 0:
            raise ValueError("Invalid composition; check batch_size and adv_ratio")

        self.base_len = int(base_len)
        self.adv_len = int(adv_len)
        self.batch_size = batch_size
        self.adv_per_batch = adv_per_batch
        self.base_per_batch = base_per_batch
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._user_g = generator  # may be None

        # Pre-compute how many full batches we can make *without* reusing samples.
        self._num_batches = self._compute_num_batches()

    # --------------------------- internal utilities ---------------------------

    def _compute_num_batches(self) -> int:
        """Compute a finite number of full batches based on pool sizes.

        We never reuse indices. Thus the number of batches is limited by how
        many full chunks of required size are available from each pool.
        """
        if self.base_per_batch == 0 and self.adv_per_batch == 0:
            return 0  # degenerate, nothing to draw

        if self.base_per_batch == 0:
            # Only adversarial samples are used.
            adv_batches = (
                self.adv_len // self.adv_per_batch if self.adv_per_batch > 0 else 0
            )
            return adv_batches

        if self.adv_per_batch == 0:
            # Only base samples are used.
            base_batches = (
                self.base_len // self.base_per_batch if self.base_per_batch > 0 else 0
            )
            return base_batches

        # General case: both pools are used every batch.
        base_batches = self.base_len // self.base_per_batch
        adv_batches = self.adv_len // self.adv_per_batch
        n = min(base_batches, adv_batches)

        if not self.drop_last:
            # We still only emit *full* batches to keep batch size constant,
            # so drop_last has no effect on count here (kept for API parity).
            pass

        return max(0, int(n))

    def _python_rng(self) -> random.Random:
        """Derive a Python RNG. If a torch.Generator is given, seed from it."""
        if self._user_g is None:
            return random
        # Draw a seed deterministically from the provided generator.
        seed = torch.randint(0, 2**31 - 1, (1,), generator=self._user_g).item()
        return random.Random(int(seed))

    # --------------------------------- API ------------------------------------

    def __len__(self) -> int:
        # Number of batches yielded by __iter__.
        return self._num_batches

    def __iter__(self) -> Iterator[BatchIndices]:
        if self._num_batches == 0:
            return
            yield  # pragma: no cover (generator formality)

        rng = self._python_rng()

        # Build index pools without reuse.
        base_idx = list(range(self.base_len))
        adv_idx = list(range(self.adv_len))
        if self.shuffle:
            rng.shuffle(base_idx)
            rng.shuffle(adv_idx)

        bptr = 0
        aptr = 0

        for _ in range(self._num_batches):
            # Safety checks (should always hold due to _num_batches computation).
            if bptr + self.base_per_batch > len(base_idx):
                break
            if aptr + self.adv_per_batch > len(adv_idx):
                break

            batch: BatchIndices = []

            if self.base_per_batch:
                bslice = base_idx[bptr : bptr + self.base_per_batch]
                bptr += self.base_per_batch
                batch.extend([("base", i) for i in bslice])

            if self.adv_per_batch:
                aslice = adv_idx[aptr : aptr + self.adv_per_batch]
                aptr += self.adv_per_batch
                batch.extend([("adv", i) for i in aslice])

            # In-batch shuffle so labels are not grouped by source.
            if self.shuffle and len(batch) > 1:
                rng.shuffle(batch)

            # Yield exactly batch_size elements every time.
            # Tests assume a constant batch size (e.g., 20).
            if len(batch) == self.batch_size:
                yield batch
            else:
                # If this ever happens, stop to avoid partial batches.
                break
