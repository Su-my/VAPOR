# SPDX-License-Identifier: Apache-2.0
"""Public algorithm settings, separate from deployment details."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class VAPORConfig:
    rank: int = 16
    cache_age: int = 512
    threshold: float = 0.10
    consistency_cutoff: float = 0.75
    low_consistency_threshold: float = 0.20
    calibrate: bool = True
    max_new_tokens: int = 8192
    prefill_query_block: int = 512
    prefill_chunk_threshold: int = 8192

    def __post_init__(self):
        if self.rank not in (4, 8, 16, 32, 64, 128):
            raise ValueError("Supported ranks: 4, 8, 16, 32, 64, 128")
        for name in ("cache_age", "max_new_tokens", "prefill_query_block", "prefill_chunk_threshold"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cache_age < 3 or self.max_new_tokens > 8192:
            raise ValueError("Require cache_age >= 3 and max_new_tokens <= 8192")
        for name in ("threshold", "consistency_cutoff", "low_consistency_threshold"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.calibrate and any(t not in (.05, .10, .15, .20, .30, .40)
                                  for t in (self.threshold, self.low_consistency_threshold)):
            raise ValueError("Calibrated thresholds must be supported by the research backend")

    @classmethod
    def internvl(cls, **overrides):
        values = dict(cache_age=256, threshold=.05, consistency_cutoff=.70,
                      low_consistency_threshold=.15)
        return cls(**(values | overrides))

    def select_threshold(self, consistency):
        if not self.calibrate:
            return self.threshold
        if not math.isfinite(consistency) or not 0 <= consistency <= 1:
            raise ValueError("Consistency must be finite and in [0, 1]")
        return self.low_consistency_threshold if consistency < self.consistency_cutoff else self.threshold

    def legacy_setting(self):
        return dict(layers=[0, 35], rank=self.rank, d=self.cache_age,
                    m1=self.threshold, confirmation_m1=self.threshold, m3_base=self.threshold)
