"""Explicit estimator choices independent of model names and trainer APIs."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CreditOptions:
    ridge_epsilon: float = 1.0e-8
    sigma_scope: str = "response"
    stats_scope: str = "global"
    bounded_packing: bool = False
    distance_chunk_size: int = 128

    @classmethod
    def from_mapping(cls, values=None):
        options = cls(**dict(values or {}))
        if not math.isfinite(options.ridge_epsilon) or options.ridge_epsilon <= 0:
            raise ValueError('credit_estimator.ridge_epsilon must be finite and positive')
        if options.sigma_scope not in {'response', 'prompt', 'micro_batch'}:
            raise ValueError('credit_estimator.sigma_scope must be response, prompt or micro_batch')
        if options.stats_scope not in {'global', 'rank'}:
            raise ValueError('credit_estimator.stats_scope must be global or rank')
        if not isinstance(options.bounded_packing, bool):
            raise ValueError('credit_estimator.bounded_packing must be boolean')
        if not isinstance(options.distance_chunk_size, int) or options.distance_chunk_size < 0:
            raise ValueError('credit_estimator.distance_chunk_size must be a nonnegative integer')
        return options
