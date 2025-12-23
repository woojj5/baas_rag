"""Latency measurement utilities."""
import time
from typing import Tuple, TypeVar, Coroutine

T = TypeVar('T')


async def measure_llm_latency(coro: Coroutine[None, None, T]) -> Tuple[T, float]:
    """
    Measure LLM call latency.
    
    Args:
        coro: Async coroutine to measure
        
    Returns:
        Tuple of (result, latency_ms)
    """
    start_time = time.time()
    result = await coro
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000.0
    return result, latency_ms

