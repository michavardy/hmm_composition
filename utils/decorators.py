import time
import psutil
import torch
import functools

def log_time_and_memory(logger):
    """
    Decorator factory that logs time and memory for a function using a provided logger.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()

            # CPU memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 ** 2)  # MB

            # GPU memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                mem_before_gpu = torch.cuda.memory_allocated() / (1024 ** 2)

            # Call the actual function
            result = func(*args, **kwargs)

            # Elapsed time
            elapsed = time.time() - start_time

            # CPU memory after
            mem_after = process.memory_info().rss / (1024 ** 2)
            # GPU peak memory
            if torch.cuda.is_available():
                mem_peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)
                logger.info(
                    f"[{func.__name__}] Time: {elapsed:.3f}s | "
                    f"CPU Δ: {mem_after - mem_before:.2f} MB | "
                    f"GPU peak: {mem_peak_gpu:.2f} MB"
                )
            else:
                logger.info(
                    f"[{func.__name__}] Time: {elapsed:.3f}s | "
                    f"CPU Δ: {mem_after - mem_before:.2f} MB"
                )

            return result

        return wrapper

    return decorator