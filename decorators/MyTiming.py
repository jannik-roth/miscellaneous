import functools
from time import perf_counter
from typing import Any, Callable

# from https://github.com/ArjanCodes/2023-decorator
def MyTiming(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to time the execution of any function
    Useful for benchmarking
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        value = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        print(f"Execution of {func.__name__} took {run_time:.2f} seconds.")
        return value
    return wrapper