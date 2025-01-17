import warnings
import functools


# Wrapper to indicate deprecation (via Claude)
def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated."""

    # # Example usage with decorator
    # @deprecated
    # def legacy_function():
    #     return "Legacy function result"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Call to deprecated function {func.__name__}. "
            "This function will be removed in future versions.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper
