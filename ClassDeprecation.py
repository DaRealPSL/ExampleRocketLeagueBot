import warnings
import functools

def deprecated_class(reason: str):
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(
                f"{cls.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator
